import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

try:
	from tqdm import tqdm
except ImportError:
	tqdm = None  # type: ignore[assignment]


PROMPT_TEMPLATE = """Given a question with the ground truth answer and a predicted answer, determine if the predicted answer is correct.
You should only focus on the final answer to the question in the predicted answer, without considering the reasoning process (if any).
Consider the predicted answer incorrect if it does not give the final answer, or if the final answer does not match the ground truth answer.
Be strict but reasonable - minor differences are acceptable if the core answer is correct (e.g., having extra or missing units is permissible).
Enclose your judgment ("Yes" or "No") in <correctness> and </correctness> tags, namely <correctness>Yes</correctness> or <correctness>No</correctness>.

Question: {question}
Ground Truth Answer: {ground_truth_answer}

Predicted Answer: {predicted_answer}
"""


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate transcription correctness for inference results")
	parser.add_argument('--result_run_dir', required=True, help='Directory containing inference run outputs (e.g. result/.../run_name)')
	parser.add_argument('--result_file', help='Path to a specific result file to evaluate (optional)')
	parser.add_argument('--output_root', default='eval_result', help='Root directory for evaluation outputs')
	parser.add_argument('--model', required=True, help='Evaluation model name')
	parser.add_argument('--base_url', required=True, help='Evaluation service base URL')
	parser.add_argument('--api_key', help='API key (optional, falls back to environment/.env)')
	parser.add_argument('--threads', type=int, default=4, help='Thread pool size for evaluation requests')
	return parser.parse_args()


def setup_logging() -> None:
	logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')


def load_entries(results_path: Path) -> List[Dict[str, Any]]:
	with results_path.open('r', encoding='utf-8') as fp:
		data = json.load(fp)
	if not isinstance(data, list):
		raise ValueError(f'{results_path} does not contain a list of entries')
	return data


class _NullProgress:
	def __init__(self, total: int) -> None:
		self.total = total

	def update(self, n: int = 1) -> None:
		return

	def close(self) -> None:
		return


def extract_content(text: Any, *, tag: str, special_tag: bool = False) -> str:
	if not isinstance(text, str):
		return ""
	if not special_tag:
		pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", flags=re.IGNORECASE | re.DOTALL)
	else:
		pattern = re.compile(rf"\\<{tag}\\>(.*?)\\</{tag}\\>", flags=re.IGNORECASE | re.DOTALL)
	matches = list(pattern.finditer(text))
	if not matches:
		return ""
	return matches[-1].group(1).strip()

def extract_final_answer(text: Any) -> str:
	# matches = list(re.finditer(r"<answer>(.*?)</answer>", text, flags=re.DOTALL))
	# if matches:
	# 	return matches[-1].group(1).strip()
	# matches = list(re.finditer(r"\\<answer\\>(.*?)\\</answer\\>", text, flags=re.DOTALL))
	# if matches:
	# 	return matches[-1].group(1).strip()

	result = extract_content(text, tag='answer')
	if result:
		return result
	result = extract_content(text, tag='answer', special_tag=True)
	if result:
		return result
	
	return text if isinstance(text, str) else ""


def build_prompt(
	entry: Dict[str, Any],
	answer_key: str = 'answer',
	predicted_key: str = 'transcription'
) -> str:
	# question = entry['question'] if 'question' in entry else entry['prompt'].split('Generate a video showing the solution process')[0].strip()
	# question = entry['prompt'].split('**Content to Display:**')[1].split('Correct Answer: _____ (fill this in after explanation)')[0].strip()
	question = entry.get('question', '')
	answer = entry.get(answer_key)
	if answer is None and answer_key != 'answer':
		answer = entry.get('answer')
	predicted = entry.get(predicted_key)
	if isinstance(predicted, str):
		predicted = extract_final_answer(predicted)
	return PROMPT_TEMPLATE.format(
		question=question,
		ground_truth_answer=answer,
		predicted_answer=predicted,
	)


def try_manual_evaluation(entry: Dict[str, Any], final_prediction: Any, answer_value: str) -> Optional[bool]:
	if not isinstance(final_prediction, str):
		return None

	pred_token = final_prediction.strip()
	answer_token = answer_value.strip()
	if not pred_token or not answer_token:
		return None
	if re.search(r"\s", pred_token) or re.search(r"\s", answer_token):
		return None

	if pred_token.isdigit() and answer_token.isdigit():
		is_correct = pred_token == answer_token
		entry['evaluation_correctness'] = 'Yes' if is_correct else 'No'
		entry['evaluation_raw'] = 'Manual token comparison (digit exact match)'
		entry['evaluation_method'] = 'manual_token_compare'
		return is_correct

	if pred_token.isalpha() and answer_token.isalpha() and len(pred_token) == 1 and len(answer_token) == 1:
		is_correct = pred_token.lower() == answer_token.lower()
		entry['evaluation_correctness'] = 'Yes' if is_correct else 'No'
		entry['evaluation_raw'] = 'Manual token comparison (single-letter option)'
		entry['evaluation_method'] = 'manual_token_compare'
		return is_correct

	if pred_token.isalpha() and answer_token.isalpha() and pred_token.lower() == answer_token.lower():
		entry['evaluation_correctness'] = 'Yes'
		entry['evaluation_raw'] = 'Manual token comparison (case-insensitive)'
		entry['evaluation_method'] = 'manual_token_compare'
		return True

	return None


def evaluate_entries(
	entries: List[Dict[str, Any]],
	client: OpenAI,
	model: str,
	threads: int,
	stage_label: str,
) -> Tuple[int, List[Tuple[int, str]], List[Tuple[int, str]]]:
	correct_count = 0
	failures: List[Tuple[int, str]] = []
	parse_errors: List[Tuple[int, str]] = []

	def run_eval(prompt: str) -> str:
		response = client.chat.completions.create(
			model=model,
			messages=[{"role": "user", "content": prompt}],
			temperature=0.0
		)
		return response.choices[0].message.content or ''

	progress = (
		tqdm(total=len(entries), desc=f"{stage_label} | eval", unit="req")
		if tqdm  # type: ignore[truthy-bool]
		else _NullProgress(len(entries))
	)
	future_to_idx: Dict[Any, int] = {}
	try:
		with ThreadPoolExecutor(max_workers=max(threads, 1)) as executor:
			for idx, entry in enumerate(entries):
				prediction_value = entry.get('prediction')
				final_prediction = extract_final_answer(prediction_value) if isinstance(prediction_value, str) else prediction_value
				entry['final_prediction'] = final_prediction
				answer_raw = entry.get('correct_answer', entry.get('answer', ''))
				answer_value = str(answer_raw).strip()

				manual_result = try_manual_evaluation(entry, final_prediction, answer_value)
				if manual_result is not None:
					if manual_result:
						correct_count += 1
					progress.update()
					continue

				entry['evaluation_method'] = 'model_judge'

				future = executor.submit(
					run_eval,
					build_prompt(
						entry,
						answer_key='correct_answer',
						predicted_key='prediction'
					)
				)
				future_to_idx[future] = idx

			for future in as_completed(future_to_idx):
				idx = future_to_idx[future]
				entry = entries[idx]
				try:
					raw_response = future.result()
					entry['evaluation_raw'] = raw_response
					correctness = raw_response.split('<correctness>')[-1].split('</correctness>')[0].strip()
					if not correctness:
						parse_errors.append((idx, 'Failed to parse correctness tag from evaluator response'))
						entry['evaluation_correctness'] = 'Error'
						continue
					entry['evaluation_correctness'] = correctness
					if correctness == 'Yes':
						correct_count += 1
				except Exception as exc:
					failures.append((idx, str(exc)))
					entry['evaluation_error'] = str(exc)
				finally:
					progress.update()
	finally:
		progress.close()

	return correct_count, failures, parse_errors


def write_dataset_report(
	entries: List[Dict[str, Any]],
	output_path: Path,
	model: str,
	base_url: str,
	correct_count: int,
) -> None:
	summary = {
		'evaluation_time': datetime.now().isoformat(),
		'model': model,
		'base_url': base_url,
		'total_samples': len(entries),
		'correct_samples': correct_count,
		'accuracy': correct_count / len(entries) if entries else 0.0,
		'entries': entries,
	}
	with output_path.open('w', encoding='utf-8') as fp:
		json.dump(summary, fp, indent=2, ensure_ascii=False)


def main() -> None:
	args = parse_args()
	setup_logging()

	if not args.api_key:
		load_dotenv()
	api_key = args.api_key or os.getenv('OPENAI_API_KEY')
	if not api_key:
		raise RuntimeError('API key not provided and OPENAI_API_KEY not set')

	client = OpenAI(api_key=api_key, base_url=args.base_url)

	result_run_dir = Path(args.result_run_dir).resolve()
	if not result_run_dir.exists() or not result_run_dir.is_dir():
		raise FileNotFoundError(f'Result run directory {result_run_dir} not found or not a directory')

	output_root = Path(args.output_root).resolve()
	output_root.mkdir(parents=True, exist_ok=True)
	run_output_dir = output_root / result_run_dir.name
	run_output_dir.mkdir(parents=True, exist_ok=True)

	results_to_process: List[Path] = []
	if args.result_file:
		result_file_path = Path(args.result_file).resolve()
		if not result_file_path.is_file():
			raise FileNotFoundError(f"Specified result file {result_file_path} not found.")
		results_to_process.append(result_file_path)
	else:
		dataset_dirs = [p for p in result_run_dir.iterdir() if p.is_dir()]
		if not dataset_dirs:
			logging.warning("No dataset directories found under %s", result_run_dir)
			return
		for p in dataset_dirs:
			if (p / 'results.json').is_file():
				results_to_process.append(p / 'results.json')
			elif (p / 'responses.json').is_file():
				results_to_process.append(p / 'responses.json')

	if not results_to_process:
		logging.warning("No result files found to evaluate.")
		return

	for results_path in results_to_process:
		dataset_name = results_path.parent.name
		if not results_path.is_file():
			logging.warning("Skipping %s (file not found)", results_path)
			continue

		logging.info("Evaluating dataset %s", dataset_name)
		try:
			entries = load_entries(results_path)
		except Exception as exc:
			logging.exception("Failed to load %s: %s", results_path, exc)
			continue

		correct_count, failures, parse_errors = evaluate_entries(
			entries,
			client,
			args.model,
			args.threads,
			dataset_name,
		)

		for idx, message in failures:
			logging.error("%s entry %d evaluation failed: %s", dataset_name, idx, message)
		for idx, message in parse_errors:
			logging.error("%s entry %d response parse failed: %s", dataset_name, idx, message)

		output_path = run_output_dir / f"{dataset_name}.json"
		write_dataset_report(entries, output_path, args.model, args.base_url, correct_count)
		logging.info("Dataset %s evaluation saved to %s", dataset_name, output_path)


if __name__ == '__main__':
	main()
