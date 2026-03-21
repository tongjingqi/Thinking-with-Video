import argparse
import base64
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]


GENERATION_SUFFIX_PATTERN = re.compile(
    r"\s*Generate a video showing the solution process\s*\.?\s*$",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch VLM requester for text-centric evaluation pipeline")
    parser.add_argument("--model", required=True, help="Model name to query")
    parser.add_argument("--base_url", required=True, help="Inference service base URL")
    parser.add_argument("--dataset_root", required=True, help="Root directory that contains dataset folders")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset names to evaluate")
    parser.add_argument("--api_key", help="API key (optional, falls back to dotenv)")
    parser.add_argument("--no_images", action="store_true", help="Do not include images in requests")
    parser.add_argument("--threads", type=int, default=4, help="Thread pool size for requests")
    parser.add_argument("--output_root", default="result", help="Root directory for test outputs")
    parser.add_argument("--max_request_attempts", type=int, default=1, help="Maximum attempts per request (includes first attempt)")
    parser.add_argument("--request_attempt_delay", type=float, default=0.0, help="Seconds to wait between request attempts")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


def load_json(json_path: Path) -> Any:
    with json_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(data: Any, json_path: Path, indent: int = 2) -> None:
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=indent, ensure_ascii=False)


class _NullProgress:
    def __init__(self, total: int) -> None:
        self.total = total

    def update(self, n: int = 1) -> None:
        return

    def close(self) -> None:
        return


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    data = load_json(dataset_path)
    if not isinstance(data, list):
        raise ValueError(f"Dataset at {dataset_path} is not a list of entries")
    return data


def build_run_directory(output_root: Path, model_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dirname = f"{model_name}_{timestamp}"
    run_dir = output_root / run_dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_dataset_dir(run_dir: Path, dataset_name: str) -> Path:
    dataset_dir = run_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def resolve_image_path(entry: Dict[str, Any], dataset_root: Path) -> Optional[Path]:
    image_value = entry.get("image")
    if not image_value:
        return None
    candidate = Path(image_value)
    if candidate.is_file():
        return candidate
    candidate = dataset_root / image_value
    if candidate.is_file():
        return candidate
    return None


def build_messages(text_prompt: str, image_base64: Optional[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt}]
    if image_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
        })
    return [{"role": "user", "content": content}]


def _format_choices(entry: Dict[str, Any]) -> str:
    choices = entry.get("choices")
    if isinstance(choices, dict) and choices:
        lines = ["Options:"]
        for key, value in choices.items():
            lines.append(f"{key}. {value}")
        return "\n".join(lines)

    options = entry.get("options")
    if isinstance(options, list) and options:
        lines = ["Options:"]
        for item in options:
            lines.append(str(item))
        return "\n".join(lines)

    prompt = entry.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        # Fallback for datasets where options are embedded in prompt text.
        match = re.search(
            r"(?is)\bOptions:\s*(.*?)(?:\n\s*Correct\s*Answer\s*:|\n\s*\*\*Requirements:\*\*|\Z)",
            prompt,
        )
        if match:
            block = match.group(1).strip()
            option_lines: List[str] = []
            for raw_line in block.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if re.match(r"^[A-Za-z][\.|\)]\s+", line) or re.match(r"^[A-Za-z]\s*[:：]\s+", line):
                    option_lines.append(line)
            if option_lines:
                return "\n".join(["Options:", *option_lines])

    return ""


def build_question(entry: Dict[str, Any]) -> Tuple[str, str]:
    question_text = ""
    used_prompt_fallback = False
    prompt_text_raw = entry.get("prompt") if isinstance(entry.get("prompt"), str) else ""

    question = entry.get("question")
    if isinstance(question, str) and question.strip():
        question_text = question.strip()
    else:
        prompt = entry.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            used_prompt_fallback = True
            prompt_clean = GENERATION_SUFFIX_PATTERN.sub("", prompt.strip())
            question_match = re.search(r"(?is)\bQuestion\s*:\s*(.*)", prompt_clean)
            if question_match:
                question_text = question_match.group(1).strip()
            else:
                question_text = prompt_clean

    if not question_text:
        raise ValueError("Entry missing usable question/prompt field")

    choices_block = _format_choices(entry)
    require_option_letter_only = bool(choices_block) or used_prompt_fallback

    if choices_block:
        question_text = f"{question_text}\n\n{choices_block}"

    if require_option_letter_only:
        question_text = (
            f"{question_text}\n\n"
            "Note that if there are options provided above, your answer must be only the option letter (e.g., A, B, C, D). Give the final answer otherwise."
        )

    prompt_text = FORMAT_ANSWER_PROMPT_TEMPLATE.format(question=question_text)
    return question_text, prompt_text


def extract_answer(raw_text: Optional[str]) -> Optional[str]:
    if not raw_text:
        return None
    matches = list(re.finditer(r"<answer>(.*?)</answer>", raw_text, flags=re.DOTALL | re.IGNORECASE))
    if not matches:
        return None
    return matches[-1].group(1).strip()

FORMAT_ANSWER_PROMPT_TEMPLATE = """{question}

Enclose the final answer within <answer> </answer> tags, i.e., <answer>answer here</answer>.
Note that using tools (e.g., python) is not allowed."""

def request_entry(
    client: OpenAI,
    model_name: str,
    entry: Dict[str, Any],
    dataset_root: Path,
    include_image: bool,
) -> Dict[str, Any]:
    entry_id = entry.get("id") or entry.get("index") or entry.get("question_id")
    prefix = f"Entry {entry_id}" if entry_id is not None else "Entry"

    modified_question, prompt_text = build_question(entry)
    if "question_original" not in entry:
        entry["question_original"] = entry.get("question")
    entry["question"] = modified_question

    image_b64: Optional[str] = None

    if include_image:
        image_path = resolve_image_path(entry, dataset_root)
        if image_path:
            entry["request_image_path"] = str(image_path)
            image_b64 = image_to_base64(image_path)
        else:
            logging.warning("%s: image path %r could not be resolved", prefix, entry.get("image"))

    messages = build_messages(prompt_text, image_b64)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0
    )

    raw_content = response.choices[0].message.content or ""
    entry["response_raw"] = raw_content
    entry["prediction"] = raw_content
    entry["extracted_answer"] = extract_answer(raw_content)
    return entry


def request_entries_for_indices(
    entries: List[Dict[str, Any]],
    indices: List[int],
    client: OpenAI,
    model_name: str,
    dataset_root: Path,
    include_image: bool,
    threads: int,
    attempt_label: str,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, str]]:
    results: Dict[int, Dict[str, Any]] = {}
    errors: Dict[int, str] = {}

    if not indices:
        return results, errors

    with ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_index = {
            executor.submit(
                request_entry,
                client,
                model_name,
                dict(entries[idx]),
                dataset_root,
                include_image,
            ): idx
            for idx in indices
        }
        total = len(future_to_index)
        progress = (
            tqdm(total=total, desc=attempt_label, unit='req')
            if tqdm  # type: ignore[truthy-bool]
            else _NullProgress(total)
        )
        try:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    errors[idx] = str(exc)
                finally:
                    progress.update()
        finally:
            progress.close()

    return results, errors


def write_outputs(entries: Iterable[Dict[str, Any]], dataset_dir: Path) -> None:
    responses_path = dataset_dir / "responses.json"
    entry_list = list(entries)
    save_json(entry_list, responses_path)
    logging.info("Responses written to %s", responses_path)

    text_only_results: List[Dict[str, Any]] = []
    for entry in entry_list:
        question_id = entry.get("id") or entry.get("index") or entry.get("question_id")
        text_only_results.append({
            "question_id": question_id,
            "question": entry.get("question"),
            "correct_answer": entry.get("correct_answer") or entry.get("answer"),
            "prediction": entry.get("prediction"),
            "extracted_answer": entry.get("extracted_answer"),
            "request_error": entry.get("request_error"),
        })
    text_only_path = dataset_dir / "answers.json"
    save_json(text_only_results, text_only_path)
    logging.info("Text answers written to %s", text_only_path)


def validate_entries(entries: Iterable[Dict[str, Any]]) -> None:
    invalid_entries: List[int] = []
    for idx, entry in enumerate(entries):
        question = entry.get("question")
        prompt = entry.get("prompt")
        has_question = isinstance(question, str) and bool(question.strip())
        has_prompt = isinstance(prompt, str) and bool(prompt.strip())
        if not has_question and not has_prompt:
            logging.error("Entry %d missing usable question/prompt fields", idx)
            invalid_entries.append(idx)
    if invalid_entries:
        raise SystemExit(
            "Input validation failed: see errors above. Aborting requests to avoid inconsistent outputs."
        )


def process_dataset(
    dataset_name: str,
    entries: List[Dict[str, Any]],
    dataset_path_obj: Path,
    dataset_dir: Path,
    client: OpenAI,
    model_name: str,
    include_image: bool,
    threads: int,
    max_attempts: int,
    attempt_delay: float,
) -> None:
    dataset_root = dataset_path_obj.parent
    pending_indices = list(range(len(entries)))
    last_errors: Dict[int, str] = {}
    attempts = max(1, max_attempts)

    validate_entries(entries)
    write_outputs(entries, dataset_dir)
    logging.info("%s: loaded %d entries from %s", dataset_name, len(entries), dataset_path_obj)

    for attempt in range(1, attempts + 1):
        if not pending_indices:
            break

        logging.info(
            "%s: requesting %d entries (attempt %d/%d)",
            dataset_name,
            len(pending_indices),
            attempt,
            attempts,
        )

        attempt_label = f"{dataset_name} | requests {attempt}/{attempts}"
        attempt_results, attempt_errors = request_entries_for_indices(
            entries,
            pending_indices,
            client,
            model_name,
            dataset_root,
            include_image,
            threads,
            attempt_label,
        )

        for idx, message in attempt_errors.items():
            last_errors[idx] = message

        for idx, result in attempt_results.items():
            entries[idx].update(result)

        write_outputs(entries, dataset_dir)

        pending_indices = [idx for idx in pending_indices if idx not in attempt_results]

        if pending_indices and attempt < attempts and attempt_delay > 0:
            logging.info(
                "%s: %d entries pending, sleeping %.2f seconds before next attempt",
                dataset_name,
                len(pending_indices),
                attempt_delay,
            )
            time.sleep(attempt_delay)

    if pending_indices:
        for idx in pending_indices:
            error_message = last_errors.get(idx, "Unknown error")
            entries[idx]["request_error"] = error_message
            if "prediction" not in entries[idx]:
                entries[idx]["prediction"] = f"Error: {error_message}"
        logging.warning(
            "%s: %d entries failed after %d attempts",
            dataset_name,
            len(pending_indices),
            attempts,
        )
        write_outputs(entries, dataset_dir)
    else:
        logging.info("%s: all entries processed successfully", dataset_name)


def main() -> None:
    args = parse_args()
    setup_logging()

    if not args.api_key:
        load_dotenv()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("API key not provided and OPENAI_API_KEY not set")

    include_image = not args.no_images
    logging.info("Starting batch run for model %s", args.model)
    run_dir = build_run_directory(Path(args.output_root), args.model)
    logging.info("Text responses will be stored under %s", run_dir)

    client = OpenAI(api_key=api_key.strip(), base_url=args.base_url)
    threads = max(args.threads, 1)
    attempts = max(args.max_request_attempts, 1)
    attempt_delay = max(args.request_attempt_delay, 0.0)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        raise RuntimeError(f"Dataset root {dataset_root} does not exist or is not a directory")

    for dataset_name in args.datasets:
        dataset_path = dataset_root / dataset_name / "data.json"
        if not dataset_path.is_file():
            logging.error(
                "Dataset %s: expected data.json at %s but file was not found; skipping",
                dataset_name,
                dataset_path,
            )
            continue

        logging.info("Loading dataset %s from %s", dataset_name, dataset_path)
        try:
            entries = load_dataset(dataset_path)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Failed to load dataset %s: %s", dataset_name, exc)
            continue

        dataset_dir = ensure_dataset_dir(run_dir, dataset_name)

        try:
            process_dataset(
                dataset_name=dataset_name,
                entries=entries,
                dataset_path_obj=dataset_path,
                dataset_dir=dataset_dir,
                client=client,
                model_name=args.model,
                include_image=include_image,
                threads=threads,
                max_attempts=attempts,
                attempt_delay=attempt_delay,
            )
        except SystemExit as exc:
            logging.error("%s: %s", dataset_name, exc)
        except Exception as exc:  # noqa: BLE001
            logging.exception("%s: unexpected error: %s", dataset_name, exc)


if __name__ == "__main__":
    main()