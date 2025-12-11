import argparse
import base64
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch text LLM requester")
    parser.add_argument("--model", required=True, help="Model name to query")
    parser.add_argument("--base_url", required=True, help="Inference service base URL")
    parser.add_argument("--tasks", nargs="+", required=True, help="Task names to evaluate")
    parser.add_argument("--data_root", default="data", help="Directory containing task folders")
    parser.add_argument("--api_key", help="API key (optional, falls back to dotenv and environment variable)")
    parser.add_argument("--threads", type=int, default=8, help="Thread pool size for requests")
    parser.add_argument("--output_root", default="llm_results", help="Root directory for result outputs")
    parser.add_argument("--max_request_attempts", type=int, default=1, help="Maximum request attempts per entry")
    parser.add_argument(
        "--request_attempt_delay",
        type=float,
        default=0.0,
        help="Seconds to wait between request attempts",
    )
    parser.add_argument("--no_images", action="store_true", help="Do not include images in requests")
    parser.add_argument("--provide_options", action="store_true", help="Include entry options in the question prompt")
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


def extract_answer(raw_text: Optional[str]) -> Optional[str]:
    if not raw_text:
        return None
    lower_text = raw_text.lower()
    start_tag = "<answer>"
    end_tag = "</answer>"
    start_idx = lower_text.rfind(start_tag)
    if start_idx == -1:
        return None
    end_idx = lower_text.find(end_tag, start_idx)
    if end_idx == -1:
        return None
    content_start = start_idx + len(start_tag)
    return raw_text[content_start:end_idx].strip()

FORMAT_ANSWER_PROMPT_TEMPLATE = """{question}

Enclose the final answer within <answer> </answer> tags, i.e., <answer>answer here</answer>.
Note that using tools (e.g., python) is not allowed."""

def request_entry(
    client: OpenAI,
    model_name: str,
    entry: Dict[str, Any],
    dataset_root: Path,
    include_image: bool,
    provide_options: bool,
) -> Dict[str, Any]:
    question_value = entry.get('question')
    if not question_value:
        raise ValueError("Entry missing question field")
    question = str(question_value)

    if provide_options:
        options_list = entry.get("options")
        if isinstance(options_list, list) and options_list:
            formatted = ", ".join(str(item) for item in options_list)
            question = f"{question}\nOptions: {formatted}"

    question = FORMAT_ANSWER_PROMPT_TEMPLATE.format(
        question=question,
    )

    message_content: List[Dict[str, Any]] = [{"type": "text", "text": question}]

    if include_image:
        image_path = resolve_image_path(entry, dataset_root)
        if image_path:
            entry["request_image_path"] = str(image_path)
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_to_base64(image_path)}"},
            })
        else:
            logging.warning(
                "Entry %s: image path could not be resolved",
                entry.get("id") or entry.get("index") or entry.get("question_id"),
            )

    messages = [{
        "role": "user",
        "content": message_content if include_image else question,
    }]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=8192
    )

    raw_content = response.choices[0].message.content or ""
    entry["prediction"] = raw_content
    extracted = extract_answer(raw_content)
    entry["extracted_answer"] = extracted
    answer_value = entry.get("answer")
    if extracted is None:
        entry["is_correct"] = False
    else:
        expected = str(answer_value).strip().lower()
        entry["is_correct"] = extracted.strip().lower() == expected
    return entry


def request_entries_for_indices(
    entries: List[Dict[str, Any]],
    indices: List[int],
    client: OpenAI,
    model_name: str,
    dataset_root: Path,
    include_image: bool,
    provide_options: bool,
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
                provide_options,
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
                    print(f"Error processing response for item {idx}: {exc}")
                finally:
                    progress.update()
        finally:
            progress.close()

    return results, errors


def write_outputs(entries: Iterable[Dict[str, Any]], dataset_dir: Path) -> None:
    entries_list = list(entries)
    total = len(entries_list)
    correct = sum(1 for entry in entries_list if entry.get("is_correct") is True)
    accuracy = correct / total if total else 0.0
    metadata = {
        "dataset_name": dataset_dir.name,
        "total_entries": total,
        "correct_count": correct,
        "accuracy": accuracy,
        "generated_at": datetime.now().isoformat(),
    }
    output_path = dataset_dir / "result.json"
    save_json({"metadata": metadata, "entries": entries_list}, output_path)
    logging.info("Results written to %s (accuracy %.2f%%)", output_path, accuracy * 100)


def process_dataset(
    dataset_name: str,
    entries: List[Dict[str, Any]],
    dataset_path: Path,
    dataset_dir: Path,
    client: OpenAI,
    model_name: str,
    dataset_root: Path,
    include_image: bool,
    provide_options: bool,
    threads: int,
    max_attempts: int,
    attempt_delay: float,
) -> None:
    pending_indices = list(range(len(entries)))
    last_errors: Dict[int, str] = {}
    attempts = max(1, max_attempts)

    logging.info("%s: loaded %d entries from %s", dataset_name, len(entries), dataset_path)

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
            provide_options,
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

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = build_run_directory(output_root, args.model)
    logging.info("Outputs will be stored under %s", run_dir)

    client = OpenAI(api_key=api_key.strip(), base_url=args.base_url)
    threads = max(args.threads, 1)
    attempts = max(args.max_request_attempts, 1)
    attempt_delay = max(args.request_attempt_delay, 0.0)
    include_image = not args.no_images
    provide_options = args.provide_options

    data_root = Path(args.data_root)

    for dataset_name in args.tasks:
        dataset_path = data_root / dataset_name / "data.json"
        if not dataset_path.is_file():
            logging.error("Task %s: dataset not found at %s", dataset_name, dataset_path)
            continue
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
                dataset_path=dataset_path,
                dataset_dir=dataset_dir,
                client=client,
                model_name=args.model,
                dataset_root=dataset_path.parent,
                include_image=include_image,
                    provide_options=provide_options,
                threads=threads,
                max_attempts=attempts,
                attempt_delay=attempt_delay,
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception("%s: unexpected error: %s", dataset_name, exc)


if __name__ == "__main__":
    main()