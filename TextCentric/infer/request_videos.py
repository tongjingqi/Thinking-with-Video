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
from tqdm import tqdm
import requests


REQUIRED_FIELDS = {"id", "question", "prompt", "answer"}


def image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch video requester for evaluation pipeline")
    parser.add_argument("--model", required=True, help="Model name to query")
    parser.add_argument("--base_url", required=True, help="Inference service base URL")
    parser.add_argument("--api_key", help="API key (optional, falls back to dotenv)")
    parser.add_argument("--dataset_root", required=True, help="Root directory that contains dataset folders")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset names to evaluate")
    parser.add_argument("--no_images", action="store_true", help="Do not include images in requests")
    parser.add_argument("--threads", type=int, default=4, help="Thread pool size for requests/downloads")
    parser.add_argument("--output_root", default="result", help="Root directory for test outputs")
    parser.add_argument("--max_request_attempts", type=int, default=1, help="Maximum attempts per video request (includes first attempt)")
    parser.add_argument("--request_attempt_delay", type=float, default=0.0, help="Seconds to wait between request attempts")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Dataset at {dataset_path} is not a list of entries")
    return data


def resolve_image_path(entry: Dict[str, Any], dataset_dir: Path) -> Optional[Path]:
    image_value = entry.get("image")
    if not image_value:
        return None
    candidate = Path(image_value)
    if candidate.is_file():
        return candidate
    candidate = dataset_dir / image_value
    return candidate if candidate.is_file() else None


def build_messages(text_prompt: str, image_base64: Optional[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt}]
    if image_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
        })
    return [{"role": "user", "content": content}]


def request_entry(
    client: OpenAI,
    model_name: str,
    entry: Dict[str, Any],
    dataset_dir: Path,
    include_image: bool,
) -> Dict[str, Any]:
    entry_id = entry.get("id") or entry.get("index") or entry.get("question_id")
    prefix = f"Entry {entry_id}" if entry_id is not None else "Entry"

    text_prompt = entry.get("prompt")
    if not text_prompt:
        raise ValueError(f"{prefix}: missing text_prompt field")

    image_b64: Optional[str] = None
    if include_image:
        image_path = resolve_image_path(entry, dataset_dir)
        if image_path:
            image_b64 = image_to_base64(image_path)
            entry["request_image_path"] = str(image_path)
        else:
            logging.warning("%s: image path %r could not be resolved", prefix, entry.get("image"))

    messages = build_messages(text_prompt, image_b64)

    logging.debug("%s: sending request", prefix)
    response = client.chat.completions.create(model=model_name, messages=messages)
    content = response.choices[0].message.content or ""

    pattern = r"\(https?://[^\s)]+\)"
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"{prefix}: response did not include a video URL")

    video_url = match.group(0)[1:-1]
    if not video_url:
        raise ValueError(f"{prefix}: unable to parse video URL from response")

    entry["response_raw"] = content
    entry["video_url"] = video_url
    return entry


def download_video(video_url: str, target_path: Path) -> None:
    response = requests.get(video_url, timeout=60)
    response.raise_for_status()
    target_path.write_bytes(response.content)


def build_run_directory(output_root: Path, model_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dirname = f"{model_name}_{timestamp}"
    run_dir = output_root / run_dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_dataset_dirs(run_dir: Path, dataset_name: str) -> Dict[str, Path]:
    dataset_dir = run_dir / dataset_name
    videos_dir = dataset_dir / "videos"
    for path in (dataset_dir, videos_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "dataset_dir": dataset_dir,
        "videos_dir": videos_dir,
    }


def request_entries_for_indices(
    entries: List[Dict[str, Any]],
    indices: List[int],
    client: OpenAI,
    model_name: str,
    dataset_dir: Path,
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
                dataset_dir,
                include_image,
            ): idx
            for idx in indices
        }
        progress = tqdm(total=len(future_to_index), desc=attempt_label, unit="req")
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


def download_videos(
    entries: List[Dict[str, Any]],
    dataset_dirs: Dict[str, Path],
    indices: List[int],
    threads: int,
    stage_label: str,
) -> List[Tuple[int, str]]:
    if not indices:
        return []

    def handle_download(idx: int, entry: Dict[str, Any]) -> None:
        video_url = entry["video_url"]
        video_filename = f"entry_{idx}.mp4"
        video_path = dataset_dirs["videos_dir"] / video_filename

        download_video(video_url, video_path)

        entry["video_path"] = str(video_path)
        entry["video_filename"] = video_filename

    download_errors: List[Tuple[int, str]] = []

    with ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_idx = {
            executor.submit(handle_download, idx, entries[idx]): idx for idx in indices
        }
        progress = tqdm(total=len(future_to_idx), desc=f"{stage_label} | downloads", unit="file")
        try:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    entries[idx]["download_error"] = str(exc)
                    download_errors.append((idx, str(exc)))
                finally:
                    progress.update()
        finally:
            progress.close()

    return download_errors


def write_outputs(
    entries: Iterable[Dict[str, Any]],
    dataset_dirs: Dict[str, Path],
    stage_label: str,
) -> None:
    dataset_results_path = dataset_dirs["dataset_dir"] / "responses.json"
    with dataset_results_path.open("w", encoding="utf-8") as fp:
        json.dump(list(entries), fp, indent=2, ensure_ascii=False)
    logging.info("%s: responses written to %s", stage_label, dataset_results_path)

    manifest = build_question_manifest(entries)
    questions_path = dataset_dirs["dataset_dir"] / "questions.json"
    with questions_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)
    logging.info(
        "%s: questions manifest written to %s (%d entries)",
        stage_label,
        questions_path,
        len(manifest),
    )


def log_stage_errors(stage_name: str, errors: List[Tuple[int, str]]) -> None:
    for idx, message in errors:
        logging.error("Entry %d %s failed: %s", idx, stage_name, message)


def validate_entries(entries: Iterable[Dict[str, Any]]) -> None:
    missing_entries: List[int] = []
    for idx, entry in enumerate(entries):
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            logging.error("Entry %d missing fields: %s", idx, ", ".join(sorted(missing)))
            missing_entries.append(idx)
    if missing_entries:
        raise SystemExit(
            "Input validation failed: see errors above. Aborting video requests to avoid inconsistent outputs."
        )


def build_question_manifest(entries: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    manifest: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        video_filename = entry.get("video_filename")
        question = entry.get("question")
        correct_answer = entry.get("correct_answer")
        if not video_filename or not question or not correct_answer:
            continue
        question_id = entry.get("id") or entry.get("index") or entry.get("question_id")
        manifest[video_filename] = {
            "question": question,
            "correct_answer": correct_answer,
            "question_id": question_id if question_id is not None else video_filename,
        }
    return manifest


def process_dataset(
    dataset_name: str,
    entries: List[Dict[str, Any]],
    dataset_path_obj: Path,
    dataset_dirs: Dict[str, Path],
    request_client: OpenAI,
    request_model: str,
    include_image: bool,
    threads: int,
    request_max_attempts: int,
    request_attempt_delay: float,
) -> None:
    dataset_root = dataset_path_obj.parent
    stage_label = dataset_name

    validate_entries(entries)
    pending_indices = list(range(len(entries)))
    last_request_errors: Dict[int, str] = {}
    max_attempts = max(1, request_max_attempts)
    aggregate_download_errors: List[Tuple[int, str]] = []

    write_outputs(entries, dataset_dirs, stage_label)

    for attempt in range(1, max_attempts + 1):
        if not pending_indices:
            break

        logging.info(
            "%s: requesting %d videos (attempt %d/%d)",
            dataset_name,
            len(pending_indices),
            attempt,
            max_attempts,
        )
        attempt_label = f"{stage_label} | requests {attempt}/{max_attempts}"
        attempt_results, attempt_errors = request_entries_for_indices(
            entries,
            pending_indices,
            request_client,
            request_model,
            dataset_root,
            include_image,
            threads,
            attempt_label,
        )

        for idx, error_message in attempt_errors.items():
            last_request_errors[idx] = error_message

        success_indices = list(attempt_results.keys())
        for idx, result in attempt_results.items():
            entries[idx].update(result)

        if success_indices:
            logging.info(
                "%s: downloading %d videos from attempt %d",
                dataset_name,
                len(success_indices),
                attempt,
            )
            download_errors = download_videos(
                entries,
                dataset_dirs,
                success_indices,
                threads,
                stage_label,
            )
            if download_errors:
                log_stage_errors("download", download_errors)
                aggregate_download_errors.extend(download_errors)
        else:
            logging.info("%s: no successful requests in attempt %d", dataset_name, attempt)

        write_outputs(entries, dataset_dirs, stage_label)

        pending_indices = [idx for idx in pending_indices if idx not in success_indices]

        if not pending_indices:
            break

        if attempt < max_attempts:
            logging.info(
                "%s: %d videos pending after attempt %d",
                dataset_name,
                len(pending_indices),
                attempt,
            )
            if request_attempt_delay > 0:
                logging.info(
                    "%s: waiting %.2f seconds before next request attempt",
                    stage_label,
                    request_attempt_delay,
                )
                time.sleep(request_attempt_delay)

    request_errors: List[Tuple[int, str]] = []
    if pending_indices:
        for idx in pending_indices:
            error_message = last_request_errors.get(idx, "Unknown error")
            entries[idx]["request_error"] = error_message
            request_errors.append((idx, error_message))
        write_outputs(entries, dataset_dirs, stage_label)
        log_stage_errors("request", request_errors)
        logging.warning(
            "%s: %d entries failed after %d attempts",
            dataset_name,
            len(pending_indices),
            max_attempts,
        )
    else:
        logging.info("%s: all %d entries succeeded", dataset_name, len(entries))

    if aggregate_download_errors:
        logging.warning(
            "%s: %d download failures encountered",
            dataset_name,
            len(aggregate_download_errors),
        )


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
    logging.info("Video assets and question manifests will be stored under %s", run_dir)

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    threads = max(args.threads, 1)
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        raise RuntimeError(f"Dataset root {dataset_root} does not exist or is not a directory")

    for dataset_name in args.datasets:
        dataset_path_obj = dataset_root / dataset_name / "data.json"
        if not dataset_path_obj.is_file():
            logging.error(
                "Dataset %s: expected data.json at %s but file was not found; skipping",
                dataset_name,
                dataset_path_obj,
            )
            continue
        dataset_dirs = ensure_dataset_dirs(run_dir, dataset_name)

        logging.info("Loading dataset %s from %s", dataset_name, dataset_path_obj)
        try:
            entries = load_dataset(dataset_path_obj)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Failed to load dataset %s: %s", dataset_name, exc)
            continue

        try:
            process_dataset(
                dataset_name,
                entries,
                dataset_path_obj,
                dataset_dirs,
                client,
                args.model,
                include_image,
                threads,
                max(args.max_request_attempts, 1),
                max(args.request_attempt_delay, 0.0),
            )
        except SystemExit as exc:
            logging.error("%s: %s", dataset_name, exc)
        except Exception as exc:  # noqa: BLE001
            logging.exception("%s: unexpected error: %s", dataset_name, exc)


if __name__ == "__main__":
    main()