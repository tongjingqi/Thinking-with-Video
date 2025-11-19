import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find the best matching frame in a video compared to a solution image.")
    parser.add_argument("--results_dir", type=str, required=True, help="Root directory where the results are stored.")
    parser.add_argument("--tasks", nargs="+", help="A list of task names to process. If not provided, all tasks in the results directory will be processed.")
    parser.add_argument("--distance_metric", type=str, default="euclidean", choices=["euclidean", "manhattan", "cielab", "coverage"], help="The metric for color difference calculation.")
    parser.add_argument("--compare_width", type=int, help="Optional width of the comparison window. If not set, the whole image is used.")
    parser.add_argument("--compare_height", type=int, help="Optional height of the comparison window. If not set, the whole image is used.")
    parser.add_argument("--compare_x", type=int, help="Optional x-coordinate of the top-left corner of the comparison window.")
    parser.add_argument("--compare_y", type=int, help="Optional y-coordinate of the top-left corner of the comparison window.")
    parser.add_argument("--frame_rate", type=int, default=1, help="The rate at which to sample frames from the video (e.g., 1 means every frame, 2 means every other frame).")
    parser.add_argument("--binarization_threshold", type=int, default=245, help="Grayscale threshold for 'coverage' metric.")
    parser.add_argument("--resize_width", type=int, help="Optional width to resize the comparison window to before comparison.")
    parser.add_argument("--resize_height", type=int, help="Optional height to resize the comparison window to before comparison.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


def get_comparison_window(image: np.ndarray, width: int, height: int, x: Optional[int] = None, y: Optional[int] = None) -> np.ndarray:
    """Extracts a window from the image."""
    img_h, img_w = image.shape[:2]
    if x is not None and y is not None:
        start_x = x
        start_y = y
    else:
        start_x = (img_w - width) // 2
        start_y = (img_h - height) // 2
    end_x = start_x + width
    end_y = start_y + height
    return image[start_y:end_y, start_x:end_x]


def calculate_difference(img1: np.ndarray, img2: np.ndarray, metric: str, threshold: int = 245) -> float:
    """Calculates the difference between two images using the specified metric."""
    if metric == "coverage":
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # Binarize
        _, bin1 = cv2.threshold(gray1, threshold, 255, cv2.THRESH_BINARY)
        _, bin2 = cv2.threshold(gray2, threshold, 255, cv2.THRESH_BINARY)
        # Calculate difference (number of non-matching pixels)
        diff = np.sum(bin1 != bin2)
    elif metric == "cielab":
        # Convert images from BGR to CIELAB
        img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
        img1_lab = img1_lab.astype(np.float32)
        img2_lab = img2_lab.astype(np.float32)
        # Calculate Euclidean distance in CIELAB space (Delta E* 76)
        diff = np.sqrt(np.sum((img1_lab - img2_lab) ** 2, axis=2))
    else:
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        if metric == "euclidean":
            diff = np.sqrt(np.sum((img1 - img2) ** 2, axis=2))
        elif metric == "manhattan":
            diff = np.sum(np.abs(img1 - img2), axis=2)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    return float(np.sum(diff))


def process_video(
    video_path: Path,
    solution_image_path: Path,
    best_frame_path: Path,
    frame_rate: int,
    metric: str,
    compare_window: Optional[Tuple[int, int]],
    binarization_threshold: int,
    compare_x: Optional[int] = None,
    compare_y: Optional[int] = None,
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
) -> Optional[float]:
    """
    Processes a single video to find the best matching frame.
    Returns the minimum difference found.
    """
    if not video_path.is_file():
        logging.warning(f"Video not found: {video_path}")
        return None
    if not solution_image_path.is_file():
        logging.warning(f"Solution image not found: {solution_image_path}")
        return None

    solution_img = cv2.imread(str(solution_image_path))
    if solution_img is None:
        logging.error(f"Failed to read solution image: {solution_image_path}")
        return None

    if compare_window:
        win_w, win_h = compare_window
        solution_img = get_comparison_window(solution_img, win_w, win_h, compare_x, compare_y)

    # Resize the solution image comparison window if specified
    if resize_width and resize_height:
        solution_img = cv2.resize(solution_img, (resize_width, resize_height))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return None

    best_frame = None
    min_diff = float("inf")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:
            if compare_window:
                win_w, win_h = compare_window
                frame_window = get_comparison_window(frame, win_w, win_h, compare_x, compare_y)
            else:
                frame_window = frame

            # Resize the frame window if specified
            if resize_width and resize_height:
                frame_window = cv2.resize(frame_window, (resize_width, resize_height))

            if frame_window.shape != solution_img.shape:
                logging.warning(f"Shape mismatch between frame ({frame_window.shape}) and solution ({solution_img.shape}). Resizing frame.")
                frame_window = cv2.resize(frame_window, (solution_img.shape[1], solution_img.shape[0]))

            diff = calculate_difference(frame_window, solution_img, metric, binarization_threshold)

            if diff < min_diff:
                min_diff = diff
                best_frame = frame
        
        frame_count += 1

    cap.release()

    if best_frame is not None:
        best_frame_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(best_frame_path), best_frame)
        logging.debug(f"Saved best frame for {video_path.name} to {best_frame_path} with difference {min_diff}")
        return min_diff
    else:
        logging.warning(f"No frames processed or found for video: {video_path}")
        return None


def main() -> None:
    args = parse_args()
    setup_logging()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    task_names = args.tasks
    if not task_names:
        task_names = [d.name for d in results_dir.iterdir() if d.is_dir()]
    
    logging.info(f"Processing tasks: {', '.join(task_names)}")

    compare_window = (args.compare_width, args.compare_height) if args.compare_width and args.compare_height else None
    
    total_diff = 0
    processed_count = 0

    for task_name in tqdm(task_names, desc="Tasks"):
        task_dir = results_dir / task_name
        results_file = task_dir / "video_result.json"
        # best_frames_dir = task_dir / "best_frames"
        best_frames_dir = task_dir / f"best_frames_{args.distance_metric}"

        if not results_file.is_file():
            logging.warning(f"Skipping task '{task_name}': video_result.json not found.")
            continue

        with open(results_file, "r", encoding="utf-8") as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON for task '{task_name}'.")
                continue
        
        for i, entry in enumerate(tqdm(entries, desc=f"Processing '{task_name}'", leave=False)):
            if not isinstance(entry, dict) or "video_path" not in entry or "solution_image_path" not in entry:
                continue

            video_path = Path(entry["video_path"])
            solution_path = Path(entry["solution_image_path"])

            # Ensure video path is absolute relative to task_dir if necessary
            if not video_path.is_absolute():
                video_path = (task_dir / video_path).resolve()
            else:
                video_path = video_path.resolve()

            # Ensure solution path is absolute. The previous tool wrote absolute paths, but handle relative defensively.
            if not solution_path.is_absolute():
                # assume solution paths are relative to the dataset folder which is sibling to task_dir's parent 'data' layout
                # fallback to task_dir if not resolvable
                possible = (task_dir / solution_path)
                if possible.is_file():
                    solution_path = possible.resolve()
                else:
                    solution_path = solution_path.resolve()
            else:
                solution_path = solution_path.resolve()

            best_frame_filename = video_path.with_suffix(".png").name
            best_frame_path = best_frames_dir / best_frame_filename

            min_diff = None
            try:
                min_diff = process_video(
                    video_path,
                    solution_path,
                    best_frame_path,
                    args.frame_rate,
                    args.distance_metric,
                    compare_window,
                    args.binarization_threshold,
                    args.compare_x,
                    args.compare_y,
                    args.resize_width,
                    args.resize_height,
                )

                # write absolute best_frame path into the in-memory entry
                if best_frame_path.is_file():
                    entry["best_frame"] = str(best_frame_path.resolve())
                else:
                    entry["best_frame"] = None
                
                entry["diff"] = min_diff
                entry["metric"] = args.distance_metric
                if min_diff is not None:
                    total_diff += min_diff
                    processed_count += 1

            except Exception as e:
                logging.error(f"Error processing video for entry {entry.get('id', 'N/A')}: {e}")
                entry["best_frame"] = None
                entry["diff"] = None
                entry["metric"] = args.distance_metric

        # after processing all entries, write result.json with added best_frame field
        result_out = task_dir / "result.json"
        try:
            with result_out.open("w", encoding="utf-8") as fo:
                json.dump(entries, fo, indent=2, ensure_ascii=False)
            logging.info(f"Wrote result.json for task {task_name} to {result_out}")
        except Exception as e:
            logging.error(f"Failed to write result.json for task {task_name}: {e}")

        # Save summary
        summary_path = results_dir / task_name / f"summary_{args.distance_metric}.txt"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write(f"Total Processed Entries: {processed_count}\n")
            f.write(f"Total Difference ({args.distance_metric}): {total_diff}\n")
            if processed_count > 0:
                average_diff = total_diff / processed_count
                f.write(f"Average Difference ({args.distance_metric}): {average_diff:.2f}\n")
            else:
                f.write("Average Difference: N/A\n")
        logging.info(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
