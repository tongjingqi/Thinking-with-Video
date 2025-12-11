# Visual Puzzles

This directory contains code for generating and testing visual puzzle tasks from the VideoThinkBench benchmark.

Visual puzzles assess pattern recognition, inductive reasoning, and visual logic capabilities through tasks involving color filling and shape drawing.

## Quick Start

```bash
# 1. Navigate to the visual_puzzles directory
cd visual_puzzles

# 2. Prepare benchmark data
mkdir -p data
cp -r ../VideoThinkBench/Vision-Centric_Reasoning/visual_puzzles/* data/

# 3. Configure your API key in scripts/run.sh
# Edit the file and replace YOUR_API_KEY_HERE with your actual API key

# 4. Run inference with Sora-2
bash scripts/run.sh

# 5. Extract best frames for evaluation
bash scripts/extract_best_frame.sh
```

## Directory Structure

```
visual_puzzles/
├── README.md                    # This file
├── data/                        # Benchmark dataset (after preparation)
├── example_data/                # Example data for testing
│   ├── color_size/             # Example color-size puzzles
│   ├── color_grid/             # Example color-grid puzzles
│   └── ...                     # Other puzzle types
├── eval/                        # Evaluation scripts
│   └── find_best_frame.py      # Extract optimal frames from videos
├── gen_data/                    # Data generation scripts
│   └── data_generation.py      # Generate new puzzle instances
├── infer/                       # Inference scripts
│   └── request_videos.py       # Request video generation from models
├── scripts/                     # Utility scripts
│   ├── run.sh                  # Run inference pipeline
│   ├── extract_best_frame.sh   # Extract best frames
│   └── generate_data.sh        # Generate new data
└── fonts/                       # Font files for rendering
```

## Benchmark Data Preparation

Before running experiments, download and prepare the benchmark data:

```bash
# From the Thinking-with-Video root directory
# 1. Download VideoThinkBench dataset (see main README.md)
hf download --repo-type dataset OpenMOSS-Team/VideoThinkBench --local-dir VideoThinkBench

# 2. Extract the visual puzzles data
cd VideoThinkBench
bash unzip_dir.sh Vision-Centric_Reasoning

# 3. Copy to visual_puzzles directory
cd ..
mkdir -p visual_puzzles/data
cp -r VideoThinkBench/Vision-Centric_Reasoning/visual_puzzles/* visual_puzzles/data/
cd visual_puzzles
```

## Task Categories

### Color-Filling Tasks (6 types)
1. **color_size**: Fill colors based on object size patterns
2. **color_grid**: Complete color grids following pattern rules
3. **color_hexagon**: Fill hexagonal grids with color patterns
4. **color_overlap_squares**: Determine colors for overlapping squares
5. **polygon_sides_color**: Color polygons based on number of sides
6. **rectangle_height_color**: Color rectangles based on their heights

### Shape-Drawing Tasks (4 types)
1. **size_grid**: Draw circles in grids based on size patterns
2. **shape_reflect**: Draw reflected shapes
3. **shape_size_grid**: Combine shape and size patterns in grids
4. **size_cycle**: Draw circles in cycle structures based on size patterns

## Usage

### Testing with Sora-2

Run inference on all visual puzzle tasks using Sora-2 or compatible video generation models:

```bash
bash scripts/run.sh
```

**Configuration Options** (edit `scripts/run.sh`):
- `--model`: Model identifier (default: `sora_video2-landscape`)
- `--tasks`: Space-separated list of tasks to evaluate
- `--data_root`: Path to input data directory
- `--base_url`: API endpoint URL
- `--api_key`: Your API key for the video generation service
- `--output_root`: Directory to save generated videos
- `--threads`: Number of parallel threads (default: 16)
- `--max_request_attempts`: Maximum retry attempts (default: 5)
- `--request_attempt_delay`: Delay between retries in seconds (default: 2)

### Extracting Best Frames

After generating videos, extract the frame that best matches the solution for each task:

```bash
bash scripts/extract_best_frame.sh
```

This script evaluates each frame in the generated videos and selects the one closest to the ground truth solution.

**Two comparing modes:**
1. **Color-Filling Tasks**: Uses RGB Euclidean distance to compare pixel-wise color similarity
2. **Shape-Drawing Tasks**: Uses coverage difference after binarization to compare shape accuracy

### Evaluating Vision-Language Models (VLMs)

We now provide `infer/test_VLM.py`  to batch-query VLMs on the visual puzzle tasks.

The companion script `scripts/run_VLM.sh` runs two batches: one that sends the baseline tasks and another that appends the available `options` lists when requested (via `--provide_options`). Each batch saves results under `vlm_output/<mode>/<model_timestamp>/result.json`.

Key CLI flags for `test_VLM.py`:
- `--model`: Name of the VLM endpoint to call
- `--base_url`: Inference service URL
- `--tasks`: List of puzzle tasks to evaluate (each task folder must contain `data.json`)
- `--data_root`: Path containing the task directories (defaults to `data`)
- `--output_root`: Directory to hold structured output (metadata + entries in `result.json`)
- `--threads`, `--max_request_attempts`, `--request_attempt_delay`: Control concurrency and retry timing (mirrors video script defaults)
- `--provide_options`: Use this flag to append `Options: ...` text for question answering

The runner logs per-task accuracy into each `result.json` and includes an `is_correct` flag on every entry, making it easy to aggregate VLM performance.

### Generating Custom Data

Create new puzzle instances for creating your own dataset.

```bash
bash scripts/generate_data.sh
```

**Generate specific puzzle types:**
```bash
# Generate a single puzzle type
python gen_data/data_generation.py create_data color_size example_data --limit 10 --seed 42

# Generate multiple types with custom resolution
for pattern in color_size size_grid color_grid; do
    python gen_data/data_generation.py create_data $pattern custom_data \
        --limit 5 \
        --seed 17 \
        --target_size "(1280, 704)"
done
```

**Parameters:**
- `create_data`: Command to generate puzzle data
- `pattern`: Puzzle type (see Task Categories below)
- `output_dir`: Directory to save generated puzzles
- `--limit`: Number of instances to generate
- `--seed`: Random seed for reproducibility
- `--target_size`: Output resolution as "(width, height)"

## Acknowledgements

The data generation code is adapted from the [PuzzleVQA](https://github.com/declare-lab/LLM-PuzzleTest) project.