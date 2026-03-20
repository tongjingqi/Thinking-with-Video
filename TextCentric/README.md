# Text-Centric Tasks

This directory provides the text-centric benchmark pipeline for two model families:

- Video generation models (e.g., Sora-2): generate videos first, then evaluate video content.
- Vision-language models (VLM): directly answer each question, then evaluate prediction correctness.

## Overview

Run commands from this `TextCentric/` directory.

Main components:

- `infer/request_videos.py`: request videos for text-centric datasets.
- `eval/src/evaluate_videos.py`: evaluate generated videos.
- `infer/test_VLM.py`: request direct text answers from a VLM.
- `eval/src/eval_VLM.py`: judge VLM predictions (manual fast-path + model-judge fallback).
- `scripts/run.sh`, `scripts/eval.sh`: example commands for video generation models.
- `scripts/run_VLM.sh`, `scripts/eval_VLM.sh`: example commands for VLM workflow.

## Data Preparation

From repository root:

```bash
cd VideoThinkBench
bash unzip_dir.sh minitest_Text-Centric_Reasoning
# or full set:
# bash unzip_dir.sh Text-Centric_Reasoning
```

Typical dataset root used by scripts:

- `../VideoThinkBench/minitest_Text-Centric_Reasoning`

## A. Video Generation Models

### 1) Inference (request videos)

```bash
bash scripts/run.sh
```

This runs `infer/request_videos.py` and writes outputs under:

- `<output_root>/<model>_<timestamp>/<dataset>/videos/*.mp4`
- `<output_root>/<model>_<timestamp>/<dataset>/responses.json`
- `<output_root>/<model>_<timestamp>/<dataset>/questions.json`

### 2) Evaluation (evaluate videos)

```bash
bash scripts/eval.sh
```

This runs `eval/src/evaluate_videos.py` on one dataset output folder.

For API-based evaluation settings, see `eval/config/api_config.yaml`

## B. Vision-Language Models (VLM)

### 1) Inference

```bash
bash scripts/run_VLM.sh
```

This runs `infer/test_VLM.py` and writes outputs under:

- `<output_root>/<model>_<timestamp>/<dataset>/responses.json`
- `<output_root>/<model>_<timestamp>/<dataset>/answers.json`

### 2) Evaluation

```bash
bash scripts/eval_VLM.sh
```

This runs `eval/src/eval_VLM.py` and writes one JSON report per dataset: `<output_root>/<run_name>/<dataset>.json`
