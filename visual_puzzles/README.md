# Visual Puzzles

## Benchmark Data Preparation

Download the benchmark data first, see the main [README.md](../README.md) for instructions. Then, copy the data to the `visual_puzzles/data` directory:

```bash
# under the Thinking-with-Video root directory
mkdir -p visual_puzzles/data
cp -r VideoThinkBench/Vision-Centric_Reasoning/visual_puzzles/* visual_puzzles/data
cd visual_puzzles
```

## Test Sora-2

```bash
bash scripts/run.sh
```

## Select the Best Frame for Manual Evaluation

```bash
bash scripts/extract_best_frame.sh
```

## Generate New Data by Yourself

```bash
bash scripts/generate_data.sh
```

## Acknowledgements

The data generation code is adapted from the [PuzzleVQA](https://github.com/declare-lab/LLM-PuzzleTest).