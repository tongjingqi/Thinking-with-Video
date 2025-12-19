<div align="center">

# Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm

</div>

<div align="center" style="font-size: 15pt">

<a href='https://arxiv.org/abs/2511.04570'><img src='https://img.shields.io/badge/Arxiv-2511.04570-purple'></a>
<a href='https://huggingface.co/papers/2511.04570'><img src='https://img.shields.io/badge/HF%20Paper-2511.04570-blue'></a>
<a href='https://thinking-with-video.github.io/'><img src='https://img.shields.io/badge/Project-Website-green'></a>
<a href='https://thinking-with-video.github.io/#leaderboard'><img src='https://img.shields.io/badge/Leaderboard-Table-E07A5F'></a>
<a href='https://huggingface.co/datasets/fnlp/VideoThinkBench'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow'></a>

</div>

<div align="center">
  <a href="https://huggingface.co/papers/week/2025-W45">
    <img src="assets/huggingface_paper_gold_week.svg"/>
  </a>
</div>

## 🎊 News <!-- omit in toc -->

- [2025.11] Our paper "Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm" has been released on arXiv! 📄 [[Paper](https://arxiv.org/abs/2511.04570)] On HuggingFace, it has achieved "#1 Paper of the Day"!
- [2025.11] 🔥We release *["minitest"](https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench)* of our VideoThinkBench, including 500 test samples of vision-centric tasks and 250 test samples of text-centric tasks.
- [2025.12] 🔥We release VideoThinkBench [Leaderboard](https://thinking-with-video.github.io/#leaderboard) that includes different models.

## 📜 Brief Introduction <!-- omit in toc -->

Moving beyond the traditional paradigms of "Thinking with Text" (e.g., Chain-of-Thought) and "Thinking with Images", we propose **"Thinking with Video"**—a new paradigm that unifies visual and textual reasoning through video generation models. It naturally enables human-like dynamic reasoning through video generation, such as **drawing and imagination**.

💡 **A New Unified Reasoning Paradigm**
&nbsp;&nbsp;&nbsp;&nbsp;"Thinking with Video" leverages video generation models to visualize dynamic processes, represent temporal evolution, and embed text within video frames. This approach achieves unified multimodal understanding and generation, overcoming the static constraints of image-based reasoning and the modality separation in traditional approaches.

📊 **VideoThinkBench: A Comprehensive Benchmark**
&nbsp;&nbsp;&nbsp;&nbsp;We developed VideoThinkBench, the first reasoning benchmark specifically designed for evaluating video generation models. It comprises vision-centric tasks (eyeballing puzzles, visual puzzles, ARC-AGI-2, mazes) that leverage dynamic visual reasoning, and text-centric tasks adapted from established benchmarks (MATH, GSM8K, MMLU, MMMU, etc.) that test text-based reasoning capabilities within generated videos.

🚀 **Surpassing VLMs on Several Tasks**
&nbsp;&nbsp;&nbsp;&nbsp;Our evaluation shows that Sora-2 demonstrates competitive reasoning capabilities across both categories. Notably, Sora-2 **surpasses state-of-the-art vision-language models on several vision-centric tasks**, showcasing the unique advantages of dynamic visual reasoning. On text-centric tasks, Sora-2 achieves strong performance including 98.9% on GSM8K, 94.0% on MATH, and 75.5% on MMMU, demonstrating the potential of "Thinking with Video" as a unified multimodal reasoning paradigm.

<div align="center">
<img src="assets/main_picture.png" width=100% />
</div>


## 📌 Contents <!-- omit in toc -->

- [Installation and Dataset Download](#installation-and-dataset-download)
- [VideoThinkBench](#videothinkbench)
- [Code and Evaluation](#code-and-evaluation)
- [Benchmark Results](#benchmark-results)
- [Takeaways](#takeaways)
- [Citation](#citation)


## Installation and Dataset Download <!-- omit in toc -->

1. Clone this repository and navigate to Thinking-with-Video folder
   ```bash
   git clone --recursive https://github.com/tongjingqi/Thinking-with-Video.git
   cd Thinking-with-Video
   ```

2. Install dependencies
   ```bash
   conda create -y -n thinking_with_video python==3.12
   conda activate thinking_with_video
   pip install -r requirements.txt
   ```
3. Download benchmark datasets from Hugging Face
   ```bash
   hf download --repo-type dataset OpenMOSS-Team/VideoThinkBench --local-dir VideoThinkBench
   cd VideoThinkBench

   # upzip the zip datasets under the `Vision-Centric_Reasoning` and `Text-Centric_Reasoning` folders
   bash unzip_dir.sh Vision-Centric_Reasoning
   bash unzip_dir.sh Text-Centric_Reasoning

   # check the statistics of the datasets
   python check.py Vision-Centric_Reasoning > vision_centric_stats.txt
   python check.py Text-Centric_Reasoning > text_centric_stats.txt
   ```

## VideoThinkBench

VideoThinkBench is a comprehensive benchmark for evaluating video generation models' reasoning capabilities, consisting of two main categories:

### Vision-Centric Tasks
- **Eyeballing Puzzles**: Spatial reasoning tasks requiring visual estimation and drawing
- **Visual Puzzles**: Pattern recognition and visual logic problems
- **ARC-AGI-2**: Abstract reasoning tasks requiring few-shot learning
- **Mazes**: Path-finding and navigation challenges

### Text-Centric Tasks
Adapted from established benchmarks including:
- **Math Reasoning**: GSM8K, MATH-500, AIME24, AIME25
- **General Knowledge Reasoning**: BBH, MMLU, MMLU-Pro, GPQA-diamond, SuperGPQA-easy
- **Multimodal Math Reasoning**: MathVista, MathVision
- **Multimodal Understanding**: MMBench, MMMU 

Dataset is available on [Hugging Face](https://huggingface.co/datasets/fnlp/VideoThinkBench).


## Code and Evaluation

### Vision-Centric Tasks

- **Eyeballing Puzzles, Mazes, ARC-AGI-2**: [`VisionCentric/`](https://github.com/betmma/VLMPuzzle) (submodule)
- **Visual Puzzles**: [`visual_puzzles/`](./visual_puzzles)

### Text-Centric Tasks

- **All Text-Centric Tasks**: [`TextCentric/`](./TextCentric)


## Benchmark Results

### Performance Comparison Across All Tasks

The table below summarizes the accuracy (%) of Sora-2 compared with state-of-the-art vision-language models across all second-level tasks in VideoThinkBench:

| **Category** | **Task** | **Sora-2** | **Gemini 2.5 Pro** | **GPT5 high** | **Claude Sonnet 4.5** |
|--------------|----------|------------|-------------------|--------------|---------------------|
| **Vision-Centric** | Eyeballing-Point | 44.7 | 27.8 | 33.6 | 36.2 |
| | Eyeballing-Line | 38.0 | 21.0 | 24.0 | 26.3 |
| | Eyeballing-Shape | 34.5 | 34.5 | 32.5 | 50.5 |
| | Visual-Color | 67.0 | 73.9 | 79.6 | 85.6 |
| | Visual-Shape | 64.9 | 92.9 | 97.5 | 68.6 |
| | ARC-AGI-2 | 1.3 | 1.9 | 0.5 | 5.3 |
| | **Average** | **41.7** | **42.0** | **44.6** | **45.4** |
| **Text-Centric** | Text-Only Math | 53.6 | 94.8 | 97.2 | 90.0 |
| | Text-Only General Knowledge | 63.1 | 84.5 | 85.2 | 86.3 |
| | Multimodal Math | 56.3 | 66.7 | 69.6 | 65.6 |
| | Multimodal General Knowledge | 49.4 | 83.0 | 80.6 | 82.3 |
| | **Average** | **55.6** | **82.3** | **83.2** | **81.1** |
| **Overall Average** | | **47.3** | **58.1** | **60.0** | **59.7** |

**Note**: For Sora-2: Eyeballing Puzzles use Major Frame evaluation; Visual Puzzles show the average of Color-Filling and Shape-Drawing tasks; Text-Centric Reasoning tasks use Video evaluation results.


## Takeaways

Our systematic evaluation on VideoThinkBench reveals seven key findings:

1. **Surpassing VLMs on Eyeballing Puzzles**: Sora-2 generally **surpasses SOTA VLMs** on eyeballing puzzles, exhibiting strong **geometric and physical reasoning** abilities. It can simulate the extension and reflection of rays and manipulate geometric elements (e.g., points and lines) to support spatial reasoning.

2. **Inductive Reasoning on Visual Puzzles**: Sora-2's performance is comparable to Claude Sonnet 4.5 on Shape-Drawing puzzles, demonstrating **inductive reasoning** capabilities. Sora-2 can recognize and apply **patterns of color, shape, and size**, solving visual puzzles involving symmetry, gradients, and compositionality.

3. **Few-Shot Learning Capabilities**: **Sora-2 is a few-shot learner**. On ARC-AGI-2, which requires finding patterns in input-output pairs, while SOTA VLMs achieve less than 5% accuracy, Sora-2 can often make **reasonable predictions**, although they do not strictly match dataset annotations.

4. **Unified Multimodal Reasoning**: On text-centric tasks, Sora-2 shows surprising performance on text and multimodal reasoning benchmarks. The video generation model can **embed text within video frames**, enabling unified multimodal understanding and generation. This demonstrates that "Thinking with Video" is potentially a **unified multimodal reasoning paradigm**.

5. **Improved In-Context Learning with More Examples**: Sora-2 achieves better in-context learning by providing more examples. Experiments show that Sora-2 performs better when provided with all examples compared to only one example, revealing an underexplored direction for analyzing and improving the in-context learning abilities of video generation models.

6. **Test-Time Scaling with Self-Consistency**: **Self-consistency can improve** Sora-2's performance on verifiable video generation reasoning tasks. This reveals an underexplored direction: **test-time scaling in video generation reasoning tasks**.

7. **Analysis of Capability Source**: We systematically analyzed the **source of Sora-2's capabilities**. Sora-2 maintains performance comparable to the original test set on adapted math problems, reducing the likelihood of test set leakage. However, Sora-2 struggles to generate coherent reasoning processes in videos, even when providing correct final answers. Through comparative experiments with Wan 2.5, we speculate that Sora-2's text-centric reasoning ability originates from its **prompt rewriter** model.


## Licenses <!-- omit in toc -->

[![Code License](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

This project is licensed under the MIT License - see the LICENSE file for details.


## Citation

If you find our work helpful, please consider citing our paper 📝 and starring us ⭐️!

```bibtex
@article{tong2025thinkingwithvideo,
    title={Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm},
    author={Jingqi Tong and Yurong Mou and Hangcheng Li and Mingzhe Li and Yongzhuo Yang and Ming Zhang and Qiguang Chen and Tianyi Liang and Xiaomeng Hu and Yining Zheng and Xinchi Chen and Jun Zhao and Xuanjing Huang and Xipeng Qiu},
    journal={arXiv preprint arXiv:2511.04570},
    year={2025}
}
```

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tongjingqi/Thinking-with-Video&type=date&legend=top-left)](https://www.star-history.com/#tongjingqi/Thinking-with-Video&type=date&legend=top-left)

---

<div align="center">
Made with ❤️ for advancing multimodal reasoning research
</div>
