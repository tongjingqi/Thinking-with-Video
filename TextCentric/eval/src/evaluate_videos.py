#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频评估主脚本
直接评估 mp4 视频文件，输出4种指标

使用示例：
    # 评估单个视频
    python src/evaluate_videos.py --video path/to/video.mp4 --question "What is 2+2?" --answer "4"

    # 批量评估视频（从JSON文件）
    python src/evaluate_videos.py --batch videos.json --output output/evaluation

    # 评估视频目录
    python src/evaluate_videos.py --video-dir output/videos --questions questions.json --output output/evaluation
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator import VideoEvaluator


def evaluate_single_video(args):
    """评估单个视频"""
    print(f"\n{'='*80}")
    print(f"单视频评估模式")
    print(f"{'='*80}")

    evaluator = VideoEvaluator(config_path=args.config)

    result = evaluator.evaluate_video(
        video_path=args.video,
        question=args.question,
        correct_answer=args.answer,
        question_id=args.id if hasattr(args, 'id') else 0,
        output_dir=args.output,
        verbose=True
    )

    # 保存结果
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        result_path = os.path.join(args.output, "result.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                "video_path": result.video_path,
                "question": result.question,
                "correct_answer": result.correct_answer,
                "last_frame_correct": result.last_frame_correct,
                "audio_correct": result.audio_correct,
                "both_correct": result.both_correct,
                "any_correct": result.any_correct,
                "status": result.status
            }, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] 结果已保存: {result_path}")


def evaluate_batch_from_json(args):
    """从JSON文件批量评估视频"""
    print(f"\n{'='*80}")
    print(f"批量评估模式 (JSON)")
    print(f"{'='*80}")

    # 加载批量评估配置
    with open(args.batch, 'r', encoding='utf-8') as f:
        batch_config = json.load(f)

    # 格式可以是：
    # [
    #   {
    #     "video_path": "path/to/video.mp4",
    #     "question": "What is 2+2?",
    #     "correct_answer": "4",
    #     "question_id": 1
    #   },
    #   ...
    # ]

    evaluator = VideoEvaluator(config_path=args.config)

    results = evaluator.evaluate_batch(
        video_infos=batch_config,
        output_dir=args.output,
        max_workers=args.threads,
        save_interval=args.save_interval
    )

    print(f"\n[OK] 批量评估完成！共评估 {len(results)} 个视频")


def evaluate_video_directory(args):
    """评估视频目录"""
    print(f"\n{'='*80}")
    print(f"视频目录评估模式")
    print(f"{'='*80}")

    # 加载题目信息
    with open(args.questions, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    # 格式可以是：
    # {
    #   "video_filename1.mp4": {
    #     "question": "What is 2+2?",
    #     "correct_answer": "4",
    #     "question_id": 1
    #   },
    #   ...
    # }
    # 或者列表格式：
    # [
    #   {
    #     "video_filename": "video1.mp4",
    #     "question": "What is 2+2?",
    #     "correct_answer": "4",
    #     "question_id": 1
    #   },
    #   ...
    # ]

    # 扫描视频目录
    video_dir = Path(args.video_dir)
    video_files = list(video_dir.glob("*.mp4"))

    print(f"发现 {len(video_files)} 个视频文件")

    # 构建评估列表
    video_infos = []

    if isinstance(questions_data, dict):
        # 字典格式
        for video_file in video_files:
            filename = video_file.name
            if filename in questions_data:
                info = questions_data[filename]
                video_infos.append({
                    "video_path": str(video_file),
                    "question": info["question"],
                    "correct_answer": info["correct_answer"],
                    "question_id": info.get("question_id", filename)
                })
    else:
        # 列表格式
        video_map = {v.name: str(v) for v in video_files}
        for item in questions_data:
            filename = item.get("video_filename")
            if filename and filename in video_map:
                video_infos.append({
                    "video_path": video_map[filename],
                    "question": item["question"],
                    "correct_answer": item["answer"],
                    "question_id": item.get("question_id", filename)
                })

    print(f"匹配到 {len(video_infos)} 个视频与题目的对应关系")

    if not video_infos:
        print("[ERROR] 没有找到匹配的视频和题目")
        return

    # 评估
    evaluator = VideoEvaluator(config_path=args.config)

    results = evaluator.evaluate_batch(
        video_infos=video_infos,
        output_dir=args.output,
        max_workers=args.threads,
        save_interval=args.save_interval
    )

    print(f"\n[OK] 目录评估完成！共评估 {len(results)} 个视频")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="视频评估工具 - 直接评估mp4视频，输出4种指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 评估单个视频:
   python src/evaluate_videos.py --video video.mp4 --question "What is 2+2?" --answer "4" --output output/

2. 从JSON文件批量评估:
   python src/evaluate_videos.py --batch videos.json --output output/ --threads 8

   JSON格式:
   [
     {
       "video_path": "path/to/video1.mp4",
       "question": "What is 2+2?",
       "correct_answer": "4",
       "question_id": 1
     },
     ...
   ]

3. 评估视频目录:
   python src/evaluate_videos.py --video-dir output/videos --questions questions.json --output output/evaluation

4种评估指标:
  1. 视频最后一帧是否显示正确答案
  2. 视频音频转文字后是否包含正确答案
  3. 两者都正确 (AND)
  4. 有一个正确 (OR)
        """
    )

    # 通用参数
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径 (默认: config/api_config.yaml)")
    parser.add_argument("--output", type=str, default="output/evaluation",
                        help="输出目录")
    parser.add_argument("--threads", type=int, default=4,
                        help="并发线程数 (批量模式)")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="保存间隔 (批量模式，每N个视频保存一次)")

    # 创建子命令
    subparsers = parser.add_subparsers(dest='mode', help='评估模式')

    # 单视频模式
    single_parser = subparsers.add_parser('single', help='评估单个视频')
    single_parser.add_argument("--video", type=str, required=True,
                              help="视频文件路径")
    single_parser.add_argument("--question", type=str, required=True,
                              help="题目文本")
    single_parser.add_argument("--answer", type=str, required=True,
                              help="正确答案")
    single_parser.add_argument("--id", type=int, default=0,
                              help="题目ID")

    # 批量模式 (JSON)
    batch_parser = subparsers.add_parser('batch', help='从JSON文件批量评估')
    batch_parser.add_argument("--batch", type=str, required=True,
                             help="批量评估JSON文件")

    # 目录模式
    dir_parser = subparsers.add_parser('directory', help='评估视频目录')
    dir_parser.add_argument("--video-dir", type=str, required=True,
                           help="视频目录")
    dir_parser.add_argument("--questions", type=str, required=True,
                           help="题目信息JSON文件")

    # 兼容旧的命令行格式（不使用子命令）
    parser.add_argument("--video", type=str, help="视频文件路径")
    parser.add_argument("--question", type=str, help="题目文本")
    parser.add_argument("--answer", type=str, help="正确答案")
    parser.add_argument("--id", type=int, default=0, help="题目ID")
    parser.add_argument("--batch", type=str, help="批量评估JSON文件")
    parser.add_argument("--video-dir", type=str, help="视频目录")
    parser.add_argument("--questions", type=str, help="题目信息JSON文件")

    args = parser.parse_args()

    # 根据参数决定模式
    if args.mode == 'single' or (args.video and args.question and args.answer):
        evaluate_single_video(args)
    elif args.mode == 'batch' or args.batch:
        evaluate_batch_from_json(args)
    elif args.mode == 'directory' or (args.video_dir and args.questions):
        evaluate_video_directory(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
