#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标计算模块
计算并统计4种评估指标
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """单个视频的评估结果"""
    video_path: str
    question_id: int
    question: str
    correct_answer: str

    # 中间结果
    lastframe_path: str = None
    audio_path: str = None
    audio_transcript: str = None

    # 判断结果
    last_frame_judgment: str = None  # "yes" or "no" or "error"
    audio_judgment: str = None       # "yes" or "no" or "error"

    # 4种指标
    last_frame_correct: bool = None  # 指标1: 视频最后一帧正确
    audio_correct: bool = None       # 指标2: 音频转文字正确
    both_correct: bool = None        # 指标3: 两者都正确（AND）
    any_correct: bool = None         # 指标4: 有一个正确（OR）

    status: str = "pending"  # pending, completed, error
    error_message: str = None


class EvaluationMetrics:
    """评估指标计算器"""

    @staticmethod
    def compute_single_metrics(
        last_frame_judgment: str,
        audio_judgment: str
    ) -> Dict[str, bool]:
        """
        计算单个视频的4种指标

        参数:
            last_frame_judgment: 最后一帧判断结果 ("yes" or "no" or "error")
            audio_judgment: 音频判断结果 ("yes" or "no" or "error")

        返回:
            包含4种指标的字典
        """
        # 转换为布尔值（error 视为 False）
        last_frame_correct = (last_frame_judgment == "yes")
        audio_correct = (audio_judgment == "yes")

        return {
            "last_frame_correct": last_frame_correct,           # 指标1
            "audio_correct": audio_correct,                     # 指标2
            "both_correct": last_frame_correct and audio_correct,  # 指标3: AND
            "any_correct": last_frame_correct or audio_correct,    # 指标4: OR
        }

    @staticmethod
    def compute_batch_statistics(results: List[EvaluationResult]) -> Dict:
        """
        计算批量评估的统计信息

        参数:
            results: 评估结果列表

        返回:
            统计信息字典
        """
        # 过滤出已完成的结果
        completed = [r for r in results if r.status == "completed"]

        if not completed:
            return {
                "total": len(results),
                "completed": 0,
                "metrics": {}
            }

        # 计算每种指标的准确率
        metrics = {}

        # 指标1: 视频最后一帧正确率
        last_frame_valid = [r for r in completed if r.last_frame_correct is not None]
        if last_frame_valid:
            metrics["last_frame"] = {
                "correct": sum(1 for r in last_frame_valid if r.last_frame_correct),
                "total": len(last_frame_valid),
                "accuracy": sum(1 for r in last_frame_valid if r.last_frame_correct) / len(last_frame_valid)
            }

        # 指标2: 音频转文字正确率
        audio_valid = [r for r in completed if r.audio_correct is not None]
        if audio_valid:
            metrics["audio"] = {
                "correct": sum(1 for r in audio_valid if r.audio_correct),
                "total": len(audio_valid),
                "accuracy": sum(1 for r in audio_valid if r.audio_correct) / len(audio_valid)
            }

        # 指标3: 两者都正确（AND）
        both_valid = [r for r in completed if r.both_correct is not None]
        if both_valid:
            metrics["both_correct"] = {
                "correct": sum(1 for r in both_valid if r.both_correct),
                "total": len(both_valid),
                "accuracy": sum(1 for r in both_valid if r.both_correct) / len(both_valid)
            }

        # 指标4: 有一个正确（OR）
        any_valid = [r for r in completed if r.any_correct is not None]
        if any_valid:
            metrics["any_correct"] = {
                "correct": sum(1 for r in any_valid if r.any_correct),
                "total": len(any_valid),
                "accuracy": sum(1 for r in any_valid if r.any_correct) / len(any_valid)
            }

        return {
            "total": len(results),
            "completed": len(completed),
            "metrics": metrics
        }

    @staticmethod
    def print_statistics(stats: Dict):
        """
        打印统计信息

        参数:
            stats: 统计信息字典
        """
        print(f"\n{'='*80}")
        print(f"评估统计")
        print(f"{'='*80}")
        print(f"总题数: {stats['total']}")
        print(f"已完成: {stats['completed']}")

        if "metrics" in stats and stats["metrics"]:
            print(f"\n4种评估指标:")

            if "last_frame" in stats["metrics"]:
                m = stats["metrics"]["last_frame"]
                print(f"  1. 视频最后一帧正确率: {m['accuracy']:.2%} ({m['correct']}/{m['total']})")

            if "audio" in stats["metrics"]:
                m = stats["metrics"]["audio"]
                print(f"  2. 音频转文字正确率:   {m['accuracy']:.2%} ({m['correct']}/{m['total']})")

            if "both_correct" in stats["metrics"]:
                m = stats["metrics"]["both_correct"]
                print(f"  3. 两者都正确 (AND):    {m['accuracy']:.2%} ({m['correct']}/{m['total']})")

            if "any_correct" in stats["metrics"]:
                m = stats["metrics"]["any_correct"]
                print(f"  4. 有一个正确 (OR):     {m['accuracy']:.2%} ({m['correct']}/{m['total']})")

        print(f"{'='*80}\n")
