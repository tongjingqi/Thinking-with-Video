#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频评估器模块
提供完整的视频评估功能，从mp4视频输入到4种指标输出
"""

import os
import json
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[WARN] tqdm 未安装，将使用简单进度显示。安装方法: pip install tqdm")

from ..config import load_config
from ..utils import VideoProcessor, AudioProcessor
from .judges import AnswerJudge
from .metrics import EvaluationResult, EvaluationMetrics


class VideoEvaluator:
    """
    视频评估器

    核心功能：
    - 输入：mp4 视频文件
    - 输出：4种评估指标
        1. 视频最后一帧是否正确
        2. 音频转文字是否正确
        3. 两者都正确（AND）
        4. 有一个正确（OR）
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化视频评估器

        参数:
            config_path: 配置文件路径（可选）
        """
        # 加载配置
        self.config_loader = load_config(config_path)
        self.openai_client = self.config_loader.get_openai_client()

        # 初始化工具
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        self.answer_judge = AnswerJudge(self.openai_client)
        self.metrics = EvaluationMetrics()

    def evaluate_video(
        self,
        video_path: str,
        question: str,
        correct_answer: str,
        question_id: int = 0,
        output_dir: Optional[str] = None,
        verbose: bool = True
    ) -> EvaluationResult:
        """
        评估单个视频

        参数:
            video_path: 视频文件路径
            question: 题目文本
            correct_answer: 正确答案
            question_id: 题目ID（可选）
            output_dir: 输出目录（可选，用于保存中间结果）
            verbose: 是否显示详细信息（默认True，批量评估时建议False）

        返回:
            EvaluationResult 对象
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"评估视频: {video_path}")
            print(f"题目ID: {question_id}")
            print(f"题目: {question[:100]}...")
            print(f"正确答案: {correct_answer}")
            print(f"{'='*80}")

        # 初始化结果对象
        result = EvaluationResult(
            video_path=video_path,
            question_id=question_id,
            question=question,
            correct_answer=correct_answer
        )

        try:
            # 步骤1: 验证视频文件
            if verbose:
                print("\n[1/6] 验证视频文件...")
            if not self.video_processor.validate_video(video_path):
                result.status = "error"
                result.error_message = "视频文件无效或不存在"
                return result

            # 准备输出路径
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = f"{question_id}"
                lastframe_path = os.path.join(output_dir, f"{base_name}_last_frame.png")
                audio_path = os.path.join(output_dir, f"{base_name}_audio.mp3")
            else:
                lastframe_path = None
                audio_path = None

            # 步骤2: 提取最后一帧
            if verbose:
                print("\n[2/6] 提取视频最后一帧...")
            lastframe_path = self.video_processor.extract_last_frame(video_path, lastframe_path)
            if not lastframe_path:
                result.status = "error"
                result.error_message = "提取最后一帧失败"
                return result
            result.lastframe_path = lastframe_path

            # 步骤3: 提取音频
            if verbose:
                print("\n[3/6] 提取音频...")
            audio_path = self.audio_processor.extract_audio(video_path, audio_path)
            if not audio_path:
                if verbose:
                    print("[WARN] 视频没有音频轨道，将跳过音频评估")
                result.audio_path = None
                result.audio_transcript = None
            else:
                result.audio_path = audio_path

                # 步骤4: 音频转文字
                if verbose:
                    print("\n[4/6] 音频转文字...")
                transcript = self.audio_processor.transcribe_audio(audio_path, self.openai_client)
                result.audio_transcript = transcript

            # 步骤5: GPT-4o 判断最后一帧
            if verbose:
                print("\n[5/6] GPT-4o 判断最后一帧...")
            last_frame_judgment = self.answer_judge.judge_by_last_frame(
                lastframe_path, question, correct_answer
            )
            result.last_frame_judgment = last_frame_judgment
            if verbose:
                print(f"结果: {last_frame_judgment}")

            # 步骤6: GPT-4o 判断音频
            if result.audio_transcript:
                if verbose:
                    print("\n[6/6] GPT-4o 判断音频...")
                audio_judgment = self.answer_judge.judge_by_audio_transcript(
                    result.audio_transcript, question, correct_answer
                )
                result.audio_judgment = audio_judgment
                if verbose:
                    print(f"结果: {audio_judgment}")
            else:
                if verbose:
                    print("\n[6/6] 跳过音频判断（无音频）")
                result.audio_judgment = "error"

            # 计算4种指标
            metrics = self.metrics.compute_single_metrics(
                last_frame_judgment, result.audio_judgment
            )
            result.last_frame_correct = metrics["last_frame_correct"]
            result.audio_correct = metrics["audio_correct"]
            result.both_correct = metrics["both_correct"]
            result.any_correct = metrics["any_correct"]

            result.status = "completed"

            # 打印结果
            if verbose:
                print(f"\n{'='*80}")
                print(f"评估结果:")
                print(f"  1. 视频最后一帧正确: {result.last_frame_correct}")
                print(f"  2. 音频转文字正确:   {result.audio_correct}")
                print(f"  3. 两者都正确 (AND): {result.both_correct}")
                print(f"  4. 有一个正确 (OR):  {result.any_correct}")
                print(f"{'='*80}\n")

        except Exception as e:
            if verbose:
                print(f"\n[ERROR] 评估失败: {e}")
            result.status = "error"
            result.error_message = str(e)

        return result

    def evaluate_batch(
        self,
        video_infos: List[Dict],
        output_dir: Optional[str] = None,
        max_workers: int = 4,
        save_interval: int = 5
    ) -> List[EvaluationResult]:
        """
        批量评估多个视频

        参数:
            video_infos: 视频信息列表，每个元素为字典，包含：
                - video_path: 视频路径
                - question: 题目
                - correct_answer: 正确答案
                - question_id: 题目ID（可选）
            output_dir: 输出目录（可选）
            max_workers: 最大并发线程数
            save_interval: 保存间隔（每完成多少个保存一次）

        返回:
            EvaluationResult 列表
        """
        print(f"\n{'#'*80}")
        print(f"# 批量评估模式")
        print(f"# 视频数量: {len(video_infos)}")
        print(f"# 并发线程数: {max_workers}")
        print(f"{'#'*80}\n")

        results = []
        results_lock = threading.Lock()

        # 准备输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            results_path = os.path.join(output_dir, "evaluation_results.json")
        else:
            results_path = None

        # 创建进度条（如果 tqdm 可用）
        if TQDM_AVAILABLE:
            pbar = tqdm(total=len(video_infos), desc="评估进度", unit="视频")
        else:
            pbar = None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_info = {
                executor.submit(
                    self.evaluate_video,
                    info['video_path'],
                    info['question'],
                    info['correct_answer'],
                    info.get('question_id', idx),
                    os.path.join(output_dir, f"video_{info.get('question_id', idx)}") if output_dir else None,
                    verbose=False  # 批量评估时关闭详细输出
                ): info
                for idx, info in enumerate(video_infos)
            }

            # 处理完成的任务
            for future in as_completed(future_to_info):
                info = future_to_info[future]

                try:
                    result = future.result()

                    with results_lock:
                        results.append(result)

                        # 更新进度条
                        if pbar:
                            # 更新进度条并显示当前状态
                            status_info = f"✓ {result.question_id}" if result.status == "completed" else f"✗ {result.question_id}"
                            pbar.set_postfix_str(status_info)
                            pbar.update(1)
                        else:
                            # 简单进度显示（无 tqdm）
                            print(f"[{len(results)}/{len(video_infos)}] 视频 #{result.question_id} 完成")

                        # 定期保存结果
                        if results_path and (len(results) % save_interval == 0 or len(results) == len(video_infos)):
                            self._save_results(results, results_path)
                            if not pbar:  # 只在没有进度条时打印保存信息
                                print(f"[保存] 已保存 {len(results)}/{len(video_infos)} 个结果")

                except Exception as e:
                    if pbar:
                        pbar.write(f"[ERROR] 评估失败: {e}")
                    else:
                        print(f"[ERROR] 评估失败: {e}")

                    # 即使失败也更新进度条
                    if pbar:
                        pbar.update(1)

        # 关闭进度条
        if pbar:
            pbar.close()

        # 最终保存
        if results_path:
            self._save_results(results, results_path)
            print(f"\n[OK] 所有结果已保存至: {results_path}")

        # 打印统计信息
        stats = self.metrics.compute_batch_statistics(results)
        self.metrics.print_statistics(stats)

        return results

    def _save_results(self, results: List[EvaluationResult], output_path: str):
        """保存结果到JSON文件"""
        results_dict = []
        for r in results:
            results_dict.append({
                "video_path": r.video_path,
                "question_id": r.question_id,
                "question": r.question,
                "correct_answer": r.correct_answer,
                "lastframe_path": r.lastframe_path,
                "audio_path": r.audio_path,
                "audio_transcript": r.audio_transcript,
                "last_frame_judgment": r.last_frame_judgment,
                "audio_judgment": r.audio_judgment,
                "last_frame_correct": r.last_frame_correct,
                "audio_correct": r.audio_correct,
                "both_correct": r.both_correct,
                "any_correct": r.any_correct,
                "status": r.status,
                "error_message": r.error_message
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
