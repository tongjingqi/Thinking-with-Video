#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理工具模块
提供视频文件处理功能，如提取最后一帧
"""

import os
import cv2
from typing import Optional


class VideoProcessor:
    """视频处理器类"""

    @staticmethod
    def extract_last_frame(video_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        提取视频最后一帧

        参数:
            video_path: 视频文件路径
            output_path: 输出图片路径（可选，默认为视频路径_last_frame.png）

        返回:
            图片路径，失败返回 None
        """
        try:
            # 如果未指定输出路径，自动生成
            if output_path is None:
                video_dir = os.path.dirname(video_path)
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(video_dir, f"{video_name}_last_frame.png")

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                cap.release()
                return None

            # 跳到最后一帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, frame = cap.read()

            if ret:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, frame)
                cap.release()
                return output_path
            else:
                cap.release()
                return None

        except Exception as e:
            return None

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        获取视频信息

        参数:
            video_path: 视频文件路径

        返回:
            视频信息字典
        """
        try:
            cap = cv2.VideoCapture(video_path)
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
            }
            cap.release()
            return info

        except Exception as e:
            return {}

    @staticmethod
    def validate_video(video_path: str) -> bool:
        """
        验证视频文件是否有效

        参数:
            video_path: 视频文件路径

        返回:
            是否有效
        """
        if not os.path.exists(video_path):
            return False

        try:
            cap = cv2.VideoCapture(video_path)
            is_valid = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
            cap.release()
            return is_valid

        except Exception as e:
            return False
