#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频处理工具模块
提供音频提取和转录功能
"""

import os
from typing import Optional

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip


class AudioProcessor:
    """音频处理器类"""

    @staticmethod
    def extract_audio(video_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        从视频中提取音频

        参数:
            video_path: 视频文件路径
            output_path: 输出音频路径（可选，默认为视频路径.mp3）

        返回:
            音频路径，失败返回 None
        """
        try:
            # 如果未指定输出路径，自动生成
            if output_path is None:
                video_dir = os.path.dirname(video_path)
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(video_dir, f"{video_name}.mp3")

            video = VideoFileClip(video_path)
            audio = video.audio

            if audio is None:
                video.close()
                return None

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # moviepy 2.x 版本的 write_audiofile 不再支持 verbose 和 logger 参数
            try:
                # 尝试使用新版本API（moviepy 2.x）
                audio.write_audiofile(output_path, logger=None)
            except TypeError:
                # 如果失败，尝试旧版本API（moviepy 1.x）
                audio.write_audiofile(output_path, verbose=False, logger=None)

            video.close()
            return output_path

        except Exception as e:
            return None

    @staticmethod
    def transcribe_audio(audio_path: str, openai_client) -> Optional[str]:
        """
        使用 Whisper API 进行语音转文字

        参数:
            audio_path: 音频文件路径
            openai_client: OpenAI 客户端实例

        返回:
            转录文本，失败返回 None
        """
        try:
            with open(audio_path, 'rb') as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            text = transcript.text
            return text

        except Exception as e:
            return None

    @staticmethod
    def validate_audio(audio_path: str) -> bool:
        """
        验证音频文件是否有效

        参数:
            audio_path: 音频文件路径

        返回:
            是否有效
        """
        if not os.path.exists(audio_path):
            return False

        # 检查文件大小（至少应该有一些字节）
        if os.path.getsize(audio_path) == 0:
            return False

        return True
