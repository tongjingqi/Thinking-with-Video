#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
答案判断器模块
使用 GPT-4o 判断视频和音频中的答案是否正确
"""

import base64
from typing import Literal


class AnswerJudge:
    """答案判断器类"""

    def __init__(self, openai_client):
        """
        初始化答案判断器

        参数:
            openai_client: OpenAI 客户端实例
        """
        self.openai_client = openai_client

    def judge_by_last_frame(
        self,
        lastframe_path: str,
        question: str,
        correct_answer: str
    ) -> Literal["yes", "no", "error"]:
        """
        使用 GPT-4o Vision 判断视频最后一帧是否显示正确答案

        参数:
            lastframe_path: 最后一帧图片路径
            question: 题目文本
            correct_answer: 正确答案

        返回:
            "yes" 或 "no" 或 "error"
        """
        try:
            # 读取图片并转为 base64
            with open(lastframe_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # System prompt: 定义角色和任务
#             system_prompt = """You are an expert answer checker for educational videos. Your task is to determine if an image (the last frame of a solution video) displays the correct answer to a given question.

# Rules:
# 1. Compare the visible answer in the image with the provided correct answer
# 2. Be strict but reasonable - minor formatting differences are acceptable if the core answer is correct
# 3. For multiple choice questions, check if the correct option (A, B, C, etc.) is clearly marked or highlighted
# 4. For numerical answers, check if the number matches (ignore minor formatting like "4" vs "4.0")
# 5. For text answers, check if the key content matches (ignore case sensitivity and minor punctuation)
# 6. You must respond with ONLY 'yes' or 'no', nothing else"""

            system_prompt = """You are an expert answer checker for educational videos. Your task is to determine if an image (the last frame of a solution video) displays the correct answer to a given question.

Rules:
0. First, determine the visible answer from the image using this priority:
   - If there is an explicit statement indicating the answer (e.g., "The answer is ..."), use that answer.
   - Else, check for an answer marked by a symbol such as box, circle, underline, arrow, etc. If multiple positions are marked but show different results, respond 'no' immediately.
   - Else, use the bottom-rightmost result in the image as the visible answer.
1. Compare the visible answer in the image with the provided correct answer
2. Be strict but reasonable - minor formatting differences are acceptable if the core answer is correct
3. For multiple choice questions, check if the answer given in the image matches the correct option (e.g., A, B, C, etc.). Marking or highlighting the correct option in the question is allowed.
4. For numerical answers, check if the number matches (ignore minor formatting like "4" vs "4.0")
5. For text answers, check if the key content matches (ignore case sensitivity and minor punctuation)
6. You must respond with ONLY 'yes' or 'no', nothing else"""

            # User prompt: 具体的问题和答案
            user_prompt = f"""Question: {question}

Correct answer: {correct_answer}

Does the image show the correct answer?"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=10,
                temperature=0  # 使用确定性输出
            )

            result = response.choices[0].message.content.strip().lower()
            return "yes" if "yes" in result else "no"

        except Exception as e:
            print(f"[ERROR] GPT-4o 视频判断失败: {e}")
            return "error"

    def judge_by_audio_transcript(
        self,
        transcript: str,
        question: str,
        correct_answer: str
    ) -> Literal["yes", "no", "error"]:
        """
        使用 GPT-4o 判断音频转录文本是否包含正确答案

        参数:
            transcript: 音频转录文本
            question: 题目文本
            correct_answer: 正确答案

        返回:
            "yes" 或 "no" 或 "error"
        """
        try:
            # System prompt: 定义角色和任务
            system_prompt = """You are an expert answer checker for educational video transcripts. Your task is to determine if an audio transcript from a solution video contains the correct answer to a given question.

Rules:
1. Check if the transcript explicitly states or clearly implies the correct answer
2. Be lenient with phrasing - the transcript may explain the answer in different words
3. For multiple choice questions, check if the correct option (A, B, C, etc.) is mentioned
4. For numerical answers, check if the number is stated (ignore surrounding explanation)
5. For text answers, check if the key concept is explained correctly
6. Common phrases like "the correct answer is...", "the answer is...", "it should be..." indicate the answer
7. You must respond with ONLY 'yes' or 'no', nothing else"""

            # User prompt: 具体的问题、答案和转录文本
            print(transcript)
            user_prompt = f"""Question: {question}

Correct answer: {correct_answer}

Audio transcript: "{transcript}"

Does the transcript provide the correct answer?"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                max_tokens=10,
                temperature=0  # 使用确定性输出
            )

            result = response.choices[0].message.content.strip().lower()
            return "yes" if "yes" in result else "no"

        except Exception as e:
            print(f"[ERROR] GPT-4o 音频判断失败: {e}")
            return "error"

    def judge_combined(
        self,
        lastframe_path: str,
        transcript: str,
        question: str,
        correct_answer: str
    ) -> Literal["yes", "no", "error"]:
        """
        综合判断视频最后一帧和音频转录（有一个正确即可）

        参数:
            lastframe_path: 最后一帧图片路径
            transcript: 音频转录文本
            question: 题目文本
            correct_answer: 正确答案

        返回:
            "yes" 或 "no" 或 "error"
        """
        try:
            # 读取图片并转为 base64
            with open(lastframe_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # System prompt: 定义角色和任务
            system_prompt = """You are an expert answer checker for educational solution videos. Your task is to determine if a video provides the correct answer through EITHER its visual content (last frame) OR audio content (transcript), or both.

Rules:
1. Check BOTH the image and the audio transcript for the correct answer
2. If EITHER the image OR the transcript contains the correct answer, respond 'yes'
3. Only respond 'no' if BOTH sources fail to provide the correct answer
4. For the image: check if the answer is visibly displayed
5. For the transcript: check if the answer is stated or clearly explained
6. Use the same lenient checking rules as single-source evaluation
7. You must respond with ONLY 'yes' or 'no', nothing else

Evaluation logic: visual_correct OR audio_correct = 'yes'"""

            # User prompt: 具体的问题、答案、图片和转录文本
            user_prompt = f"""Question: {question}

Correct answer: {correct_answer}

Audio transcript: "{transcript}"

Based on the provided image (last frame) and audio transcript, does the video provide the correct answer through either or both channels?"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=10,
                temperature=0  # 使用确定性输出
            )

            result = response.choices[0].message.content.strip().lower()
            return "yes" if "yes" in result else "no"

        except Exception as e:
            print(f"[ERROR] GPT-4o 综合判断失败: {e}")
            return "error"
