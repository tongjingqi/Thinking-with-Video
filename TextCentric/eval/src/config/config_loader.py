#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载器模块
负责加载和管理 API 配置
"""

import os
import yaml
from typing import Dict, Optional
from openai import OpenAI


class ConfigLoader:
    """配置加载器类"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器

        参数:
            config_path: 配置文件路径，默认为 config/api_config.yaml
        """
        if config_path is None:
            # 默认配置路径
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(base_dir, "config", "api_config.yaml")

        self.config_path = config_path
        self._config = None
        self._openai_client = None

    def load(self) -> Dict:
        """
        加载配置文件

        返回:
            配置字典
        """
        if self._config is None:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        return self._config

    def get(self, key: str, default=None):
        """
        获取配置项

        参数:
            key: 配置键
            default: 默认值

        返回:
            配置值
        """
        config = self.load()
        return config.get(key, default)

    def get_openai_client(self) -> OpenAI:
        """
        获取 OpenAI 客户端实例（单例模式）

        返回:
            OpenAI 客户端
        """
        if self._openai_client is None:
            config = self.load()
            self._openai_client = OpenAI(
                api_key=config.get('openai_api_key'),
                base_url=config.get('base_url')
            )
        return self._openai_client


# 全局配置加载器实例
_global_config_loader = None


def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    加载配置（单例模式）

    参数:
        config_path: 配置文件路径，默认为 config/api_config.yaml

    返回:
        ConfigLoader 实例
    """
    global _global_config_loader

    if _global_config_loader is None:
        _global_config_loader = ConfigLoader(config_path)

    return _global_config_loader
