"""
日志系统模块

提供统一的日志管理接口，支持控制台和文件输出。
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "llm_optimization",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """
    配置并返回日志记录器。

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径，为 None 时仅输出到控制台
        level: 日志级别
        fmt: 日志格式字符串

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_timestamp() -> str:
    """
    获取当前时间戳字符串，用于日志文件命名。

    Returns:
        格式化的时间戳字符串
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class LoggerContext:
    """
    日志上下文管理器，用于在代码块中临时设置日志级别。

    Example:
        with LoggerContext(logger, logging.DEBUG):
            logger.debug("这条消息会输出")
    """

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._original_level = logger.level

    def __enter__(self):
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self._original_level)
        return False