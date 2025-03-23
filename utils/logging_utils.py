import os
import logging
from typing import Optional


def setup_logger(log_file: Optional[str] = None, console: bool = True) -> logging.Logger:
    """
    设置并返回日志记录器
    
    Args:
        log_file: 日志文件路径，不提供则仅控制台输出
        console: 是否同时输出到控制台
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger('logistics_rag')
    logger.setLevel(logging.INFO)
    # 清除现有的处理器，防止重复添加
    logger.handlers = []

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 添加日志文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')  # 使用覆盖模式
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 添加控制台处理器
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
