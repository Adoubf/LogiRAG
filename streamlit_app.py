"""
Streamlit 应用入口文件
使用 streamlit run streamlit_app.py 启动
"""
import sys
import os
import logging

# 确保当前目录在Python路径中，使所有模块都能正确导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 初始化日志
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/streamlit_app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding='utf-8',  # 确保使用UTF-8编码
    filemode='w'  # 使用覆盖模式
)
logging.info("启动Streamlit应用...")

# 导入并运行应用
from ui.streamlit_app import run_app
run_app()
