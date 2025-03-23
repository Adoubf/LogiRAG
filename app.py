"""
物流行业信息咨询系统 - 传统入口
兼容 streamlit run app.py 命令
"""
import sys
import os
import logging

# 确保当前目录在Python路径中，使所有模块都能正确导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 初始化日志
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding='utf-8',  # 确保使用UTF-8编码
    filemode='w'  # 使用覆盖模式
)
logging.info("通过app.py启动Streamlit应用...")
logging.info("提示: 建议使用 streamlit run streamlit_app.py 启动")

# 导入并运行应用
from ui.streamlit_app import run_app
run_app()
