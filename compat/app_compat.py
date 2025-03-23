"""
物流行业信息咨询系统 - 兼容性入口
此文件保留用于向后兼容，建议使用 run.py 作为主入口
"""
import os
import sys
import subprocess
from utils.logging_utils import setup_logger


def run_legacy_app():
    """运行旧版应用入口"""
    logger = setup_logger(log_file="./logs/app_compat.log")
    logger.info("正在启动物流行业信息咨询系统(兼容模式)...")
    logger.info("提示: 建议使用 run.py 作为新的入口点")

    # 获取当前app.py的路径
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(current_dir, "app.py")

    try:
        # 使用subprocess启动streamlit
        logger.info(f"执行命令: streamlit run {app_path}")
        subprocess.run(["streamlit", "run", app_path], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Streamlit启动失败: {e}")
        print(f"启动失败: {e}")
    except FileNotFoundError:
        logger.error("未找到streamlit命令，请确认已安装: pip install streamlit")
        print("错误: 未找到streamlit命令，请确认已安装: pip install streamlit")


if __name__ == "__main__":
    run_legacy_app()
