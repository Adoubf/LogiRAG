"""
物流行业信息咨询系统 - 主入口文件
"""
import argparse
import os
import sys
import subprocess

# 确保当前目录在Python路径中，使所有模块都能正确导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 添加配置项导入
from config.settings import (
    QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT, FAISS_SAVE_PATH
)
from utils.logging_utils import setup_logger


def start_streamlit():
    """使用streamlit命令启动应用"""
    logger = setup_logger(log_file="./logs/app.log")
    logger.info("使用Streamlit启动应用...")

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_app = os.path.join(current_dir, "streamlit_app.py")

    try:
        # 使用subprocess启动streamlit
        logger.info(f"执行命令: streamlit run {streamlit_app}")
        subprocess.run(["streamlit", "run", streamlit_app], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Streamlit启动失败: {e}")
        print(f"启动失败: {e}")
    except FileNotFoundError:
        logger.error("未找到streamlit命令，请确认已安装: pip install streamlit")
        print("错误: 未找到streamlit命令，请确认已安装: pip install streamlit")


def main():
    """处理命令行参数并执行相应功能"""
    parser = argparse.ArgumentParser(description="物流行业信息咨询系统")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 应用启动命令
    app_parser = subparsers.add_parser("app", help="启动问答界面")
    app_parser.add_argument(
        "-l", "--log",
        help="日志文件路径",
        default="./logs/app.log"
    )
    
    # 向量构建命令
    build_parser = subparsers.add_parser("build", help="构建向量库")
    build_parser.add_argument(
        "--append", action="store_true",
        help="追加到现有向量库而非重建"
    )
    build_parser.add_argument(
        "--resume", action="store_true",
        help="从上次中断点继续"
    )
    build_parser.add_argument(
        "--clear-cache", action="store_true",
        help="清除嵌入缓存"
    )
    build_parser.add_argument(
        "-l", "--log",
        help="日志文件路径",
        default="./logs/vector_build.log"
    )
    
    # 向量管理命令
    manage_parser = subparsers.add_parser("manage", help="管理向量库")
    manage_subparsers = manage_parser.add_subparsers(dest="manage_cmd", help="管理操作")
    
    # 删除向量
    delete_parser = manage_subparsers.add_parser("delete", help="删除向量")
    delete_parser.add_argument("sources", nargs="+", help="要删除的源文件名列表")
    
    # 列出源文件
    list_parser = manage_subparsers.add_parser("list", help="列出源文件")
    
    # 显示统计信息
    stats_parser = manage_subparsers.add_parser("stats", help="显示统计信息")
    
    # 兼容模式命令
    compat_parser = subparsers.add_parser("compat", help="兼容模式")
    
    # 保留旧的参数格式，但是重定向到新的子命令
    parser.add_argument(
        "-m", "--mode",
        choices=["app", "build", "compat"],
        help="运行模式（已弃用，请使用子命令）"
    )
    
    args = parser.parse_args()
    
    # 处理旧的参数格式
    if args.mode and not args.command:
        args.command = args.mode
    
    # 如果没有指定任何命令，默认启动应用
    if not args.command:
        args.command = "app"
    
    # 初始化日志器
    log_file = getattr(args, "log", f"./logs/{args.command}.log")
    logger = setup_logger(log_file=log_file)
    logger.info(f"执行命令: {args.command}")
    
    # 执行对应的命令
    if args.command == "app":
        start_streamlit()
    elif args.command == "build":
        from tools.vector_builder import main as build_main
        build_main(
            append_mode=getattr(args, "append", False),
            resume=getattr(args, "resume", False),
            clear_cache=getattr(args, "clear_cache", False)
        )
    elif args.command == "manage":
        if args.manage_cmd == "delete":
            from tools.vector_manager import delete_by_source, list_source_files
            
            # 首先列出所有可用源文件
            sources = list_source_files()
            
            # 验证输入的源文件是否存在
            invalid_sources = [src for src in args.sources if src not in sources]
            if invalid_sources:
                print(f"警告: 以下文件在向量库中不存在: {invalid_sources}")
                print(f"可用的源文件有: {sources}")
                
                # 过滤出有效的源文件
                valid_sources = [src for src in args.sources if src in sources]
                if not valid_sources:
                    print("没有有效的源文件可删除")
                    return
                
                # 确认是否继续
                print(f"将继续删除这些文件: {valid_sources}")
                args.sources = valid_sources
            
            # 执行删除
            try:
                deleted = delete_by_source(args.sources)
                print(f"已删除 {deleted} 个向量")
            except Exception as e:
                logger.error(f"删除失败: {e}")
                print(f"错误: 删除失败 - {e}")
        elif args.manage_cmd == "list":
            from tools.vector_manager import list_source_files
            sources = list_source_files()
            if sources:
                print(f"找到 {len(sources)} 个源文件:")
                for source in sources:
                    print(f"  - {source}")
            else:
                print("未找到任何源文件")
        elif args.manage_cmd == "stats":
            from tools.vector_manager import get_collection_stats
            stats = get_collection_stats()
            print("向量库统计信息:")
            if stats["exists"]:
                print(f"  Qdrant集合: {QDRANT_COLLECTION}")
                print(f"  向量数量: {stats['count']}")
                print(f"  向量维度: {stats['vectors']['size']}")
                print(f"  距离度量: {stats['vectors']['distance']}")
                
                if "faiss" in stats:
                    if stats["faiss"]["exists"]:
                        if "error" in stats["faiss"]:
                            print(f"  FAISS索引: 存在但无法读取 ({stats['faiss']['error']})")
                        else:
                            print(f"  FAISS索引: 包含 {stats['faiss']['count']} 个向量")
                    else:
                        print("  FAISS索引: 不存在")
            else:
                print(f"  Qdrant集合 {QDRANT_COLLECTION} 不存在")
        else:
            logger.error(f"未知的管理命令: {args.manage_cmd}")
            print(f"错误: 未知的管理命令 '{args.manage_cmd}'")
    elif args.command == "compat":
        from compat.app_compat import run_legacy_app
        run_legacy_app()
    else:
        logger.error(f"未知命令: {args.command}")
        print(f"错误: 未知命令 '{args.command}'")
        parser.print_help()


if __name__ == "__main__":
    main()
