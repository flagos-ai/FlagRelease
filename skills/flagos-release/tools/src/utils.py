"""
工具函数
"""
import os
import sys
from datetime import datetime
from typing import Dict, Any, List


def print_banner():
    """打印 Banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗██╗      █████╗  ██████╗ ██████╗ ███████╗██╗        ║
║   ██╔════╝██║     ██╔══██╗██╔════╝ ██╔══██╗██╔════╝██║        ║
║   █████╗  ██║     ███████║██║  ███╗██████╔╝█████╗  ██║        ║
║   ██╔══╝  ██║     ██╔══██║██║   ██║██╔══██╗██╔══╝  ██║        ║
║   ██║     ███████╗██║  ██║╚██████╔╝██║  ██║███████╗███████╗   ║
║   ╚═╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝   ║
║                                                               ║
║              Model Release Pipeline Tool v1.0                 ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_config_summary(config) -> None:
    """打印配置摘要"""
    print("\n配置摘要:")
    print(f"  容器名称: {config.container_name}")
    print(f"  模型名称: {config.model_info.output_name}")
    print(f"  供应商: {config.model_info.vendor}")
    print(f"  执行阶段: {', '.join(config.stages_to_run)}")
    print()


def print_stage_summary(results: List[Dict[str, Any]]) -> None:
    """打印阶段执行摘要"""
    print("\n" + "="*60)
    print("执行摘要")
    print("="*60)

    total_duration = 0
    all_success = True

    for result in results:
        stage_name = result.get('stage_name', 'Unknown')
        success = result.get('success', False)
        duration = result.get('total_duration', 0)
        total_duration += duration

        status_icon = "+" if success else "x"
        status_text = "成功" if success else "失败"

        print(f"  {status_icon} {stage_name}: {status_text} ({duration:.2f}s)")

        if not success:
            all_success = False
            if result.get('error'):
                print(f"      错误: {result['error']}")

    print("-"*60)
    print(f"  总耗时: {total_duration:.2f}s")
    print(f"  最终状态: {'+ 全部成功' if all_success else 'x 存在失败'}")
    print("="*60 + "\n")


def format_duration(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.0f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}小时{minutes}分"


def ensure_dir(path: str) -> None:
    """确保目录存在"""
    dir_path = os.path.dirname(path) if not os.path.isdir(path) else path
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def get_timestamp() -> str:
    """获取时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
