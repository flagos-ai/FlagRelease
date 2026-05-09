#!/usr/bin/env python3
"""
FlagRelease - 模型发布流水线工具
主入口文件
"""
import sys
import time
import argparse
from typing import List, Dict, Any

from src.config import load_config_from_context, validate_config, auto_fill_config, PipelineConfig
from src.stages import PublishStage
from src.utils import print_banner, print_config_summary, print_stage_summary


class Pipeline:
    """发布流水线"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results: List[Dict[str, Any]] = []

    def run(self) -> bool:
        """执行流水线"""
        print_banner()
        print_config_summary(self.config)

        start_time = time.time()
        all_success = True

        # 依次执行各阶段
        stages_map = {
            "publish": PublishStage
        }

        for stage_name in self.config.stages_to_run:
            if stage_name not in stages_map:
                print(f"警告: 未知的阶段 '{stage_name}'，跳过")
                continue

            # 检查阶段是否启用
            stage_config = getattr(self.config, stage_name, None)
            if stage_config and hasattr(stage_config, 'enabled') and not stage_config.enabled:
                print(f"跳过阶段: {stage_name} (配置禁用)")
                continue

            # 创建并执行阶段
            stage_class = stages_map[stage_name]
            stage = stage_class(self.config)

            try:
                result = stage.run()
                result_dict = {
                    "stage_name": result.stage_name,
                    "success": result.success,
                    "total_duration": result.total_duration,
                    "error": result.error,
                    "steps": [
                        {
                            "name": s.step_name if hasattr(s, 'step_name') else s.get('step_name'),
                            "status": s.status.value if hasattr(s, 'status') else s.get('status'),
                        }
                        for s in result.steps
                    ]
                }
                self.results.append(result_dict)

                if not result.success:
                    all_success = False
                    print(f"\n✗ 阶段 '{stage_name}' 失败，停止后续执行")
                    break

            except Exception as e:
                print(f"\n✗ 阶段 '{stage_name}' 执行异常: {e}")
                self.results.append({
                    "stage_name": stage_name,
                    "success": False,
                    "error": str(e),
                    "total_duration": 0
                })
                all_success = False
                break

        # 打印摘要
        total_duration = time.time() - start_time
        print_stage_summary(self.results)

        summary = {
            "总耗时": f"{total_duration:.2f}秒",
            "执行阶段数": len(self.results),
            "成功阶段数": sum(1 for r in self.results if r.get('success')),
        }
        print(f"\n流水线总结: {summary}")

        return all_success


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="FlagRelease - 模型发布流水线工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 context.yaml 自动生成配置并执行发布
  python main.py --from-context /flagos-workspace/shared/context.yaml

  # 覆盖容器名
  python main.py --from-context /flagos-workspace/shared/context.yaml --container-name mycontainer

  # 只生成 README
  python main.py --from-context /flagos-workspace/shared/context.yaml --only-readme

  # 干运行（不实际执行）
  python main.py --from-context /flagos-workspace/shared/context.yaml --dry-run
"""
    )

    parser.add_argument(
        "--from-context",
        required=True,
        help="FlagOS context.yaml 路径，从中自动生成发布配置"
    )
    parser.add_argument(
        "-s", "--stages",
        help="要执行的阶段，逗号分隔 (publish)"
    )
    parser.add_argument(
        "--container-name",
        help="覆盖 context.yaml 中的容器名称"
    )
    parser.add_argument(
        "--only-readme",
        action="store_true",
        help="只生成 README，不执行其他发布步骤"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干运行模式，只打印配置不实际执行"
    )
    parser.add_argument(
        "--plugin-mode",
        action="store_true",
        help="Plugin 发布模式：镜像 tag 追加 -plugin，仓库名追加 -plugin，发布后更新已发布仓库 README"
    )
    parser.add_argument(
        "--only-harbor",
        action="store_true",
        help="只执行 Harbor 推送（commit→tag→push），跳过 README/ModelScope/HuggingFace"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出模式"
    )

    args = parser.parse_args()

    # 加载配置
    try:
        config = load_config_from_context(args.from_context)
    except FileNotFoundError:
        print(f"错误: 文件不存在: {args.from_context}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 加载配置失败: {e}")
        sys.exit(1)

    # 命令行参数覆盖配置
    if args.stages:
        config.stages_to_run = [s.strip() for s in args.stages.split(",")]

    if args.container_name:
        config.container_name = args.container_name

    if args.only_readme:
        config.stages_to_run = ["publish"]
        config.publish.tag_image = False
        config.publish.push_harbor = False
        config.publish.publish_modelscope = False
        config.publish.publish_huggingface = False

    if args.plugin_mode:
        config.plugin_image_mode = True
        config.publish.existing_harbor_image = ""

    if args.only_harbor:
        config.stages_to_run = ["publish"]
        config.publish.generate_readme = False
        config.publish.publish_modelscope = False
        config.publish.publish_huggingface = False

    # 自动填充配置
    print("正在检测环境信息...")
    config = auto_fill_config(config)
    print(f"检测到芯片厂商: {config.model_info.vendor or config.chip.vendor}")

    # 验证配置
    errors = validate_config(config)
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    # 干运行模式
    if args.dry_run:
        print_banner()
        print("\n[干运行模式] 配置验证通过，以下是自动生成的配置:\n")
        print(f"  芯片厂商: {config.chip.vendor} ({config.model_info.vendor})")
        print(f"  模型名称: {config.model_info.output_name}")
        print(f"  FlagRelease名称: {config.model_info.flagrelease_name}")
        print(f"  镜像目标Tag: {config.publish.image_target_tag}")
        print(f"  ModelScope ID: {config.publish.modelscope_model_id}")
        print(f"  HuggingFace ID: {config.publish.huggingface_repo_id}")
        print(f"  Docker版本: {config.model_info.docker_version}")
        print(f"  系统版本: {config.model_info.ubuntu_version}")
        print(f"  驱动版本: {config.chip.driver_version}")
        print(f"  SDK版本: {config.chip.sdk_version}")
        print(f"  PyTorch版本: {config.chip.torch_version}")
        print(f"  Python版本: {config.chip.python_version}")
        print(f"  GPU型号: {config.chip.gpu_model}")
        print("\n实际执行时将运行以下阶段:")
        for stage in config.stages_to_run:
            print(f"  - {stage}")
        print("\n使用 --help 查看更多选项")
        sys.exit(0)

    # 执行流水线
    pipeline = Pipeline(config)
    success = pipeline.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
