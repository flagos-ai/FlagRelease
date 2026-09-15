#!/usr/bin/env python3
"""
精度调优 checkpoint 持久化工具

每轮评测成功后立即调用，写入中间结果到 context.yaml，确保会话超时时不丢失调优进度。

用法:
  python3 persist_tuning_checkpoint.py <round> <score> <baseline> <disabled_ops_comma_separated>

示例:
  python3 persist_tuning_checkpoint.py 1 22.0 22.0 "addmm,mm,bmm"
  python3 persist_tuning_checkpoint.py 2 20.0 22.0 "addmm,mm,bmm,softmax"
"""
import sys
import subprocess
import datetime


def main():
    if len(sys.argv) < 5:
        print("用法: persist_tuning_checkpoint.py <round> <score> <baseline> <disabled_ops_comma_separated>")
        print("示例: persist_tuning_checkpoint.py 1 22.0 22.0 'addmm,mm,bmm'")
        sys.exit(1)

    round_num = sys.argv[1]
    score = float(sys.argv[2])
    baseline = float(sys.argv[3])
    disabled_ops = sys.argv[4]  # 逗号分隔，可能为空字符串

    drop_pct = round(baseline - score, 1)
    passed = drop_pct <= 5.0

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    # 写入本轮结果到 context.yaml
    cmd = [
        "python3", "/flagos-workspace/scripts/update_context.py",
        "--set", f"accuracy_tuning.round_{round_num}_score={score}",
        "--set", f"accuracy_tuning.round_{round_num}_baseline={baseline}",
        "--set", f"accuracy_tuning.round_{round_num}_drop_pct={drop_pct}",
        "--set", f"accuracy_tuning.round_{round_num}_passed={str(passed).lower()}",
        "--set", f"accuracy_tuning.round_{round_num}_disabled_ops={disabled_ops}",
        "--set", f"accuracy_tuning.round_{round_num}_timestamp={timestamp}",
        "--set", f"accuracy_tuning.latest_round={round_num}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"✗ 写入 checkpoint 失败: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"✓ Round {round_num} checkpoint 已写入: score={score}%, drop={drop_pct}%, passed={passed}")

    # 如果本轮达标，标记调优完成
    if passed:
        cmd_final = [
            "python3", "/flagos-workspace/scripts/update_context.py",
            "--set", "accuracy_tuning.completed=true",
            "--set", f"accuracy_tuning.final_disabled_ops={disabled_ops}",
            "--set", f"accuracy_tuning.final_round={round_num}",
            "--set", "workflow.accuracy_ok=true",
        ]
        result = subprocess.run(cmd_final, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"✗ 标记调优完成失败: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        print(f"✓ 精度调优达标（Round {round_num}），已标记 accuracy_ok=true")
    else:
        print(f"  本轮未达标，继续下一轮调优")


if __name__ == "__main__":
    main()
