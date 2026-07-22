#!/bin/bash
# FlagRelease 旁路汇报入口。只投递/消费事件，绝不启动或控制迁移 pipeline。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/progress_worker.py" "$@"
