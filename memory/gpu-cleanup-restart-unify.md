---
name: gpu-cleanup-restart-unify
description: 显存清理统一为docker restart优先(任务结束层已改);容器内无docker只能pkill的约束
metadata:
  type: project
---

2026-07-13(test分支)显存清理/服务重启机制梳理与第1+2层改动。

**机制全貌(清理点分两类,按执行环境)**:
- 宿主机层(有docker): run_pipeline.sh `cleanup_gpu_services`(trap EXIT+流程结尾显式调用)、safe_restart_service.sh
- 容器内层(无docker,只能pkill): operator_search.py/operator_reduction.py/operator_expansion.py/baseline_selector.py/persist_op_config.py/start_service.sh。这些经 docker exec 调起,容器内跑。

**硬约束**: docker run模板未挂docker.sock,且NVIDIA平台严禁加模板外参数(SKILL.md:146),所以**容器内无法执行docker restart**。operator_search.py:1164已有的docker restart fallback此前还因 /flagos-workspace/.container_name 从未被写入而拿不到容器名,双重失效。

**本次改动(第1+2层)**:
1. run_pipeline.sh cleanup_gpu_services(~721): 从 pkill 改为**优先docker restart**(宿主机执行最彻底,回收僵尸worker),失败降级pkill -9。有docker inspect守卫。
2. setup_workspace.sh(~394): 容器就绪后写 `/flagos-workspace/.container_name`(紧邻.mount_mode),激活容器内已有的docker restart fallback。

**诚实局限**: 标准NVIDIA模板下容器内仍无docker CLI,改动2的fallback只在"容器能访问docker"的平台生效(run_cmd容错check=False不崩)。彻底解决容器内pkill清不净需第3层信号机制(容器内写信号→宿主机restart),本次未做。任务结束(宿主机层)是主场景,已彻底解决。

关联: [[rollback-overflow-fix]] 同属 run_pipeline.sh 编排层修复。