---
name: hygon-pipeline-bug-fixes
description: 海光机器上四个 pipeline 缺陷的根因与已实施的修复（V4被kill/V5误跳/DEBUG行污染/报告缺算子）
metadata:
  type: project
---

海光(Hygon DCU)运行时暴露的四个 pipeline 缺陷。绝对时间：2026-07 修复前三个。

## 问题1：V4 后台跑 + Claude 提前结束 → pipeline kill 掉在跑的脚本（最严重，已修）
- 根因：段4.5 prompt 只文字要求"前台执行"靠自觉；Claude 转后台跑、看到有输出就发 result 结束会话 → shell 段结束流程 kill 掉仍在跑的 operator_reduction.py。gemma3/glm4 因此 V4 中断。
- 修复（三重保险）：
  1. operator_reduction.py 所有退出路径写完成标记 `results/v4_reduction.done`（含 exit_code/success/final_composite/时间戳）；起始清旧标记；中断/崩溃则无标记。
  2. run_pipeline.sh 段4.5 Claude 返回后**轮询校验**：`.done` 存在=真完成；不存在但 `pgrep operator_reduction.py` 活=继续等（上限 15h30m）；进程死且无标记=判 incomplete/告警不静默跳过。
  3. prompt 强化：禁止后台化(&/nohup/disown/run_in_background)、禁止"看起来在跑"就结束、命令返回后必须回读 .done 确认。

## 问题2：V5 被误跳过（字段名脆弱，已修）
- 根因：V5 跳过判据(run_pipeline.sh ~2091)只读 `optimization.disabled_ops` 单字段；Claude 曾把禁用算子写到 `workflow.v2_disabled_ops`（代码里无此字段，纯写错）→ 读空 → 误判无禁用 → 跳过 V5。glm4 中招。
- 修复：改**多源兜底**——optimization.disabled_ops + eval.excluded_ops_accuracy + 常见错误字段(v2_disabled_ops等) + 初始全量集vs当前启用集差集(最可靠不依赖字段名)，任一显示有禁用即触发 V5。

## 问题3：DEBUG行被当算子名 → other组全关 → 崩溃（已修）
- 根因：diagnose_ops.py generate_accuracy_groups 加载 ops_list.json 零校验，污染行(如 `[DEBUG] flag_gems.ops.add.add: GEMS ADD`)原样进 all_ops → 匹配不上任何组 → 全掉进 other 组 → 整组关闭=flaggems全关=v1.3 → 触发环境bug崩溃跳过v3-v5。hunyuan7B 43算子全掉 other 组中招。
- 修复：新增 `sanitize_ops_list()`——DEBUG行用 `flag_gems\.ops\.(\w+)` 提取真算子名、其余要求合法短标识符(`^[a-z][a-z0-9_]{1,39}$`)、剔污染去重；净化后全空则报错。加 **other组熔断**：占比>50% 时拆成逐算子单测(other_<op>)，避免整组关闭等价全关触发崩溃。返回加 dropped_count/other_fuse_warning。

## 问题4：总结报告缺算子列表（未单独修，判为问题1下游后果）
- 现状：shared/generate_report.py 已有多级回退(op_config→context.yaml→oplist txt，commit 18ca63b)。V4被kill→无 operator_config_v4.json/v4_oplist.txt→报告V4栏空。修好问题1后应自然缓解。待用户提供实际报告确认哪版本栏空再定是否单独改。

## 验证状态
三处编译/语法均通过；问题2/3 有单元自测通过。均未在海光真机验证。相关：[[tuning-logic-fixes-plan]]

## 追加：镜像模式复用旧容器事故（真机排查，已修 2026-07）
- 现象：tasks.txt 换新镜像跑批，agent 发现同名旧容器（如 gemma-3-1b-it_flagos，旧镜像建的）直接复用，且镜像不在本地时从未 docker pull → 新任务全程跑在旧镜像上，结果错误归属。
- 三个缺陷：①流程无 docker pull 步骤（隐含假设镜像已在本地）；②容器名冲突"追加时间戳"只是 prompt 文字，agent 靠自觉；③docker run 失败的降级路径（"借鉴已有容器"）被滑坡成复用旧容器。
- 修复（确定性归 shell，agent 只消费）：
  1. run_pipeline.sh 镜像模式 pre-flight 新增：docker image inspect 存在性检查 → 不在本地 docker pull（直连失败按 PROXY_LIST 逐个重试）→ 全失败明确 exit 1（拒绝降级，注明"复用旧容器=旧镜像跑新任务比失败更糟"）。
  2. shell 预生成容器名 CONTAINER_NAME_PRE（冲突必然追加 _MMDD_HHMM，同分钟再冲突加秒），注入 STEP1 prompt，agent 禁止自行生成/判断/修改。
  3. STEP1 prompt：禁止 docker pull（编排层已做）、降级策略改为"仅抄挂载参数拼新命令，容器名不变"、绝对禁止复用已存在容器。
  4. container-preparation SKILL.md 入口2 同步（镜像就绪保证归编排层、容器名注入制、禁止行为写明事故教训）。
- mock docker 四场景模拟验证通过：本地有镜像/需拉取/拉取全失败exit1/同名容器强制时间戳。未真机验证。
