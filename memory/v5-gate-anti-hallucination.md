---
name: v5-gate-anti-hallucination
description: 段5 V5精度探测强制闸门——用产物痕迹而非agent判据防止臆断跳过operator_expansion精度调优，并把V5发布从agent会话剥离到shell
metadata:
  type: project
---

2026-07 修复:摩尔迁移 LFM2.5-1.2B-Thinking,V5 开启最多算子精度不达标。**本应**逐个探测算子、精度不达标的标 accuracy_harmful 回退(operator_expansion.py 逻辑本身是对的)。**实际**:agent 不信任 prompt 显式规则,从 git commit message 推断出错误逻辑,跳过精度调优阶段直接发布 V5。

**根因**(与 [[step7-gate-anti-hallucination]] 同构):段5 把"精度探测一定执行"托付给 agent 自觉,发布(步骤15)又在 agent 会话内,中间无确定性检查。agent 臆断跳过就能抢先发布不合格 V5 到 Harbor(flagrelease-project),事后补跑救不回已推的镜像。

**用户明确否定"加提示词"**:这次 bug 本质就是 agent 无视已有 prompt 规则,再加一条"禁止从git推断"是用同一失效机制打补丁,且推断来源(git/注释/旧经验)堵不全。治本只能靠确定性闸门。

**用户红线(2026-07)**:不在乎产出镜像的标签,只要"三组件齐全+精度达标"就能交付;但"明明可以调优却没调优、精度不达标还强行产出"绝不容忍。→ 闸门要守的是**结果(精度达标)**,不只是手段(探测跑没跑)。

**修复(纯确定性闸门,不加任何 prompt 规则)**:v5_gate.py 双子命令。
- **gate 子命令(精度探测该不该补跑)**:两类 agent 无法伪造的事实。
  (1) 是否有算子可扩展——**四源兜底**判断(对齐 run_pipeline.sh HAS_DISABLED_OPS,不因单一字段没写就漏判):源1 optimization.disabled_ops / 源2 eval.excluded_ops_accuracy / 源3 别名字段(workflow/optimization 的 v2_disabled_ops/disabled_ops/excluded_ops) / **源4 service.initial_operator_list 与 optimization.enabled_ops 的差集(不依赖字段名,最难写偏)**。任一源显示有禁用即要求探测必须跑。
  (2) 精度探测是否真跑过——operator_config_v5.json 的 actual_rounds>0 / probed_ops / tier_results / completed=true
  输出:no_expansion(四源均空→V5=当前最优,放行交精度终检)/done(有算子且真探测过)/needed(有算子无痕迹→臆断跳过,补跑)/no_data
- **accuracy 子命令(发布前精度终检,守红线的最后一道关)**:shell 独立读 gpqa_v5.json(V5最终精度)与 gpqa_native.json(V1基线),算 diff=v1-v5,diff>threshold(5.0,与 operator_expansion.run_accuracy_check 同义)→ 拒发。输出 pass(达标/no_expansion由V3保证)/fail(不达标,拒发)/no_result(做过扩展却缺gpqa_v5.json,不确定即不发)/no_data(缺基线,保守拒发)。**即便 agent 绕过所有探测/补跑,最终镜像精度不达标也发不出去。**
- **关键边界**:operator_expansion.py 在"无禁用算子"时走早退分支(operator_expansion.py:552,**不写 state 文件**),所以"合法无需扩展"与"臆断跳过"都表现为 state 文件缺失,必须靠四源判断这个独立事实源区分,不能只看 state 是否存在。
- **为什么四源而非单源**:用户质疑"若上游 agent 没把 disabled_ops 正确写入,闸门是否照样漏判"——对,单读 optimization.disabled_ops 会因该字段没写而误判 no_expansion→跳过调优。四源兜底(尤其源4差集不依赖字段名)把误判从"任一字段写对才安全"收紧到"所有来源都空才判无需扩展",拉平到仓库现有最强判据。**根本局限**:若上游所有来源全没写,任何下游闸门都兜不住(闸门=核对事实源,源全错则失基准),但 accuracy 终检仍能兜住"精度不达标"这个最终结果。
- **V5 发布从 agent 会话剥离到 shell**(用户选定):PROMPT_SEG5 改为只跑步骤14扩展、禁止发布;shell 在 v5_gate 通过(done/no_expansion,或经补跑/兜底)后才 `main.py --version-tag v5`。needed→先让 agent 补跑扩展会话→复检仍 needed/no_data→shell 直接 docker exec 调 operator_expansion.py 兜底,绕开 agent。发布前置条件由 shell 确定性保证,agent 无法抢先发不合格 V5。
- **断点续跑重入保护**(剥离发布引入的回归防护):shell 发布前读 workflow.v5_released,=true 则跳过;发布成功后 shell 写 workflow.v5_released=true(容器内 update_context.py + 回写 snapshot)。

接线位置:run_pipeline.sh 段5 会话结束、context 同步后、`else` 分支前。V5→flagrelease-project 仓库路由由 config.py:440 自动处理,shell 只传 --version-tag v5。SCRIPT_DIR 指向 prompts/,v5_gate.py 与 step7_gate.py 同目录。

**深度审查又抓到的两个致命 bug(2026-07,均已修)**:
1. **精度终检读错文件系统**:gpqa_v5.json/gpqa_native.json 由 operator_expansion.py 写在**容器内** /flagos-workspace/results/,容器与宿主机**无 volume 挂载**(两套独立 fs,靠 docker cp 同步)。终检原读宿主机路径 → gpqa_v5.json 从来不存在 → 每个达标 V5 判 no_result 全误杀拒发。修:终检前 docker cp gpqa_native/gpqa_v5 从容器同步到宿主机。宿主机全 workspace 搜不到任何 gpqa_v5.json 就是这个原因(V5 结果从不落宿主机)。
2. **旧文件残留读旧精度**:断点续跑/重试时宿主机可能残留上轮 gpqa_v5.json,若本轮 cp 失败(容器没生成)→ 读到上轮旧精度 → 可能把本轮不达标 V5 误判达标发布(破红线)。修:docker cp 前 rm -f 宿主机旧 gpqa_v5.json，确保读到本轮的或明确缺失(no_result 拒发)。
一致性确认(非bug):脚本 write_control_file(enabled_ops)写容器控制文件→测 gpqa_v5.json→发布 docker commit 打包该容器,三者同源,终检精度=发布镜像精度。

**真实数据抓到的 bug(2026-07 快速验证)**:真实 context 里 optimization.disabled_ops 存的是**字符串** `'[zeros,ones,full,...]'`(带方括号),不是 YAML list;enabled_ops 常为 `[]`。原 `_as_list` 只认 list → 源1把字符串当空 + 源4差集因 enabled 空而 falsy 短路 → 四源全失效 → 误判 no_expansion → 精度探测被跳过(正是本 bug 的另一条触发路径)。修复:新增 `_as_ops()` 容忍 list/`'[a,b]'`/`'a,b'` 三种写法;源4 改为 `initial and (initial-enabled)`(不再要求 enabled 非空)。用真实 ERNIE-4.5-21B 数据在容器 gems_tree_plugin(py3.12.3)内实跑确认:gate→needed(修复前 no_expansion)、accuracy(v1=90) v5=86→pass / v5=75→fail拒发 / 缺v5→no_result。**教训:构造数据测不出真实字段格式,必须用真实产物验证。**

验证:gate 五场景(源1/源2/源4任一有算子+无探测=needed / 源4+真探测=done / 四源全空=no_expansion)全过;accuracy 五场景(达标=pass / 退化>5=fail拒发 / 缺v5结果=no_result拒发 / 无算子=pass / 缺基线=no_data)全过;端到端集成四场景(摩尔事故重演=needed拦截 / 探测完但精度不达标=fail拒发 / 达标=发布 / 无算子=V3保证发布)全过;v5_released 重入2场景过;bash -n + py_compile 过。未真机验证。未提交。

相关:[[step7-gate-anti-hallucination]] [[anti-freewheel-gates-v1-v2]] [[no-block-full-chain-v2-v5]] [[v5-operator-expansion-whitelist-bug]] [[v4-v5-baseline-optimization-enabled-ops]]
