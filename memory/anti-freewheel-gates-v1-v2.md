---
name: anti-freewheel-gates-v1-v2
description: 防Claude自由发挥——V1三选强制闸门+V2精度条件兜底两道产物级闸门
metadata:
  type: project
---

2026-07 加固:排查分支B流程中"Claude不用规定脚本自作主张"的口子,补两道产物级闸门(与 [[step7-gate-anti-hallucination]] 同范式:只信agent无法伪造的产物事实,不读Claude写的ok字段)。用户拍板先做这两个,发布镜像校验(口子3)暂不做。完整方案见仓库 防自由发挥加固方案.md。

**口子1:V1三选强制闸门(上游根因)**
问题:baseline_selector.py 全程无shell强制调用点,Claude可能自起服务测一下就把 baseline.v1_variant/v1_available 写成臆断值,导致下游V2分支(2.1/2.2)、合成基线触发、精度基线回退NV全建立在未验证判据上。
- 新增 prompts/v1_gate.py:只认 v1_baseline_selection.json 里 baseline_selector 产出的 attempts[] 真实痕迹(每变体含 variant/service_ok/smoke_passed,且必含v1.1起点)。输出 ok/needed。Claude手写context或伪造简单json都产生不了完整attempts。
- run_pipeline.sh ~1272(段2 context同步后、精度评测兜底前,**仅 PIPELINE_BRANCH=B**):v1_gate needed → shell docker exec 直调 baseline_selector.py(--vendor-plugin auto),同步产出后复检。
- 坑:read_context 只返回3字段(container|env|last_step)**不含port**,原想 cut -f4 取port是错的;改为额外 docker exec 读 context 的 service.port + model.name。

**口子2:V2精度条件兜底(零成本复用)**
问题:run_eval_if_missing 已是通用函数但只调过一次(gpqa_native.json=V1),V2(gpqa_flagos.json)没兜。
- 关键:V2精度**必须在FlagGems服务(USE_FLAGGEMS=1)下测**,段2结束时服务模式不确定。直接照搬V1兜底会拿native/错误配置服务测出污染的V2分数,比不兜底更糟。
- 用户选"条件兜底+retry回退":新增 is_flaggems_service()(判据=startup_default.log软链指向startup_flagos* 或 /etc/environment有USE_FLAGGEMS=1)。V2产物无效时:当前是FlagGems服务→run_eval_if_missing直接测;否则**不硬测**,输出提示交由现有段2 retry让Claude用skill正确切服务后重测。

均 bash -n / py_compile 通过,v1_gate 六场景验证通过。未真机,未提交(与前几批攒一起)。相关:[[v1-3-stale-service-false-negative]]、[[step7-gate-anti-hallucination]]。