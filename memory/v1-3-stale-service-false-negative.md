---
name: v1-3-stale-service-false-negative
description: 国产厂商V1.3被误判服务启动失败——残留服务检测过度依赖service_ready日志短语
metadata:
  type: project
---

2026-07 修复:多个国产厂商上自动化流程认为 V1.3(VLLM_PLUGINS=fl USE_FLAGGEMS=0)启动失败,但手动 curl 验证服务正常,导致这些厂商被判无 V1 基线。

**根因**(wait_for_service.sh 残留服务检测):
- LOG_CONFIRMED_READY 只在日志出现 `Application startup complete`/`Ready to serve`(原生 vLLM uvicorn 短语)时置 true
- 残留检测旧逻辑:端口 /v1/models 已响应 **但** LOG_CONFIRMED_READY=false → 连续3次判 stale_service 假失败 exit 2
- 国产厂商 vLLM 分支(vllm-ascend/寒武纪/海光等)就绪短语与原生 vLLM 不同,抓不到 service_ready → 服务真起来了(端口能响应=手动验证正常)却被误判残留(注:本流程启动路径均为 vLLM / vLLM+fl plugin / vLLM+厂商插件,不经 FlagScale)
- V1.3 尤其中招:依赖 fl plugin 注册 platform,启动路径/日志格式与原生 vLLM 差异大

**修复(用户选"观测到PROGRESS即放行")**:
在残留检测条件加 `[ -z "$PHASES_OBSERVED" ]`——只有本次启动**无任何进度信号**才判残留。
- 真残留=旧进程占端口、本次没起新服务 → PHASES_OBSERVED 空 → 仍拦截
- 国产V1.3=端口响应+观测到 loading_weights/gpu_initialized/port_bound 等进度、仅缺 service_ready 短语 → PHASES_OBSERVED 非空 → 判就绪放行
- port_bound(Uvicorn running on)等进度总在端口可响应前打出,时序上可靠

**已知残留风险**(未覆盖):日志有内容但无任何一条匹配 PROGRESS 正则的极端情况仍会误判。baseline_selector.py 的 start_variant 已把 stdout/stderr 重定向到 log_path,不存在日志完全空的情况。

另注:smoke_test 关键词仅 ["北京","Beijing","beijing"](baseline_selector.py:41),reasoning 模型 <think> 包裹或异常输出结构可能取不到 content——本次未改,若后续仍有冒烟误判可查此处。相关:[[plugin-inheritance-cold-injection]]。未真机,未提交。