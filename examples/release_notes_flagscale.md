# Release v1.0.0-alpha.0

**Changes since v0.9.0**

## New Features

- Native support for PI0 Training and Inference ([#871](https://github.com/flagos-ai/FlagScale/pull/871)) by @Hchnr
- Native Support for PI0 Serving ([#877](https://github.com/flagos-ai/FlagScale/pull/877)) by @Hchnr
- Fix qwen2.5-vl bug in the latest megatron backend ([#924](https://github.com/flagos-ai/FlagScale/pull/924)) by @heavyrain-lzy
- Robobrain-X0 Training Support ([#925](https://github.com/flagos-ai/FlagScale/pull/925)) by @Hchnr
- Fix qwen25vl config ([#929](https://github.com/flagos-ai/FlagScale/pull/929)) by @ceci3
- Support training of Qwen3-VL ([#951](https://github.com/flagos-ai/FlagScale/pull/951)) by @heavyrain-lzy

## Bug Fixes

- Fix the assignment logic of tp_split_dim in megatron DistributedOptimizer ([#992](https://github.com/flagos-ai/FlagScale/pull/992)) by @Darryl233
- Fix the assignment logic of tp_split_dim in megatron DistributedOptimizer for Metax ([#1014](https://github.com/flagos-ai/FlagScale/pull/1014)) by @yanqingliu35-stack

## Improvements

- Remove hardware ([#1018](https://github.com/flagos-ai/FlagScale/pull/1018)) by @cyber-pioneer

## Documentation

- Modify install section in README ([#1026](https://github.com/flagos-ai/FlagScale/pull/1026)) by @Darryl233

## CI/Infrastructure

- Add dependencies in the training environment ([#859](https://github.com/flagos-ai/FlagScale/pull/859)) by @zihugithub
- Fix the while loop exit logic ([#868](https://github.com/flagos-ai/FlagScale/pull/868)) by @zihugithub
- Change the random temporary directory to a fixed temporary directory ([#869](https://github.com/flagos-ai/FlagScale/pull/869)) by @wzj09
- Update gold value ([#870](https://github.com/flagos-ai/FlagScale/pull/870)) by @zihugithub
- Configure Gemini ([#874](https://github.com/flagos-ai/FlagScale/pull/874)) by @zihugithub
- Annotation inference use case ([#895](https://github.com/flagos-ai/FlagScale/pull/895)) by @zihugithub
- Add pr template ([#900](https://github.com/flagos-ai/FlagScale/pull/900)) by @cyber-pioneer
- Rename gemini-code-assist config file ([#902](https://github.com/flagos-ai/FlagScale/pull/902)) by @zihugithub
- Fix slow downloads in Megatron data tests ([#967](https://github.com/flagos-ai/FlagScale/pull/967)) by @Darryl233

## Hardware Support

- Support aquila training ([#906](https://github.com/flagos-ai/FlagScale/pull/906)) by @cifar10
- Add Cambricon_MLU and NVIDIA_CambriconMLU patch history ([#908](https://github.com/flagos-ai/FlagScale/pull/908)) by @shuyq10
- Update vllm for Qwen infer ([#950](https://github.com/flagos-ai/FlagScale/pull/950)) by @cifar10

## Other

- Update readme for v0.9.0 ([#850](https://github.com/flagos-ai/FlagScale/pull/850)) by @aoyulong
- Remove redundant logging in ChainedOptimizer ([#852](https://github.com/flagos-ai/FlagScale/pull/852)) by @chai-xiaonan
- Add another Submodule URL for PI0 ([#857](https://github.com/flagos-ai/FlagScale/pull/857)) by @Hchnr
- Pi0 Serving Support ([#862](https://github.com/flagos-ai/FlagScale/pull/862)) by @Hchnr
- Merge the `Megatron-LM` repository on Sep. 22nd ([#833](https://github.com/flagos-ai/FlagScale/pull/833)) by @heavyrain-lzy
- Update hygon patch history ([#861](https://github.com/flagos-ai/FlagScale/pull/861)) by @yanqingliu35-stack
- Add taylorseer transformation ([#844](https://github.com/flagos-ai/FlagScale/pull/844)) by @legitnull
- Update PI0 Codes and Examples ([#875](https://github.com/flagos-ai/FlagScale/pull/875)) by @Hchnr
- Refactor multiple-instance serve of openai type, and support tool call parameters ([#804](https://github.com/flagos-ai/FlagScale/pull/804)) by @luoyc123
- Fix model loading ([#883](https://github.com/flagos-ai/FlagScale/pull/883)) by @legitnull
- Update to v0.11.0 ([#887](https://github.com/flagos-ai/FlagScale/pull/887)) by @zhaoyinglia
- Update readme ([#890](https://github.com/flagos-ai/FlagScale/pull/890)) by @zhaoyinglia
- Update FlagScale for late patch of huawei ([#891](https://github.com/flagos-ai/FlagScale/pull/891)) by @heavyrain-lzy
- Refine git apply logic in unpatch.py to ensure patch hash consistency ([#899](https://github.com/flagos-ai/FlagScale/pull/899)) by @yanqingliu35-stack
- Adapt the training code backend for NVIDIA + Cambricon ([#907](https://github.com/flagos-ai/FlagScale/pull/907)) by @shuyq10
- Data Processing (JSON -> WebDataset) for PI0 and Robotics ([#903](https://github.com/flagos-ai/FlagScale/pull/903)) by @Hchnr
- Support RoboBrain-X0 Serving ([#912](https://github.com/flagos-ai/FlagScale/pull/912)) by @Hchnr
- Support robobrain_x0.5 serving ([#915](https://github.com/flagos-ai/FlagScale/pull/915)) by @Hchnr
- Deprecate vllm v0 ([#918](https://github.com/flagos-ai/FlagScale/pull/918)) by @zhaoyinglia
- Support serving with sglang backend on multiple nodes with different args ([#856](https://github.com/flagos-ai/FlagScale/pull/856)) by @luoyc123
- Support qwen2.5-Vl for huawei NPU ([#919](https://github.com/flagos-ai/FlagScale/pull/919)) by @heavyrain-lzy
- Request args support general standard types ([#923](https://github.com/flagos-ai/FlagScale/pull/923)) by @cyber-pioneer
- [Emu3.5] Support Emu3.5 Inference with vLLM ([#920](https://github.com/flagos-ai/FlagScale/pull/920)) by @zhaoyinglia
- Support serving for Emu3_5 model ([#911](https://github.com/flagos-ai/FlagScale/pull/911)) by @luoyc123
- Add qwen2.5-10b and qwen3-10b model yaml ([#941](https://github.com/flagos-ai/FlagScale/pull/941)) by @shuyq10
- Fix the bug of vllm adaptation ([#943](https://github.com/flagos-ai/FlagScale/pull/943)) by @luoyc123
- Remove third_party/lerobot ([#944](https://github.com/flagos-ai/FlagScale/pull/944)) by @legitnull
- Integrate TransformerEngine-FL based on FlagOS to Megatron-LM ([#917](https://github.com/flagos-ai/FlagScale/pull/917)) by @lxd-cumt
- Fix log collection and diagnostic for multi-node ([#882](https://github.com/flagos-ai/FlagScale/pull/882)) by @wanglei19991004
- Update te_fl yaml ([#952](https://github.com/flagos-ai/FlagScale/pull/952)) by @lxd-cumt
- Improve the usability of  installation tools ([#949](https://github.com/flagos-ai/FlagScale/pull/949)) by @Darryl233
- [AutoTuner] Enhance HeteroSearcher with advanced recompute, layer splitting, hetero_dp and heterogeneous memory model ([#854](https://github.com/flagos-ai/FlagScale/pull/854)) by @shuyq10
- Update mlu patch history ([#963](https://github.com/flagos-ai/FlagScale/pull/963)) by @yanqingliu35-stack
- Adapt to new MUSA env and support model Qwen2.5VL and Qwen3-8B. ([#961](https://github.com/flagos-ai/FlagScale/pull/961)) by @lingfengqiu
- Update vllm to support flaggems ([#970](https://github.com/flagos-ai/FlagScale/pull/970)) by @Rowman-G
- Fix vllm in online serving ([#969](https://github.com/flagos-ai/FlagScale/pull/969)) by @luoyc123
- Update TE-FL arguments ([#972](https://github.com/flagos-ai/FlagScale/pull/972)) by @lxd-cumt
- Update musa and hygon patch history ([#971](https://github.com/flagos-ai/FlagScale/pull/971)) by @yanqingliu35-stack
- Fix the bug when using pipeline parallel in Qwen3-VL ([#977](https://github.com/flagos-ai/FlagScale/pull/977)) by @heavyrain-lzy
- Correct flagscale commit id in hardware/MUSA_S5000/vllm. ([#980](https://github.com/flagos-ai/FlagScale/pull/980)) by @lingfengqiu
- Update MUSA patch history ([#981](https://github.com/flagos-ai/FlagScale/pull/981)) by @yanqingliu35-stack
- Add muon optimizer ([#978](https://github.com/flagos-ai/FlagScale/pull/978)) by @Caozhou1995
- Force to set `pipeline_model_parallel_size=1` in vision model ([#985](https://github.com/flagos-ai/FlagScale/pull/985)) by @heavyrain-lzy
- Fix license file for the project ([#989](https://github.com/flagos-ai/FlagScale/pull/989)) by @tengqm
- Add COC document ([#988](https://github.com/flagos-ai/FlagScale/pull/988)) by @tengqm
- Edit llvm compilation configs of MUSA kernels to support model… ([#997](https://github.com/flagos-ai/FlagScale/pull/997)) by @lingfengqiu
- Update MUSA patch_history ([#999](https://github.com/flagos-ai/FlagScale/pull/999)) by @yanqingliu35-stack
- Update vllm to v0.10.1 ([#1001](https://github.com/flagos-ai/FlagScale/pull/1001)) by @DannyP0
- Update iluvatar patch_history ([#1002](https://github.com/flagos-ai/FlagScale/pull/1002)) by @yanqingliu35-stack
- Support RoboBrain-X0 Training on Huawei_Atlas800TA3 ([#1000](https://github.com/flagos-ai/FlagScale/pull/1000)) by @Hchnr
- Refactor Runner of all tasks ([#996](https://github.com/flagos-ai/FlagScale/pull/996)) by @cyber-pioneer
- Update te-fl usage in flagscale ([#1009](https://github.com/flagos-ai/FlagScale/pull/1009)) by @lxd-cumt
- Update JPG Download URL for VLA Client ([#1005](https://github.com/flagos-ai/FlagScale/pull/1005)) by @Hchnr
- Update OpenSeek-10B ([#987](https://github.com/flagos-ai/FlagScale/pull/987)) by @RZJM
- Update Metax patch history ([#1012](https://github.com/flagos-ai/FlagScale/pull/1012)) by @yanqingliu35-stack
- Update Metax patch history ([#1016](https://github.com/flagos-ai/FlagScale/pull/1016)) by @yanqingliu35-stack
- Remove the training and serving of robotics ([#1015](https://github.com/flagos-ai/FlagScale/pull/1015)) by @Hchnr
- Refactor FlagScale by integrating Megatron-LM-FL ([#958](https://github.com/flagos-ai/FlagScale/pull/958)) by @lxd-cumt
- Refactor config files ([#1011](https://github.com/flagos-ai/FlagScale/pull/1011)) by @cyber-pioneer
- Rm backend and use vllm-fl repo ([#1004](https://github.com/flagos-ai/FlagScale/pull/1004)) by @zhaoyinglia
- Remove backends, runner/estimator ([#1019](https://github.com/flagos-ai/FlagScale/pull/1019)) by @lxd-cumt
- Remove third party ([#1022](https://github.com/flagos-ai/FlagScale/pull/1022)) by @lxd-cumt
- Support RoboBrain-X0.5 training ([#1020](https://github.com/flagos-ai/FlagScale/pull/1020)) by @Hchnr
- Remove unnecessary log ([#1024](https://github.com/flagos-ai/FlagScale/pull/1024)) by @cyber-pioneer
- Support PI0/PI0.5 ([#1021](https://github.com/flagos-ai/FlagScale/pull/1021)) by @legitnull
- Remove backend install ([#1025](https://github.com/flagos-ai/FlagScale/pull/1025)) by @Darryl233

## Contributors

Thanks to all contributors who made this release possible:

- @Hchnr (13 PRs)
- @yanqingliu35-stack (10 PRs)
- @heavyrain-lzy (7 PRs)
- @lxd-cumt (7 PRs)
- @cyber-pioneer (6 PRs)
- @zihugithub (6 PRs)
- @Darryl233 (5 PRs)
- @luoyc123 (5 PRs)
- @zhaoyinglia (5 PRs)
- @legitnull (4 PRs)
- @shuyq10 (4 PRs)
- @lingfengqiu (3 PRs)
- @cifar10 (2 PRs)
- @tengqm (2 PRs)
- @Caozhou1995
- @DannyP0
- @RZJM
- @Rowman-G
- @aoyulong
- @ceci3
- @chai-xiaonan
- @wanglei19991004
- @wzj09

---
*Generated by [Release Notes Generator](https://github.com/flagos-ai/FlagRelease/tree/main/tools)*