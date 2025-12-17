🌐 Language:
[English](README.md) | [简体中文](README.zh-CN.md)

# FlagRelease

FlagRelease is a large-model automated migration, adaptation, and release platform developed by the Beijing Academy of Artificial Intelligence (BAAI) for multi-architecture artificial intelligence chips. The platform aims to enable mainstream large models to be migrated, validated, and released on diverse domestic AI hardware with lower cost and higher efficiency through automated, standardized, and intelligent adaptation workflows.
Built upon the unified and open-source AI system software stack FlagOS, which provides cross-hardware adaptation capabilities, FlagRelease establishes a standardized pipeline that supports automatic migration of large models to different hardware architectures, automated evaluation of migration results, built-in automated deployment and tuning, and multi-chip model packaging and release.
The artifacts released through the FlagRelease platform are published on ModelScope and Hugging Face under the FlagRelease organization, where users can obtain different hardware-specific versions of open-source large models. These models can be downloaded and used directly on the corresponding hardware environments without requiring users to perform model migration themselves, significantly reducing the migration cost for end users.
Currently, the outputs of the FlagRelease platform include validated, hardware-adapted model files and integrated Docker images. Each image contains the core components of FlagOS along with all required model dependencies, allowing users to deploy and use the models directly on the target chips. In addition, each model release provides evaluation results as technical references, enabling users to clearly understand the model’s correctness and performance characteristics across different hardware platforms.
Furthermore, every released model is accompanied by configuration and usage instructions for AnythingLLM, helping users quickly verify the availability of the migrated models and facilitating downstream development and application based on these models.
The overall architecture of FlagOS is illustrated in the figure below:
   
![](assets/flagos.jpeg)

## Release Notes

<!-- START:models -->

<!-- END:models -->

## Example Usage of Released Artifacts
The outputs of FlagRelease include validated large-model files and integrated FlagOS Docker images. By using these artifacts, users can rapidly deploy and run large models on different hardware platforms without performing model migration themselves or configuring complex software environments.
Example Workflow
1. Download Open-Source Model Weights
  - Visit the FlagRelease pages on ModelScope or Hugging Face, select the required large model and the corresponding hardware-specific version, and download the model weight files directly.
2. Download the FlagOS Image
  - Obtain the officially provided integrated FlagOS Docker image, which includes the unified software stack and built-in hardware adaptation support.
3. Deployment and Execution
  - Combine the downloaded model weights with the FlagOS image to run the model directly on the target hardware.
  - FlagOS automatically manages hardware resources and supports multi-chip parallel execution, eliminating the need for manual environment configuration.
Example Application Scenarios
- Research and experimentation: rapidly deploy large models for inference without concern for underlying hardware differences.
- Production environments: directly deploy hardware-specific versions of models as services, ensuring performance and stability across different AI chips.

