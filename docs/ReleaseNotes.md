# Release Notes

## 1.5.0 Release
- New code agent sample
- New modern C++ code samples
- New custom plugin creation tutorial
- Performance improvements
>KNOWN ISSUES:
>- The NVIGI core unit tests (`nvigi.tests.exe`) currently does not support AMD hardware; this will be fixed in a future release.
>- The NVIGI SDK's [Function Call sample](SamplesFcall.md) may hang at first inference when built from source while using older compiler components; this will be fixed in a future release.
>- In detailed testing of the [3D Sample](Samples3D.md) in Vulkan rendering with the ASR GGML Vulkan backend, we have found an intermittent crash in (only) the first inference on RTX4090.  When run in a Vulkan-based 3D application, we recommend using the GGML CUDA ASR backend, rather than the GGML VK ASR backend with this release.  We are currently investigating this issue.

## 1.4.0 Release
- New models supported: Nemotron Nano 9B v2, Qwen3 0.6B, Qwen3 4B, Qwen3 8B
- Performance improvements
- Documentation improvements

## 1.3.0 Release
- This release of NVIGI includes a significant upgrade to how the SDK pack is laid out and organized.  The Plugin SDK is now the top-level directory, and the existing 3D Sample has now been moved into the same directory tree and build solution as all of the command-line samples.  This simplifies the build, run and debugging workflow for the 3D sample, making it a first-class citizen of the SDK.

## 1.2.0 Release
- We are investigating a potential issue on some configurations where the 3D Sample can crash the first time a model is loaded. If you hit this problem, please reboot and rerun the sample.
- The models for the Riva Magpie Flow TTS plugins are shipped in the pack itself; we expect that in future releases, the models will be downloadable via the `download_data.bat` script.
- The previous ASquaredFlow TTS model has been updated and is now known as "Riva Magpie-TTS-Flow".  The plugin itself retains the original name to avoid problems for existing applications.  However, the new model must be used, as the ones shipped with the 1.1.1 release are no longer compatible with the updated plugin.
- This release adds Vulkan and D3D12 GGML backend plugins for ASR and GPT; they are experimental and not intended for production use.  They allow GPU GPT and ASR on general GPU architectures/vendors.
- This release adds Vulkan support to the 3D Sample via the `-vk` command-line option; it is experimental and not intended for production use.  It allows the sample to run on general GPU architectures/vendors.
- The D3D12 GGML plugins (ASR and GPT) are not currently rebuildable from source.  They are available precompiled in the `nvigi_pack` only.  Source builds required the DLLs to be copied from a binary pack.
- This release adds a GGML-based ASquaredFlow TTS plugin for running Riva Magpie-TTS-Flow on GGML.  This plugin required the renaming of the existing TRT-based ASquaredFlow plugin (was `nvigi.plugin.tts.asqflow.trt.dll`) to `nvigi.plugin.tts.asqflow-trt.dll`.  Note this when copying the DLLs, especially if a 1.1.1 release was in use prior to this release.  Delete any old `nvigi.plugin.tts.asqflow.trt.dll` files before copying the new DLLs.

## 1.1.1 Release
- This release is intended to add the AsqFlow TTS plugin.
- Currently, AsqFlow TTS plugin is not compatible with CiG on Blackwell architecture. This issue will be resolved in the next release.
