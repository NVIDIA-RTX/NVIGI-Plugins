# Getting Started

## What is NVIGI?

The NVIDIA In-Game Inferencing (NVIGI) SDK streamlines AI model deployment and integration for PC application developers. The SDK pre-configures the PC with the necessary AI models, engines, and dependencies. It allows an application to orchestrate AI inference seamlessly across PC and cloud via a unified inference API. It can support major inference backends across different hardware accelerators (GPU, NPU, CPU), depending upon plugin implementation.  

The system is meant to be integrated into end-user applications and games to provide selection between models running locally or in the cloud (i.e. hybrid).

High level objectives are:
- Allow models to execute across a variety of backends, devices and runtimes
- Support a wide range of models and pipelines
- Provide a seamless way for application developers to run in cloud or locally
- Efficient in-app integration

The NVIDIA IGI SDK  is architected as a suite of plugins, containing both inferencing plugins as well as helper plugins, that are to be integrated into end-user applications. . The "helper" plugins are shared amongst the various inference plugins. Examples of "helper" plugins include network functionalities like gRPC or D3D12 device/queue/command list management for integration of 3D workloads and AI workloads. Provided AI inferencing plugins implement many different models using multiple runtimes, but all of them share the same creation and inference APIs as one another. As a result, all of the LLM plugins, all of the ASR (Speech Recognition) plugins, and all of the TTS (Text-to-Speech) plugins share functionality-specific APIs and can be easily swapped in and out for one another by an application with minor code modifications. All of this is possible with the core plugin architecture by creating interfaces that are shared by all plugins that implement the specific functionality.

## Where Do I Start?

The recommended order of "getting started" with the pack is to:
- Download models via `download_all_data.bat` in the top-level of the pack.
	- Downloading all of the models via the `download_all_data.bat` script will require approximately **10GB**.
	- For model details, see [Getting Models](NVIGIDeveloperPack.md#getting-models).

- Run the 3D Sample by running `nvigi.3d.exe`
	- Read the [3D Sample's docs](Samples) in detail for additional features and instructions on interaction
	- If you wish to try the cloud plugins, follow the instructions in [Setting up the GPT Cloud Plugin](Samples.md#setting-up-the-gpt-cloud-plugin).

- Rebuild the 3D Sample from source.  This will allow you to step through the code in the debugger and understand how NVIGI is integrated into the sample
	1. Opening a VS2022 Developer Command Prompt
	1. Running `setup.bat` to generate the build files
	1. Opening `_project/vs2022/nvigi.sln` with VS2022
	1. Compiling the solution

- Follow on with the rest of the top-level documents, especially those directly following under The Basics

## The Current Release of the Developer Pack

The NVIDIA IGI Developer Pack is a release of several components: the IGI Core APIs SDK, plugins for a wide range of inference features, command-line samples, a 3D Sample and models that are designed to show the architecture and application integration of NVIGI with interactive applications. The pack supports a group of local inference plugins along with a cloud inference plugin.  The set of plugins includes:
- GGML-based LLMs on CPU or GPU (CUDA, D3D12 or Vulkan) [https://github.com/ggerganov/ggml](https://github.com/ggerganov/ggml)
- GGML-based Speech Recognition on CPU or GPU (CUDA, D3D12 or Vulkan) [https://github.com/ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- GGML-based embeddings on CPU or GPU (CUDA)
- GGML- or TensorRT-based Text-to-Speech (TTS) via Riva Magpie Flow (was ASquaredFlow) on GPU (CUDA or Vulkan)
- NVIDIA Cloud-based GPT/LLM inference via [https://build.nvidia.com/explore/discover](https://build.nvidia.com/explore/discover)
- GPU scheduling optimizations with 3D workloads [nvigi_core/docs/GpuSchedulingForAI](nvigi_core/docs/GpuSchedulingForAI)

In addition, several samples exist in two locations.  Source code for the samples is provided, precompiled, which allow for instant experimentation with the plugins.:
- [Command-line samples](Samples) for specific plugins, located in `source/samples` with (pre-)built binaries in `bin/x64`.  These include:
  - **Basic (nvigi.basic):** A command-line sample that shows the use of individual ASR, GPT and TTS plugins to implement conversational AI.  The user can provide input by typing their queries or by using a microphone to pass their verbal query to speech recognition.  The GPT plugin will respond to the query, with conversational context.  The GPT plugin may be switched from local to cloud models via the command line.  The response will then be turned into audio via TTS and played to the audio output
  - **FCall (nvigi.fcall):** A command-line sample that shows the use of a GPT plugin to implement function calling, showing the basics of having an LLM call a function and returning the results of that function for the LLM to then use to answer the user's query.
  - **Pipeline (nvigi.pipeline):** A command-line sample that shows the use of a pipeline plugin, capable of running a sequence of ASR and GPT plugins via a single evaluation call.  This sample uses audio input from an audio file
  - **RAG (nvigi.rag):** A command-line sample that shows how to use GPT and embedding to implement Retrieval Automated Generation, or RAG.  Specifically, the sample takes a text file to use as its reference, or "corpus" when answering queries, along with a prompt to guide how it uses the corpus.  The user may type in queries for the RAG.
  - **Reload (nvigi.reload):** A command-line sample that shows how to "hot-swap" or "unload" model data for D3D12 GGML-based models, avoiding wasted VRAM or the need to reload models from scratch.
- **A 3D sample (nvigi.3d)** exists in the SDK itself, along with the command-line samples.  It includes a wider range of plugins, as well as a GUI for interaction and a 3D scene rendered at the same time
  - Support for local and cloud GPT
  - Support for ASR via GUI-based recording. 
  - Support for TTS (Text-to-Speech)

## Known Issues and Important Notes

### 1.3.0 Release
- TODO
- This release of NVIGI includes a significant upgrade to how the SDK pack is laid out and organized.  The Plugin SDK is now the top-level directory, and the existing 3D Sample has now been moved into the same directory tree and build solution as all of the command-line samples.  This simplifies the build, run and debugging workflow for the 3D sample, making it a first-class citizen of the SDK.

### 1.2.0 Release
- We are investigating a potential issue on some configurations where the 3D Sample can crash the first time a model is loaded. If you hit this problem, please reboot and rerun the sample.
- The models for the Riva Magpie Flow TTS plugins are shipped in the pack itself; we expect that in future releases, the models will be downloadable via the `download_data.bat` script.
- The previous ASquaredFlow TTS model has been updated and is now known as "Riva Magpie-TTS-Flow".  The plugin itself retains the original name to avoid problems for existing applications.  However, the new model must be used, as the ones shipped with the 1.1.1 release are no longer compatible with the updated plugin.
- This release adds Vulkan and D3D12 GGML backend plugins for ASR and GPT; they are experimental and not intended for production use.  They allow GPU GPT and ASR on general GPU architectures/vendors.
- This release adds Vulkan support to the 3D Sample via the `-vk` command-line option; it is experimental and not intended for production use.  It allows the sample to run on general GPU architectures/vendors.
- The D3D12 GGML plugins (ASR and GPT) are not currently rebuildable from source.  They are available precompiled in the `nvigi_pack` only.  Source builds required the DLLs to be copied from a binary pack.
- This release adds a GGML-based ASquaredFlow TTS plugin for running Riva Magpie-TTS-Flow on GGML.  This plugin required the renaming of the existing TRT-based ASquaredFlow plugin (was `nvigi.plugin.tts.asqflow.trt.dll`) to `nvigi.plugin.tts.asqflow-trt.dll`.  Note this when copying the DLLs, especially if a 1.1.1 release was in use prior to this release.  Delete any old `nvigi.plugin.tts.asqflow.trt.dll` files before copying the new DLLs.

### 1.1.1 Release
- This release is intended to add the AsqFlow TTS plugin.
- Currently, AsqFlow TTS plugin is not compatible with CiG on Blackwell architecture. This issue will be resolved in the next release.

### 1.1.0 Release
- 

### 1.0.0 Release
- Initial public release

### Beta 1

- A significant change in the Beta 1 release compared to early-access releases is that independent components of NVIGI have been split into their own directories.  Specifically, the core APIs and core DLLs for NVIGI are now in the `nvigi_core` package, and the AI Plugins are in `SDK`.  The Beta 1 pack generally ships all of these, zipped as sibling directories, in a single release pack.  However, future releases may be decoupled, especially w.r.t. `nvigi_core`, which should change much less frequently.  Additional, new plugins may also be distributed independently. 
- Owing to a quirk of conversational/interactive mode in GGML, the GGML GPT plugin can in some cases produce truncated responses or an empty response.  This can be seen in the Basic Sample or in the 3D Sample if the user interacts with the GPT for several iterations without resetting the conversation.  If desired, the frequency of this can be lowered by increasing the `nvigi::GPTRuntimeParameters::tokensToPredict` value.  A solution to this is being investigated.
- When using CiG, it is currently important to keep a reference to the `nvigi::plugin::hwi::cuda` for the life of the application.  This ensures that if plugins using CiG are created and destroyed multiple times, the shared CUDA context stays active.  It is important that the `nvigi::plugin::hwi::cuda` plugin not be unloaded and reloaded multiple times in the application.  Keeping a reference via an active interface for the life of the application will do this.  See the CiG code in the 3D Sample for an example of this.
