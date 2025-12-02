# Introduction

## What is NVIGI?

The NVIDIA In-Game Inferencing (NVIGI) SDK streamlines AI model deployment and integration for PC application developers. The SDK pre-configures the PC with the necessary AI models, engines, and dependencies. It allows an application to orchestrate AI inference seamlessly across PC and cloud via a unified inference API. It can support major inference backends across different hardware accelerators (GPU, NPU, CPU), depending upon plugin implementation.  

The system is meant to be integrated into end-user applications and games to provide selection between models running locally or in the cloud (i.e. hybrid).

High level objectives are:
- Allow models to execute across a variety of backends, devices and runtimes
- Support a wide range of models and pipelines
- Provide a seamless way for application developers to run in cloud or locally
- Efficient in-app integration

The NVIDIA IGI SDK  is architected as a suite of plugins, containing both inferencing plugins as well as helper plugins, that are to be integrated into end-user applications. . The "helper" plugins are shared amongst the various inference plugins. Examples of "helper" plugins include network functionalities like gRPC or D3D12 device/queue/command list management for integration of 3D workloads and AI workloads. Provided AI inferencing plugins implement many different models using multiple runtimes, but all of them share the same creation and inference APIs as one another. As a result, all of the LLM plugins, all of the ASR (Speech Recognition) plugins, and all of the TTS (Text-to-Speech) plugins share functionality-specific APIs and can be easily swapped in and out for one another by an application with minor code modifications. All of this is possible with the core plugin architecture by creating interfaces that are shared by all plugins that implement the specific functionality.

## The Current Release of the Developer Pack

The NVIDIA IGI Developer Pack is a release of several components: the IGI Core APIs SDK, plugins for a wide range of inference features, command-line samples, a 3D Sample and models that are designed to show the architecture and application integration of NVIGI with interactive applications. The pack supports a group of local inference plugins along with a cloud inference plugin.  The set of plugins includes:
- GGML-based LLMs on CPU or GPU (CUDA, D3D12 or Vulkan) [https://github.com/ggerganov/ggml](https://github.com/ggerganov/ggml)
- GGML-based Speech Recognition on CPU or GPU (CUDA, D3D12 or Vulkan) [https://github.com/ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- GGML-based embeddings on CPU or GPU (CUDA)
- GGML- or TensorRT-based Text-to-Speech (TTS) via Riva Magpie Flow (was ASquaredFlow) on GPU (CUDA or Vulkan)
- NVIDIA Cloud-based GPT/LLM inference via [https://build.nvidia.com/explore/discover](https://build.nvidia.com/explore/discover)
- GPU scheduling optimizations with 3D workloads [nvigi_core/docs/GpuSchedulingForAI](nvigi_core/docs/GpuSchedulingForAI)

In addition, several samples exist in two locations. Source code for the samples is provided and precompiled, which allow for instant experimentation with the plugins.:
- Command-line samplesfor specific plugins, located in `source/samples` with (pre-)built binaries in `bin/x64`.  These include:
  - [Basic (nvigi.basic)](SamplesBasic.md): A command-line sample that shows the use of individual ASR, GPT and TTS plugins to implement conversational AI.  The user can provide input by typing their queries or by using a microphone to pass their verbal query to speech recognition.  The GPT plugin will respond to the query, with conversational context.  The GPT plugin may be switched from local to cloud models via the command line.  The response will then be turned into audio via TTS and played to the audio output
  - **C++ Basic Samples**: Modern C++ wrapper-based samples demonstrating individual plugin usage with async/polling APIs, perfect for game integration:
    - [Basic ASR (nvigi.basic.cxx.asr)](SampleBasicCxxASR.md): Demonstrates automatic speech recognition with both complete audio mode and real-time streaming. Shows microphone recording, transcription, language detection, translation, and WAV file export.
    - [Basic GPT (nvigi.basic.cxx.gpt)](SampleBasicCxxGPT.md): Demonstrates language model inference with chat interface, streaming responses, and non-blocking polling operations. Supports local (D3D12/CUDA/Vulkan) and cloud backends (OpenAI, NVIDIA NIM).
    - [Basic TTS (nvigi.basic.cxx.tts)](SampleBasicCxxTTS.md): Demonstrates text-to-speech synthesis with voice cloning, real-time audio playback, and async generation modes. Features multiple language support and adjustable speech speed.
  - [FCall (nvigi.fcall)](SamplesFcall.md): A command-line sample that shows the use of a GPT plugin to implement function calling, showing the basics of having an LLM call a function and returning the results of that function for the LLM to then use to answer the user's query.
  - [Pipeline (nvigi.pipeline)](SamplesPipeline.md): A command-line sample that shows the use of a pipeline plugin, capable of running a sequence of ASR and GPT plugins via a single evaluation call.  This sample uses audio input from an audio file
  - [RAG (nvigi.rag)](SamplesRAG.md): A command-line sample that shows how to use GPT and embedding to implement Retrieval Automated Generation, or RAG.  Specifically, the sample takes a text file to use as its reference, or "corpus" when answering queries, along with a prompt to guide how it uses the corpus.  The user may type in queries for the RAG.
  - [Reload (nvigi.reload)](SamplesReload.md): A command-line sample that shows how to "hot-swap" or "unload" model data for D3D12 GGML-based models, avoiding wasted VRAM or the need to reload models from scratch.
- A [3D sample (nvigi.3d)](Samples3D.md) exists in the SDK itself, along with the command-line samples.  It includes a wider range of plugins, as well as a GUI for interaction and a 3D scene rendered at the same time
  - Support for local and cloud GPT
  - Support for ASR via GUI-based recording. 
  - Support for TTS (Text-to-Speech)
