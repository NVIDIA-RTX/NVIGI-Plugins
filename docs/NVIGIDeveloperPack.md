# Top-Level Items

## Models (A.K.A. Model Data)


### Getting Models

In order to avoid making the prebuilt binary release pack very large, the pack does not include any model data files.  Models for NVIGI fall into one of a few categories
- **Cloud models**; these consist only of a configuration JSON file.  These are shipped with this pack under `nvigi.models/<plugin name>/{model GUID}`
- **SDK Pack-included, public models**; these consist of a configuration JSON file and pre-downloaded model files, zipped into the `nvigi_pack` itself.  These are shipped in this pack under `nvigi.models/<plugin name>/{model GUID}`.  The model files are ready to use without any further download steps, but do make the pack larger.
- **Manually-downloadable, public models**; these consist of a configuration JSON file and a Windows batch file `download.bat`.  These are shipped in this pack under `nvigi.models/<plugin name>/{model GUID}`, and the `download.bat` can be double-clicked to use `curl` to download the model without any form of authentication.  In some cases, the batch file will also extract the required files from zip, if the downloaded file is a zipfile.  Depending upon the security settings on the local system, Windows may ask for confirmation when running the batch file.
- **Manually-downloadable, licensed models**; these consist of a configuration JSON file and a README file with information on how to download the model.  However, these models require authentication via an NVIDIA Developer Relations representative.

### Cloud Models

The supported cloud models in this release include:

| Plugin | Model Name | GUID | URL |
| ------ | ---------- | ---- | --- |
| nvigi.plugin.gpt.cloud.rest | Llama3.2 3b Instruct | 01F43B70-CE23-42CA-9606-74E80C5ED0B6 | https://integrate.api.nvidia.com/v1/chat/completions |
| | Nemotron Mini 4B | 8E31808B-C182-4016-9ED8-64804FF5B40D | https://integrate.api.nvidia.com/v1/chat/completions |
| | gpt-3.5-turbo | E9102ACB-8CD8-4345-BCBF-CCF6DC758E58 | https://api.openai.com/v1/chat/completions | 

### SDK Pack-included, Public Models

| Plugin | Model Name | GUID | Source |
| ------ | ---------- | ---- | ------ |
| nvigi.plugin.tts.asqflow-trt | Riva Magpie-TTS-Flow | 81320D1D-DF3C-4CFC-B9FA-4D3FF95FC35F | Model in pack |
| nvigi.plugin.tts.asqflow-ggml | Riva Magpie-TTS-Flow (Q4) | 3D52FDC0-5B6D-48E1-B108-84D308818602 | Model in pack |
| | Riva Magpie-TTS-Flow (FP16) | 33E000D6-35A2-46D8-BCB5-E10F8CA137C0 | Model in pack |

### Manually-Downloadable, Public Models

| Plugin | Model Name | GUID | Source |
| ------ | ---------- | ---- | ------ |
| nvigi.plugin.asr.ggml.* | Whisper Small | 5CAD3A03-1272-4D43-9F3D-655417526170 | [Model Page](https://huggingface.co/ggerganov/whisper.cpp) |
| nvigi.plugin.embed.ggml.* | E5 Large Unsupervised | 5D458A64-C62E-4A9C-9086-2ADBF6B241C7 | [Model Page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nvigisdk/models/e5-large-unsupervised) |
| nvigi.plugin.gpt.ggml.* | Llama3.2 3b Instruct | 01F43B70-CE23-42CA-9606-74E80C5ED0B6 | [Model Page](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| | Nemotron Mini 4B | 8E31808B-C182-4016-9ED8-64804FF5B40D | [Model Page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ucs-ms/resources/nemotron-mini-4b-instruct) |
| | Qwen3 8B (Q4) | 545F7EC2-4C29-499B-8FC8-61720DF3C626 | [Model Page](https://huggingface.co/Qwen/Qwen3-8B-GGUF) |

### Manually-Downloadable, Licensed Models

To gain access to the following models, contact your NVIDIA Developer Relations representative.

| Plugin | Model Name | GUID | Source |
| ------ | ---------- | ---- | ------ |
| nvigi.plugin.gpt.ggml.* | Nemovision 4B Instruct FP16 | 0BAEDD5C-F2CA-49AA-9892-621C40030D12 | [Model Page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ace/models/nemovision-4b-instruct) |
| | Nemovision 4B Instruct Q4 | 0BAEDD5C-F2CA-49AA-9892-621C40030D13 | [Model Page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ace/models/nemovision-4b-instruct) |

## Important Documentation in the Developer Pack

The main documentation for using the pack includes:

- [Samples](Samples)
	- The best "starting" doc, as the sample is provided pre-compiled and ready-to-run.  It describes how to run, recompile, and debug in the sample, which shows both speech recognition and LLMs for text interaction.
- `nvigi_core`
	- Detailed documentation on the components of the NVIGI core library, including:
    - [Architecture Guide](nvigi_core/docs/Architecture)
		- Discusses the high-level architecture of the entire SDK, the data flow, etc
    - [GPU Scheduling For AI Guide](nvigi_core/docs/GpuSchedulingForAI)
		- A detailed document on how advanced applications can assist in causing GPU AI work to be best scheduled along with 3D work
- [Plugin SDK Guide](Plugins)
	- The core documentation for getting started with the details of the AI plguins.  This describes how to run a set of much more minimal samples and how to run the samples in the debugger.
	- Detailed documentation on the components of the SDK plugins, including:
    - [ASR Whisper Plugins Programming Guide](ProgrammingGuideASRWhisper)
		- Detailed docs on how to program for the speech recognition plugins
    - [Embedding Plugin Programming Guide](ProgrammingGuideEmbed)
		- Detailed documentation on how to program for the Embedding plugins
    - [GPT Plugins Programming Guide](ProgrammingGuideGPT)
		- Detailed documentation on how to program for the GPT/LLM plugins
	- [TTS  Riva Magpie-TTS-Flow (was ASquaredFlow) Plugin Programming Guide](ProgrammingGuideTTSASqFlow.md)
		- Detailed documentation on how to program for using the  Riva Magpie-TTS-Flow model via the TTS AsqFlow plugin

## Contents of the Developer Pack

The pack consists of:
- **AI Plugins:** `include` and `bin/x64`
	- The headers (`include`) and DLLs (`bin/x64`) that comprise the AI Plugin functionalities such as ASR, GPT, and Embedding.
- **Command-line Samples:** `source/samples` and `bin/x64`
	- Basic, precompiled command-line samples(basic, fcall, pipeline, rag, reload) that run a sequence of AI workloads
- **3D Sample:** `source/samples/nvigi.3d` and `bin/x64`
	- A precompiled, runnable 3D+GUI sample that allows easy experimentation with components of the SDK, including speech input and text input to an LLM
- **Core:** `nvigi_core`
	- The NVIGI Core components, including the headers, libraries and DLLs that make up the main functionality of NVIGI and basic documentation on the core architecture
- **Models:** `data/nvigi.models`
	- AI Models for use with the above, including:
		- Llama-3 GPT LLM, downloadable manually
		- Whisper Speech Recognition, downloadable manually
- **Data:** `data/nvigi.test`
	- Basic AI Test Data, including:
		- A WAV file for use as input to speech recognition in the command-line sample
- **Docs:** `docs`
	- Documentation for all of the above components, repackaged and linked from this top-level document.
