# Plugin SDK Command-line Samples

The NVIGI Plugins SDK includes a set of command-line samples that show the basics of several common app use cases.  The following sections describe how to configure and run these samples, whether they were shipped in a binary "standard layout" app development pack or built from the public GitHub source tree.

## Configuring and Running the Basic Sample

The basic sample, `nvigi.basic` shows the basics of running a workflow of WAV (microphone) audio -> ASR (speech recognition) -> GPT (LLM).  It is automatically built as a part of the SDK build.  It allows direct typing "to" the LLM or "talking to" the LLM via a microphone and ASR.

Because of the SDK layout, once the components are built they will be under the `_artifacts` tree; in order for the SDK to run an app like the sample, all of these DLLs and the executable must be in the same directory.  We do this by copying the DLLs and EXE into the `bin\x64` directory, and running the app from within that directory, so all plugins are available.

Normally, whether the plugins have been built from a binary, "standard layout" app developer pack or from GitHub source, this is done via the `copy_sdk_binaries.bat`, whose use is described in the base documentation for the binary app developer pack or GitHub README.  The rest of this document assumes that the binaries are up to date and have been copied.

These instructions make reference to a set of directories; the location of these directories differ between binary app developer pack and GitHub source.  The documentation for each of these define the locations for these directories in the particular use:
- `<SDK_PLUGINS>`: the root of the SDK Plugins tree, which contains the `bin` directory for the plugins
- `<SDK_MODELS>`: the root of the models tree for the SDK plugins, normally named `nvigi.models`
- `<SDK_TEST>`: : the root of the test data tree for the SDK plugins, normally named `nvigi.test`


### Download Required Models

The basic sample requires the following models:

| Plugin | Model Name | GUID |
| ------ | ---------- | ---- |
| nvigi.plugin.asr.ggml.* | Whisper Small | 5CAD3A03-1272-4D43-9F3D-655417526170 |
| nvigi.plugin.gpt.ggml.* | Llama3.2 3b Instruct | 01F43B70-CE23-42CA-9606-74E80C5ED0B6 |

See the top-level documentation that shipped with your development pack for information on how to download these models.

### Cloud Models

| Plugin | Model Name | GUID | URL |
| ------ | ---------- | ---- | ---- |
| nvigi.plugin.gpt.cloud.rest | gpt-3.5-turbo | E9102ACB-8CD8-4345-BCBF-CCF6DC758E58 | https://api.openai.com/v1/chat/completions | 
| nvigi.plugin.gpt.cloud.rest | Llama 3.2 3B Instruct | 01F43B70-CE23-42CA-9606-74E80C5ED0B6 | https://integrate.api.nvidia.com/v1/chat/completions |

> NOTE: These are just two popular models, for more details on model repository please read the `ProgrammingGuideAI` located in the NVIGI Core SDK.

### How to Use the Basic Sample

When run, the sample should launch a console (or use the one from which it was run); it will await user input.  The user may do one of three things at the prompt:
- Type a chat query as text and press enter.  This will be passed directly to the LLM and the LLM response printed.
- Type an exit command; "Q", "q" or "quit".  This will cause the sample to exit.
- Press enter with no text.  This will start recording from the default Windows/DirectX recording device (e.g. a headset microphone).  Pressing enter again will stop recording.  Once the recording is complete, the audio will be sent to the ASR plugin and then the result of ASR printed to the console and passed to the LLM for response.

### Run at Command Line

To run `nvigi.basic` from the command line, take the following steps (`--models` is a required argument):

1. Open a command prompt in `<SDK_PLUGINS>`
2. Run the command:
```sh
bin\x64\nvigi.basic.exe --models <SDK_MODELS>
```
3. In a standard layout binary development pack or GitHub source tree, launching from a current working directory of the `<SDK_PLUGINS>` directory, this is:
```sh
bin\x64\nvigi.basic.exe --models data/nvigi.models
```

Here are the command line options:

```console
Usage: nvigi.basic [options]

  -m, --models              model repo location (REQUIRED)
  --asr-guid                asr model guid in registry format (default: {5CAD3A03-1272-4D43-9F3D-655417526170})
  -a, --audio               audio file location (default: )
  --gpt                     gpt mode, 'local' or 'cloud' - model GUID determines the cloud endpoint (default: local)
  --gpt-guid                gpt model guid in registry format (default: {01F43B70-CE23-42CA-9606-74E80C5ED0B6})
  -s, --sdk                 sdk location, (default: exe location)
  -t, --token               authorization token for the cloud provider (default: )
  --vram                    the amount of vram to use in MB (default: 8192)
```

### Run in Debugger

To run `nvigi.basic` in the debugger, we must ensure that all of the plugins and the `nvigi.basic.exe` app are copied into one directory, the SDK bin directory.  Then, MSVC's debugger must know to launch the copy, not the original, or it will not find the plugins.  Take the following steps:

1. One-time setup in the project file (needs to be redone if `_project` is deleted):
    1. In the MSVC IDE, edit the project config settings for `aiinferencemananger/samples/nvigi.basic`
    1. Navigate to the "Debugging" settings
    1. Set "Command" to `<SDK_PLUGINS>\bin\x64\nvigi.basic.exe`
    1. Set "Command Arguments" as needed (see the command line options in the above section)
    1. Set "Working Directory" to `<SDK_PLUGINS>/bin/x64`
1. Build the desired non-production config of the SDK; Release is recommended (it is optimized, but contains symbols)
1. After each (re-)build, re-run `copy_sdk_binaries.bat <cfg>`
1. The sample can now be run in the debugger.

## Configuring and Running the Pipeline Sample

The AI Pipeline sample, `nvigi.pipeline` shows the basics of running a multi-plugin workflow via a single eval call using the pipeline plugin.  In this case, the workflow is WAV (file) audio -> ASR (speech recognition) -> GPT (LLM).  It is automatically built as a part of the SDK build.  It runs multiple AI plugins in a single evaluate call, using the AI Pipeline plugin.

Because of the SDK layout, once the components are built they will be under the `_artifacts` tree; in order for the SDK to run an app like the sample, all of these DLLs and the executable must be in the same directory.  We do this by copying the DLLs and EXE into the `bin\x64` directory, and running the app from within that directory, so all plugins are available.

### Download Required Models

The pipeline sample requires the following models:
| Plugin | Model Name | GUID |
| ------ | ---------- | ---- |
| nvigi.plugin.asr.ggml.* | Whisper) | 5CAD3A03-1272-4D43-9F3D-655417526170 |
| nvigi.plugin.gpt.ggml.* | Llama3.2 3b Instruct | 01F43B70-CE23-42CA-9606-74E80C5ED0B6 |

See the top-level documentation that shipped with your development pack for information on how to download these models.

### How to Use the Pipeline Sample

When run, the sample will load the specified wav file and pass it through a pipeline of ASR followed by an LLM, logging the result of each step.

### Run at Command Line

To run `nvigi.pipeline` from the command line, take the following steps:

1. Open a command prompt in `<SDK_PLUGINS>`
2. Run the command:
```sh
bin\x64\nvigi.pipeline.exe <SDK_MODELS> <SDK_TEST>/nvigi.asr/jfk.wav
```
3. In a standard layout binary development pack or GitHub source tree, launching from a current working directory of the `<SDK_PLUGINS>` directory, this is:
```sh
bin\x64\nvigi.pipeline.exe data/nvigi.models data/nvigi.test/nvigi.asr/jfk.wav
```

### Run in Debugger

To run `nvigi.pipeline` in the debugger, we must ensure that all of the plugins and the `nvigi.pipeline.exe` app are copied into one directory, the SDK bin directory.  Then, MSVC's debugger must know to launch the copy, not the original, or it will not find the plugins.  Take the following steps:

1. One-time setup in the project file (needs to be redone if `_project` is deleted):
    1. In the MSVC IDE, edit the project config settings for `aiinferencemananger/samples/nvigi.pipeline`
    1. Navigate to the "Debugging" settings
    1. Set "Command" to `<SDK_PLUGINS>\bin\x64\nvigi.pipeline.exe`
    1. Set "Command Arguments" to `<SDK_MODELS> <SDK_TEST>/nvigi.asr/jfk.wav (use forward slashes in all cases)` (use forward slashes in all cases)
    1. Set "Working Directory" to `<SDK_PLUGINS>/bin/x64`
1. Build the desired non-production config of the SDK; Release is recommended (it is optimized, but contains symbols)
1. After each (re-)build, re-run `copy_sdk_binaries.bat <cfg>`
1. The sample can now be run in the debugger.

## Configuring and Running the RAG Sample

The RAG sample, `nvigi.rag` shows an example of an LLM-based chatbot that uses RAG to provide the LLM with detailed context from which it should answer questions.  The built-in system prompt and provided context text (`data/nvigi.test/nvigi.rag/LegendOfDoria_Corpus.txt`) file describes a fantasy game world.  The user can type in questions that will be passed to the LLM to be answered in-game.

Because of the SDK layout, once the components are built they will be under the `_artifacts` tree; in order for the SDK to run an app like the sample, all of these DLLs and the executable must be in the same directory.  We do this by copying the DLLs and EXE into the `bin\x64` directory, and running the app from within that directory, so all plugins are available.

### Download Required Models

The RAG sample requires the following models:

| Plugin | Model Name | GUID |
| ------ | ---------- | ---- |
| nvigi.plugin.embed.ggml.* | E5 Large Unsupervised | 5D458A64-C62E-4A9C-9086-2ADBF6B241C7 |
| nvigi.plugin.gpt.ggml.* | Nemotron Mini 4B | 8E31808B-C182-4016-9ED8-64804FF5B40D |

See the top-level documentation that shipped with your development pack for information on how to download these models.

### How to Use the RAG Sample

When run, the sample should launch a console (or use the one from which it was run); it will await user input.  The user may do one of two things:
- Type in "exit" then Enter, which will exit the sample
- Type in a Question to the LLM, which will answer them based upon context passed in via the provided text file.

### Run at Command Line

To run `nvigi.rag` from the command line, take the following steps:

1. Open a command prompt in `<SDK_PLUGINS>`
2. Run the command:
```sh
bin\x64\nvigi.rag.exe <SDK_MODELS> <text file>
```
3. In a standard layout binary development pack or GitHub source tree, launching from a current working directory of the `<SDK_PLUGINS>` directory, this might be:
```sh
bin\x64\nvigi.rag.exe data/nvigi.models data/nvigi.test/nvigi.rag/LegendOfDoria_Corpus.txt
```

### Run in Debugger

To run `nvigi.rag` in the debugger, we must ensure that all of the plugins and the `nvigi.rag.exe` app are copied into one directory, the SDK bin directory.  Then, MSVC's debugger must know to launch the copy, not the original, or it will not find the plugins.  Take the following steps:

1. One-time setup in the project file (needs to be redone if `_project` is deleted):
    1. In the MSVC IDE, edit the project config settings for `aiinferencemananger/samples/nvigi.rag`
    1. Navigate to the "Debugging" settings
    1. Set "Command" to `<SDK_PLUGINS>/bin/x64/nvigi.rag.exe`
    1. Set "Command Arguments" to `<SDK_MODELS> <text file>` (use forward slashes in all cases)
    1. Set "Working Directory" to `<SDK_PLUGINS>/bin/x64`
1. Build the desired non-production config of the SDK; Release is recommended (it is optimized, but contains symbols)
1. After each (re-)build, re-run `copy_sdk_binaries.bat <cfg>`
1. The sample can now be run in the debugger.

