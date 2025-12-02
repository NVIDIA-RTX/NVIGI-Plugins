# The Command-Line Pipeline Sample

The AI Pipeline sample, `nvigi.pipeline` shows the basics of running a multi-plugin workflow via a single eval call using the pipeline plugin.  In this case, the workflow is WAV (file) audio -> ASR (speech recognition) -> GPT (LLM).  It is automatically built as a part of the SDK build.  It runs multiple AI plugins in a single evaluate call, using the AI Pipeline plugin.

**NOTE**: This sample makes use of a CUDA-based backend, and therefore will not work on non-NVIDIA hardware. This will be fixed in future releases.

Because of the SDK layout, once the components are built they will be under the `_artifacts` tree; in order for the SDK to run an app like the sample, all of these DLLs and the executable must be in the same directory.  We do this by copying the DLLs and EXE into the `bin\x64` directory, and running the app from within that directory, so all plugins are available.

## Download Required Models

The pipeline sample requires the following models:
| Plugin | Model Name | GUID |
| ------ | ---------- | ---- |
| nvigi.plugin.asr.ggml.* | Whisper | 5CAD3A03-1272-4D43-9F3D-655417526170 |
| nvigi.plugin.gpt.ggml.* | Llama3.2 3b Instruct | 01F43B70-CE23-42CA-9606-74E80C5ED0B6 |

See the top-level documentation that shipped with your development pack for information on how to download these models.

## How to Use the Pipeline Sample

When run, the sample will load the specified wav file and pass it through a pipeline of ASR followed by an LLM, logging the result of each step.

## Run at Command Line

To run `nvigi.pipeline` from the command line, take the following steps:

1. Open a command prompt in `<SDK_ROOT>`
2. Run the command:
```sh
bin\x64\nvigi.pipeline.exe <SDK_MODELS> <SDK_TEST>/nvigi.asr/jfk.wav
```
3. In a standard layout binary development pack or GitHub source tree, launching from a current working directory of the `<SDK_ROOT>` directory, this is:
```sh
bin\x64\nvigi.pipeline.exe data/nvigi.models data/nvigi.test/nvigi.asr/jfk.wav
```

## Run in Debugger

To run `nvigi.pipeline` in the debugger, we must ensure that all of the plugins and the `nvigi.pipeline.exe` app are copied into one directory, the SDK bin directory.  Then, MSVC's debugger must know to launch the copy, not the original, or it will not find the plugins.  Take the following steps:

1. One-time setup in the project file (needs to be redone if `_project` is deleted):
    1. In the MSVC IDE, edit the project config settings for `nvigi/samples/nvigi.pipeline`
    1. Navigate to the "Debugging" settings
    1. Set "Command" to `<SDK_ROOT>\bin\x64\nvigi.pipeline.exe`
    1. Set "Command Arguments" to `<SDK_MODELS> <SDK_TEST>/nvigi.asr/jfk.wav` (use forward slashes in all cases)
    1. Set "Working Directory" to `<SDK_ROOT>/bin/x64`
1. Build the desired non-production config of the SDK; Release is recommended (it is optimized, but contains symbols)
1. After each (re-)build, re-run `copy_sdk_binaries.bat <cfg>`
1. The sample can now be run in the debugger.
