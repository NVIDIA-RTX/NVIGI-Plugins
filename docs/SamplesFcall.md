# The Command-Line Function Calling Sample

The function calling sample, `nvigi.fcall` shows the basics of having an LLM call a function and returning the results of that function for the LLM to then use to answer the user's query.

**NOTE**: This sample makes use of a CUDA-based backend, and therefore will not work on non-NVIDIA hardware. This will be fixed in future releases.

Because of the SDK layout, once the components are built they will be under the `_artifacts` tree; in order for the SDK to run an app like the sample, all of these DLLs and the executable must be in the same directory.  We do this by copying the DLLs and EXE into the `bin\x64` directory, and running the app from within that directory, so all plugins are available.

## Download Required Models

The function calling sample requires the following models:

| Plugin | Model Name | GUID |
| ------ | ---------- | ---- |
| nvigi.plugin.gpt.ggml.* | Qwen3 8B Instruct | 545F7EC2-4C29-499B-8FC8-61720DF3C626 |

See the top-level documentation that shipped with your development pack for information on how to download these models.

> NOTE: This is just a popular model.  Other models trained on function calling, such as Qwen3-4B-instruct-2507, may be used. For more details on model repository please read the `ProgrammingGuideAI` located in the NVIGI Core SDK.

## How to Use the Function Calling Sample

When run, the sample should launch a console (or use the one from which it was run); it will await user input.  The user may do one of four things, which are also described when the app starts:
- Type in "exit" then Enter, which will exit the sample
- Type "disable_tools" then Enter.  This will not allow any tool usage by the model.  It will also reset the internal chat history.
- Type "enable_tools" then Enter.  This will allow tool usage by the model.  It will also reset the internal chat history.  This is the default state on start
- Type in a Question to the LLM.  Depending on the nature of the question, the LLM will have the option of using one of two tools to answer the question.  The two tools are either a fake CurrentTempTool or a WikiSearchTool.  The CurrentTempTool is hardcoded to return fake temperatures in three cities: Durham, Austin and Santa Clara.  CurrentTempTool serves as a basic example of how all the plumbing is hooked up. WikiSearchTool will use libCurl and an internet connection to perform a naive search on wikipedia, find the first search result, and use the extract (first few sentences of the opening summary) to help answer the question.  This tool uses a bit more assistance in constructing the tool schema.

## Run at Command Line

To run `nvigi.fcall` from the command line, take the following steps:

1. Open a command prompt in `<SDK_ROOT>`
2. Run the command:
```sh
bin\x64\nvigi.fcall.exe <SDK_MODELS>
```
3. In a standard layout binary development pack or GitHub source tree, launching from a current working directory of the `<SDK_ROOT>` directory, this is:
```sh
bin\x64\nvigi.fcall.exe data/nvigi.models
```

## Run in Debugger

To run `nvigi.fcall` in the debugger, we must ensure that all of the plugins and the `nvigi.fcall.exe` app are copied into one directory, the SDK bin directory.  Then, MSVC's debugger must know to launch the copy, not the original, or it will not find the plugins.  Take the following steps:

1. One-time setup in the project file (needs to be redone if `_project` is deleted):
    1. In the MSVC IDE, edit the project config settings for `nvigi/samples/nvigi.fcall`
    1. Navigate to the "Debugging" settings
    1. Set "Command" to `<SDK_ROOT>\bin\x64\nvigi.fcall.exe`
    1. Set "Command Arguments" to `<SDK_MODELS>` (use forward slashes in all cases)
    1. Set "Working Directory" to `<SDK_ROOT>/bin/x64`
1. Build the desired non-production config of the SDK; Release is recommended (it is optimized, but contains symbols)
1. After each (re-)build, re-run `copy_sdk_binaries.bat <cfg>`
1. The sample can now be run in the debugger.
