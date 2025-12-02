# The Command-Line RAG Sample

The RAG sample, `nvigi.rag` shows an example of an LLM-based chatbot that uses RAG to provide the LLM with detailed context from which it should answer questions.  The built-in system prompt and provided context text (`data/nvigi.test/nvigi.rag/LegendOfDoria_Corpus.txt`) file describes a fantasy game world.  The user can type in questions that will be passed to the LLM to be answered in-game.

**NOTE**: This sample makes use of a CUDA-based backend, and therefore will not work on non-NVIDIA hardware. This will be fixed in future releases.

Because of the SDK layout, once the components are built they will be under the `_artifacts` tree; in order for the SDK to run an app like the sample, all of these DLLs and the executable must be in the same directory.  We do this by copying the DLLs and EXE into the `bin\x64` directory, and running the app from within that directory, so all plugins are available.

## Download Required Models

The RAG sample requires the following models:

| Plugin | Model Name | GUID |
| ------ | ---------- | ---- |
| nvigi.plugin.embed.ggml.* | E5 Large Unsupervised | 5D458A64-C62E-4A9C-9086-2ADBF6B241C7 |
| nvigi.plugin.gpt.ggml.* | Nemotron Mini 4B | 8E31808B-C182-4016-9ED8-64804FF5B40D |

See the top-level documentation that shipped with your development pack for information on how to download these models.

## How to Use the RAG Sample

When run, the sample should launch a console (or use the one from which it was run); it will await user input.  The user may do one of two things:
- Type in "exit" then Enter, which will exit the sample
- Type in a Question to the LLM, which will answer them based upon context passed in via the provided text file.

## Run at Command Line

To run `nvigi.rag` from the command line, take the following steps:

1. Open a command prompt in `<SDK_ROOT>`
2. Run the command:
```sh
bin\x64\nvigi.rag.exe <SDK_MODELS> <text file>
```
3. In a standard layout binary development pack or GitHub source tree, launching from a current working directory of the `<SDK_ROOT>` directory, this might be:
```sh
bin\x64\nvigi.rag.exe data/nvigi.models data/nvigi.test/nvigi.rag/LegendOfDoria_Corpus.txt
```

## Run in Debugger

To run `nvigi.rag` in the debugger, we must ensure that all of the plugins and the `nvigi.rag.exe` app are copied into one directory, the SDK bin directory.  Then, MSVC's debugger must know to launch the copy, not the original, or it will not find the plugins.  Take the following steps:

1. One-time setup in the project file (needs to be redone if `_project` is deleted):
    1. In the MSVC IDE, edit the project config settings for `nvigi/samples/nvigi.rag`
    1. Navigate to the "Debugging" settings
    1. Set "Command" to `<SDK_ROOT>/bin/x64/nvigi.rag.exe`
    1. Set "Command Arguments" to `<SDK_MODELS> <text file>` (use forward slashes in all cases)
    1. Set "Working Directory" to `<SDK_ROOT>/bin/x64`
1. Build the desired non-production config of the SDK; Release is recommended (it is optimized, but contains symbols)
1. After each (re-)build, re-run `copy_sdk_binaries.bat <cfg>`
1. The sample can now be run in the debugger.
