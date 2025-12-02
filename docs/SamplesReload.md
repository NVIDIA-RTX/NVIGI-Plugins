# The Command-Line Reload Sample

The D3D12 model-reloading sample, `nvigi.reload` shows how a D3D12-based NVIGI application can manually load and unload models for D3D12 GGML-based plugins to/from VRAM.  This allows an application to reclaim VRAM when inference is not in use, or to have two model instances ready to quickly swap in and out of VRAM as desired, avoiding having to choose between time-consuming load-from file and keeping multiple models in VRAM at once.  It is automatically built as a part of the SDK build.  It allows direct typing "to" an LLM as well as the ability to select when to switch models without reloading from file.  It shows the current VRAM use along the way.

Because of the SDK layout, once the components are built they will be under the `_artifacts` tree; in order for the SDK to run an app like the sample, all of these DLLs and the executable must be in the same directory.  We do this by copying the DLLs and EXE into the `bin\x64` directory, and running the app from within that directory, so all plugins are available.

Normally, whether the plugins have been built from a binary, "standard layout" app developer pack or from GitHub source, this is done via the `copy_sdk_binaries.bat`, whose use is described in the base documentation for the binary app developer pack or GitHub README.  The rest of this document assumes that the binaries are up to date and have been copied.

These instructions make reference to a set of directories; the location of these directories differ between binary app developer pack and GitHub source.  The documentation for each of these define the locations for these directories in the particular use:
- `<SDK_ROOT>`: the root of the SDK, which contains the `bin` directory for the SDK
- `<SDK_MODELS>`: the root of the models tree for the SDK, normally named `data/nvigi.models`
- `<SDK_TEST>`: the root of the test data tree for the SDK, normally named `data/nvigi.test`


## Download Required Models

The reload sample requires the following models:

| Plugin | Model Name | GUID |
| ------ | ---------- | ---- |
| nvigi.plugin.gpt.ggml.* | Llama3.2 3b Instruct | 01F43B70-CE23-42CA-9606-74E80C5ED0B6 |
| nvigi.plugin.gpt.ggml.* | Nemotron Mini 4B | 8E31808B-C182-4016-9ED8-64804FF5B40D |

See the top-level documentation that shipped with your development pack for information on how to download these models.

## How to Use the Command-Line Reload Sample

When run, the sample should launch a console (or use the one from which it was run); it will await user input.  The user may do one of three things at the prompt:
- Type a chat query as text and press enter.  This will be passed directly to the LLM and the LLM response printed.
- Type an exit command; "Q", "q" or "quit".  This will cause the sample to exit.
- Type `<unload>` and press enter.  This will unload all loaded models.
- Type `<llama3>` and press enter.  This will load the Llama3.2 3b Instruct model if it is not already loaded, or switch to it if it is already loaded.  It will unload the other model.
- Type `<nemotron>` and press enter.  This will load the Nemotron Mini 4B model if it is not already loaded, or switch to it if it is already loaded.  It will unload the other model.

## Run at Command Line

To run `nvigi.reload` from the command line, take the following steps (`--models` is a required argument):

1. Open a command prompt in `<SDK_ROOT>`
2. Run the command:
```sh
bin\x64\nvigi.reload.exe --models <SDK_MODELS>
```
3. In a standard layout binary development pack or GitHub source tree, launching from a current working directory of the `<SDK_ROOT>` directory, this is:
```sh
bin\x64\nvigi.reload.exe --models data/nvigi.models
```

Here are the command line options:

```console
Usage: nvigi.reload [options]

  -m, --models              model repo location (REQUIRED)
  -s, --sdk                 sdk location, (default: exe location)
  --vram                    the amount of vram to use in MB (default: 8192)
```

## Run in Debugger

To run `nvigi.reload` in the debugger, we must ensure that all of the plugins and the `nvigi.reload.exe` app are copied into one directory, the SDK bin directory.  Then, MSVC's debugger must know to launch the copy, not the original, or it will not find the plugins.  Take the following steps:

1. One-time setup in the project file (needs to be redone if `_project` is deleted):
    1. In the MSVC IDE, edit the project config settings for `nvigi/samples/nvigi.reload`
    1. Navigate to the "Debugging" settings
    1. Set "Command" to `<SDK_ROOT>\bin\x64\nvigi.reload.exe`
    1. Set "Command Arguments" as needed (see the command line options in the above section)
    1. Set "Working Directory" to `<SDK_ROOT>/bin/x64`
1. Build the desired non-production config of the SDK; Release is recommended (it is optimized, but contains symbols)
1. After each (re-)build, re-run `copy_sdk_binaries.bat <cfg>`
1. The sample can now be run in the debugger.
