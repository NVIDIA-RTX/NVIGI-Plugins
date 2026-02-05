# The Command-Line Basic Sample

The basic sample, `nvigi.basic` shows the basics of running a workflow of WAV (microphone) audio -> ASR (speech recognition) -> GPT (LLM).  It is automatically built as a part of the SDK build.  It allows direct typing "to" the LLM or "talking to" the LLM via a microphone and ASR.

Because of the SDK layout, once the components are built they will be under the `_artifacts` tree; in order for the SDK to run an app like the sample, all of these DLLs and the executable must be in the same directory.  We do this by copying the DLLs and EXE into the `bin\x64` directory, and running the app from within that directory, so all plugins are available.

Normally, whether the plugins have been built from a binary, "standard layout" app developer pack or from GitHub source, this is done via the `copy_sdk_binaries.bat`, whose use is described in the base documentation for the binary app developer pack or GitHub README.  The rest of this document assumes that the binaries are up to date and have been copied.

These instructions make reference to a set of directories; the location of these directories differ between binary app developer pack and GitHub source.  The documentation for each of these define the locations for these directories in the particular use:
- `<SDK_ROOT>`: the root of the SDK Plugins tree, which contains the `bin` directory for the plugins
- `<SDK_MODELS>`: the root of the models tree for the SDK plugins, normally named `nvigi.models`
- `<SDK_TEST>`: the root of the test data tree for the SDK plugins, normally named `nvigi.test`


## Download Required Models

The basic sample requires the following models:

| Plugin | Model Name | GUID |
| ------ | ---------- | ---- |
| nvigi.plugin.asr.ggml.* | Whisper Small | 5CAD3A03-1272-4D43-9F3D-655417526170 |
| nvigi.plugin.gpt.ggml.* | Llama3.2 3b Instruct | 01F43B70-CE23-42CA-9606-74E80C5ED0B6 |
| nvigi.plugin.tts.asqflow-trt | Riva Magpie-TTS-Flow | 81320D1D-DF3C-4CFC-B9FA-4D3FF95FC35F |

See the top-level documentation that shipped with your development pack for information on how to download these models.

## Cloud Models

| Plugin | Model Name | GUID | URL |
| ------ | ---------- | ---- | ---- |
| nvigi.plugin.gpt.cloud.rest | gpt-3.5-turbo | E9102ACB-8CD8-4345-BCBF-CCF6DC758E58 | https://api.openai.com/v1/chat/completions | 
| nvigi.plugin.gpt.cloud.rest | Llama 3.2 3B Instruct | 01F43B70-CE23-42CA-9606-74E80C5ED0B6 | https://integrate.api.nvidia.com/v1/chat/completions |

> NOTE: These are just two popular models, for more details on model repository please read the `ProgrammingGuideAI` located in the NVIGI Core SDK.

## How to Use the Basic Sample

When run, the sample should launch a console (or use the one from which it was run); it will await user input.  The user may do one of three things at the prompt:
- Type a chat query as text and press enter.  This will be passed directly to the LLM and the LLM response printed.
- Type an exit command; "Q", "q" or "quit".  This will cause the sample to exit.
- Press enter with no text.  This will start recording from the default Windows/DirectX recording device (e.g. a headset microphone).  Pressing enter again will stop recording.  Once the recording is complete, the audio will be sent to the ASR plugin and then the result of ASR printed to the console and passed to the LLM for response.

## Run at Command Line

To run `nvigi.basic` from the command line, take the following steps (`--models` is a required argument):

1. Open a command prompt in `<SDK_ROOT>`
2. Run the command:
```sh
bin\x64\nvigi.basic.exe --models <SDK_MODELS> --targetPathSpectrogram <SDK_TEST>/nvigi.tts/asqflow/mel_spectrograms_targets/03_M-Tom_Sawyer_15s_se.bin
```
3. In a standard layout binary development pack or GitHub source tree, launching from a current working directory of the `<SDK_ROOT>` directory, this is:
```sh
bin\x64\nvigi.basic.exe --models data/nvigi.models --targetPathSpectrogram data/nvigi.test/nvigi.tts/asqflow/mel_spectrograms_targets/03_M-Tom_Sawyer_15s_se.bin
```

Here are the command line options:

```console
Usage: nvigi.basic [options]

  -m, --models              model repo location (REQUIRED)
  --asr                     asr backend, 'cpu', 'cuda', or 'vulkan' (default: cuda)
  --asr-guid                asr model guid in registry format, in quotes (default: "{5CAD3A03-1272-4D43-9F3D-655417526170}")
  -a, --audio               audio file location (default: )
  --extendedPhonemeDict     path to the extendend phonemes dictionary for ASqFlow TTS model
  --gpt                     gpt backend, 'cpu', 'cuda', 'vulkan', or 'cloud' - model GUID determines cloud endpoint (default: cuda)
  --gpt-guid                gpt model guid in registry format, in quotes (default: "{01F43B70-CE23-42CA-9606-74E80C5ED0B6}")
  -s, --sdk                 sdk location, (default: exe location)
  -t, --token               authorization token for the cloud provider (default: )
  --targetPathSpectrogram   target path of the spectrogram of the voice you want to clone
  --tts                     tts backend, 'cuda', 'vulkan', or 'trt' (default: trt)
  --tts-guid                tts model guid in registry format, in quotes (default: auto-selected based on backend)
  --vram                    the amount of vram to use in MB (default: 8192)
```

**TTS Model Selection**: If `--tts-guid` is not specified, the sample will automatically select an appropriate model based on the `--tts` backend:
  - TRT backend (`--tts trt`): Automatically uses `{81320D1D-DF3C-4CFC-B9FA-4D3FF95FC35F}`
  - GGML backends (`--tts cuda` or `--tts vulkan`): Automatically uses `{16EEB8EA-55A8-4F40-BECE-CE995AF44101}`

If you specify a custom `--tts-guid`, ensure it is compatible with your chosen backend. Using mismatched backend/model pairs will result in errors.

## Run in Debugger

To run `nvigi.basic` in the debugger, we must ensure that all of the plugins and the `nvigi.basic.exe` app are copied into one directory, the SDK bin directory.  Then, MSVC's debugger must know to launch the copy, not the original, or it will not find the plugins.  Take the following steps:

1. One-time setup in the project file (needs to be redone if `_project` is deleted):
    1. In the MSVC IDE, edit the project config settings for `nvigi/samples/nvigi.basic`
    1. Navigate to the "Debugging" settings
    1. Set "Command" to `<SDK_ROOT>\bin\x64\nvigi.basic.exe`
    1. Set "Command Arguments" as needed (see the command line options in the above section)
    1. Set "Working Directory" to `<SDK_ROOT>/bin/x64`
1. Build the desired non-production config of the SDK; Release is recommended (it is optimized, but contains symbols)
1. After each (re-)build, re-run `copy_sdk_binaries.bat <cfg>`
1. The sample can now be run in the debugger.
