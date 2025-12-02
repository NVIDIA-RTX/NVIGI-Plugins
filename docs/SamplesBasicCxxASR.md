# The C++ Basic ASR Sample

The C++ ASR sample, `nvigi.basic.asr.cxx`, demonstrates automatic speech recognition (ASR) using the NVIGI SDK with modern C++ interfaces. It showcases both complete audio transcription and real-time streaming modes, allowing you to record audio from a microphone and convert speech to text.

**NOTE**: This sample supports multiple backends (D3D12, CUDA, and Vulkan), allowing it to run on various hardware configurations.

## Features

- **Complete Audio Mode**: Record a fixed duration (5 seconds) and transcribe the complete audio clip
- **Streaming Mode**: Real-time continuous audio streaming with incremental transcription
- **Multiple Backend Support**: D3D12, CUDA, or Vulkan backends
- **Language Support**: Auto-detection or specific language selection (en, es, fr, etc.)
- **Translation**: Automatic translation to English if desired
- **WAV File Export**: Save recorded audio to WAV files for debugging

## Download Required Models

The ASR sample requires the following model:

| Plugin | Model Name | GUID |
| ------ | ---------- | ---- |
| nvigi.plugin.asr.ggml.* | Whisper Small | 5CAD3A03-1272-4D43-9F3D-655417526170 |

See the top-level documentation that shipped with your development pack for information on how to download these models.

## Building the Sample

The sample is built as part of the SDK build process. After building, copy the binaries using:

```sh
copy_sdk_binaries.bat <cfg>
```

This ensures all DLLs and the executable are in the same directory (`bin\x64`).

## How to Use the ASR Sample

### Complete Audio Mode (Default)

In this mode, you record a fixed duration (5 seconds) of audio and then transcribe it as a complete clip.

1. Open a command prompt in `<SDK_ROOT>`
2. Run the command:

```sh
bin\x64\nvigi.basic.asr.cxx.exe --models <SDK_MODELS> --sdk bin\x64
```

3. In a standard layout binary development pack or GitHub source tree:

```sh
bin\x64\nvigi.basic.asr.cxx.exe --models data/nvigi.models --sdk bin\x64
```

4. At the prompt, type `record` to start a 5-second recording
5. Speak into your microphone during the recording
6. The transcribed text will be displayed after recording completes
7. Type `quit` or `exit` to exit the application

### Streaming Mode

Streaming mode provides real-time continuous audio streaming with incremental transcription.

1. Run the sample with the `--streaming` flag:

```sh
bin\x64\nvigi.basic.asr.cxx.exe --models data/nvigi.models --sdk bin\x64 --streaming
```

2. At the prompt, type `stream` to start real-time streaming
3. Speak into your microphone - you'll see transcription appear in real-time
4. Press Enter to stop streaming
5. Type `quit` or `exit` to exit the application

## Command Line Options

```console
Usage: nvigi.basic.asr.cxx [options]

  --sdk                  sdk location, if none provided assuming exe location
  --plugin               plugin location, if none provided assuming sdk location
  -m, --models           model repo location (REQUIRED)
  -t, --threads          number of threads (default: 8)
  --fa, --flash-attention   use flash attention
  --backend              backend to use - d3d12, cuda, vulkan (default: d3d12)
  --guid                 ASR model guid in registry format (default: {5CAD3A03-1272-4D43-9F3D-655417526170})
  --vram                 the amount of vram to use in MB (default: 2048)
  --log-level            logging level 0-2 (default: 0)
  --language             language code (en, es, fr, auto, etc.) (default: en)
  --detect-lang          auto-detect language
  --translate            translate to English
  --streaming            use streaming mode (experimental)
  --save-wav             save audio to file (default: c:/test.wav)
  --print-system-info    print system information
```

## Examples

### Basic transcription with English language:

```sh
bin\x64\nvigi.basic.asr.cxx.exe --models data/nvigi.models --sdk bin\x64 --language en
```

### Auto-detect language and translate to English:

```sh
bin\x64\nvigi.basic.asr.cxx.exe --models data/nvigi.models --sdk bin\x64 --detect-lang --translate
```

### Use Vulkan backend with streaming mode:

```sh
bin\x64\nvigi.basic.asr.cxx.exe --models data/nvigi.models --sdk bin\x64 --backend vulkan --streaming
```

### Save recorded audio to a WAV file:

```sh
bin\x64\nvigi.basic.asr.cxx.exe --models data/nvigi.models --sdk bin\x64 --save-wav recording.wav
```

## Audio Format

The sample uses the following audio format:
- Sample Rate: 16000 Hz (16 kHz)
- Bit Depth: 16-bit PCM
- Channels: Mono
- Input: Windows DirectX default recording device

## Troubleshooting

### "Failed to start recording"
- Check that a microphone is connected and enabled in Windows sound settings
- Verify the default recording device is set correctly in Windows

### No audio recorded
- Ensure the microphone is not muted
- Check microphone permissions in Windows privacy settings
- Try speaking louder or moving closer to the microphone

### Out of memory errors
- Try reducing the `--vram` parameter value
- Close other GPU-intensive applications

## Programming Notes

This sample demonstrates:
- Modern C++ wrapper interfaces (`nvigi::asr::Instance`)
- Blocking transcription with callbacks
- Non-blocking async operations with polling
- Real-time audio streaming with incremental results
- Builder pattern for configuration
- RAII-based resource management
- `std::expected` for error handling

## Run in Debugger

To run `nvigi.basic.asr.cxx` in the debugger:

1. One-time setup in the project file:
    1. In the MSVC IDE, edit the project config settings for `nvigi/samples/nvigi.basic.cxx/asr`
    1. Navigate to the "Debugging" settings
    1. Set "Command" to `<SDK_ROOT>\bin\x64\nvigi.basic.asr.cxx.exe`
    1. Set "Command Arguments" as needed (see command line options above)
    1. Set "Working Directory" to `<SDK_ROOT>/bin/x64`
1. Build the desired non-production config; Release is recommended
1. After each (re-)build, re-run `copy_sdk_binaries.bat <cfg>`
1. The sample can now be run in the debugger

