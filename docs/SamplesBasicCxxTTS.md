# The C++ Basic TTS Sample

The C++ TTS sample, `nvigi.basic.tts.cxx`, demonstrates text-to-speech synthesis using the NVIGI SDK with modern C++ interfaces. It showcases voice cloning, real-time audio playback, and both synchronous and asynchronous generation modes with support for multiple backends.

**NOTE**: This sample supports multiple backends (D3D12, CUDA, and Vulkan), allowing it to run on various hardware configurations.

## Features

- **Voice Cloning**: Clone any voice using a target spectrogram
- **Real-Time Playback**: Play generated audio in real-time using DirectSound (Windows)
- **WAV File Output**: Save generated speech to WAV files
- **Async/Polling API**: Non-blocking operations perfect for game loops
- **Multiple Backends**: D3D12, CUDA, or Vulkan backends
- **Language Support**: Multiple languages (en, en-us, en-uk, es, de)
- **Speed Control**: Adjust speech speed from 0.5x to 2.0x
- **Quality Control**: Configure timesteps for quality/speed tradeoff

## Download Required Models

The TTS sample requires the following model:

| Plugin | Model Name | GUID |
| ------ | ---------- | ---- |
| nvigi.plugin.tts.asqflow-trt | Riva Magpie-TTS-Flow | 81320D1D-DF3C-4CFC-B9FA-4D3FF95FC35F |
| nvigi.plugin.tts.asqflow-ggml.* | Riva Magpie-TTS-Flow (GGML) | 16EEB8EA-55A8-4F40-BECE-CE995AF44101 |

**Important**: You also need a target voice spectrogram file. The SDK test data includes sample spectrograms in `<SDK_TEST>/nvigi.tts/asqflow/mel_spectrograms_targets/`.

See the top-level documentation that shipped with your development pack for information on how to download these models and test data.

## Building the Sample

The sample is built as part of the SDK build process. After building, copy the binaries using:

```sh
copy_sdk_binaries.bat <cfg>
```

This ensures all DLLs and the executable are in the same directory (`bin\x64`).

## How to Use the TTS Sample

### Basic Speech Generation

1. Open a command prompt in `<SDK_ROOT>`
2. Run the command:

```sh
bin\x64\nvigi.basic.tts.cxx.exe --models <SDK_MODELS> --sdk bin\x64 --target <SDK_TEST>/nvigi.tts/asqflow/mel_spectrograms_targets/03_M-Tom_Sawyer_15s_se.bin
```

3. In a standard layout binary development pack or GitHub source tree:

```sh
bin\x64\nvigi.basic.tts.cxx.exe --models data/nvigi.models --sdk bin\x64 --target data/nvigi.test/nvigi.tts/asqflow/mel_spectrograms_targets/03_M-Tom_Sawyer_15s_se.bin
```

4. The sample will generate speech for the default text and save it to `output.wav`
5. You can play the generated WAV file to hear the synthesized speech

### Real-Time Playback Mode

To hear the generated speech in real-time as it's being generated:

```sh
bin\x64\nvigi.basic.tts.cxx.exe --models data/nvigi.models --sdk bin\x64 --target data/nvigi.test/nvigi.tts/asqflow/mel_spectrograms_targets/03_M-Tom_Sawyer_15s_se.bin --play
```

### Async Mode (Non-Blocking)

For game integration or when you need to continue other processing:

```sh
bin\x64\nvigi.basic.tts.cxx.exe --models data/nvigi.models --sdk bin\x64 --target data/nvigi.test/nvigi.tts/asqflow/mel_spectrograms_targets/03_M-Tom_Sawyer_15s_se.bin --async --play
```

## Command Line Options

```console
Usage: nvigi.basic.tts.cxx [options]

  --sdk                     sdk location, if none provided assuming exe location
  --plugin                  plugin location, if none provided assuming sdk location
  -m, --models              model repo location (REQUIRED)
  -t, --threads             number of threads (default: 8)
  --backend                 backend to use - d3d12, cuda, vulkan (default: d3d12)
  --guid                    TTS model guid in registry format, in quotes (default: "{16EEB8EA-55A8-4F40-BECE-CE995AF44101}")
  --vram                    the amount of vram to use in MB (default: 2048)
  --log-level               logging level 0-2 (default: 0)
  --text                    text to synthesize (default: "Hello! This is a test of the text to speech system.")
  --target                  path to target voice spectrogram (REQUIRED)
  --output                  output WAV file path (default: output.wav)
  --speed                   speech speed (0.5 - 2.0) (default: 1.0)
  --language                language code (en, en-us, en-uk, es, de) (default: en)
  --timesteps               number of timesteps for TTS inference (16-32) (default: 16)
  --async                   use async mode (polled, non-blocking)
  --play                    play audio in real-time using DirectSound
  --print-system-info       print system information
```

## Examples

### Generate speech with custom text:

```sh
bin\x64\nvigi.basic.tts.cxx.exe --models data/nvigi.models --sdk bin\x64 --target data/nvigi.test/nvigi.tts/asqflow/mel_spectrograms_targets/03_M-Tom_Sawyer_15s_se.bin --text "Welcome to the NVIGI SDK text to speech system."
```

### Use a different voice:

```sh
bin\x64\nvigi.basic.tts.cxx.exe --models data/nvigi.models --sdk bin\x64 --target data/nvigi.test/nvigi.tts/asqflow/mel_spectrograms_targets/01_F-Jennifer_20s_se.bin --text "Hello world!"
```

### Adjust speech speed (slower):

```sh
bin\x64\nvigi.basic.tts.cxx.exe --models data/nvigi.models --sdk bin\x64 --target data/nvigi.test/nvigi.tts/asqflow/mel_spectrograms_targets/03_M-Tom_Sawyer_15s_se.bin --speed 0.75
```

### Adjust speech speed (faster):

```sh
bin\x64\nvigi.basic.tts.cxx.exe --models data/nvigi.models --sdk bin\x64 --target data/nvigi.test/nvigi.tts/asqflow/mel_spectrograms_targets/03_M-Tom_Sawyer_15s_se.bin --speed 1.5
```

### Higher quality (more timesteps, slower):

```sh
bin\x64\nvigi.basic.tts.cxx.exe --models data/nvigi.models --sdk bin\x64 --target data/nvigi.test/nvigi.tts/asqflow/mel_spectrograms_targets/03_M-Tom_Sawyer_15s_se.bin --timesteps 32
```

### Lower quality (fewer timesteps, faster):

```sh
bin\x64\nvigi.basic.tts.cxx.exe --models data/nvigi.models --sdk bin\x64 --target data/nvigi.test/nvigi.tts/asqflow/mel_spectrograms_targets/03_M-Tom_Sawyer_15s_se.bin --timesteps 16
```

### Use Vulkan backend:

```sh
bin\x64\nvigi.basic.tts.cxx.exe --models data/nvigi.models --sdk bin\x64 --target data/nvigi.test/nvigi.tts/asqflow/mel_spectrograms_targets/03_M-Tom_Sawyer_15s_se.bin --backend vulkan
```

### Spanish language:

```sh
bin\x64\nvigi.basic.tts.cxx.exe --models data/nvigi.models --sdk bin\x64 --target data/nvigi.test/nvigi.tts/asqflow/mel_spectrograms_targets/03_M-Tom_Sawyer_15s_se.bin --language es --text "Hola mundo"
```

### Save to custom output file:

```sh
bin\x64\nvigi.basic.tts.cxx.exe --models data/nvigi.models --sdk bin\x64 --target data/nvigi.test/nvigi.tts/asqflow/mel_spectrograms_targets/03_M-Tom_Sawyer_15s_se.bin --output my_speech.wav
```

## Audio Format

The generated audio uses the following format:
- Sample Rate: 22050 Hz (22.05 kHz)
- Bit Depth: 16-bit PCM
- Channels: Mono

## API Patterns Demonstrated

### Synchronous (Blocking) Generation

The sample demonstrates a simple blocking pattern where generation blocks until complete:

```cpp
instance->generate(
    text,
    target_path,
    config,
    [&wav_writer](const int16_t* audio, size_t samples, ExecutionState state) -> ExecutionState {
        if (state == ExecutionState::DataPending || state == ExecutionState::Done) {
            // Write audio chunk to file
            wav_writer.write_samples(audio, samples);
            
            // Optionally play in real-time
            AudioPlayer::play_audio(audio, samples);
        }
        return state;  // Continue
    }
);
```

### Asynchronous (Non-Blocking) Generation

The sample also demonstrates a polling-based async pattern perfect for game loops:

```cpp
auto op = instance->generate_async(text, target_path, config);

// Game loop
while (!op.is_complete()) {
    // Try to get results (non-blocking)
    if (auto result = op.try_get_results()) {
        if (!result->audio.empty()) {
            // Write audio chunk to file
            wav_writer.write_samples(result->audio.data(), result->audio.size());
            
            // Play in real-time
            AudioPlayer::play_audio(result->audio.data(), result->audio.size());
            
            if (result->state == ExecutionState::Done) {
                break;
            }
        }
    }
    
    // Continue game logic
    render_frame();
    update_physics();
    process_input();
    
    // Small sleep to avoid busy-wait
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}
```

## Voice Cloning

Voice cloning is achieved using target spectrogram files. These files contain the acoustic characteristics of the voice you want to clone. The SDK test data includes several sample voices:

- `01_F-Jennifer_20s_se.bin` - Female voice (Jennifer)
- `03_M-Tom_Sawyer_15s_se.bin` - Male voice (Tom Sawyer)
- Additional samples in `<SDK_TEST>/nvigi.tts/asqflow/mel_spectrograms_targets/`

To use a different voice, simply change the `--target` parameter to point to a different spectrogram file.

## Quality vs Speed Tradeoff

The `--timesteps` parameter controls the quality/speed tradeoff:

- **16 timesteps**: Faster generation, slightly lower quality (default)
- **24 timesteps**: Balanced quality and speed
- **32 timesteps**: Highest quality, slower generation

For real-time applications, 16-24 timesteps is recommended. For offline generation where quality is paramount, use 32 timesteps.

## Troubleshooting

### "Failed to create TTS instance"
- Verify the model GUID exists in your models directory
- Check that you have sufficient VRAM (try reducing `--vram`)
- Ensure the backend is properly installed (e.g., D3D12 requires Windows 10+)

### Missing target spectrogram file
- Verify the path to the spectrogram file is correct
- Check that the test data has been downloaded
- Use an absolute path if relative paths aren't working

### No audio playback with `--play`
- Real-time playback only works on Windows with DirectSound
- Check that your audio output device is working
- Try without `--play` and verify the WAV file is generated correctly

### Out of memory errors
- Reduce VRAM budget: `--vram 1024`
- Close other GPU-intensive applications
- Try a smaller text input

### Poor audio quality
- Increase timesteps: `--timesteps 32`
- Check the quality of the target spectrogram file
- Verify the language parameter matches your text

## Programming Notes

This sample demonstrates:
- Modern C++ wrapper interfaces (`nvigi::tts::Instance`)
- Blocking generation with streaming callbacks
- Non-blocking async operations with polling (perfect for games)
- Real-time audio playback using DirectSound
- WAV file writing with proper headers
- Builder pattern for runtime configuration
- RAII-based resource management
- `std::expected` for error handling

## Run in Debugger

To run `nvigi.basic.tts.cxx` in the debugger:

1. One-time setup in the project file:
    1. In the MSVC IDE, edit the project config settings for `nvigi/samples/nvigi.basic.cxx/tts`
    1. Navigate to the "Debugging" settings
    1. Set "Command" to `<SDK_ROOT>\bin\x64\nvigi.basic.tts.cxx.exe`
    1. Set "Command Arguments" as needed (see command line options above)
    1. Set "Working Directory" to `<SDK_ROOT>/bin/x64`
1. Build the desired non-production config; Release is recommended
1. After each (re-)build, re-run `copy_sdk_binaries.bat <cfg>`
1. The sample can now be run in the debugger

