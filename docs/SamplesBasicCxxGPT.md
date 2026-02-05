# The C++ Basic GPT Sample

The C++ GPT sample, `nvigi.basic.gpt.cxx`, demonstrates language model inference using the NVIGI SDK with modern C++ interfaces. It showcases both local and cloud-based LLM inference with support for chat conversations, streaming responses, and non-blocking polling-based operations perfect for game integration.

**NOTE**: This sample supports multiple backends (D3D12, CUDA, Vulkan, and Cloud), providing maximum flexibility for different deployment scenarios.

## Features

- **Chat Interface**: Interactive multi-turn conversations with message history
- **Streaming Responses**: Token-by-token streaming for real-time display
- **Async/Polling API**: Non-blocking operations perfect for game loops
- **Multiple Backends**: D3D12, CUDA, Vulkan (local), or Cloud (REST API)
- **Cloud Provider Support**: OpenAI, NVIDIA NIM, and other OpenAI-compatible APIs
- **LoRA Support**: Load and apply LoRA adapters for model customization
- **KV Cache Quantization**: FP32, FP16, Q4_0, Q8_0 for memory optimization
- **Custom JSON Requests**: Full control over cloud API requests

## Download Required Models

### Local Models

For local inference, the sample requires one of the following models:

| Plugin | Model Name | GUID |
| ------ | ---------- | ---- |
| nvigi.plugin.gpt.ggml.* | Llama3.2 3b Instruct | 01F43B70-CE23-42CA-9606-74E80C5ED0B6 |
| nvigi.plugin.gpt.ggml.* | Phi-3.5 Mini Instruct | 8E31808B-C182-4016-9ED8-64804FF5B40D |

### Cloud Models

For cloud inference, configure the model in your cloud provider dashboard:

| Plugin | Model Name | GUID | URL |
| ------ | ---------- | ---- | ---- |
| nvigi.plugin.gpt.cloud.rest | gpt-3.5-turbo | E9102ACB-8CD8-4345-BCBF-CCF6DC758E58 | https://api.openai.com/v1/chat/completions | 
| nvigi.plugin.gpt.cloud.rest | Llama 3.2 3B Instruct | 01F43B70-CE23-42CA-9606-74E80C5ED0B6 | https://integrate.api.nvidia.com/v1/chat/completions |

See the top-level documentation that shipped with your development pack for information on how to download these models.

## Building the Sample

The sample is built as part of the SDK build process. After building, copy the binaries using:

```sh
copy_sdk_binaries.bat <cfg>
```

This ensures all DLLs and the executable are in the same directory (`bin\x64`).

## How to Use the GPT Sample

### Basic Chat (Local Model)

1. Open a command prompt in `<SDK_ROOT>`
2. Run the command:

```sh
bin\x64\nvigi.basic.gpt.cxx.exe --models <SDK_MODELS> --sdk bin\x64
```

3. In a standard layout binary development pack or GitHub source tree:

```sh
bin\x64\nvigi.basic.gpt.cxx.exe --models data/nvigi.models --sdk bin\x64
```

4. Wait for the initial system prompt to complete
5. Type your message at the `User>` prompt and press Enter
6. The AI response will stream token-by-token to the console
7. Continue the conversation or type `quit`/`exit` to exit

### Cloud Provider Usage

To use a cloud provider (e.g., OpenAI):

```sh
bin\x64\nvigi.basic.gpt.cxx.exe --models data/nvigi.models --sdk bin\x64 --backend cloud --token YOUR_API_KEY --guid "{E9102ACB-8CD8-4345-BCBF-CCF6DC758E58}"
```

To use NVIDIA NIM:

```sh
bin\x64\nvigi.basic.gpt.cxx.exe --models data/nvigi.models --sdk bin\x64 --backend cloud --token YOUR_NVIDIA_API_KEY --guid "{01F43B70-CE23-42CA-9606-74E80C5ED0B6}"
```

## Command Line Options

```console
Usage: nvigi.basic.gpt.cxx [options]

  --sdk                  sdk location, if none provided assuming exe location
  --plugin               plugin location, if none provided assuming sdk location
  -m, --models           model repo location (REQUIRED)
  -t, --threads          number of threads (default: 1)
  --backend              backend to use - d3d12, cuda, vulkan, cloud (default: d3d12)
  --guid                 gpt model guid in registry format, in quotes (default: "{8E31808B-C182-4016-9ED8-64804FF5B40D}")
  --url                  URL to use, if none provided default is taken from model JSON
  --json                 custom JSON body for cloud request (path to JSON file)
  --token                authorization token for the cloud provider
  --vram                 the amount of vram to use in MB (default: 8192)
  --cache-type           KV cache quantization type: fp16, fp32, q4_0, q8_0 (default: fp16)
  --log-level            logging level 0-2 (default: 0)
  --print-system-info    print system information
```

## Examples

### Use a different local model:

```sh
bin\x64\nvigi.basic.gpt.cxx.exe --models data/nvigi.models --sdk bin\x64 --guid "{01F43B70-CE23-42CA-9606-74E80C5ED0B6}"
```

### Use Vulkan backend with 4-bit cache quantization:

```sh
bin\x64\nvigi.basic.gpt.cxx.exe --models data/nvigi.models --sdk bin\x64 --backend vulkan --cache-type q4_0
```

### Custom VRAM budget for large models:

```sh
bin\x64\nvigi.basic.gpt.cxx.exe --models data/nvigi.models --sdk bin\x64 --vram 16384
```

### Cloud with custom JSON request body:

```sh
bin\x64\nvigi.basic.gpt.cxx.exe --models data/nvigi.models --sdk bin\x64 --backend cloud --token YOUR_KEY --json custom_request.json
```

## API Patterns Demonstrated

### Blocking Chat

The sample demonstrates a simple blocking chat pattern where each message blocks until the full response is generated:

```cpp
chat.send_message(
    { .role = User, .content = "Hello!" },
    [](std::string_view response, ExecutionState state) -> ExecutionState {
        std::cout << response;  // Stream tokens to console
        return state;  // Continue
    }
);
```

### Non-Blocking Polling (Game Loop Pattern)

The sample also demonstrates a polling-based async pattern perfect for game loops:

```cpp
auto op = chat.send_message_polled({ .role = User, .content = "Hello!" });

// Game loop
while (game_running) {
    // Poll for tokens (non-blocking)
    if (auto result = op.try_get_results()) {
        std::cout << result->tokens;  // Display immediately
        
        if (result->state == ExecutionState::Done) {
            chat.finalize_async_response(op);
            break;
        }
    }
    
    // Continue game logic
    render_frame();
    update_physics();
    process_input();
    
    // Can cancel at any time
    if (user_pressed_cancel) {
        op.cancel();
    }
}
```

## KV Cache Quantization

The sample supports different KV cache quantization types to balance memory usage and quality:

- **fp32**: Full precision (highest quality, most memory)
- **fp16**: Half precision (default, good balance)
- **q4_0**: 4-bit quantization (significant memory savings)
- **q8_0**: 8-bit quantization (moderate memory savings)

Use `--cache-type` to select:

```sh
bin\x64\nvigi.basic.gpt.cxx.exe --models data/nvigi.models --sdk bin\x64 --cache-type q4_0
```

## Cloud Provider Configuration

### OpenAI

```sh
bin\x64\nvigi.basic.gpt.cxx.exe --models data/nvigi.models --sdk bin\x64 ^
    --backend cloud ^
    --token sk-... ^
    --guid "{E9102ACB-8CD8-4345-BCBF-CCF6DC758E58}"
```

### NVIDIA NIM

```sh
bin\x64\nvigi.basic.gpt.cxx.exe --models data/nvigi.models --sdk bin\x64 ^
    --backend cloud ^
    --token nvapi-... ^
    --guid "{01F43B70-CE23-42CA-9606-74E80C5ED0B6}"
```

### Custom OpenAI-Compatible API

```sh
bin\x64\nvigi.basic.gpt.cxx.exe --models data/nvigi.models --sdk bin\x64 ^
    --backend cloud ^
    --url https://your-api.com/v1/chat/completions ^
    --token your-token ^
    --guid "{YOUR-MODEL-GUID}"
```

## Troubleshooting

### "Failed to create inference instance"
- Verify the model GUID exists in your models directory
- Check that you have sufficient VRAM (try reducing `--vram`)
- Ensure the backend is properly installed (e.g., D3D12 requires Windows 10+)

### Cloud authentication errors
- Verify your API token is valid and not expired
- Check that the URL is correct for your provider
- Ensure you have an active internet connection

### Out of memory errors
- Reduce VRAM budget: `--vram 4096`
- Use more aggressive KV cache quantization: `--cache-type q4_0`
- Try a smaller model

### Slow inference
- Ensure GPU is being used (check Task Manager > Performance > GPU)
- Try increasing threads for CPU models: `--threads 8`
- For cloud, check network latency

## Programming Notes

This sample demonstrates:
- Modern C++ wrapper interfaces (`nvigi::gpt::Instance`)
- Chat conversation management with message history
- Streaming token-by-token responses
- Blocking callbacks for simple use cases
- Non-blocking async operations with polling (perfect for games)
- Builder pattern for runtime configuration
- RAII-based resource management
- `std::expected` for error handling
- Cloud and local inference with unified API

## Run in Debugger

To run `nvigi.basic.gpt.cxx` in the debugger:

1. One-time setup in the project file:
    1. In the MSVC IDE, edit the project config settings for `nvigi/samples/nvigi.basic.cxx/gpt`
    1. Navigate to the "Debugging" settings
    1. Set "Command" to `<SDK_ROOT>\bin\x64\nvigi.basic.gpt.cxx.exe`
    1. Set "Command Arguments" as needed (see command line options above)
    1. Set "Working Directory" to `<SDK_ROOT>/bin/x64`
1. Build the desired non-production config; Release is recommended
1. After each (re-)build, re-run `copy_sdk_binaries.bat <cfg>`
1. The sample can now be run in the debugger

