# NVIGI C++ Wrappers - Shared Library

## Overview

This directory contains shared C++ wrapper headers for the NVIGI SDK. These wrappers provide a modern C++ interface over the C API, with RAII, type safety, and improved ergonomics.

## Location

```
source/samples/shared/cxx_wrappers/
```

## Used By

- `source/samples/nvigi.basic.cxx/` - Basic samples demonstrating individual features
- `source/samples/nvigi.3d/` - Full 3D rendering sample with AI integration

## Files

### Core Infrastructure
- **core.hpp** - Core NVIGI initialization and system information
  - `nvigi::Core` - RAII wrapper for SDK initialization
  - `nvigi::SystemInfo` - Query system capabilities and plugins
  - `nvigi::Adapter` - Graphics adapter information
  - `nvigi::Plugin` - Plugin information and status

- **d3d12.hpp** - Direct3D 12 integration helpers
  - D3D12 device and queue configuration
  - Memory allocation tracking
  - Leak detection utilities

- **vulkan.hpp** - Vulkan integration helpers
  - Vulkan instance, device, and queue configuration
  - Memory allocation tracking
  - Leak detection utilities

- **io_helpers.hpp** - Custom I/O for model loading
  - Custom file I/O callbacks
  - Memory buffer loading
  - File wrapper utilities

- **clparser.hpp** - Command-line argument parsing
  - Simple argument parser for samples

### Feature-Specific Wrappers

- **gpt/gpt.hpp** - GPT (Generative Pre-trained Transformer) wrapper
  - `nvigi::gpt::Instance` - GPT model instance management
  - `nvigi::gpt::ModelConfig` - Model configuration
  - `nvigi::gpt::RuntimeConfig` - Runtime parameters
  - `nvigi::gpt::Chat` - Chat conversation management
  - `nvigi::gpt::AsyncOperation` - Async inference tracking

- **asr/asr.hpp** - ASR (Automatic Speech Recognition) wrapper
  - `nvigi::asr::Instance` - ASR model instance management
  - `nvigi::asr::ModelConfig` - Model configuration
  - `nvigi::asr::RuntimeConfig` - Runtime parameters
  - `nvigi::asr::Stream` - Audio stream management
  - `nvigi::asr::AsyncOperation` - Async transcription tracking

- **asr/audio.hpp** - Audio utilities for ASR
  - Audio capture and playback
  - WAV file I/O
  - Audio format conversion

- **tts/tts.hpp** - TTS (Text-to-Speech) wrapper
  - `nvigi::tts::Instance` - TTS model instance management
  - `nvigi::tts::ModelConfig` - Model configuration
  - `nvigi::tts::RuntimeConfig` - Runtime parameters
  - `nvigi::tts::WAVWriter` - WAV file output
  - `nvigi::tts::AudioPlayer` - Audio playback
  - `nvigi::tts::AsyncOperation` - Async synthesis tracking

## Usage

### Include in Your Project

Add this directory to your include paths:

**CMake:**
```cmake
target_include_directories(your_target PRIVATE 
    ${PROJECT_SOURCE_DIR}/source/samples/shared/cxx_wrappers
)
```

**Premake:**
```lua
includedirs {
    "%{wks.location}/source/samples/shared/cxx_wrappers"
}
```

### Basic Example

```cpp
#include "core.hpp"
#include "gpt/gpt.hpp"

int main() {
    // Initialize NVIGI
    nvigi::Core core(nvigi::Core::Config{
        .sdkPath = ".",
        .logLevel = nvigi::LogLevel::eInfo,
        .showConsole = true
    });
    
    // Print system info
    core.getSystemInfo().print();
    
    // Create GPT instance
    auto gpt = nvigi::gpt::Instance::create(
        core,
        nvigi::plugin::gpt::ggml::d3d12::kId,
        nvigi::gpt::ModelConfig{
            .modelRoot = "./models",
            .modelGUID = "{...}"
        }
    );
    
    if (!gpt) {
        std::cerr << "Failed to create GPT: " << gpt.error().message << "\n";
        return 1;
    }
    
    // Use GPT instance
    // ...
    
    return 0;
}
```

## Design Principles

1. **RAII** - Resources are automatically managed via constructors/destructors
2. **Type Safety** - Strong typing and compile-time checks where possible
3. **Error Handling** - Uses `std::expected` for fallible operations
4. **Zero-Cost Abstraction** - Wrappers compile to equivalent C code
5. **Header-Only** - No separate compilation required

## Error Handling

The wrappers use `std::expected<T, Error>` for operations that can fail:

```cpp
auto result = nvigi::gpt::Instance::create(...);
if (!result) {
    std::cerr << "Error: " << result.error().message << "\n";
    return;
}

auto& instance = *result;
// Use instance...
```

## Memory Management

- **Core** - Manages DLL lifetime and nvigiInit/nvigiShutdown
- **Instance wrappers** - Manage model instance lifetime
- **Smart pointers** - Used throughout for automatic cleanup
- **No manual new/delete** - RAII handles all allocations

## Thread Safety

The wrappers themselves are not thread-safe (matching the underlying C API). Users must serialize access to instances or use separate instances per thread.

## Migration from C API

See `SHARED_WRAPPERS_MIGRATION.md` in `nvigi.3d` for a guide on migrating existing C API code to use these wrappers.

## Contributing

When updating these wrappers:

1. **Maintain backward compatibility** - Don't break existing samples
2. **Update all samples** - Test changes with both `nvigi.basic.cxx` and `nvigi.3d`
3. **Document changes** - Update this README and migration guides
4. **Follow the style** - Match existing code patterns

## Version History

- **v1.0** (Dec 2025) - Initial shared wrapper library
  - Moved from `nvigi.basic.cxx` to shared location
  - Used by multiple samples

## License

SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT

