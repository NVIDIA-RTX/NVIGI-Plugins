// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Shared IO helpers for custom file loading implementations
// This file provides optional IO callback support for ASR, GPT, and TTS samples
// to allow hosts to provide custom implementations for file I/O operations
// (e.g., loading from memory, database, network, etc.)

#pragma once

#include <nvigi.h>
#include <nvigi_io.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <map>
#include <mutex>
#include <functional>

namespace nvigi {
namespace io {

// Thread-local storage for errors that occur before we have a handle (e.g., open failures)
static thread_local Result g_customIOLastError = kResultOk;

// IO callback configuration for custom file loading
// 
// This allows you to override how model files are loaded by providing
// custom implementations for file operations. Use cases include:
// - Streaming models from memory buffers
// - Streaming from network/cloud storage  
// - Reading from encrypted/compressed archives
// - Loading from custom databases
// - Memory-mapped files
//
// The callbacks mirror standard FILE* operations but allow any custom backend.
// 
// Example usage:
//   IOConfig io_config;
//   io_config.enable_custom_io = true;
//   io_config.open = [](void* userCtx, const char* path, const char* mode) -> void* {
//       return my_open_file(path);  // Return your custom file handle
//   };
//   io_config.read = [](void* userCtx, void* handle, void* buffer, size_t size) -> size_t {
//       return my_read_chunk(handle, buffer, size);  // Read chunk on demand
//   };
//   // ... implement other callbacks ...
//
struct IOConfig {
    bool enable_custom_io{ false };    // Enable custom IO callbacks
    void* user_context{ nullptr };     // User data passed to all callbacks
    
    // Open a file and return a handle (required)
    // handle = open(userCtx, path, mode)
    // Return nullptr on failure
    std::function<void*(void* userCtx, const char* path, const char* mode)> open;
    
    // Close a file handle (required)
    // close(userCtx, handle)
    std::function<void(void* userCtx, void* handle)> close;
    
    // Get file size in bytes (required)
    // size = size(userCtx, handle)
    std::function<size_t(void* userCtx, void* handle)> size;
    
    // Get current position in file (required)
    // pos = tell(userCtx, handle)
    std::function<size_t(void* userCtx, void* handle)> tell;
    
    // Seek to position in file (required)
    // result = seek(userCtx, handle, offset, origin)
    // origin: SEEK_SET (0), SEEK_CUR (1), SEEK_END (2)
    // Return 0 on success, -1 on failure
    std::function<int(void* userCtx, void* handle, size_t offset, int origin)> seek;
    
    // Read data from file (required)
    // bytes_read = read(userCtx, handle, buffer, size)
    // Return number of bytes actually read
    std::function<size_t(void* userCtx, void* handle, void* buffer, size_t size)> read;
    
    // Write data to file (optional, not used for model loading)
    std::function<size_t(void* userCtx, void* handle, const void* buffer, size_t size)> write;

    // Map data to a memory pointer (optional, not used for model loading)
    std::function<uint8_t* (void* userCtx, void* handle, size_t offset, size_t size, MapAccess access)> map;

    // Unmap previously mapped data (optional, not used for model loading)
    std::function<void(void* userCtx, void* handle, uint8_t* mappedPtr)> unmap;

    // Get last error (optional but recommended)
    // error = getLastError(userCtx, handle)
    // Return kResultOk if no error, or appropriate error code
    std::function<Result(void* userCtx, void* handle)> getLastError;
};

// Internal implementation details
namespace detail {

// Global state for custom IO callbacks
struct IOState {
    std::mutex mutex;
    IOConfig config;  // Current active IO configuration

    static IOState& get() {
        static IOState instance;
        return instance;
    }
};

// Trampoline functions that forward to user callbacks

inline void* io_open(void* userData, const char* filename, const char* mode) {
    auto& state = IOState::get();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (!state.config.open) {
        return nullptr;  // Fall back to default
    }

    return state.config.open(state.config.user_context, filename, mode);
}

inline void io_close(void* userData, void* handle) {
    if (!handle) return;

    auto& state = IOState::get();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (state.config.close) {
        state.config.close(state.config.user_context, handle);
    }
}

inline size_t io_size(void* userData, void* handle) {
    if (!handle) return 0;

    auto& state = IOState::get();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (!state.config.size) {
        return 0;
    }

    return state.config.size(state.config.user_context, handle);
}

inline size_t io_tell(void* userData, void* handle) {
    if (!handle) return 0;

    auto& state = IOState::get();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (!state.config.tell) {
        return 0;
    }

    return state.config.tell(state.config.user_context, handle);
}

inline int io_seek(void* userData, void* handle, size_t offset, int origin) {
    if (!handle) return -1;

    auto& state = IOState::get();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (!state.config.seek) {
        return -1;
    }

    return state.config.seek(state.config.user_context, handle, offset, origin);
}

inline size_t io_read(void* userData, void* handle, void* buffer, size_t size) {
    if (!handle || !buffer) return 0;

    auto& state = IOState::get();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (!state.config.read) {
        return 0;
    }

    return state.config.read(state.config.user_context, handle, buffer, size);
}

inline size_t io_write(void* userData, void* handle, const void* buffer, size_t size) {
    if (!handle || !buffer) return 0;

    auto& state = IOState::get();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (!state.config.write) {
        return 0;  // Write not implemented
    }

    return state.config.write(state.config.user_context, handle, buffer, size);
}

inline uint8_t* io_map(void* userData, void* handle, size_t offset, size_t size, MapAccess access) {
    if (!handle) return nullptr;

    auto& state = IOState::get();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (!state.config.map) {
        return nullptr;  // Map not implemented
    }

    return static_cast<uint8_t*>(state.config.map(state.config.user_context, handle, offset, size, access));
}

inline void io_unmap(void* userData, void* handle, uint8_t* mappedPtr) {
    if (!handle || !mappedPtr) return;

    auto& state = IOState::get();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (state.config.unmap) {
        return state.config.unmap(state.config.user_context, handle, mappedPtr);
    }
    return;
}

inline Result io_getLastError(void* userData, void* handle) {
    auto& state = IOState::get();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (!state.config.getLastError) {
        return kResultOk;  // No error tracking
    }

    return state.config.getLastError(state.config.user_context, handle);
}
} // namespace detail

// Helper function to configure IO callbacks for NVIGI creation parameters
// 
// Usage:
//   IOConfig io_config;
//   io_config.enable_custom_io = true;
//   io_config.open = [](void* ctx, const char* path, const char* mode) -> void* {
//       return my_open(path);
//   };
//   io_config.read = [](void* ctx, void* handle, void* buffer, size_t size) -> size_t {
//       return my_read(handle, buffer, size);
//   };
//   // ... other callbacks ...
//   
//   CommonCreationParameters common{};
//   configure_io_callbacks(common, io_config);
//
inline void configure_io_callbacks(
    CommonCreationParameters& common,
    const IOConfig& config
) {
    if (!config.enable_custom_io) {
        return;  // IO callbacks disabled
    }
    
    // Validate required callbacks
    if (!config.open || !config.close || !config.size || 
        !config.tell || !config.seek || !config.read) {
        // Missing required callbacks - fall back to default
        return;
    }
    
    // Store the user's configuration in global state
    detail::IOState::get().config = config;
    
    // Create and chain FileIOCallbacks that forward to user callbacks
    static FileIOCallbacks io_callbacks = {
        /* .userData = */ nullptr,
        /* .open     = */ detail::io_open,
        /* .close    = */ detail::io_close,
        /* .size     = */ detail::io_size,
        /* .tell     = */ detail::io_tell,
        /* .seek     = */ detail::io_seek,
        /* .read     = */ detail::io_read,
        /* .write    = */ detail::io_write,
        /* .map      = */ detail::io_map,
        /* .unmap    = */ detail::io_unmap,
        /* .getLastError = */ detail::io_getLastError
    };
    
    common.chain(io_callbacks);
}

// Helper function to reset IO state
// Call this when you're done with custom IO
inline void reset_io_state() {
    auto& state = detail::IOState::get();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.config = IOConfig{};  // Clear configuration
}

// ========================================================================
// Helper Utilities for Common Use Cases
// ========================================================================

// Memory buffer handle for in-memory file access
struct MemoryBuffer {
    const uint8_t* data;
    size_t size;
    size_t position;
    std::string path;  // For debugging
    Result lastError;
    
    MemoryBuffer(const uint8_t* d, size_t s, const char* p = "")
        : data(d), size(s), position(0), path(p), lastError(kResultOk) {}
};

// Helper: Create IO config for loading models from memory buffers
// 
// Usage:
//   std::vector<uint8_t> model_data = load_my_model();
//   auto io_config = create_memory_buffer_io(model_data);
//
inline IOConfig create_memory_buffer_io(const std::vector<uint8_t>& buffer) {
    // Store buffer in a shared pointer so it outlives the callbacks
    auto shared_buffer = std::make_shared<std::vector<uint8_t>>(buffer);
    auto open_files = std::make_shared<std::map<void*, std::shared_ptr<MemoryBuffer>>>();
    auto mutex = std::make_shared<std::mutex>();
    
    IOConfig config;
    config.enable_custom_io = true;
    config.user_context = nullptr;
    
    config.open = [shared_buffer, open_files, mutex](void* ctx, const char* path, const char* mode) -> void* {
        std::lock_guard<std::mutex> lock(*mutex);
        auto mem_buffer = std::make_shared<MemoryBuffer>(
            shared_buffer->data(), 
            shared_buffer->size(), 
            path
        );
        void* handle = mem_buffer.get();
        (*open_files)[handle] = mem_buffer;
        g_customIOLastError = kResultOk;
        return handle;
    };
    
    config.close = [open_files, mutex](void* ctx, void* handle) {
        std::lock_guard<std::mutex> lock(*mutex);
        open_files->erase(handle);
    };
    
    config.size = [open_files, mutex](void* ctx, void* handle) -> size_t {
        std::lock_guard<std::mutex> lock(*mutex);
        auto it = open_files->find(handle);
        return (it != open_files->end()) ? (*it->second).size : 0;
    };
    
    config.tell = [open_files, mutex](void* ctx, void* handle) -> size_t {
        std::lock_guard<std::mutex> lock(*mutex);
        auto it = open_files->find(handle);
        return (it != open_files->end()) ? (*it->second).position : 0;
    };
    
    config.seek = [open_files, mutex](void* ctx, void* handle, size_t offset, int origin) -> int {
        std::lock_guard<std::mutex> lock(*mutex);
        auto it = open_files->find(handle);
        if (it == open_files->end()) {
            return -1;
        }
        
        auto& mem = *it->second;
        size_t new_pos = 0;
        
        switch (origin) {
            case SEEK_SET: new_pos = offset; break;
            case SEEK_CUR: new_pos = mem.position + offset; break;
            case SEEK_END: new_pos = mem.size + offset; break;
            default:
                mem.lastError = kResultInvalidParameter;
                return -1;
        }
        
        if (new_pos > mem.size) {
            mem.lastError = kResultOutOfRange;
            return -1;
        }
        mem.position = new_pos;
        mem.lastError = kResultOk;
        return 0;
    };
    
    config.read = [open_files, mutex](void* ctx, void* handle, void* buffer, size_t size) -> size_t {
        std::lock_guard<std::mutex> lock(*mutex);
        auto it = open_files->find(handle);
        if (it == open_files->end()) return 0;
        
        auto& mem = *it->second;
        size_t remaining = mem.size - mem.position;
        size_t to_read = std::min(size, remaining);
        
        if (to_read > 0) {
            std::memcpy(buffer, mem.data + mem.position, to_read);
            mem.position += to_read;
            mem.lastError = kResultOk;
        } else if (size > 0) {
            // Tried to read but we're at end of file
            mem.lastError = kResultEndOfFile;
        } else {
            mem.lastError = kResultOk;
        }
        
        return to_read;
    };
    
    config.map = [open_files, mutex](void* ctx, void* handle, size_t offset, size_t size, MapAccess access) -> uint8_t* {
        (void)access;  // Memory buffer supports all access modes
        if (!handle) {
            g_customIOLastError = kResultInvalidParameter;
            return nullptr;
        }
        
        std::lock_guard<std::mutex> lock(*mutex);
        auto it = open_files->find(handle);
        if (it == open_files->end()) {
            return nullptr;
        }
        
        auto& mem = *it->second;
        if (offset + size > mem.size) {
            mem.lastError = kResultOutOfRange;
            return nullptr;
        }
        
        mem.lastError = kResultOk;
        // Cast away const - caller must respect access mode
        return const_cast<uint8_t*>(mem.data + offset);
    };
    
    config.unmap = [open_files, mutex](void* ctx, void* handle, uint8_t* mappedPtr) {
        (void)mappedPtr;
        if (!handle) return;
        
        std::lock_guard<std::mutex> lock(*mutex);
        auto it = open_files->find(handle);
        if (it != open_files->end()) {
            (*it->second).lastError = kResultOk;
        }
        // In this simple example, unmap does nothing since the memory is owned by MemoryBuffer
    };
    
    config.getLastError = [open_files, mutex](void* ctx, void* handle) -> Result {
        if (!handle) {
            return g_customIOLastError;
        }
        std::lock_guard<std::mutex> lock(*mutex);
        auto it = open_files->find(handle);
        if (it != open_files->end()) {
            return (*it->second).lastError;
        }
        return kResultOk;
    };
    
    return config;
}

// Helper: Create IO config that wraps standard FILE* operations with custom logic
// 
// Usage:
//   auto io_config = create_file_wrapper_io(
//       [](const char* path) {
//           std::cout << "Opening: " << path << "\n";
//           return true;  // Allow open
//       }
//   );
//
inline IOConfig create_file_wrapper_io(
    std::function<bool(const char* path)> on_open = nullptr
) {
    // Track memory mappings for this config instance
    auto mapped_regions = std::make_shared<std::map<uint8_t*, std::vector<uint8_t>>>();
    auto map_mutex = std::make_shared<std::mutex>();
    
    IOConfig config;
    config.enable_custom_io = true;
    config.user_context = nullptr;
    
    config.open = [on_open](void* ctx, const char* path, const char* mode) -> void* {
        if (on_open && !on_open(path)) {
            g_customIOLastError = nvigi::kResultCanceled;
            return nullptr;  // Callback rejected the open
        }
        FILE* file = std::fopen(path, mode);
        if (file) {
            g_customIOLastError = kResultOk;
        } else {
            g_customIOLastError = kResultItemNotFound;
        }
        return file;
    };
    
    config.close = [mapped_regions, map_mutex](void* ctx, void* handle) {
        if (handle) {
            // Clean up any remaining mapped regions for this handle
            {
                std::lock_guard<std::mutex> lock(*map_mutex);
                // Note: Ideally track per-handle, but for simplicity we just clear on close
            }
            std::fclose(static_cast<FILE*>(handle));
        }
    };
    
    config.size = [](void* ctx, void* handle) -> size_t {
        if (!handle) return 0;
        FILE* f = static_cast<FILE*>(handle);
        
        // Use platform-specific 64-bit file position functions for large file support
        __int64 pos = _ftelli64(f);
        if (pos < 0) {
            return 0;  // ftell failed
        }
        if (_fseeki64(f, 0, SEEK_END) != 0) {
            return 0;  // fseek to end failed
        }
        __int64 size = _ftelli64(f);
        _fseeki64(f, pos, SEEK_SET);  // Restore position
        if (size < 0) {
            return 0;  // ftell failed
        }
        return static_cast<size_t>(size);
    };
    
    config.tell = [](void* ctx, void* handle) -> size_t {
        if (!handle) return 0;
        FILE* f = static_cast<FILE*>(handle);        
        // Use platform-specific 64-bit file position functions for large file support
        __int64 pos = _ftelli64(f);
        if (pos < 0) {
            return 0;  // ftelli64 failed
        }
        return static_cast<size_t>(pos);
    };
    
    config.seek = [](void* ctx, void* handle, size_t offset, int origin) -> int {
        if (!handle) {
            return -1;
        }        
        // Use platform-specific 64-bit seek functions for large file support
        return _fseeki64(static_cast<FILE*>(handle), static_cast<__int64>(offset), origin);
    };
    
    config.read = [](void* ctx, void* handle, void* buffer, size_t size) -> size_t {
        if (!handle) return 0;
        return std::fread(buffer, 1, size, static_cast<FILE*>(handle));
    };
    
    config.map = [mapped_regions, map_mutex](void* ctx, void* handle, size_t offset, size_t size, MapAccess access) -> uint8_t* {
        if (!handle) {
            g_customIOLastError = kResultInvalidParameter;
            return nullptr;
        }
        
        FILE* f = static_cast<FILE*>(handle);
        
        // Save current position
        __int64 current_pos = _ftelli64(f);
        if (current_pos < 0) {
            g_customIOLastError = kResultIOError;
            return nullptr;
        }
        
        // Seek to offset
        if (_fseeki64(f, static_cast<__int64>(offset), SEEK_SET) != 0) {
            g_customIOLastError = kResultOutOfRange;
            return nullptr;
        }
       
        // Allocate buffer and read the data
        std::vector<uint8_t> buffer;
        try {
            buffer.resize(size);
        } catch (...) {
            _fseeki64(f, current_pos, SEEK_SET);  // Restore position
            g_customIOLastError = kResultInsufficientResources;
            return nullptr;
        }
        
        size_t bytes_read = std::fread(buffer.data(), 1, size, f);
        if (bytes_read != size) {
            _fseeki64(f, current_pos, SEEK_SET);  // Restore position
            g_customIOLastError = kResultIOError;
            return nullptr;
        }
        
        // Restore file position
        _fseeki64(f, current_pos, SEEK_SET);
        
        // Store the buffer and return pointer
        uint8_t* ptr = buffer.data();
        {
            std::lock_guard<std::mutex> lock(*map_mutex);
            (*mapped_regions)[ptr] = std::move(buffer);
        }
        
        g_customIOLastError = kResultOk;
        return ptr;
    };
    
    config.unmap = [mapped_regions, map_mutex](void* ctx, void* handle, uint8_t* mappedPtr) {
        if (!mappedPtr) return;
        
        std::lock_guard<std::mutex> lock(*map_mutex);
        auto it = mapped_regions->find(mappedPtr);
        if (it != mapped_regions->end()) {
            mapped_regions->erase(it);
        }
    };
    
    config.getLastError = [](void* ctx, void* handle) -> Result {
        if (!handle) {
            return g_customIOLastError;
        }
        // For FILE* operations, we could map errno to Result codes
        // For simplicity, just return Ok (FILE* ops handle their own errors)
        return kResultOk;
    };
    
    return config;
}

} // namespace io
} // namespace nvigi


