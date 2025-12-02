// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <string>
#include <string_view> 
#include <vector>
#include <memory>
#include <functional>
#include <optional>
#include <expected>
#include <source_location>
#include <format>
#include <mutex>
#include <cstdint>
#include <fstream>

#ifdef NVIGI_WINDOWS
#include "nvigi_core/source/utils/nvigi.dsound/player.h"
#endif

namespace nvigi::tts {

// Forward declarations
class Instance;

// Inference execution states
// 
// Used in callbacks to provide inference status and allow cancellation.
// The callback receives the current state and audio data, and can return:
// - The same state to continue normally
// - ExecutionState::Cancel to stop inference immediately
//
// Example:
//   [](const int16_t* audio, size_t samples, ExecutionState state) -> ExecutionState {
//       if (state == ExecutionState::DataPending) {
//           // Play or save audio chunk
//       }
//       return should_stop ? ExecutionState::Cancel : state;
//   }
enum class ExecutionState : uint32_t {
    Invalid = 0,        // Invalid/error state
    DataPending = 1,    // Data is pending/available (new audio chunk)
    DataPartial = 2,    // Partial data (not used for TTS currently)
    Done = 3,           // Inference completed successfully
    Cancel = 4          // Request to cancel/stop inference (return this to stop)
};

// Modern error handling
class Error {
public:
    Error(std::string msg) : message_(std::move(msg)) {}
    const std::string& what() const { return message_; }
private:
    std::string message_;
};

using Result = std::expected<void, Error>;

// Audio format constants
constexpr int kSampleRate = 22050;      // 22.05 kHz
constexpr int kBitsPerSample = 16;       // 16-bit PCM
constexpr int kNumChannels = 1;          // Mono

// WAV file header structure
struct WAVHeader {
    // RIFF Chunk Descriptor
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunkSize = 0;
    char wave[4] = {'W', 'A', 'V', 'E'};
    
    // fmt sub-chunk
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t subchunk1Size = 16;
    uint16_t audioFormat = 1;           // PCM
    uint16_t numChannels = kNumChannels;
    uint32_t sampleRate = kSampleRate;
    uint32_t byteRate = 0;
    uint16_t blockAlign = 0;
    uint16_t bitsPerSample = kBitsPerSample;
    
    // data sub-chunk
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t dataSize = 0;
};

// Helper class to write WAV files
class WAVWriter {
public:
    explicit WAVWriter(const std::string& filename) {
        if (!filename.empty()) {
            file_.open(filename, std::ios::binary);
            if (file_.is_open()) {
                // Write placeholder header (will be updated on close)
                WAVHeader header;
                file_.write(reinterpret_cast<const char*>(&header), sizeof(header));
            }
        }
    }
    
    ~WAVWriter() {
        close();
    }
    
    // Non-copyable
    WAVWriter(const WAVWriter&) = delete;
    WAVWriter& operator=(const WAVWriter&) = delete;
    
    // Movable
    WAVWriter(WAVWriter&&) noexcept = default;
    WAVWriter& operator=(WAVWriter&&) noexcept = default;
    
    void write_samples(const int16_t* samples, size_t count) {
        if (file_.is_open()) {
            file_.write(reinterpret_cast<const char*>(samples), count * sizeof(int16_t));
            total_samples_ += count;
        }
    }
    
    void close() {
        if (file_.is_open() && total_samples_ > 0) {
            // Update header with correct sizes
            WAVHeader header;
            header.numChannels = kNumChannels;
            header.sampleRate = kSampleRate;
            header.bitsPerSample = kBitsPerSample;
            header.byteRate = header.sampleRate * header.numChannels * (header.bitsPerSample / 8);
            header.blockAlign = header.numChannels * (header.bitsPerSample / 8);
            header.dataSize = static_cast<uint32_t>(total_samples_ * sizeof(int16_t));
            header.chunkSize = 36 + header.dataSize;
            
            file_.seekp(0);
            file_.write(reinterpret_cast<const char*>(&header), sizeof(header));
            file_.close();
        }
    }
    
    bool is_open() const { return file_.is_open(); }
    
private:
    std::ofstream file_;
    size_t total_samples_ = 0;
};

#ifdef NVIGI_WINDOWS
// Helper class for real-time audio playback using DirectSound
class AudioPlayer {
public:
    AudioPlayer() = default;
    
    // Play audio samples (blocking until playback completes)
    // This is thread-safe and can be called from callbacks
    static void play_audio(const int16_t* samples, size_t count) {
        std::lock_guard<std::mutex> lock(playback_mutex_);
        
        try {
            nvigi::utils::Player player(kBitsPerSample, kSampleRate);
            if (!player.initialized) {
                return;
            }
            
            DWORD buffer_size = static_cast<DWORD>(count * sizeof(int16_t));
            nvigi::utils::Buffer buffer(player, samples, buffer_size);
            
            if (buffer.initialized) {
                buffer.Play();
                buffer.Wait();
            }
        } catch (...) {
            // Silently ignore playback errors to not interrupt TTS generation
        }
    }
    
private:
    static inline std::mutex playback_mutex_;
};
#endif

struct RuntimeConfig {
    float speed{1.0f};                          // Speech speed (0.5 - 2.0)
    int min_chunk_size{100};                    // Min chunk size in characters
    int max_chunk_size{200};                    // Max chunk size in characters
    int seed{-1};                               // Random seed (-1 = auto)
    int n_timesteps{16};                        // TTS timesteps (16-32, GGML only)
    int sampler{1};                             // 0=EULER, 1=DPM++ (GGML only)
    int dpmpp_order{2};                         // DPM++ order 1-3 (GGML only)
    bool use_flash_attention{true};             // Enable flash attention (GGML only)
    std::string_view language{"en"};            // Language code (en, en-us, en-uk, es, de)
    
    // Builder pattern methods
    RuntimeConfig& set_speed(float s) {
        speed = s;
        return *this;
    }
    
    RuntimeConfig& set_chunk_size(int min, int max) {
        min_chunk_size = min;
        max_chunk_size = max;
        return *this;
    }
    
    RuntimeConfig& set_seed(int s) {
        seed = s;
        return *this;
    }
    
    RuntimeConfig& set_timesteps(int steps) {
        n_timesteps = steps;
        return *this;
    }
    
    RuntimeConfig& set_sampler(int s) {
        sampler = s;
        return *this;
    }
    
    RuntimeConfig& set_dpmpp_order(int order) {
        dpmpp_order = order;
        return *this;
    }
    
    RuntimeConfig& set_flash_attention(bool enable) {
        use_flash_attention = enable;
        return *this;
    }
    
    RuntimeConfig& set_language(std::string_view lang) {
        language = lang;
        return *this;
    }
};

struct ModelConfig {
    std::string_view backend;                   // Backend: cuda, d3d12, vulkan
    std::string_view guid;                      // Model GUID
    std::string_view model_path;                // Path to models directory
    int32_t num_threads{1};                     // Number of threads
    size_t vram_budget_mb{0};                   // VRAM budget in MB (0 = auto)
    bool warm_up_models{true};                  // Warm up models on creation
    
    ModelConfig& set_threads(int32_t threads) {
        num_threads = threads;
        return *this;
    }
    
    ModelConfig& set_vram(size_t mb) {
        vram_budget_mb = mb;
        return *this;
    }
    
    ModelConfig& set_warm_up(bool enable) {
        warm_up_models = enable;
        return *this;
    }
};

// Main TTS class implementation
class Instance {
public:
    struct Impl {
        InferenceInstance* instance{nullptr};
        ModelConfig config;
        nvigi::PluginID plugin_id;
        PFun_nvigiUnloadInterface* unloader;
        std::mutex mutex;
        std::vector<std::string> supported_languages;
    };

    // Singleton-like interface management
    struct InterfaceManager {
        static inline ITextToSpeech* itts{nullptr};
        static inline IPolledInferenceInterface* ipolled{nullptr};
        static inline std::mutex mutex;
        static inline int reference_count{0};
        static inline std::string current_backend;
    };

    static std::expected<std::unique_ptr<Instance>, Error> create(
        const ModelConfig& config,
        const nvigi::d3d12::D3D12Config& d3d12_config = {},
        const nvigi::vulkan::VulkanConfig& vulkan_config = {},
        PFun_nvigiLoadInterface* loader = nullptr,
        PFun_nvigiUnloadInterface* unloader = nullptr,
        const std::string_view& plugin_path = {},
        std::source_location loc = std::source_location::current()
    ) {
        auto instance = std::unique_ptr<Instance>(new Instance());
        instance->impl_ = std::make_unique<Impl>();
        
        // Check for valid backend
        if (config.backend != "d3d12" && config.backend != "cuda" && config.backend != "vulkan") {
            return std::unexpected(Error(
                std::format("Unsupported backend '{}', must be 'd3d12', 'cuda', or 'vulkan' at {}:{}",
                    config.backend, loc.file_name(), loc.line())
            ));
        }

        instance->impl_->unloader = unloader;
        instance->impl_->plugin_id = config.backend == "d3d12" ? plugin::tts::asqflow_ggml::d3d12::kId :
            config.backend == "vulkan" ? plugin::tts::asqflow_ggml::vulkan::kId :
            plugin::tts::asqflow_ggml::cuda::kId;

        // Initialize singleton interfaces if not already done or if backend changed
        {
            std::lock_guard<std::mutex> lock(InterfaceManager::mutex);
            
            if (InterfaceManager::reference_count == 0 || 
                InterfaceManager::current_backend != config.backend) {
                
                // Release old interfaces if backend changed
                if (InterfaceManager::reference_count > 0 && 
                    InterfaceManager::current_backend != config.backend) {
                    return std::unexpected(Error(
                        std::format("Cannot switch backend from '{}' to '{}' while instances exist",
                            InterfaceManager::current_backend, config.backend)
                    ));
                }
                
                // Initialize TTS interface
                auto result = nvigiGetInterfaceDynamic(
                    instance->impl_->plugin_id,
                    &InterfaceManager::itts,
                    loader, plugin_path.empty() ? nullptr : plugin_path.data()
                );
                
                if (!InterfaceManager::itts) {
                    return std::unexpected(Error(
                        std::format("Failed to get TTS interface for backend '{}' at {}:{}", 
                            config.backend, loc.file_name(), loc.line())
                    ));
                }
                
                // Get polled interface
                result = nvigiGetInterfaceDynamic(
                    instance->impl_->plugin_id,
                    &InterfaceManager::ipolled,
                    loader
                );

                if (!InterfaceManager::ipolled) {
                    return std::unexpected(Error(
                        std::format("Failed to get polled interface for backend '{}' at {}:{}",
                            config.backend, loc.file_name(), loc.line())
                    ));
                }
                
                InterfaceManager::current_backend = std::string(config.backend);
            }
            
            InterfaceManager::reference_count++;
        }

        // Chain parameters
        CommonCreationParameters common{};
        TTSCreationParameters tts_params{};
        D3D12Parameters d3d12params{};
        VulkanParameters vkparams{};
        
        // Chain parameters (unused are simply ignored)
        tts_params.chain(common);
        common.chain(d3d12params);
        d3d12params.chain(vkparams);

        // Set base parameters
        common.utf8PathToModels = config.model_path.data();
        common.numThreads = config.num_threads;
        common.vramBudgetMB = config.vram_budget_mb;
        common.modelGUID = config.guid.data();

        tts_params.warmUpModels = config.warm_up_models;
        
        // Set D3D12 specific parameters
        d3d12params.device = d3d12_config.device;
        d3d12params.queueCompute = d3d12_config.command_queue;
        
        // Set D3D12 memory allocation callbacks
        d3d12params.createCommittedResourceCallback = d3d12_config.create_committed_resource_callback;
        d3d12params.destroyResourceCallback = d3d12_config.destroy_resource_callback;
        d3d12params.createCommitResourceUserContext = d3d12_config.create_resource_user_context;
        d3d12params.destroyResourceUserContext = d3d12_config.destroy_resource_user_context;
        
        // Set Vulkan specific parameters (ignored if not using Vulkan)
        vkparams.instance = vulkan_config.instance;
        vkparams.physicalDevice = vulkan_config.physical_device;
        vkparams.device = vulkan_config.device;
        vkparams.queueCompute = vulkan_config.compute_queue;
        vkparams.queueTransfer = vulkan_config.transfer_queue;
        
        // Set Vulkan memory allocation callbacks
        vkparams.allocateMemoryCallback = vulkan_config.allocate_memory_callback;
        vkparams.freeMemoryCallback = vulkan_config.free_memory_callback;

        // Get capabilities to retrieve supported languages
        nvigi::TTSCapabilitiesAndRequirements* caps{};
        if (NVIGI_FAILED(result, getCapsAndRequirements(InterfaceManager::itts, tts_params, &caps))) {
            std::lock_guard<std::mutex> lock(InterfaceManager::mutex);
            InterfaceManager::reference_count--;
            if (InterfaceManager::reference_count == 0) {
                instance->impl_->unloader(instance->impl_->plugin_id, InterfaceManager::itts);
                instance->impl_->unloader(instance->impl_->plugin_id, InterfaceManager::ipolled);
                InterfaceManager::itts = nullptr;
                InterfaceManager::ipolled = nullptr;
                InterfaceManager::current_backend.clear();
            }
            return std::unexpected(Error("Failed to get TTS capabilities"));
        }

        // Extract supported languages
        if (caps && caps->supportedLanguages && caps->n_languages > 0) {
            for (uint32_t i = 0; i < caps->n_languages; ++i) {
                instance->impl_->supported_languages.push_back(caps->supportedLanguages[i]);
            }
        }

        auto creation_result = InterfaceManager::itts->createInstance(
            tts_params,
            &instance->impl_->instance
        );
        
        if (creation_result != kResultOk || !instance->impl_->instance) {
            // Decrement reference count on failure
            std::lock_guard<std::mutex> lock(InterfaceManager::mutex);
            InterfaceManager::reference_count--;
            if (InterfaceManager::reference_count == 0) {
                instance->impl_->unloader(instance->impl_->plugin_id, InterfaceManager::itts);
                instance->impl_->unloader(instance->impl_->plugin_id, InterfaceManager::ipolled);
                InterfaceManager::itts = nullptr;
                InterfaceManager::ipolled = nullptr;
                InterfaceManager::current_backend.clear();
            }
            return std::unexpected(Error("Failed to create TTS inference instance"));
        }
        
        return instance;
    }

    ~Instance() {
        if (impl_) {
            if (impl_->instance) {
                InterfaceManager::itts->destroyInstance(impl_->instance);
            }
            
            // Decrement reference count and release interfaces if needed
            std::lock_guard<std::mutex> lock(InterfaceManager::mutex);
            InterfaceManager::reference_count--;
            if (InterfaceManager::reference_count == 0) {
                // Release interfaces when all instances are destroyed
                impl_->unloader(impl_->plugin_id, InterfaceManager::itts);
                impl_->unloader(impl_->plugin_id, InterfaceManager::ipolled);
                InterfaceManager::itts = nullptr;
                InterfaceManager::ipolled = nullptr;
                InterfaceManager::current_backend.clear();
            }
        }
    }

    // Context for async operations
    struct AsyncContext {
        InferenceExecutionContext ctx;
        InferenceDataSlotArray inputs;
        std::vector<InferenceDataSlot> slots;
        TTSASqFlowRuntimeParameters runtime_params;
        std::vector<CpuData> cpu_data;  // Store CpuData objects to keep them alive
        std::vector<InferenceDataText> slot_data;
        std::string text_copy;
        std::string end_stream_copy;
        std::string target_spec_copy;
        std::string language_copy;
    };

    // Generate speech from text (blocking)
    Result generate(
        std::string_view text,
        std::string_view target_spectrogram_path,
        const RuntimeConfig& config = {},
        std::function<ExecutionState(const int16_t*, size_t, ExecutionState)> callback = nullptr
    ) {
        std::lock_guard<std::mutex> lock(impl_->mutex);

        // Callback implementation
        auto cb = [](const InferenceExecutionContext* ctx, 
                    InferenceExecutionState state, void* data) -> InferenceExecutionState {
            auto* callback = static_cast<std::function<ExecutionState(const int16_t*, size_t, ExecutionState)>*>(data);
            if (callback && ctx && ctx->outputs) {
                const InferenceDataByteArray* audio_data{};
                if (ctx->outputs->findAndValidateSlot(kTTSDataSlotOutputAudio, &audio_data)) {
                    CpuData* cpu_buffer = castTo<CpuData>(audio_data->bytes);
                    const int16_t* samples = reinterpret_cast<const int16_t*>(cpu_buffer->buffer);
                    size_t num_samples = cpu_buffer->sizeInBytes / sizeof(int16_t);
                    
                    // Convert internal state to wrapper enum
                    ExecutionState wrapper_state = ExecutionState::Invalid;
                    if (state == kInferenceExecutionStateDataPending) {
                        wrapper_state = ExecutionState::DataPending;
                    } else if (state == kInferenceExecutionStateDataPartial) {
                        wrapper_state = ExecutionState::DataPartial;
                    } else if (state == kInferenceExecutionStateDone) {
                        wrapper_state = ExecutionState::Done;
                    }
                    
                    // Call user callback and convert result back
                    ExecutionState result = (*callback)(samples, num_samples, wrapper_state);
                    if (result == ExecutionState::Cancel) {
                        return kInferenceExecutionStateCancel;
                    } else if (result == ExecutionState::Invalid) {
                        return kInferenceExecutionStateInvalid;
                    }
                }
            }
            return state;
        };

        InferenceExecutionContext ctx{};
        TTSASqFlowRuntimeParameters runtime{};
        runtime.speed = config.speed;
        runtime.minChunkSize = config.min_chunk_size;
        runtime.maxChunkSize = config.max_chunk_size;
        runtime.seed = config.seed;
        runtime.n_timesteps = config.n_timesteps;
        runtime.sampler = config.sampler;
        runtime.dpmpp_order = config.dpmpp_order;
        runtime.use_flash_attention = config.use_flash_attention;
        runtime.language = config.language.data();

        ctx.runtimeParameters = runtime;
        ctx.instance = impl_->instance;

        // Setup input data
        InferenceDataTextSTLHelper text_data(text.data());
        InferenceDataTextSTLHelper target_spec_data(target_spectrogram_path.data());
        
        std::vector<InferenceDataSlot> slots = {
            {kTTSDataSlotInputText, text_data},
            {kTTSDataSlotInputTargetSpectrogramPath, target_spec_data}
        };
        InferenceDataSlotArray inputs{slots.size(), slots.data()};
        ctx.inputs = &inputs;

        if (callback) {
            ctx.callback = cb;
            ctx.callbackUserData = &callback;
        }

        auto result = impl_->instance->evaluate(&ctx);
        if (result != kResultOk) {
            return std::unexpected(Error("Speech generation failed"));
        }

        return {};
    }

    // Polling-based async handle for non-blocking operations
    class AsyncOperation {
    public:
        enum class State {
            Pending,      // Not started yet
            Running,      // Currently executing
            HasResults,   // Has new audio/results available
            Completed,    // Successfully finished
            Failed        // Error occurred
        };

        // Check current state (non-blocking)
        State get_state() const {
            if (!context_) return State::Failed;
            
            InferenceExecutionState exec_state = kInferenceExecutionStateInvalid;
            auto result = ipolled_->getResults(&context_->ctx, false, &exec_state);
            
            if (result == kResultNotReady) {
                return first_poll_ ? State::Pending : State::Running;
            }
            
            if (result == kResultOk) {
                if (exec_state == kInferenceExecutionStateDone) {
                    return State::Completed;
                }
                else if (exec_state == kInferenceExecutionStateInvalid) {
                    return State::Failed;
                }
                return State::HasResults;
            }
            
            return State::Failed;
        }

        // Result returned from try_get_results
        struct Result {
            std::vector<int16_t> audio;     // Audio samples (PCM 16-bit)
            ExecutionState state;           // Current inference state
        };

        // Try to get available results without blocking
        std::optional<Result> try_get_results() {
            if (!context_) return std::nullopt;
            
            InferenceExecutionState exec_state = kInferenceExecutionStateInvalid;
            auto result = ipolled_->getResults(&context_->ctx, false, &exec_state);
            
            first_poll_ = false;
            
            if (result == kResultNotReady) {
                return std::nullopt; // No results yet
            }
            
            if (result == kResultOk && context_->ctx.outputs) {
                std::vector<int16_t> audio;
                const InferenceDataByteArray* audio_data{};
                if (context_->ctx.outputs->findAndValidateSlot(kTTSDataSlotOutputAudio, &audio_data)) {
                    CpuData* cpu_buffer = castTo<CpuData>(audio_data->bytes);
                    const int16_t* samples = reinterpret_cast<const int16_t*>(cpu_buffer->buffer);
                    size_t num_samples = cpu_buffer->sizeInBytes / sizeof(int16_t);
                    
                    audio.assign(samples, samples + num_samples);
                    
                    // Accumulate audio
                    if (audio_buffer_) {
                        audio_buffer_->insert(audio_buffer_->end(), audio.begin(), audio.end());
                    }
                }
                
                // Convert internal state to wrapper enum
                ExecutionState wrapper_state = ExecutionState::Invalid;
                if (exec_state == kInferenceExecutionStateDataPending) {
                    wrapper_state = ExecutionState::DataPending;
                } else if (exec_state == kInferenceExecutionStateDataPartial) {
                    wrapper_state = ExecutionState::DataPartial;
                } else if (exec_state == kInferenceExecutionStateDone) {
                    wrapper_state = ExecutionState::Done;
                    is_complete_ = true;
                } else if (exec_state == kInferenceExecutionStateInvalid) {
                    wrapper_state = ExecutionState::Invalid;
                    is_failed_ = true;
                }
                
                // Handle cancellation
                InferenceExecutionState state_to_release = exec_state;
                if (cancel_requested_) {
                    state_to_release = kInferenceExecutionStateCancel;
                    is_complete_ = true;
                    wrapper_state = ExecutionState::Cancel;
                    cancel_requested_ = false;
                }
                
                ipolled_->releaseResults(&context_->ctx, state_to_release);
                return Result{ std::move(audio), wrapper_state };
            }
            
            return std::nullopt;
        }

        // Request cancellation of the async operation
        void cancel() {
            cancel_requested_ = true;
        }

        // Check if operation is complete
        bool is_complete() const { 
            return is_complete_ || is_failed_; 
        }

        // Check if operation failed
        bool is_failed() const { 
            return is_failed_; 
        }

        // Get full accumulated audio so far
        const std::vector<int16_t>& get_accumulated_audio() const {
            static const std::vector<int16_t> empty;
            return audio_buffer_ ? *audio_buffer_ : empty;
        }

        // Move accumulated audio out
        std::vector<int16_t> take_audio() {
            if (audio_buffer_) {
                return std::move(*audio_buffer_);
            }
            return {};
        }

        // Reset/clear the operation
        void reset() {
            context_.reset();
            audio_buffer_.reset();
            is_complete_ = true;
            is_failed_ = false;
            cancel_requested_ = false;
        }

    private:
        friend class Instance;
        
        AsyncOperation(
            IPolledInferenceInterface* ipolled,
            std::shared_ptr<AsyncContext> context,
            std::shared_ptr<std::vector<int16_t>> audio_buffer
        ) : ipolled_(ipolled)
          , context_(std::move(context))
          , audio_buffer_(std::move(audio_buffer))
          , is_complete_(false)
          , is_failed_(false)
          , first_poll_(true)
          , cancel_requested_(false)
        {}

        IPolledInferenceInterface* ipolled_;
        std::shared_ptr<AsyncContext> context_;
        std::shared_ptr<std::vector<int16_t>> audio_buffer_;
        bool is_complete_;
        bool is_failed_;
        bool first_poll_;
        bool cancel_requested_;
    };

    // Generate speech asynchronously (non-blocking)
    std::expected<AsyncOperation, Error> generate_async(
        std::string_view text,
        std::string_view target_spectrogram_path,
        const RuntimeConfig& config = {}
    ) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        
        // Create async context
        auto context = std::make_shared<AsyncContext>();
        auto audio_buffer = std::make_shared<std::vector<int16_t>>();
        
        // Store copies of input data
        context->text_copy = std::string(text);
        context->target_spec_copy = std::string(target_spectrogram_path);
        context->language_copy = std::string(config.language);
        
        // Setup runtime parameters
        context->runtime_params.speed = config.speed;
        context->runtime_params.minChunkSize = config.min_chunk_size;
        context->runtime_params.maxChunkSize = config.max_chunk_size;
        context->runtime_params.seed = config.seed;
        context->runtime_params.n_timesteps = config.n_timesteps;
        context->runtime_params.sampler = config.sampler;
        context->runtime_params.dpmpp_order = config.dpmpp_order;
        context->runtime_params.use_flash_attention = config.use_flash_attention;
        context->runtime_params.language = context->language_copy.c_str();
        
        context->ctx.runtimeParameters = context->runtime_params;
        context->ctx.instance = impl_->instance;
        
        context->end_stream_copy = std::string(END_PROMPT_ASYNC);

        // Setup input data
        context->cpu_data.reserve(3);
        context->slot_data.reserve(3);
        context->cpu_data.emplace_back(context->text_copy.size() + 1, context->text_copy.data());
        context->slot_data.emplace_back(context->cpu_data.back());
        context->cpu_data.emplace_back(context->target_spec_copy.size() + 1, context->target_spec_copy.data());  
        context->slot_data.emplace_back(context->cpu_data.back());
        context->cpu_data.emplace_back(context->end_stream_copy.size() + 1, context->end_stream_copy.data());
        context->slot_data.emplace_back(context->cpu_data.back());

        context->slots = {
            {kTTSDataSlotInputText, context->slot_data[0]},
            {kTTSDataSlotInputTargetSpectrogramPath, context->slot_data[1]}
        };
        context->inputs = { context->slots.size(), context->slots.data() };
        context->ctx.inputs = &context->inputs;
        
        // Start async evaluation with the text
        auto result = impl_->instance->evaluateAsync(&context->ctx);
        if (result != kResultOk) {
            return std::unexpected(Error("Failed to start async speech generation"));
        }
        
        // Send END_PROMPT_ASYNC to signal end of stream
        context->slots[0] = {kTTSDataSlotInputText, context->slot_data[2] };
        
        result = impl_->instance->evaluateAsync(&context->ctx);
        if (result != kResultOk) {
            return std::unexpected(Error("Failed to send END_PROMPT_ASYNC marker"));
        }
        
        return AsyncOperation(InterfaceManager::ipolled, context, audio_buffer);
    }

    // Get list of supported languages
    const std::vector<std::string>& get_supported_languages() const {
        return impl_->supported_languages;
    }

private:
    Instance() = default;
    std::unique_ptr<Impl> impl_;
};

} // namespace nvigi::tts

