// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <string>
#include <string_view> 
#include <vector>
#include <memory>
#include <functional>
#include <future>
#include <optional>
#include <span>
#include <expected>
#include <source_location>
#include <format>
#include <mutex>
#include <thread>
#include <atomic>

// NVIGI core headers
#include <nvigi.h>
#include <nvigi_struct.h>
#include <nvigi_types.h>
#include <nvigi_asr_whisper.h>
#include <nvigi_d3d12.h>
#include <nvigi_vulkan.h>

// Import shared helpers
#include "../d3d12.hpp"
#include "../vulkan.hpp"
#include "../io_helpers.hpp"

namespace nvigi::asr {

// Forward declarations
class Instance;
class Stream;

// Inference execution states
// 
// Used in callbacks to provide inference status and allow cancellation.
// The callback receives the current state and response text, and can return:
// - The same state to continue normally
// - ExecutionState::Cancel to stop inference immediately
//
// Example:
//   [](std::string_view text, ExecutionState state) -> ExecutionState {
//       if (state == ExecutionState::DataPending) {
//           std::cout << text;  // Display transcribed text
//       }
//       return should_stop ? ExecutionState::Cancel : state;
//   }
enum class ExecutionState : uint32_t {
    Invalid = 0,        // Invalid/error state
    DataPending = 1,    // Data is pending/available (new transcription)
    DataPartial = 2,    // Partial data (e.g., intermediate results)
    Done = 3,           // Inference completed successfully
    Cancel = 4          // Request to cancel/stop inference (return this to stop)
};

// ASR Sampling strategies
enum class SamplingStrategy : uint32_t {
    Greedy = 0,         // Greedy decoding (faster, deterministic)
    BeamSearch = 1      // Beam search (higher quality, slower)
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

struct RuntimeConfig {
    // Sampling parameters
    SamplingStrategy sampling{SamplingStrategy::Greedy};
    int32_t best_of{1};         // For greedy sampling, number of candidates
    int32_t beam_size{-1};      // For beam search, number of beams (-1 = disabled)
    
    // Transcription parameters
    const char* prompt{nullptr};              // Optional prompt to guide transcription
    bool debug{ false };                     // Enable debug logging
    bool no_context{true};                    // Don't use previous context
    bool suppress_blank{true};                // Suppress blank token
    bool suppress_non_speech_tokens{false};   // Suppress non-speech tokens
    float temperature{0.0f};                  // Sampling temperature (0.0 = greedy)
    float entropy_threshold{2.4f};            // Entropy-based suppression threshold
    float logprob_threshold{-1.0f};           // Log-probability suppression threshold
    float no_speech_threshold{0.6f};          // No-speech detection threshold
    
    // Builder pattern methods
    RuntimeConfig& set_sampling(SamplingStrategy s) {
        sampling = s;
        return *this;
    }
    
    RuntimeConfig& set_best_of(int32_t n) {
        best_of = n;
        return *this;
    }
    
    RuntimeConfig& set_beam_size(int32_t size) {
        beam_size = size;
        return *this;
    }
    
    RuntimeConfig& set_prompt(const char* p) {
        prompt = p;
        return *this;
    }
    
    RuntimeConfig& set_no_context(bool enable) {
        no_context = enable;
        return *this;
    }
    
    RuntimeConfig& set_suppress_blank(bool enable) {
        suppress_blank = enable;
        return *this;
    }
    
    RuntimeConfig& set_suppress_non_speech_tokens(bool enable) {
        suppress_non_speech_tokens = enable;
        return *this;
    }
    
    RuntimeConfig& set_temperature(float temp) {
        temperature = temp;
        return *this;
    }
    
    RuntimeConfig& set_entropy_threshold(float threshold) {
        entropy_threshold = threshold;
        return *this;
    }
    
    RuntimeConfig& set_logprob_threshold(float threshold) {
        logprob_threshold = threshold;
        return *this;
    }
    
    RuntimeConfig& set_no_speech_threshold(float threshold) {
        no_speech_threshold = threshold;
        return *this;
    }

    RuntimeConfig& set_debug(bool enable) {
        debug = enable;
        return *this;
    }
};

struct ModelConfig {
    std::string_view backend;
    std::string_view guid;
    std::string_view model_path;
    int32_t num_threads{1};
    size_t vram_budget_mb{0};
    bool flash_attention{true};
    std::string_view language{"en"};      // Language code (e.g., "en", "es", "auto")
    bool translate{false};                 // Translate to English
    bool detect_language{false};           // Auto-detect language
    int32_t lengthMs = 10000; // length of audio segments, in milliseconds
    int32_t keepMs = 200; // amount of previous audio to keep as context for streaming mode, in milliseconds
    int32_t stepMs = 3000; // step size for streaming mode, in milliseconds

    ModelConfig& set_threads(int32_t threads) {
        num_threads = threads;
        return *this;
    }
    ModelConfig& set_vram(size_t mb) {
        vram_budget_mb = mb;
        return *this;
    }
    ModelConfig& set_flash_attention(bool enable) {
        flash_attention = enable;
        return *this;
    }
    ModelConfig& set_language(std::string_view lang) {
        language = lang;
        return *this;
    }
    ModelConfig& set_translate(bool enable) {
        translate = enable;
        return *this;
    }
    ModelConfig& set_detect_language(bool enable) {
        detect_language = enable;
        return *this;
    }
    ModelConfig& set_length_ms(int32_t length) {
        lengthMs = length;
        return *this;
    }
    ModelConfig& set_keep_ms(int32_t keep) {
        keepMs = keep;
        return *this;
    }
    ModelConfig& set_step_ms(int32_t step) {
        stepMs = step;
        return *this;
    }
};

// Helper function to convert backend string to PluginID
// Returns null PluginID if backend string is unknown
inline nvigi::PluginID backend_to_plugin_id(std::string_view backend) {
    if (backend == "cuda")   return nvigi::plugin::asr::ggml::cuda::kId;
    if (backend == "d3d12")  return nvigi::plugin::asr::ggml::d3d12::kId;
    if (backend == "vulkan") return nvigi::plugin::asr::ggml::vulkan::kId;
    if (backend == "cpu")    return nvigi::plugin::asr::ggml::cpu::kId;
    return nvigi::PluginID{};  // Invalid/null
}

// Check if a backend string is valid
inline bool is_valid_backend(std::string_view backend) {
    return backend == "cuda" || backend == "d3d12" || backend == "vulkan" || backend == "cpu";
}

// Main ASR class implementation
class Instance {
public:
    struct Impl {
        InferenceInstance* instance{nullptr};
        ModelConfig config;
        nvigi::PluginID plugin_id;
        PFun_nvigiUnloadInterface* unloader;
        std::mutex mutex;
    };

    // Singleton-like interface management
    struct InterfaceManager {
        static inline IAutoSpeechRecognition* iasr{nullptr};
        static inline IPolledInferenceInterface* ipolled{nullptr};
        static inline std::mutex mutex;
        static inline int reference_count{0};
        static inline std::string current_backend;
    };

    static std::expected<std::unique_ptr<Instance>, Error> create(
        const ModelConfig& config,
        const nvigi::d3d12::D3D12Config& d3d12_config = {},
        const nvigi::vulkan::VulkanConfig& vulkan_config = {},
        const nvigi::io::IOConfig& io_config = {},
        PFun_nvigiLoadInterface* loader = nullptr,
        PFun_nvigiUnloadInterface* unloader = nullptr,
        const std::string_view& plugin_path = {},
        std::source_location loc = std::source_location::current()
    ) {
        auto instance = std::unique_ptr<Instance>(new Instance());
        instance->impl_ = std::make_unique<Impl>();
        
        // Check for valid backend (no cloud for ASR)
        if (!is_valid_backend(config.backend)) {
            return std::unexpected(Error(
                std::format("Unsupported backend '{}', must be 'd3d12', 'cuda', 'vulkan', or 'cpu' at {}:{}",
                    config.backend, loc.file_name(), loc.line())
            ));
        }

        instance->impl_->unloader = unloader;
        instance->impl_->plugin_id = backend_to_plugin_id(config.backend);

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
                
                // Initialize ASR interface
                auto result = nvigiGetInterfaceDynamic(
                    instance->impl_->plugin_id,
                    &InterfaceManager::iasr,
                    loader, plugin_path.empty() ? nullptr : plugin_path.data()
                );
                
                if (!InterfaceManager::iasr) {
                    return std::unexpected(Error(
                        std::format("Failed to get ASR interface for backend '{}' at {}:{}", 
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
        ASRWhisperCreationParameters params{};
        D3D12Parameters d3d12params{};
        VulkanParameters vkparams{};
        
        // Chain parameters (unused are simply ignored)
        d3d12params.chain(common);        
        common.chain(params);
        params.chain(vkparams);

        // Set base parameters
        common.utf8PathToModels = config.model_path.data();
        common.numThreads = config.num_threads;
        common.vramBudgetMB = config.vram_budget_mb;
        common.modelGUID = config.guid.data();

        params.language = config.language.data();
        params.flashAtt = config.flash_attention;
        params.translate = config.translate;
        params.detectLanguage = config.detect_language;
        
        // Set D3D12 specific parameters
        d3d12params.device = d3d12_config.device;
        d3d12params.queue = d3d12_config.direct_queue;
        d3d12params.queueCompute = d3d12_config.compute_queue;
        d3d12params.queueCopy = d3d12_config.transfer_queue;
        
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

        // Configure IO callbacks if custom file loading is enabled
        nvigi::io::configure_io_callbacks(common, io_config);

        auto creation_result = InterfaceManager::iasr->createInstance(
            d3d12params,
            &instance->impl_->instance
        );
        
        if (creation_result != kResultOk || !instance->impl_->instance) {
            // Decrement reference count on failure
            std::lock_guard<std::mutex> lock(InterfaceManager::mutex);
            InterfaceManager::reference_count--;
            if (InterfaceManager::reference_count == 0) {
                instance->impl_->unloader(instance->impl_->plugin_id, InterfaceManager::iasr);
                instance->impl_->unloader(instance->impl_->plugin_id, InterfaceManager::ipolled);
                InterfaceManager::iasr = nullptr;
                InterfaceManager::ipolled = nullptr;
                InterfaceManager::current_backend.clear();
            }
            return std::unexpected(Error("Failed to create ASR inference instance"));
        }
        
        return instance;
    }

    ~Instance() {
        if (impl_) {
            if (impl_->instance) {
                InterfaceManager::iasr->destroyInstance(impl_->instance);
            }
            
            // Decrement reference count and release interfaces if needed
            std::lock_guard<std::mutex> lock(InterfaceManager::mutex);
            InterfaceManager::reference_count--;
            if (InterfaceManager::reference_count == 0) {
                // Release interfaces when all instances are destroyed
                impl_->unloader(impl_->plugin_id, InterfaceManager::iasr);
                impl_->unloader(impl_->plugin_id, InterfaceManager::ipolled);
                InterfaceManager::iasr = nullptr;
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
        std::vector<CpuData> cpu_data;
        std::vector<InferenceDataAudio> audio_data;
        ASRWhisperRuntimeParameters runtime_params;
        StreamingParameters stream_params;
        std::vector<uint8_t> audio_buffer_copy;  // Store audio data
    };

    // Transcribe complete audio (blocking)
    Result transcribe(
        const void* audio_data,
        size_t audio_size_bytes,
        const RuntimeConfig& config = {},
        std::function<ExecutionState(std::string_view, ExecutionState)> callback = nullptr
    ) {
        std::lock_guard<std::mutex> lock(impl_->mutex);

        // Callback implementation
        auto cb = [](const InferenceExecutionContext* ctx, 
                    InferenceExecutionState state, void* data) -> InferenceExecutionState {
            auto* callback = static_cast<std::function<ExecutionState(std::string_view, ExecutionState)>*>(data);
            if (callback && ctx && ctx->outputs) {
                const InferenceDataText* text{};
                if (ctx->outputs->findAndValidateSlot(kASRWhisperDataSlotTranscribedText, &text)) {
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
                    ExecutionState result = (*callback)(text->getUTF8Text(), wrapper_state);
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
        ASRWhisperRuntimeParameters runtime{};
        runtime.sampling = config.sampling == SamplingStrategy::Greedy ? 
            ASRWhisperSamplingStrategy::eGreedy : ASRWhisperSamplingStrategy::eBeamSearch;
        runtime.bestOf = config.best_of;
        runtime.beamSize = config.beam_size;
        runtime.prompt = config.prompt;
        runtime.noContext = config.no_context;
        runtime.suppressBlank = config.suppress_blank;
        runtime.suppressNonSpeechTokens = config.suppress_non_speech_tokens;
        runtime.temperature = config.temperature;
        runtime.entropyThold = config.entropy_threshold;
        runtime.logprobThold = config.logprob_threshold;
        runtime.noSpeechThold = config.no_speech_threshold;

        ctx.runtimeParameters = runtime;
        ctx.instance = impl_->instance;

        // Setup audio input
        CpuData audio_cpu(audio_size_bytes, audio_data);
        InferenceDataAudio audio(audio_cpu);
        
        std::vector<InferenceDataSlot> slots = {{kASRWhisperDataSlotAudio, audio}};
        InferenceDataSlotArray inputs{slots.size(), slots.data()};
        ctx.inputs = &inputs;

        if (callback) {
            ctx.callback = cb;
            ctx.callbackUserData = &callback;
        }

        auto result = impl_->instance->evaluate(&ctx);
        if (result != kResultOk) {
            return std::unexpected(Error("Transcription failed"));
        }

        return {};
    }

    // Polling-based async handle for non-blocking operations
    class AsyncOperation {
    public:
        enum class State {
            Pending,      // Not started yet
            Running,      // Currently executing
            HasResults,   // Has new text/results available
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
            std::string text;             // Transcribed text
            ExecutionState state;         // Current inference state
        };

        // Try to get available results without blocking
        std::optional<Result> try_get_results() {
            if (!context_ || is_complete_) return std::nullopt;
            
            InferenceExecutionState exec_state = kInferenceExecutionStateInvalid;
            auto result = ipolled_->getResults(&context_->ctx, false, &exec_state);
            
            first_poll_ = false;
            
            if (result == kResultNotReady) {
                return std::nullopt; // No results yet
            }
            
            if (result == kResultOk && context_->ctx.outputs) {
                std::string text;
                const InferenceDataText* text_data{};
                if (context_->ctx.outputs->findAndValidateSlot(kASRWhisperDataSlotTranscribedText, &text_data)) {
                    text = text_data->getUTF8Text();
                    
                    // Accumulate response
                    if (response_buffer_) {
                        response_buffer_->append(text);
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
                return Result{ std::move(text), wrapper_state };
            }
            
            return std::nullopt;
        }

        // Request cancellation of the async operation
        // 
        // This calls the cancelAsyncEvaluation API which blocks until cancellation completes.
        // If that succeeds, the operation is fully cancelled and cleaned up.
        // Otherwise, you should continue calling try_get_results() until is_complete() returns true.
        // 
        // After calling cancel(), you MUST continue calling try_get_results() 
        // until is_complete() returns true to properly drain the pipeline
        void cancel() {
            if (!context_ || is_complete_) return;
            
            // Set cancel_requested first for the fallback polling path
            cancel_requested_ = true;
            
            // Call the explicit cancelAsyncEvaluation API on the instance
            // This interrupts the inference loop as early as possible and handles all cleanup
            if (context_->ctx.instance && context_->ctx.instance->cancelAsyncEvaluation) {
                // Set is_complete_ BEFORE calling cancelAsyncEvaluation to prevent
                // concurrent try_get_results() from interfering during the blocking call
                is_complete_ = true;
                
                auto result = context_->ctx.instance->cancelAsyncEvaluation(&context_->ctx);

                if (result == kResultOk) {
                    // Cancel succeeded - cleanup already handled
                    cancel_requested_ = false;
                    return;
                }
                
                // Failed - restore state and fall through to polling approach
                is_complete_ = false;
                // If kResultNoImplementation, no async job was running, fall through to polling approach
            }

            // Fallback: try to send cancel signal immediately via non-blocking poll
            // This is used if cancelAsyncEvaluation is not available or returned kResultNoImplementation
            InferenceExecutionState exec_state = kInferenceExecutionStateInvalid;
            auto poll_result = ipolled_->getResults(&context_->ctx, false, &exec_state);
            
            if (poll_result == kResultOk) {
                // Got a result, send cancel signal immediately
                ipolled_->releaseResults(&context_->ctx, kInferenceExecutionStateCancel);
                is_complete_ = true;
                cancel_requested_ = false;
            }
            // If kResultNotReady, the cancel will be sent on next try_get_results()
        }

        // Check if operation is complete
        bool is_complete() const { 
            return is_complete_ || is_failed_; 
        }

        // Check if operation failed
        bool is_failed() const { 
            return is_failed_; 
        }

        // Get full accumulated response so far
        std::string_view get_accumulated_text() const {
            return response_buffer_ ? *response_buffer_ : std::string_view{};
        }

        // Move accumulated response out
        std::string take_text() {
            if (response_buffer_) {
                return std::move(*response_buffer_);
            }
            return {};
        }

        // Reset/clear the operation
        void reset() {
            context_.reset();
            response_buffer_.reset();
            is_complete_ = true;
            is_failed_ = false;
            cancel_requested_ = false;
        }

        // Move constructor (public for std::expected compatibility)
        AsyncOperation(AsyncOperation&& other) noexcept
            : ipolled_(other.ipolled_)
            , context_(std::move(other.context_))
            , response_buffer_(std::move(other.response_buffer_))
            , is_complete_(other.is_complete_.load())
            , is_failed_(other.is_failed_.load())
            , first_poll_(other.first_poll_.load())
            , cancel_requested_(other.cancel_requested_.load())
        {
            other.ipolled_ = nullptr;
        }
        
        // Move assignment (public for std::expected compatibility)
        AsyncOperation& operator=(AsyncOperation&& other) noexcept {
            if (this != &other) {
                ipolled_ = other.ipolled_;
                context_ = std::move(other.context_);
                response_buffer_ = std::move(other.response_buffer_);
                is_complete_.store(other.is_complete_.load());
                is_failed_.store(other.is_failed_.load());
                first_poll_.store(other.first_poll_.load());
                cancel_requested_.store(other.cancel_requested_.load());
                other.ipolled_ = nullptr;
            }
            return *this;
        }

        // Delete copy operations
        AsyncOperation(const AsyncOperation&) = delete;
        AsyncOperation& operator=(const AsyncOperation&) = delete;

    private:
        friend class Instance;
        friend class Stream;
        
        AsyncOperation(
            IPolledInferenceInterface* ipolled,
            std::shared_ptr<AsyncContext> context,
            std::shared_ptr<std::string> response_buffer
        ) : ipolled_(ipolled)
          , context_(std::move(context))
          , response_buffer_(std::move(response_buffer))
          , is_complete_(false)
          , is_failed_(false)
          , first_poll_(true)
          , cancel_requested_(false)
        {}

        IPolledInferenceInterface* ipolled_;
        std::shared_ptr<AsyncContext> context_;
        std::shared_ptr<std::string> response_buffer_;
        std::atomic<bool> is_complete_;
        std::atomic<bool> is_failed_;
        std::atomic<bool> first_poll_;
        std::atomic<bool> cancel_requested_;
    };

    // Stream class for continuous audio streaming
    class Stream {
    public:
        // Send audio chunk (blocking)
        Result send_audio(
            const void* audio_data,
            size_t audio_size_bytes,
            bool is_first_chunk,
            bool is_last_chunk,
            std::function<ExecutionState(std::string_view, ExecutionState)> callback = nullptr
        ) {
            // Callback implementation
            auto cb = [](const InferenceExecutionContext* ctx, 
                        InferenceExecutionState state, void* data) -> InferenceExecutionState {
                auto* callback = static_cast<std::function<ExecutionState(std::string_view, ExecutionState)>*>(data);
                if (callback && ctx && ctx->outputs) {
                    const InferenceDataText* text{};
                    if (ctx->outputs->findAndValidateSlot(kASRWhisperDataSlotTranscribedText, &text)) {
                        ExecutionState wrapper_state = ExecutionState::Invalid;
                        if (state == kInferenceExecutionStateDataPending) {
                            wrapper_state = ExecutionState::DataPending;
                        } else if (state == kInferenceExecutionStateDataPartial) {
                            wrapper_state = ExecutionState::DataPartial;
                        } else if (state == kInferenceExecutionStateDone) {
                            wrapper_state = ExecutionState::Done;
                        }
                        
                        ExecutionState result = (*callback)(text->getUTF8Text(), wrapper_state);
                        if (result == ExecutionState::Cancel) {
                            return kInferenceExecutionStateCancel;
                        }
                    }
                }
                return state;
            };

            InferenceExecutionContext ctx{};
            ctx.runtimeParameters = config_.runtime_params;
            ctx.instance = instance_.impl_->instance;

            // Setup streaming parameters
            StreamingParameters stream_params{};
            if (is_first_chunk) {
                stream_params.signal = StreamSignal::eStreamSignalStart;
            } else if (is_last_chunk) {
                stream_params.signal = StreamSignal::eStreamSignalStop;
            } else {
                stream_params.signal = StreamSignal::eStreamSignalData;
            }
            config_.runtime_params.chain(stream_params);

            // Setup audio input
            CpuData audio_cpu(audio_size_bytes, audio_data);
            InferenceDataAudio audio(audio_cpu);
            
            std::vector<InferenceDataSlot> slots = {{kASRWhisperDataSlotAudio, audio}};
            InferenceDataSlotArray inputs{slots.size(), slots.data()};
            ctx.inputs = &inputs;

            if (callback) {
                ctx.callback = cb;
                ctx.callbackUserData = &callback;
            }

            auto result = instance_.impl_->instance->evaluate(&ctx);
            if (result != kResultOk) {
                return std::unexpected(Error("Audio streaming failed"));
            }

            return {};
        }

        // Send audio chunk async (non-blocking, returns immediately)
        // IMPORTANT: Call this only ONCE to get AsyncOperation, then just feed chunks
		// This function is not thread safe - ensure single-threaded calls per Stream instance
        std::expected<AsyncOperation, Error> send_audio_async(
            const void* audio_data,
            size_t audio_size_bytes,
            bool is_first_chunk,
            bool is_last_chunk
        ) {            
            // Initialize on first chunk
            if (!context_) {
                context_ = std::make_shared<AsyncContext>();
                response_buffer_ = std::make_shared<std::string>();
                
                // Setup runtime parameters once
                context_->runtime_params = config_.runtime_params;
                context_->runtime_params.chain(context_->stream_params);
                context_->ctx.runtimeParameters = context_->runtime_params;
                context_->ctx.instance = instance_.impl_->instance;
            }
            
            // Clear vectors before reuse (critical!)
            context_->cpu_data.clear();
            context_->audio_data.clear();
            context_->slots.clear();
            
            // Copy current audio chunk (NVIGI will copy to its circular buffer immediately)
            context_->audio_buffer_copy.assign(
                static_cast<const uint8_t*>(audio_data),
                static_cast<const uint8_t*>(audio_data) + audio_size_bytes
            );

            // Update streaming signal
            if (is_first_chunk) {
                context_->stream_params.signal = StreamSignal::eStreamSignalStart;
            } else if (is_last_chunk) {
                context_->stream_params.signal = StreamSignal::eStreamSignalStop;
                is_last_sent_ = true;
            } else {
                context_->stream_params.signal = StreamSignal::eStreamSignalData;
            }

            // Setup audio input (fresh each time)
            context_->cpu_data.emplace_back(CpuData(context_->audio_buffer_copy.size(),
                context_->audio_buffer_copy.data()));
            context_->audio_data.emplace_back(InferenceDataAudio(context_->cpu_data.back()));

            context_->slots.push_back({kASRWhisperDataSlotAudio, context_->audio_data.back()});
            context_->inputs = { context_->slots.size(), context_->slots.data() };
            context_->ctx.inputs = &context_->inputs;

            // Call evaluateAsync - NVIGI copies audio to circular buffer immediately
            auto result = instance_.impl_->instance->evaluateAsync(&context_->ctx);
            if (result != kResultOk) {
                return std::unexpected(Error("Failed to start async transcription"));
            }

            // Return AsyncOperation - all calls return operations referencing same context
            // This is OK because NVIGI has a single background thread and poll context
            return AsyncOperation(InterfaceManager::ipolled, context_, response_buffer_);
        }

    private:
        friend class Instance;
        
        struct StreamConfig {
            ASRWhisperRuntimeParameters runtime_params;
        };
        
        Stream(Instance& instance, const RuntimeConfig& config)
            : instance_(instance)
            , is_last_sent_(false) {
            // Setup runtime parameters
            config_.runtime_params.sampling = config.sampling == SamplingStrategy::Greedy ? 
                ASRWhisperSamplingStrategy::eGreedy : ASRWhisperSamplingStrategy::eBeamSearch;
            config_.runtime_params.bestOf = config.best_of;
            config_.runtime_params.beamSize = config.beam_size;
            config_.runtime_params.prompt = config.prompt;
            config_.runtime_params.noContext = config.no_context;
            config_.runtime_params.suppressBlank = config.suppress_blank;
            config_.runtime_params.suppressNonSpeechTokens = config.suppress_non_speech_tokens;
            config_.runtime_params.temperature = config.temperature;
            config_.runtime_params.entropyThold = config.entropy_threshold;
            config_.runtime_params.logprobThold = config.logprob_threshold;
            config_.runtime_params.noSpeechThold = config.no_speech_threshold;
        }
        
        Instance& instance_;
        StreamConfig config_;
        std::shared_ptr<AsyncContext> context_;
        std::shared_ptr<std::string> response_buffer_;
        bool is_last_sent_;                          // Track if final chunk sent
    };

    Stream create_stream(const RuntimeConfig& config = {}) {
        return Stream(*this, config);
    }

private:
    Instance() = default;
    std::unique_ptr<Impl> impl_;
};

} // namespace nvigi::asr

