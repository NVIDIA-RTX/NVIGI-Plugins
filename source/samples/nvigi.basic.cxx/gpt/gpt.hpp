// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace nvigi::gpt {

// Forward declarations
class Instance;
class Context;

// KV cache quantization types
enum class CacheType : int32_t {
    FP32 = 0,  // Full precision (32-bit float)
    FP16 = 1,  // Half precision (16-bit float) - default
    Q4_0 = 2,  // 4-bit quantization
    Q8_0 = 8   // 8-bit quantization
};

// Inference execution states
// 
// Used in callbacks to provide inference status and allow cancellation.
// The callback receives the current state and response text, and can return:
// - The same state to continue normally
// - ExecutionState::Cancel to stop inference immediately
//
// Example:
//   [](std::string_view response, ExecutionState state) -> ExecutionState {
//       if (state == ExecutionState::DataPending) {
//           std::cout << response;  // Display new token
//       }
//       return should_stop ? ExecutionState::Cancel : state;
//   }
enum class ExecutionState : uint32_t {
    Invalid = 0,        // Invalid/error state
    DataPending = 1,    // Data is pending/available (new tokens)
    DataPartial = 2,    // Partial data (e.g., intermediate results)
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

struct RuntimeConfig {
    // Base parameters
    uint32_t seed{0xFFFFFFFF};     // RNG seed
    int32_t tokens_to_predict{-1};  // new tokens to predict
    int32_t batch_size{512};       // batch size for prompt processing (must be >=32 to use BLAS)
    int32_t tokens_to_keep{0};     // number of tokens to keep from initial prompt
    int32_t tokens_to_draft{16};   // number of tokens to draft during speculative decoding
    int32_t num_chunks{-1};        // max number of chunks to process (-1 = unlimited)
    int32_t num_parallel{1};       // number of parallel sequences to decode
    int32_t num_sequences{1};      // number of sequences to decode
    float temperature{0.2f};       // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float top_p{0.7f};            // 1.0 = disabled
    bool interactive{true};        // chat mode by default
    std::string_view reverse_prompt; // reverse prompt for the interactive mode
    std::string_view prefix;       // prefix for the user input
    std::string_view suffix;       // suffix for the user input
    float frame_time_ms{0.0f};    // optional, used to limit the token generation rate, disabled by default
    int32_t target_tokens_per_second{-1}; // optional, used to limit the token generation rate, disabled by default
    bool prompt_pretemplatized{false}; // if true, will not attempt to apply any sort of prompt templatization
    
    // LoRA parameters
    struct Lora {
        std::string_view name;
        float scale{1.0f};
    };
    std::vector<Lora> loras;
    
    // Jinja template parameters
    bool use_jinja{true};          // if true use jinja formatting for prompt templates
    std::string_view chat_template; // custom template to use instead of internal one

    std::string_view json_body; // raw JSON body to send to the server, if set will override all other parameters

    // Builder pattern methods
    RuntimeConfig& set_seed(uint32_t s) {
        seed = s;
        return *this;
    }
    
    RuntimeConfig& set_tokens(int32_t tokens) {
        tokens_to_predict = tokens;
        return *this;
    }

    RuntimeConfig& set_batch_size(int32_t size) {
        batch_size = size;
        return *this;
    }

    RuntimeConfig& set_tokens_to_keep(int32_t tokens) {
        tokens_to_keep = tokens;
        return *this;
    }

    RuntimeConfig& set_tokens_to_draft(int32_t tokens) {
        tokens_to_draft = tokens;
        return *this;
    }

    RuntimeConfig& set_num_chunks(int32_t chunks) {
        num_chunks = chunks;
        return *this;
    }

    RuntimeConfig& set_num_parallel(int32_t parallel) {
        num_parallel = parallel;
        return *this;
    }

    RuntimeConfig& set_num_sequences(int32_t sequences) {
        num_sequences = sequences;
        return *this;
    }

    RuntimeConfig& set_temperature(float temp) {
        temperature = temp;
        return *this;
    }

    RuntimeConfig& set_top_p(float p) {
        top_p = p;
        return *this;
    }

    RuntimeConfig& set_interactive(bool enable) {
        interactive = enable;
        return *this;
    }

    RuntimeConfig& set_reverse_prompt(std::string_view prompt) {
        reverse_prompt = prompt;
        return *this;
    }

    RuntimeConfig& set_prefix(std::string_view pre) {
        prefix = pre;
        return *this;
    }

    RuntimeConfig& set_suffix(std::string_view suf) {
        suffix = suf;
        return *this;
    }

    RuntimeConfig& set_frame_time_ms(float time) {
        frame_time_ms = time;
        return *this;
    }

    RuntimeConfig& set_target_tokens_per_second(int32_t tokens) {
        target_tokens_per_second = tokens;
        return *this;
    }

    RuntimeConfig& set_prompt_pretemplatized(bool enable) {
        prompt_pretemplatized = enable;
        return *this;
    }

    RuntimeConfig& add_lora(std::string_view name, float scale = 1.0f) {
        loras.push_back(Lora{name, scale});
        return *this;
    }

    RuntimeConfig& set_use_jinja(bool enable) {
        use_jinja = enable;
        return *this;
    }

    RuntimeConfig& set_chat_template(std::string_view templ) {
        chat_template = templ;
        return *this;
    }
};

struct CloudConfig {
    std::string_view url;     // Custom URL for cloud provider
    std::string_view token;   // Authorization token for cloud provider
    bool verbose{ false };      // Enable verbose logging for cloud operations
    bool streaming{ true };    // Enable streaming responses (if supported by backend)
    CloudConfig& set_token(std::string_view t) {
        token = t;
        return *this;
    }
    CloudConfig& set_url(std::string_view u) {
        url = u;
        return *this;
    }
    CloudConfig& set_verbose(bool v) {
        verbose = v;
        return *this;
    }
    CloudConfig& set_streaming(bool s) {
        streaming = s;
        return *this;
    }
};

// Specialized model configs that inherit from base ModelConfig
struct ModelConfig {
    std::string_view backend;
    std::string_view guid;
    std::string_view model_path;
    int32_t context_size{ 4096 };
    int32_t num_threads{ 1 };
    size_t vram_budget_mb{ 0 };
    bool flash_attention{ true };
    std::string_view cache_type{ "fp16" };

    ModelConfig& set_context_size(int32_t size) {
        context_size = size;
        return *this;
    }
    ModelConfig& set_threads(int32_t threads) {
        num_threads = threads;
        return *this;
    }
    ModelConfig& set_vram(size_t mb) {
        vram_budget_mb = mb;
        return *this;
    }
};


// Main GPT class implementation
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
        static inline IGeneralPurposeTransformer* igpt{nullptr};
        static inline IPolledInferenceInterface* ipolled{nullptr};
        static inline std::mutex mutex;
        static inline int reference_count{0};
        static inline std::string current_backend;
    };

    static std::expected<std::unique_ptr<Instance>, Error> create(
        const ModelConfig& config,
        const nvigi::d3d12::D3D12Config& d3d12_config = {},
        const nvigi::vulkan::VulkanConfig& vulkan_config = {},
        const CloudConfig& cloud_config = {},
        PFun_nvigiLoadInterface* loader = nullptr,
        PFun_nvigiUnloadInterface* unloader = nullptr,
        const std::string_view& plugin_path = {},
        std::source_location loc = std::source_location::current()
    ) {
        auto instance = std::unique_ptr<Instance>(new Instance());
        instance->impl_ = std::make_unique<Impl>();
        
        // Check for valid backend
        if (config.backend != "d3d12" && config.backend != "cuda" && config.backend != "vulkan" && config.backend != "cloud") {
            return std::unexpected(Error(
                std::format("Unsupported backend '{}', must be 'd3d12', 'cuda', 'vulkan' or 'cloud' at {}:{}",
                    config.backend, loc.file_name(), loc.line())
            ));
        }

        // Parse cache type from command line        
        CacheType cache_type = CacheType::FP16; // default
        if (config.cache_type == "fp32") {
            cache_type = CacheType::FP32;
        }
        else if (config.cache_type == "fp16") {
            cache_type = CacheType::FP16;
        }
        else if (config.cache_type == "q4_0") {
            cache_type = CacheType::Q4_0;
        }
        else if (config.cache_type == "q8_0") {
            cache_type = CacheType::Q8_0;
        }
        else
        {
            return std::unexpected(Error(
                std::format("Unsupported cache type '{}', must be 'fp32', 'fp16', 'q4_0' or 'q8_0' at {}:{}",
                    config.cache_type, loc.file_name(), loc.line())
            ));
        }
        instance->impl_->unloader = unloader;
        instance->impl_->plugin_id = config.backend == "d3d12" ? plugin::gpt::ggml::d3d12::kId :
            config.backend == "vulkan" ? plugin::gpt::ggml::vulkan::kId : config.backend == "cloud" ? plugin::gpt::cloud::rest::kId :
            plugin::gpt::ggml::cuda::kId;

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
                
                // Initialize GPT interface
                auto result = nvigiGetInterfaceDynamic(
                    instance->impl_->plugin_id,
                    &InterfaceManager::igpt,
                    loader, plugin_path.empty() ? nullptr : plugin_path.data()
                );
                
                if (!InterfaceManager::igpt) {
                    return std::unexpected(Error(
                        std::format("Failed to get GPT interface for backend '{}' at {}:{}", 
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
        GPTCreationParameters params{};
        D3D12Parameters d3d12params{};
        VulkanParameters vkparams{};
        RESTParameters restparams{};
        
        // Chain parameters (unused are simply ignored)
        d3d12params.chain(common);        
        common.chain(params);
        params.chain(restparams);
        restparams.chain(vkparams);

        // Set base parameters
        common.utf8PathToModels = config.model_path.data();
        common.numThreads = config.num_threads;
        common.vramBudgetMB = config.vram_budget_mb;
        common.modelGUID = config.guid.data();

        params.contextSize = config.context_size;
        params.flashAttention = config.flash_attention;
        params.cacheTypeK = static_cast<int32_t>(cache_type);
        params.cacheTypeV = static_cast<int32_t>(cache_type);
        
        // Set cloud parameters
        if (config.backend == "cloud") {
            nvigi::CommonCapabilitiesAndRequirements* caps{};
            if (NVIGI_FAILED(result, getCapsAndRequirements(InterfaceManager::igpt, common, &caps)))
            {
                return std::unexpected(Error("Failed to get caps and requirements"));
            }

            auto* ccaps = findStruct<nvigi::CloudCapabilities>(*caps);
            if (!ccaps)
            {
                return std::unexpected(Error("Failed to find cloud capabilities"));
            }
            
            if (cloud_config.token.empty())
            {
                return std::unexpected(Error("Cloud token is empty"));
            }

            restparams.url = cloud_config.url.empty() ? ccaps->url : cloud_config.url.data();
            restparams.authenticationToken = cloud_config.token.data();
            restparams.verboseMode = cloud_config.verbose;
            restparams.useStreaming = cloud_config.streaming;
        }
        
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

        auto creation_result = InterfaceManager::igpt->createInstance(
            d3d12params,
            &instance->impl_->instance
        );
        
        if (creation_result != kResultOk || !instance->impl_->instance) {
            // Decrement reference count on failure
            std::lock_guard<std::mutex> lock(InterfaceManager::mutex);
            InterfaceManager::reference_count--;
            if (InterfaceManager::reference_count == 0) {
                instance->impl_->unloader(instance->impl_->plugin_id, InterfaceManager::igpt);
                instance->impl_->unloader(instance->impl_->plugin_id, InterfaceManager::ipolled);
                InterfaceManager::igpt = nullptr;
                InterfaceManager::ipolled = nullptr;
                InterfaceManager::current_backend.clear();
            }
            return std::unexpected(Error("Failed to create inference instance"));
        }
        
        return instance;
    }

    ~Instance() {
        if (impl_) {
            if (impl_->instance) {
                InterfaceManager::igpt->destroyInstance(impl_->instance);
            }
            
            // Decrement reference count and release interfaces if needed
            std::lock_guard<std::mutex> lock(InterfaceManager::mutex);
            InterfaceManager::reference_count--;
            if (InterfaceManager::reference_count == 0) {
                // Release interfaces when all instances are destroyed
                impl_->unloader(impl_->plugin_id, InterfaceManager::igpt);
                impl_->unloader(impl_->plugin_id, InterfaceManager::ipolled);
                InterfaceManager::igpt = nullptr;
                InterfaceManager::ipolled = nullptr;
                InterfaceManager::current_backend.clear();
            }
        }
    }

    // Update AsyncContext to include the new message storage
    struct AsyncContext {
        InferenceExecutionContext ctx;
        InferenceDataSlotArray inputs;
        std::vector<InferenceDataSlot> slots;
        std::vector<InferenceDataText> message_data;
        std::vector<CpuData> cpu_data;  // Store CpuData objects to keep them alive
        GPTRuntimeParameters runtime_params;  // Store runtime parameters to keep them alive
        std::vector<float> lora_scales;
        std::vector<std::string> lora_name_copies;  // Store LoRA name copies
        std::vector<const char*> lora_names;  // Pointers to lora_name_copies
        std::string system_copy;
        std::string user_copy;
        std::string assistant_copy;
        std::string json_copy;
        std::string reverse_prompt_copy;
    };

    Result generate(
        std::string_view system_msg,
        std::string_view user_msg,
        std::string_view assistant_msg,
        const RuntimeConfig& config = {},
        std::function<ExecutionState(std::string_view, ExecutionState)> callback = nullptr
    ) {
        std::lock_guard<std::mutex> lock(impl_->mutex);

        // Callback implementation - convert between internal and wrapper states
        auto cb = [](const InferenceExecutionContext* ctx, 
                    InferenceExecutionState state, void* data) -> InferenceExecutionState {
            auto* callback = static_cast<std::function<ExecutionState(std::string_view, ExecutionState)>*>(data);
            if (callback && ctx && ctx->outputs) {
                const InferenceDataText* text{};
                if (ctx->outputs->findAndValidateSlot(kGPTDataSlotResponse, &text)) {
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
        GPTRuntimeParameters runtime{};
        runtime.seed = config.seed;
        runtime.tokensToPredict = config.tokens_to_predict;
        runtime.interactive = config.interactive;
        if (!config.reverse_prompt.empty()) {
            runtime.reversePrompt = config.reverse_prompt.data();
        }

        // Setup LoRA if needed - keep vectors in scope for the entire function
        std::vector<float> scales;
        std::vector<const char*> names;
        if (!config.loras.empty()) {
            for (const auto& lora : config.loras) {
                scales.push_back(lora.scale);
                names.push_back(lora.name.data());
            }
            runtime.loraScales = scales.data();
            runtime.loraNames = names.data();
            runtime.numLoras = static_cast<int32_t>(config.loras.size());
        }

        ctx.runtimeParameters = runtime;
        ctx.instance = impl_->instance;

        // Setup input data with all message slots
        std::vector<InferenceDataText> message_data;
        std::vector<InferenceDataSlot> slots;
        message_data.reserve(4); // Reserve space for system, user, assistant, json
        slots.reserve(4);

        if (!system_msg.empty()) {
            message_data.emplace_back(CpuData(system_msg.size() + 1, system_msg.data()));
            slots.push_back({kGPTDataSlotSystem, message_data.back()});
        }
        
        if (!user_msg.empty()) {
            message_data.emplace_back(CpuData(user_msg.size() + 1, user_msg.data()));
            slots.push_back({kGPTDataSlotUser, message_data.back()});
        }
        
        if (!assistant_msg.empty()) {
            message_data.emplace_back(CpuData(assistant_msg.size() + 1, assistant_msg.data()));
            slots.push_back({kGPTDataSlotAssistant, message_data.back()});
        }

        if (!config.json_body.empty()) {
            message_data.emplace_back(CpuData(config.json_body.size() + 1, config.json_body.data()));
            slots.push_back({ kGPTDataSlotJSON, message_data.back() });
        }

        InferenceDataSlotArray inputs{slots.size(), slots.data()};
        ctx.inputs = &inputs;

        if (callback) {
            ctx.callback = cb;
            ctx.callbackUserData = &callback;
        }

        auto result = impl_->instance->evaluate(&ctx);
        if (result != kResultOk) {
            return std::unexpected(Error("Generation failed"));
        }

        return {};
    }    

    // Polling-based async handle for non-blocking operations
    class AsyncOperation {
    public:
        enum class State {
            Pending,      // Not started yet
            Running,      // Currently executing
            HasResults,   // Has new tokens/results available
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
        // Contains the response tokens and current execution state
        struct Result {
            std::string tokens;           // New token(s) from inference
            ExecutionState state;         // Current inference state
        };

        // Try to get available results without blocking (perfect for game loops!)
        // 
        // Returns:
        //   - std::nullopt if no results are available yet (inference still processing)
        //   - Result{ tokens, state } if new data is available
        //
        // The state indicates:
        //   - ExecutionState::DataPending: New token(s) ready
        //   - ExecutionState::DataPartial: Intermediate results
        //   - ExecutionState::Done: Inference completed
        //   - ExecutionState::Invalid: Error occurred
        std::optional<Result> try_get_results() {
            if (!context_) return std::nullopt;
            
            InferenceExecutionState exec_state = kInferenceExecutionStateInvalid;
            auto result = ipolled_->getResults(&context_->ctx, false, &exec_state);
            
            first_poll_ = false;
            
            if (result == kResultNotReady) {
                return std::nullopt; // No results yet
            }
            
            if (result == kResultOk && context_->ctx.outputs) {
                std::string tokens;
                const InferenceDataText* text{};
                if (context_->ctx.outputs->findAndValidateSlot(kGPTDataSlotResponse, &text)) {
                    tokens = text->getUTF8Text();
                    
                    // Accumulate response
                    if (response_buffer_) {
                        response_buffer_->append(tokens);
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
                
                // If cancel was requested, signal it to the inference engine
                // by passing kInferenceExecutionStateCancel to releaseResults()
                InferenceExecutionState state_to_release = exec_state;
                if (cancel_requested_) {
                    // Override the state we send back to the engine with Cancel
                    state_to_release = kInferenceExecutionStateCancel;
                    is_complete_ = true;
                    wrapper_state = ExecutionState::Cancel;
                    cancel_requested_ = false;
                }
                
                // releaseResults() receives our state (potentially Cancel) and acts accordingly
                ipolled_->releaseResults(&context_->ctx, state_to_release);
                return Result{ std::move(tokens), wrapper_state };
            }
            
            return std::nullopt;
        }

        // Request cancellation of the async operation
        // 
        // Call this to stop inference early (e.g., user pressed ESC, timeout, etc.)
        // The cancellation will be signaled to the inference engine on the next try_get_results() call.
        //
        // Example:
        //   if (user_pressed_cancel) {
        //       op.cancel();
        //   }
        void cancel() {
            cancel_requested_ = true;
            // The cancellation will be sent to the inference engine on next try_get_results()
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
        std::string_view get_accumulated_response() const {
            return response_buffer_ ? *response_buffer_ : std::string_view{};
        }

        // Move accumulated response out
        std::string take_response() {
            if (response_buffer_) {
                return std::move(*response_buffer_);
            }
            return {};
        }

        // Reset/clear the operation (useful for reusing the variable)
        void reset() {
            context_.reset();
            response_buffer_.reset();
            is_complete_ = true;
            is_failed_ = false;
            cancel_requested_ = false;
        }

    private:
        friend class Instance;
        friend class Chat;
        
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
        bool is_complete_;
        bool is_failed_;
        bool first_poll_;
        bool cancel_requested_;
    };

    class Chat {
    public:
        struct Message {
            enum class Role { System, User, Assistant };
            Role role;
            std::string content;
        };

        Result send_message(
            Message message,
            std::function<ExecutionState(std::string_view, ExecutionState)> callback = nullptr
        ) {
            messages_.push_back(message);

            // Format chat history into a single prompt
            std::string system, user, assistant;
            
            switch (message.role) {
                case Message::Role::System:
                    system = message.content;
                    break;
                case Message::Role::User:
                    user = message.content;
                    break;
                case Message::Role::Assistant:
                    assistant = message.content;
                    break;
            }

            auto result = instance_.generate(system, user, assistant, config_,
                [this, callback](std::string_view response, ExecutionState state) -> ExecutionState {
                    current_response_.append(response);
                    if (callback) {
                        // Call user callback and let them control the flow
                        return callback(response, state);
                    }
                    return state; // Continue with current state by default
                }
            );

            if (result) {
                messages_.push_back(Message{
                    Message::Role::Assistant, 
                    std::move(current_response_)
                });
                current_response_.clear();
            }

            return result;
        }

        // New polling-based async method - perfect for game loops!
        std::expected<AsyncOperation, Error> send_message_polled(
            Message message
        ) {
            messages_.push_back(message);

            // Format chat history into a single prompt
            std::string system, user, assistant;

            switch (message.role) {
            case Message::Role::System:
                system = message.content;
                break;
            case Message::Role::User:
                user = message.content;
                break;
            case Message::Role::Assistant:
                assistant = message.content;
                break;
            }

            // Create async context
            auto context = std::make_shared<AsyncContext>();
            auto response_buffer = std::make_shared<std::string>();
            
            // Store copies of all messages
            if (!system.empty()) {
                context->system_copy = system;
            }
            if (!user.empty()) {
                context->user_copy = user;
            }
            if (!assistant.empty()) {
                context->assistant_copy = assistant;
            }
            if (!config_.json_body.empty()) {
                context->json_copy = std::string(config_.json_body);
            }
            
            // Setup runtime parameters
            context->runtime_params.seed = config_.seed;
            context->runtime_params.tokensToPredict = config_.tokens_to_predict;
            context->runtime_params.interactive = config_.interactive;
            if (!config_.reverse_prompt.empty()) {
                context->reverse_prompt_copy = std::string(config_.reverse_prompt);
                context->runtime_params.reversePrompt = context->reverse_prompt_copy.c_str();
            }

            // Setup LoRA if needed
            if (!config_.loras.empty()) {
                context->lora_scales.reserve(config_.loras.size());
                context->lora_name_copies.reserve(config_.loras.size());
                context->lora_names.reserve(config_.loras.size());
                
                // First, copy all LoRA names to ensure they remain valid
                for (const auto& lora : config_.loras) {
                    context->lora_scales.push_back(lora.scale);
                    context->lora_name_copies.emplace_back(lora.name);
                }
                
                // Then, create pointers to the copied names
                for (const auto& name : context->lora_name_copies) {
                    context->lora_names.push_back(name.c_str());
                }
                
                context->runtime_params.loraScales = context->lora_scales.data();
                context->runtime_params.loraNames = context->lora_names.data();
                context->runtime_params.numLoras = static_cast<int32_t>(config_.loras.size());
            }

            context->ctx.runtimeParameters = context->runtime_params;
            
            context->message_data.reserve(4);
            context->cpu_data.reserve(4);

            // Setup input data with message slots
            if (!context->system_copy.empty()) {
                context->cpu_data.emplace_back(CpuData(context->system_copy.size() + 1, 
                                                       context->system_copy.data()));
                context->message_data.emplace_back(context->cpu_data.back());
                context->slots.push_back({kGPTDataSlotSystem, context->message_data.back()});
            }
            
            if (!context->user_copy.empty()) {
                context->cpu_data.emplace_back(CpuData(context->user_copy.size() + 1, 
                                                       context->user_copy.data()));
                context->message_data.emplace_back(context->cpu_data.back());
                context->slots.push_back({kGPTDataSlotUser, context->message_data.back()});
            }
            
            if (!context->assistant_copy.empty()) {
                context->cpu_data.emplace_back(CpuData(context->assistant_copy.size() + 1, 
                                                       context->assistant_copy.data()));
                context->message_data.emplace_back(context->cpu_data.back());
                context->slots.push_back({kGPTDataSlotAssistant, context->message_data.back()});
            }
            
            if (!context->json_copy.empty()) {
                context->cpu_data.emplace_back(CpuData(context->json_copy.size() + 1, 
                                                       context->json_copy.data()));
                context->message_data.emplace_back(context->cpu_data.back());
                context->slots.push_back({kGPTDataSlotJSON, context->message_data.back()});
            }

            context->inputs = { context->slots.size(), context->slots.data() };
            context->ctx.inputs = &context->inputs;
            context->ctx.instance = instance_.impl_->instance;

            // Start async evaluation
            auto result = instance_.impl_->instance->evaluateAsync(&context->ctx);
            if (result != kResultOk) {
                return std::unexpected(Error("Failed to start async generation"));
            }

            // Return polling handle
            return AsyncOperation(InterfaceManager::ipolled, context, response_buffer);
        }

        // Helper to finalize a completed async operation
        void finalize_async_response(AsyncOperation& op) {
            if (op.is_complete() && !op.is_failed()) {
                messages_.push_back(Message{
                    Message::Role::Assistant,
                    op.take_response()
                });
            }
        }

        const std::vector<Message>& history() const { return messages_; }
        
    private:
        friend class Instance;
        Chat(Instance& instance, const RuntimeConfig& config)
            : instance_(instance), config_(config) {}
        
        Instance& instance_;
        RuntimeConfig config_;
        std::vector<Message> messages_;
        std::string current_response_;
    };

    Chat create_chat(const RuntimeConfig& config = {}) {
        return Chat(*this, config);
    }

private:
    Instance() = default;
    std::unique_ptr<Impl> impl_;
};

} // namespace nvigi::gpt