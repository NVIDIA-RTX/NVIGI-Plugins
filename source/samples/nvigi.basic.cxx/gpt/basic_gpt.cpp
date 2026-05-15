// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <iostream>
#include <format>

// Windows and D3D12 Agility SDK exports
#include <windows.h>
extern "C" __declspec(dllexport) UINT         D3D12SDKVersion = 615;
extern "C" __declspec(dllexport) const char* D3D12SDKPath = ".\\D3D12\\";

// NVIGI includes
#include <nvigi.h>
#include "nvigi_gpt.h"
#include "nvigi_d3d12.h"
#include "nvigi_vulkan.h"
#include "nvigi_cloud.h"

// C++ wrappers (from shared location)
#include "cxx_wrappers/clparser.hpp"
#include "cxx_wrappers/core.hpp"
#include "cxx_wrappers/d3d12.hpp"
#include "cxx_wrappers/vulkan.hpp"
#include "cxx_wrappers/gpt/gpt.hpp"

using namespace nvigi::gpt;


int main(int argc, char** argv) {
    // We throw exceptions on errors
    try {

        // Add commands with fluent interface
        auto parser = clparser::create_parser();
        parser.add_command("", "sdk", " sdk location, if none provided assuming exe location", "", true)
            .add_command("", "plugin", " plugin location, if none provided assuming sdk location", "")
            .add_command("m", "models", " model repo location", "")
            .add_command("", "model-card", " path to model card JSON file (replaces --models + --guid)", "")
            .add_command("t", "threads", " number of threads", "1")
            .add_command("", "backend", " backend to use for local execution - d3d12, cuda, vulkan, cloud", "d3d12")
            .add_command("", "guid", " gpt model guid in registry format", "{8E31808B-C182-4016-9ED8-64804FF5B40D}")
            .add_command("", "url", " URL to use, if none provided default is taken from model JSON", "")
            .add_command("", "json", " custom JSON body for cloud request", "")
            .add_command("", "token", " authorization token for the cloud provider", "")
            .add_command("", "vram", " the amount of vram to use in MB", "8192")
            .add_command("", "cache-type", " KV cache quantization type: fp16, fp32, q4_0, q8_0", "fp16")
            .add_command("", "log-level", " logging level 0-2", "0")
            .add_command("", "print-system-info", " print system information", "")
            // Sampler parameters
            .add_command("", "top-k", " top-k sampling (<=0 to use vocab size)", "40")
            .add_command("", "min-p", " min-p sampling (0.0 = disabled)", "0.05")
            .add_command("", "penalty-repeat", " repeat penalty (1.0 = disabled)", "1.0")
            .add_command("", "penalty-freq", " frequency penalty (0.0 = disabled)", "0.0")
            .add_command("", "penalty-present", " presence penalty (0.0 = disabled)", "0.0")
            .add_command("", "penalty-last-n", " last n tokens to penalize (0 = disable, -1 = context size)", "64")
            .add_command("", "mirostat", " mirostat sampling mode (0 = disabled, 1 = mirostat, 2 = mirostat 2.0)", "0")
            .add_command("", "mirostat-tau", " mirostat target entropy", "5.0")
            .add_command("", "mirostat-eta", " mirostat learning rate", "0.1")
            .add_command("", "grammar", " BNF-like grammar to constrain sampling", "")
            .add_command("", "persistent-kv", " persistent KV cache (keeps context between calls)", "");
        parser.parse(argc, argv);

        {
            // Initialize NVIGI core
            nvigi::Core core({ .sdkPath = parser.get("sdk"), .logLevel = static_cast<nvigi::LogLevel>(parser.get_int("log-level")), .showConsole = true });

            // Print system info if requested
            if (parser.has("print-system-info")) {
                core.getSystemInfo().print();
                std::cout << std::flush;
            }

            // Default empty configs
            nvigi::vulkan::VulkanConfig vk_config{};
            nvigi::d3d12::D3D12Config d3d12_config{};
            CloudConfig cloud_config{};
            nvigi::vulkan::VulkanObjects vk_objects{};
            nvigi::d3d12::DeviceAndQueue deviceAndQueue{};

            // Initialize configs as needed
            if (parser.get("backend") == "vulkan")
            {
                vk_objects = nvigi::vulkan::VulkanHelper::create_best_compute_device();
                vk_config = {
                    .instance = vk_objects.instance,
                    .physical_device = vk_objects.physical_device,
                    .device = vk_objects.device,
                    .compute_queue = vk_objects.compute_queue,
                    .transfer_queue = vk_objects.transfer_queue,
                    .allocate_memory_callback = nvigi::vulkan::default_allocate_memory,
                    .free_memory_callback = nvigi::vulkan::default_free_memory
                };
            }
            else if (parser.get("backend") == "cloud")
            {
                cloud_config = {
                    .url = parser.get("url"),       // optional, default URL is stored in model JSON
                    .token = parser.get("token"),   // mandatory
                    .verbose = false,
                    .streaming = true
                };
            }
            else // d3d12 or cuda
            {
                // NOTE: CUDA backend still requires D3D12 device to demonstrate proper setup for CIG
                deviceAndQueue = nvigi::d3d12::D3D12Helper::create_best_compute_device();
                d3d12_config = {
                    .device = deviceAndQueue.device.Get(),
                    .compute_queue = deviceAndQueue.compute_queue.Get(),
                    .create_committed_resource_callback = nvigi::d3d12::default_create_committed_resource,
                    .destroy_resource_callback = nvigi::d3d12::default_destroy_resource,
                    .create_resource_user_context = nullptr,
                    .destroy_resource_user_context = nullptr
                };
            }

            // ===================================================================
            // Model Card JSON Support
            // ===================================================================
            // NVIGI supports two ways to specify which model to load:
            //
            // 1. Traditional approach (--models + --guid):
            //    Provide a path to the model repository and a GUID that identifies
            //    the specific model in the registry. The SDK locates and loads the
            //    model using its built-in file system access.
            //
            // 2. Model card JSON approach (--model-card):
            //    Provide a JSON file that contains all model metadata including
            //    file paths, configuration, and capabilities. This approach gives
            //    the host application full control over model discovery and loading.
            //    When using this approach, --models and --guid are ignored because
            //    the JSON already contains all the information the SDK needs.
            //
            // IMPORTANT constraint on FileIOCallbacks:
            //    Custom IO callbacks (FileIOCallbacks) can ONLY be used together
            //    with the model card JSON approach. When using the traditional
            //    --models + --guid approach, FileIOCallbacks MUST be nullptr
            //    because the SDK manages file I/O internally in that mode.
            //    Passing FileIOCallbacks without a model card JSON is not supported.
            // ===================================================================

            // Read the model card JSON file if the --model-card option was provided.
            // parser.get_file() reads the file at the given path and caches its contents.
            // The returned string_view remains valid for the lifetime of the parser.
            std::string model_card_json_contents;
            if (parser.is_set("model-card")) {
                model_card_json_contents = std::string(parser.get_file("model-card"));
                std::cout << "Model card JSON loaded from: " << parser.get("model-card") << "\n";
                std::cout << "Model card size: " << model_card_json_contents.size() << " bytes\n";
            } else if (!parser.is_set("models")) {
                // Neither --model-card nor --models was provided — cannot proceed.
                // One of these two approaches must be used to identify the model.
                throw std::runtime_error(
                    "Either --models (with --guid) or --model-card must be provided.\n"
                    "  --models/-m <path>      Traditional: model repository path + GUID\n"
                    "  --model-card <path>     Alternative: model card JSON file (replaces --models + --guid)"
                );
            }

            // ===================================================================
            // File IO Callbacks Configuration
            // ===================================================================
            // FileIOCallbacks allow the host to intercept all file operations
            // (open, read, seek, close, etc.) that the SDK performs when loading
            // model data. This enables custom loading strategies such as:
            //   - Streaming from network/cloud storage
            //   - Reading from encrypted or compressed archives
            //   - Loading from memory buffers or databases
            //   - Logging/auditing which files are accessed
            //
            // CRITICAL: FileIOCallbacks are ONLY valid when a model card JSON is
            // provided. Without a model card, the SDK uses its own internal file
            // system layer and does not call FileIOCallbacks. Passing non-null
            // FileIOCallbacks without a model card JSON results in undefined behavior.
            //
            // When io_config has enable_custom_io = false (the default), the helper
            // configure_io_callbacks() skips chaining FileIOCallbacks entirely,
            // which is equivalent to passing nullptr to the SDK.
            // ===================================================================
            nvigi::io::IOConfig io_config{};  // Default: no custom IO → FileIOCallbacks = nullptr
            if (!model_card_json_contents.empty()) {
                // Model card JSON mode: provide FileIOCallbacks so the SDK can
                // load model files through our file system wrapper.
                // create_file_wrapper_io() wraps standard FILE* operations and
                // accepts an optional open callback for logging/filtering.
                io_config = nvigi::io::create_file_wrapper_io(
                    [](const char* path) {
                        std::cout << "  [IO] Opening: " << path << "\n";
                        return true;  // Return false to reject/block a file open
                    }
                );
            }
            // When model_card_json_contents is empty (traditional mode), io_config
            // stays default-initialized with enable_custom_io = false. This causes
            // configure_io_callbacks() to return early without chaining any
            // FileIOCallbacks, effectively passing nullptr to the SDK.

            // Create GPT instance.
            // ModelConfig supports two mutually exclusive model specification modes:
            //   - Traditional (guid + model_path): SDK resolves the model internally
            //   - Model card JSON (model_card_json): host provides full model metadata
            // When model_card_json is non-empty, the wrapper automatically ignores
            // guid and model_path (sets them to empty in CommonCreationParameters).
            std::unique_ptr<Instance> instance;
            instance = Instance::create(
                ModelConfig{
                    .backend = parser.get("backend"),
                    // guid and model_path are used in traditional mode. When model_card_json
                    // is provided, these are overridden to empty by the wrapper internally.
                    .guid = parser.get("guid"),
                    .model_path = parser.get("models"),
                    // model_card_json: when non-empty, this JSON string replaces the
                    // traditional guid + model_path approach. The JSON contains all
                    // model metadata (file paths, parameters, capabilities) that the
                    // SDK needs. CommonCreationParameters.modelCardJSON is set to this
                    // value, and modelGUID / utf8PathToModels are cleared.
                    .model_card_json = model_card_json_contents,
                    .context_size = 4096,
                    .num_threads = parser.get_int("threads"),
                    .vram_budget_mb = parser.get_size_t("vram"),
                    .flash_attention = true,
                    .cache_type = parser.get("cache-type")
                },
                d3d12_config,
                vk_config,
                cloud_config,
                // io_config: contains active FileIOCallbacks ONLY when model card JSON
                // is set. In traditional mode (guid + model_path), this is a default
                // IOConfig with enable_custom_io=false → no FileIOCallbacks (nullptr).
                io_config,
                core.loadInterface(),
                core.unloadInterface(),
                parser.get("plugin")
            ).value();

            // Configure sampler parameters from command-line options
            SamplerConfig sampler;
            sampler.set_top_k(parser.get_int("top-k"))
                   .set_min_p(parser.get_float("min-p"))
                   .set_penalty_repeat(parser.get_float("penalty-repeat"))
                   .set_penalty_freq(parser.get_float("penalty-freq"))
                   .set_penalty_present(parser.get_float("penalty-present"))
                   .set_penalty_last_n(parser.get_int("penalty-last-n"))
                   .set_mirostat(parser.get_int("mirostat"))
                   .set_mirostat_tau(parser.get_float("mirostat-tau"))
                   .set_mirostat_eta(parser.get_float("mirostat-eta"))
                   .set_persistent_kv_cache(parser.has("persistent-kv"));

            // Set grammar if provided
            if (!parser.get("grammar").empty()) {
                sampler.set_grammar(parser.get("grammar"));
            }

            // Print sampler configuration
            std::cout << "\n=== Sampler Configuration ===\n";
            std::cout << std::format("Top-K: {}\n", sampler.top_k);
            std::cout << std::format("Min-P: {}\n", sampler.min_p);
            std::cout << std::format("Penalty Repeat: {}\n", sampler.penalty_repeat);
            std::cout << std::format("Penalty Frequency: {}\n", sampler.penalty_freq);
            std::cout << std::format("Penalty Presence: {}\n", sampler.penalty_present);
            std::cout << std::format("Penalty Last N: {}\n", sampler.penalty_last_n);
            std::cout << std::format("Mirostat: {}\n", sampler.mirostat);
            if (sampler.mirostat > 0) {
                std::cout << std::format("Mirostat TAU: {}\n", sampler.mirostat_tau);
                std::cout << std::format("Mirostat ETA: {}\n", sampler.mirostat_eta);
            }
            std::cout << std::format("Persistent KV Cache: {}\n", sampler.persistent_kv_cache ? "enabled" : "disabled");
            if (!sampler.grammar.empty()) {
                std::cout << std::format("Grammar: {}\n", sampler.grammar);
            }
            std::cout << "============================\n\n";

            // Simple chat example, start chat with specific parameters including sampler config
            RuntimeConfig runtime_config;
            runtime_config.set_tokens(256)
                          .set_batch_size(2048)
                          .set_temperature(0.3f)
                          .set_top_p(0.8f)
                          .set_interactive(true)
                          .set_reverse_prompt("\nAssistant:")
                          .set_sampler(sampler)  // Attach our sampler configuration
                          .set_json_body(parser.get_file_or("json", ""));  // Optional custom JSON body for cloud requests

            auto chat = instance->create_chat(runtime_config);

            // Start operation (doesn't block!)
            auto op = chat.send_message_polled(
                { .role = Instance::Chat::Message::Role::System,
                  .content = "You are a helpful AI assistant who answers questions concisely."
                }
            ).value();

            // Example of polling results while your game is running in a loop
            bool game_running = true;
            while (game_running) {
                // Poll for tokens (non-blocking - returns immediately!)
                if (auto result = op.try_get_results()) {
                    std::cout << result->tokens; // Show immediately!

                    // Can inspect state for debugging or control flow
                    if (result->state == ExecutionState::Done) {
                        std::cout << "\n[System prompt inference complete]";
                    }
                    else if (result->state == ExecutionState::Cancel) {
                        std::cout << "\n[System prompt inference cancelled]";
                    }
                }

                // Check if done
                if (op.is_complete()) {
                    chat.finalize_async_response(op);
                    op.reset(); // Optional here since we're breaking, but good practice if reusing variable
                    break;
                }

                // For example, game continues running smoothly
                // 
                // render_frame();      // Renders every frame at 60 FPS
                // update_physics();    // Physics keeps running
                // process_input();     // Player can still move
                //
                // Example cancellation:
                // if (user_pressed_cancel) {
                //     op.cancel();     // Sets flag; next try_get_results() signals cancellation to engine
                // }
            }

            // User interaction
            std::string input;
            while (true) {
                std::cout << "\nUser> ";
                std::getline(std::cin, input);

                if (input == "quit" || input == "exit")
                    break;

                // Blocking call, will return when the message is complete
                chat.send_message(
                    { .role = Instance::Chat::Message::Role::User,
                      .content = input
                    },
                    [](std::string_view response, ExecutionState state) -> ExecutionState {
                        // Callback will be called for each chunk of the response with current state
                        std::cout << response;

                        // Example: Cancel if needed
                        // if (some_condition) {
                        //     return ExecutionState::Cancel;
                        // }

                        // Return state to continue normally, or ExecutionState::Cancel to stop
                        return state;
                    }
                );
            }
        }
        
        // At this point nvigi::Core is destroyed and all resources are released

        // Print memory tracking statistics before exit
        if (parser.get("backend") == "d3d12") {
            std::cout << "\n\n=== Memory Tracking Statistics ===\n";
            std::cout << std::format("D3D12 Resources Active: {}\n", nvigi::d3d12::g_resource_count.load());
            std::cout << std::format("D3D12 Total Bytes Allocated: {} MB\n", 
                nvigi::d3d12::g_total_allocation_bytes.load() / (1024.0 * 1024.0));
        } else if (parser.get("backend") == "vulkan") {
            std::cout << "\n\n=== Memory Tracking Statistics ===\n";
            std::cout << std::format("Vulkan Allocations Active: {}\n", nvigi::vulkan::g_memory_allocation_count.load());
            std::cout << std::format("Vulkan Total Bytes Allocated: {} MB\n", 
                nvigi::vulkan::g_total_allocation_bytes.load() / (1024.0 * 1024.0));
        }
        std::cout << std::flush;
    }
    catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << std::flush;
        return -1;
    }

    return 0;
}