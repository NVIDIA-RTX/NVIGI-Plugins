// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// C++ wrappers
#include "clparser.hpp"
#include "core.hpp"
#include "d3d12.hpp"
#include "vulkan.hpp"
#include "gpt.hpp"

using namespace nvigi::gpt;


int main(int argc, char** argv) {
    // We throw exceptions on errors
    try {

        // Add commands with fluent interface
        auto parser = clparser::create_parser();
        parser.add_command("", "sdk", " sdk location, if none provided assuming exe location", "", true)
            .add_command("", "plugin", " plugin location, if none provided assuming sdk location", "")
            .add_command("m", "models", " model repo location", "", true)
            .add_command("t", "threads", " number of threads", "1")
            .add_command("", "backend", " backend to use for local execution - d3d12, cuda, vulkan, cloud", "d3d12")
            .add_command("", "guid", " gpt model guid in registry format", "{8E31808B-C182-4016-9ED8-64804FF5B40D}")
            .add_command("", "url", " URL to use, if none provided default is taken from model JSON", "")
            .add_command("", "json", " custom JSON body for cloud request", "")
            .add_command("", "token", " authorization token for the cloud provider", "")
            .add_command("", "vram", " the amount of vram to use in MB", "8192")
            .add_command("", "cache-type", " KV cache quantization type: fp16, fp32, q4_0, q8_0", "fp16")
            .add_command("", "log-level", " logging level 0-2", "0")
            .add_command("", "print-system-info", " print system information", "");
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
                    .command_queue = deviceAndQueue.compute_queue.Get(),
                    .create_committed_resource_callback = nvigi::d3d12::default_create_committed_resource,
                    .destroy_resource_callback = nvigi::d3d12::default_destroy_resource,
                    .create_resource_user_context = nullptr,
                    .destroy_resource_user_context = nullptr
                };
            }

            // Create GPT instance
            std::unique_ptr<Instance> instance;
            instance = Instance::create(
                ModelConfig{
                    .backend = parser.get("backend"),
                    .guid = parser.get("guid"),
                    .model_path = parser.get("models"),
                    .context_size = 4096,
                    .num_threads = parser.get_int("threads"),
                    .vram_budget_mb = parser.get_size_t("vram"),
                    .flash_attention = true,
                    .cache_type = parser.get("cache-type")
                },
                d3d12_config,
                vk_config,
                cloud_config,
                core.loadInterface(),
                core.unloadInterface(),
                parser.get("plugin") // Optional custom plugin path, by default SDK location is used
            ).value(); // Will throw if creation fails

            // Simple chat example, start chat with specific parameters
            auto chat = instance->create_chat(
                {
                    .tokens_to_predict = 256,
                    .batch_size = 2048,
                    .temperature = 0.3f,
                    .top_p = 0.8f,
                    .interactive = true,
                    .reverse_prompt = "\nAssistant:",
                    .json_body = parser.get_file_or("json", "") // Optional custom JSON body for cloud requests, empty or file contents
                });

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