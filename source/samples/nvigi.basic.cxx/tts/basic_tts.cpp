// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <iostream>
#include <format>
#include <chrono>
#include <thread>

// Windows and D3D12 Agility SDK exports
#include <windows.h>
extern "C" __declspec(dllexport) UINT         D3D12SDKVersion = 615;
extern "C" __declspec(dllexport) const char* D3D12SDKPath = ".\\D3D12\\";

// NVIGI includes
#include <nvigi.h>
#include <nvigi_tts.h>
#include <nvigi_d3d12.h>
#include <nvigi_vulkan.h>
#include <nvigi_stl_helpers.h>

// C++ wrappers
#include "clparser.hpp"
#include "core.hpp"
#include "d3d12.hpp"
#include "vulkan.hpp"
#include "tts.hpp"

using namespace nvigi::tts;

int main(int argc, char** argv) {
    try {
        // Parse command line arguments
        auto parser = clparser::create_parser();
        parser.add_command("", "sdk", " sdk location, if none provided assuming exe location", "", true)
            .add_command("", "plugin", " plugin location, if none provided assuming sdk location", "")
            .add_command("m", "models", " model repo location", "", true)
            .add_command("t", "threads", " number of threads", "8")
            .add_command("", "backend", " backend to use - d3d12, cuda, vulkan", "d3d12")
            .add_command("", "guid", " TTS model guid in registry format", "{16EEB8EA-55A8-4F40-BECE-CE995AF44101}")
            .add_command("", "vram", " the amount of vram to use in MB", "2048")
            .add_command("", "log-level", " logging level 0-2", "0")
            .add_command("", "text", " text to synthesize", "Hello! This is a test of the text to speech system.")
            .add_command("", "target", " path to target voice spectrogram", "", true)
            .add_command("", "output", " output WAV file path", "output.wav")
            .add_command("", "speed", " speech speed (0.5 - 2.0)", "1.0")
            .add_command("", "language", " language code (en, en-us, en-uk, es, de)", "en")
            .add_command("", "timesteps", " number of timesteps for TTS inference (16-32)", "16")
            .add_command("", "async", " use async mode (polled, non-blocking)")
            .add_command("", "play", " play audio in real-time using DirectSound")
            .add_command("", "print-system-info", " print system information", "");
        parser.parse(argc, argv);

        {
            // Initialize NVIGI core
            nvigi::Core core({ 
                .sdkPath = parser.get("sdk"), 
                .logLevel = static_cast<nvigi::LogLevel>(parser.get_int("log-level")), 
                .showConsole = true 
            });

            // Print system info if requested
            if (parser.has("print-system-info")) {
                core.getSystemInfo().print();
                std::cout << std::flush;
            }

            // Default empty configs
            nvigi::vulkan::VulkanConfig vk_config{};
            nvigi::d3d12::D3D12Config d3d12_config{};
            nvigi::vulkan::VulkanObjects vk_objects{};
            nvigi::d3d12::DeviceAndQueue deviceAndQueue{};

            // Initialize configs as needed
            if (parser.get("backend") == "vulkan") {
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
            } else {
                // D3D12 or CUDA
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

            // Create TTS instance
            std::cout << "\n=== Creating TTS Instance ===\n";
            std::unique_ptr<Instance> instance;
            instance = Instance::create(
                ModelConfig{
                    .backend = parser.get("backend"),
                    .guid = parser.get("guid"),
                    .model_path = parser.get("models"),
                    .num_threads = parser.get_int("threads"),
                    .vram_budget_mb = parser.get_size_t("vram"),
                    .warm_up_models = true
                },
                d3d12_config,
                vk_config,
                core.loadInterface(),
                core.unloadInterface(),
                parser.get("plugin")
            ).value();

            std::cout << "TTS instance created successfully!\n";
            std::cout << "\n=== NVIGI TTS Basic Sample ===\n";
            std::cout << "Backend: " << parser.get("backend") << "\n";
            std::cout << "Model GUID: " << parser.get("guid") << "\n";
            
            // Print supported languages
            auto supported_langs = instance->get_supported_languages();
            if (!supported_langs.empty()) {
                std::cout << "Supported Languages: ";
                for (size_t i = 0; i < supported_langs.size(); ++i) {
                    std::cout << supported_langs[i];
                    if (i < supported_langs.size() - 1) std::cout << ", ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";

            std::string text = parser.get("text").data();
            std::string target_path = parser.get("target").data();
            std::string output_path = parser.get("output").data();
            float speed = parser.get_float("speed");
            std::string language = parser.get("language").data();
            int timesteps = parser.get_int("timesteps");
            bool play_audio = parser.has("play");

            std::cout << "Text: \"" << text << "\"\n";
            std::cout << "Target Voice: " << target_path << "\n";
            std::cout << "Output: " << output_path << "\n";
            std::cout << "Speed: " << speed << "x\n";
            std::cout << "Language: " << language << "\n";
            std::cout << "Timesteps: " << timesteps << "\n";
#ifdef NVIGI_WINDOWS
            std::cout << "Real-time Playback: " << (play_audio ? "Enabled" : "Disabled") << "\n";
#endif
            std::cout << "\n";

            // Configure runtime parameters
            auto config = RuntimeConfig{}
                .set_speed(speed)
                .set_language(language)
                .set_timesteps(timesteps)
                .set_flash_attention(true);

            bool use_async = parser.has("async");

            if (use_async) {
                std::cout << "=== Async Mode (Polled, Non-blocking) ===\n";
                std::cout << "Starting async speech generation...\n";

                // Start async operation
                auto op = instance->generate_async(text, target_path, config).value();

                // Create WAV writer
                WAVWriter wav_writer(output_path);
                if (!wav_writer.is_open()) {
                    std::cerr << "Failed to create output WAV file: " << output_path << "\n";
                    return -1;
                }

                size_t total_samples = 0;
                auto start_time = std::chrono::steady_clock::now();

                // Poll for results in a game-loop style
                std::cout << "Generating";
                std::cout.flush();

                while (!op.is_complete()) {
                    // Try to get results (non-blocking!)
                    if (auto result = op.try_get_results()) {
                        if (!result->audio.empty()) {
                            // Write audio chunk to file
                            wav_writer.write_samples(result->audio.data(), result->audio.size());
                            total_samples += result->audio.size();
                            
#ifdef NVIGI_WINDOWS
                            // Play audio in real-time if requested
                            if (play_audio) {
                                AudioPlayer::play_audio(result->audio.data(), result->audio.size());
                            }
#endif
                            
                            if (result->state == ExecutionState::DataPending) {
                                std::cout << "." << std::flush;
                            }
                        }
                        
                        if (result->state == ExecutionState::Done) {
                            std::cout << " Done!\n";
                        } else if (result->state == ExecutionState::Invalid) {
                            std::cerr << "\nError during speech generation!\n";
                            return -1;
                        }
                    }

                    // Simulate game loop - do other work here
                    // In a real game, you'd be rendering frames, updating physics, etc.
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }

                wav_writer.close();

                auto end_time = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

                std::cout << "\n=== Generation Complete ===\n";
                std::cout << "Total Samples: " << total_samples << "\n";
                std::cout << "Duration: " << (total_samples / static_cast<float>(kSampleRate)) << " seconds\n";
                std::cout << "Generation Time: " << (duration.count() / 1000.0f) << " seconds\n";
                std::cout << "Output saved to: " << output_path << "\n";

            } else {
                std::cout << "=== Sync Mode (Blocking) ===\n";
                std::cout << "Starting speech generation...\n";

                // Create WAV writer
                WAVWriter wav_writer(output_path);
                if (!wav_writer.is_open()) {
                    std::cerr << "Failed to create output WAV file: " << output_path << "\n";
                    return -1;
                }

                size_t total_samples = 0;
                auto start_time = std::chrono::steady_clock::now();

                std::cout << "Generating";
                std::cout.flush();

                // Generate speech (blocking call)
                auto result = instance->generate(
                    text,
                    target_path,
                    config,
                    [&wav_writer, &total_samples, play_audio](const int16_t* audio, size_t samples, ExecutionState state) -> ExecutionState {
                        if (state == ExecutionState::DataPending || state == ExecutionState::Done) {
                            // Write audio chunk to file
                            wav_writer.write_samples(audio, samples);
                            total_samples += samples;
                            
#ifdef NVIGI_WINDOWS
                            // Play audio in real-time if requested
                            if (play_audio) {
                                AudioPlayer::play_audio(audio, samples);
                            }
#endif
                            
                            if (state == ExecutionState::DataPending) {
                                std::cout << "." << std::flush;
                            }
                        }
                        return state; // Continue normally
                    }
                );

                wav_writer.close();

                auto end_time = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

                if (!result) {
                    std::cerr << "\nError: " << result.error().what() << "\n";
                    return -1;
                }

                std::cout << " Done!\n";
                std::cout << "\n=== Generation Complete ===\n";
                std::cout << "Total Samples: " << total_samples << "\n";
                std::cout << "Duration: " << (total_samples / static_cast<float>(kSampleRate)) << " seconds\n";
                std::cout << "Generation Time: " << (duration.count() / 1000.0f) << " seconds\n";
                std::cout << "Output saved to: " << output_path << "\n";
            }

            std::cout << "\n";
        }
        
        // Print memory statistics
        if (parser.get("backend") == "d3d12") {
            std::cout << "=== Memory Tracking Statistics ===\n";
            std::cout << std::format("D3D12 Resources Active: {}\n", nvigi::d3d12::g_resource_count.load());
            std::cout << std::format("D3D12 Total Bytes Allocated: {:.2f} MB\n", 
                nvigi::d3d12::g_total_allocation_bytes.load() / (1024.0 * 1024.0));
        } else if (parser.get("backend") == "vulkan") {
            std::cout << "=== Memory Tracking Statistics ===\n";
            std::cout << std::format("Vulkan Allocations Active: {}\n", nvigi::vulkan::g_memory_allocation_count.load());
            std::cout << std::format("Vulkan Total Bytes Allocated: {:.2f} MB\n", 
                nvigi::vulkan::g_total_allocation_bytes.load() / (1024.0 * 1024.0));
        }
    }
    catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

