// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <iostream>
#include <fstream>
#include <format>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstring>
#include <optional>

// Windows and D3D12 Agility SDK exports
#include <windows.h>
#include <mmeapi.h>
extern "C" __declspec(dllexport) UINT         D3D12SDKVersion = 615;
extern "C" __declspec(dllexport) const char* D3D12SDKPath = ".\\D3D12\\";

// NVIGI includes
#include <nvigi.h>
#include "nvigi_asr_whisper.h"
#include "nvigi_d3d12.h"
#include "nvigi_vulkan.h"

// C++ wrappers
#include "clparser.hpp"
#include "core.hpp"
#include "d3d12.hpp"
#include "vulkan.hpp"
#include "asr.hpp"
#include "audio.hpp"

using namespace nvigi::asr;

int main(int argc, char** argv) {
    try {
        // Parse command line arguments
        auto parser = clparser::create_parser();
        parser.add_command("", "sdk", " sdk location, if none provided assuming exe location", "", true)
            .add_command("", "plugin", " plugin location, if none provided assuming sdk location", "")
            .add_command("m", "models", " model repo location", "", true)
            .add_command("t", "threads", " number of threads", "8")
            .add_command("fa", "flash-attention", " use flash attention")
            .add_command("", "backend", " backend to use - d3d12, cuda, vulkan", "d3d12")
            .add_command("", "guid", " ASR model guid in registry format", "{5CAD3A03-1272-4D43-9F3D-655417526170}")
            .add_command("", "vram", " the amount of vram to use in MB", "2048")
            .add_command("", "log-level", " logging level 0-2", "0")
            .add_command("", "language", " language code (en, es, fr, auto, etc.)", "en")
            .add_command("", "detect-lang", " auto-detect language")
            .add_command("", "translate", " translate to English")
            .add_command("", "streaming", " use streaming mode (experimental)")
            .add_command("", "save-wav", " save audio to file", "c:/test.wav")
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

            bool use_streaming = parser.has("streaming");

            // Create ASR instance
            std::unique_ptr<Instance> instance;
            instance = Instance::create(
                ModelConfig{
                    .backend = parser.get("backend"),
                    .guid = parser.get("guid"),
                    .model_path = parser.get("models"),
                    .num_threads = parser.get_int("threads"),
                    .vram_budget_mb = parser.get_size_t("vram"),
                    .flash_attention = parser.has("flash-attention"),
                    .language = parser.get("language"),
                    .translate = parser.has("translate"),
                    .detect_language = parser.has("detect-lang"),   
                },
                d3d12_config,
                vk_config,
                core.loadInterface(),
                core.unloadInterface(),
                parser.get("plugin")
            ).value();

            std::cout << "\n=== NVIGI ASR Basic Sample ===\n";
            std::cout << "Backend: " << parser.get("backend") << "\n";
            std::cout << "Language: " << parser.get("language") << "\n";
            std::cout << "Model GUID: " << parser.get("guid") << "\n\n";

            if (use_streaming) {
                std::cout << "=== Real-Time Streaming Mode ===\n";
                std::cout << "This mode demonstrates continuous real-time audio streaming.\n";
                std::cout << "Audio is captured and transcribed continuously in small chunks.\n";
                std::cout << "Note: Streaming is experimental and may not work with all models.\n\n";

                // Create a stream for audio processing
                auto stream = instance->create_stream(
                    RuntimeConfig{}
                        .set_sampling(SamplingStrategy::Greedy)
                        .set_temperature(0.0f)
                        .set_best_of(2)
                );

                std::cout << "Commands:\n";
                std::cout << "  Type 'stream' to start real-time streaming\n";
                std::cout << "  Type 'quit' or 'exit' to quit\n\n";

                std::string input;
                while (true) {
                    std::cout << "> ";
                    std::getline(std::cin, input);

                    if (input == "quit" || input == "exit") {
                        break;
                    }

                    if (input == "stream") {
                        std::cout << "\n=== Starting Real-Time Streaming ===\n";
                        std::cout << "Speak into your microphone. Press Enter to stop.\n";
                        
                        // Current approach: accumulation mode (backward compatible)
                        auto* recording = AudioRecorder::StartRecording();
                        if (!recording) {
                            std::cerr << "Failed to start recording!\n";
                            std::cerr << "Please check your microphone settings in Windows.\n";
                            continue;
                        }

                        // Create WAV file to dump chunks for verification
                        std::string wav_filename = parser.has("save-wav") ? parser.get("save-wav").data() : "";
                        WavWriter::WavFile wav_file(wav_filename);
                        if (parser.has("save-wav"))
                        {
                            if (!wav_file.is_open()) {
                                std::cerr << "Warning: Failed to create WAV file for chunk dumping.\n";
                            }
                            else {
                                std::cout << "Recording chunks to: " << wav_filename << "\n\n";
                            }
                        }

                        // Audio format: 16000 Hz, 16-bit, mono = 32000 bytes/second
                        const size_t chunk_size = size_t(0.2f * 32000); // 200ms
                        
                        std::atomic<bool> stop_streaming{false};
                        std::thread stop_thread([&stop_streaming]() {
                            std::cin.get(); // Wait for Enter key
                            stop_streaming = true;
                        });

                        std::cout << "Transcription: ";
                        std::cout.flush();

                        size_t chunk_count = 0;
                        size_t bytes_processed = 0;
                        std::optional< Instance::AsyncOperation> async_op;
                        std::string accumulated_text;
                        while (!stop_streaming) {
                            // Poll stream for results (manages operations internally!)
                            if (async_op.has_value()) {
                                if (auto result = async_op->try_get_results())
                                {
                                    if (!result->text.empty() && result->text != "[BLANK_AUDIO]") {
                                        if (result->state == ExecutionState::DataPartial)
                                        {
                                            printf("\33[2K\r");
                                            // print long empty line to clear the previous line
                                            printf("%s", std::string(result->text.size(), ' ').c_str());
                                            printf("\33[2K\r");
                                        }
                                        std::cout << result->text << std::flush;
                                        
                                        if (result->state != ExecutionState::DataPartial) {
                                            accumulated_text += result->text;
                                        }
                                    }
                                }
                            }
                            
                            // Send new chunk if available
                            std::vector<uint8_t> chunk_data;
                            
                            {
                                // Thread-safe: Read current size and copy chunk data
                                std::lock_guard<std::mutex> lock(recording->mutex);
                                size_t current_size = recording->bytesWritten;
                                
                                // Check if we have new data
                                if (current_size > bytes_processed) {
                                    size_t available = current_size - bytes_processed;
                                    
                                    // Only process if we have at least chunk_size of new data
                                    if (available >= chunk_size) {
                                        // Copy the new chunk while holding the lock
                                        chunk_data.resize(chunk_size);
                                        memcpy(chunk_data.data(), 
                                               recording->audioBuffer.data() + bytes_processed, 
                                               chunk_size);
                                        bytes_processed += chunk_size;
                                    }
                                }
                            }
                            // Lock released
                            
                            if (!chunk_data.empty()) {
                                bool is_first = (chunk_count == 0);
                                
                                // Write chunk to WAV file for verification
                                if (parser.has("save-wav") && wav_file.is_open()) {
                                    wav_file.write_samples(chunk_data.data(), chunk_data.size());
                                }
                                
                                // Send chunk to ASR (non-blocking, managed internally!)
                                async_op = stream.send_audio_async(
                                    chunk_data.data(),
                                    chunk_data.size(),
                                    is_first,
                                    false  // Not last chunk (streaming continues)
                                ).value();

                                chunk_count++;
                            }
                            
                            // Small sleep to avoid busy-wait
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        }

                        // Send final signal and remaining audio
                        auto audioData = AudioRecorder::StopRecording(recording);
                        
                        // Explicitly close and finalize the WAV file
                        if (parser.has("save-wav") && wav_file.is_open()) {
                            wav_file.close();
                        }
                        
                        stop_thread.join();
                        std::cout << "\n\n[Streaming complete]\n";
                        std::cout << "Processed " << chunk_count << " chunks, total " 
                                  << audioData.size() << " bytes\n";
                        std::cout << "Final transcription: " << accumulated_text << "\n\n";
                        
                    } else {
                        std::cout << "Unknown command. Type 'stream' to start streaming, or 'quit' to exit.\n";
                    }
                }
            } else {
                std::cout << "=== Complete Audio Mode ===\n";
                std::cout << "This mode processes complete audio clips after recording finishes.\n\n";

                std::cout << "Commands:\n";
                std::cout << "  Type 'record' to start recording (5 seconds)\n";
                std::cout << "  Type 'quit' or 'exit' to quit\n\n";

                std::string input;
                while (true) {
                    std::cout << "> ";
                    std::getline(std::cin, input);

                    if (input == "quit" || input == "exit") {
                        break;
                    }

                    if (input == "record") {
                        std::cout << "\nStarting recording... (speak now)\n";
                        
                        auto* recording = AudioRecorder::StartRecording();
                        if (!recording) {
                            std::cerr << "Failed to start recording!\n";
                            std::cerr << "Please check your microphone settings in Windows.\n";
                            continue;
                        }

                        std::cout << "Recording for 5 seconds...\n";
                        std::this_thread::sleep_for(std::chrono::seconds(5));

                        auto audioData = AudioRecorder::StopRecording(recording);
                        
                        if (audioData.empty()) {
                            std::cout << "No audio recorded.\n";
                            continue;
                        }

                        std::cout << "Recorded " << audioData.size() << " bytes\n";
                        std::cout << "Transcribing...\n\n";

                        // Transcribe complete audio (blocking)
                        std::string transcription;
                        auto result = instance->transcribe(
                            audioData.data(),
                            audioData.size(),
                            RuntimeConfig{}
                                .set_sampling(SamplingStrategy::Greedy)
                                .set_temperature(0.0f),
                            [&transcription](std::string_view text, ExecutionState state) -> ExecutionState {
                                if (state == ExecutionState::DataPending || state == ExecutionState::Done) {
                                    transcription += text;
                                    std::cout << text << std::flush;
                                }
                                return state;
                            }
                        );

                        if (!result) {
                            std::cerr << "\nError: " << result.error().what() << "\n";
                        } else {
                            std::cout << "\n\n[Transcription complete]\n";
                            std::cout << "Final text: " << transcription << "\n";
                        }

                        std::cout << "\n";
                    } else {
                        std::cout << "Unknown command. Type 'record', 'async', or 'quit'.\n";
                    }
                }
            }
        }
        
        // Print memory statistics
        if (parser.get("backend") == "d3d12") {
            std::cout << "\n=== Memory Tracking Statistics ===\n";
            std::cout << std::format("D3D12 Resources Active: {}\n", nvigi::d3d12::g_resource_count.load());
            std::cout << std::format("D3D12 Total Bytes Allocated: {:.2f} MB\n", 
                nvigi::d3d12::g_total_allocation_bytes.load() / (1024.0 * 1024.0));
        } else if (parser.get("backend") == "vulkan") {
            std::cout << "\n=== Memory Tracking Statistics ===\n";
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

