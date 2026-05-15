// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// C++ wrappers (from shared location)
#include "cxx_wrappers/clparser.hpp"
#include "cxx_wrappers/core.hpp"
#include "cxx_wrappers/d3d12.hpp"
#include "cxx_wrappers/vulkan.hpp"
#include "cxx_wrappers/asr/asr.hpp"
#include "cxx_wrappers/asr/audio.hpp"

using namespace nvigi::asr;

int main(int argc, char** argv) {
    try {
        // Parse command line arguments
        auto parser = clparser::create_parser();
        parser.add_command("", "sdk", " sdk location, if none provided assuming exe location", "", true)
            .add_command("", "plugin", " plugin location, if none provided assuming sdk location", "")
            .add_command("m", "models", " model repo location", "")
            .add_command("", "model-card", " path to model card JSON file (replaces --models + --guid)", "")
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
            .add_command("", "print-system-info", " print system information", "")
            .add_command("", "length", " length of audio segments in milliseconds", "5000")
            .add_command("", "step", " step size for streaming mode in milliseconds", "500")
            .add_command("", "keep", " amount of previous audio to keep as context for streaming", "200");
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
                    .compute_queue = deviceAndQueue.compute_queue.Get(),
                    .create_committed_resource_callback = nvigi::d3d12::default_create_committed_resource,
                    .destroy_resource_callback = nvigi::d3d12::default_destroy_resource,
                    .create_resource_user_context = nullptr,
                    .destroy_resource_user_context = nullptr
                };
            }

            bool use_streaming = parser.has("streaming");

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

            // Create ASR instance.
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
                    .num_threads = parser.get_int("threads"),
                    .vram_budget_mb = parser.get_size_t("vram"),
                    .flash_attention = parser.has("flash-attention"),
                    .language = parser.get("language"),
                    .translate = parser.has("translate"),
                    .detect_language = parser.has("detect-lang"),
                    .lengthMs = parser.get_int("length"),
                    .keepMs = parser.get_int("keep"),
                    .stepMs = parser.get_int("step"),
                },
                d3d12_config,
                vk_config,
                // io_config: contains active FileIOCallbacks ONLY when model card JSON
                // is set. In traditional mode (guid + model_path), this is a default
                // IOConfig with enable_custom_io=false → no FileIOCallbacks (nullptr).
                io_config,
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
                std::cout << "Important: You can experience delay or a time-out on the first run due to shader compilation.\n\n";

                std::cout << "Commands:\n";
                std::cout << "  Press enter (empty line) to start real-time streaming\n";
                std::cout << "  Type 'quit' or 'exit' to quit\n\n";

                std::string input;
                while (true) {
                    std::cout << "> ";
                    std::getline(std::cin, input);

                    if (input == "quit" || input == "exit") {
                        break;
                    }

                    if (input == "") {
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

                        // Create a stream for audio processing
                        auto stream = instance->create_stream(
                            RuntimeConfig{}
                            .set_sampling(SamplingStrategy::Greedy)
                            .set_temperature(0.0f)
                            .set_best_of(2)
                        );

                        size_t chunk_count = 0;
                        size_t bytes_processed = 0;
                        std::optional< Instance::AsyncOperation> async_op;
                        std::string accumulated_text;
                        while (!stop_streaming) {
                            // Poll stream for results (manages operations internally!)
                            if (async_op.has_value()) {
                                if (auto result = async_op->try_get_results())
                                {
                                    // Handle different execution states (matches whisper.cpp stream.cpp sliding window mode)
                                    if (!result->text.empty()) {
                                        if (result->state == ExecutionState::DataPartial)
                                        {
                                            // INTERIM RESULT (whisper.cpp prints with \r to overwrite line)
                                            // This result will be refined/updated in next iteration
                                            // DON'T accumulate yet - just display with overwrite
                                            printf("\33[2K\r");  // Clear line
                                            printf("%s", result->text.c_str());  // Print text
                                            std::cout << std::flush;
                                            // NO accumulation - this is interim!
                                        }
                                        else if (result->state == ExecutionState::DataPending)
                                        {
                                            // FINALIZED RESULT (whisper.cpp prints with \n for new line)
                                            // This phrase is complete - keep it
                                            printf("\33[2K\r");  // Clear the interim line first
                                            std::cout << result->text << std::endl;  // Print with newline
                                            accumulated_text += result->text;  // NOW accumulate
                                        }
                                    }
                                    
                                    // Done state signals stream has completely ended
                                    if (result->state == ExecutionState::Done) {
                                        // Stream completed - nothing more to do
                                        // (text already accumulated from DataPending states)
                                        break;
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
                        
                        // Send final chunk with is_last=true to signal end of stream
                        // This ensures the background thread in the plugin terminates properly
                        if (audioData.size() > bytes_processed && async_op.has_value()) {
                            // Send any remaining audio that wasn't processed in chunks
                            size_t remaining = audioData.size() - bytes_processed;
                            if (remaining > 0) {
                                async_op = stream.send_audio_async(
                                    audioData.data() + bytes_processed,
                                    remaining,
                                    chunk_count == 0,  // is_first only if we never sent any chunks
                                    true  // is_last = true to signal end of stream
                                ).value();
                            }
                        } else if (async_op.has_value()) {
                            // No remaining audio, but still need to send empty final chunk to signal stop
                            // Create small silent chunk to signal end
                            std::vector<uint8_t> final_chunk(chunk_size, 0);
                            async_op = stream.send_audio_async(
                                final_chunk.data(),
                                final_chunk.size(),
                                chunk_count == 0,  // is_first only if we never sent any chunks
                                true  // is_last = true to signal end of stream
                            ).value();
                        }
                        
                        // Drain any remaining results from the pipeline
                        if (async_op.has_value()) {
                            // Increased timeout to 15 seconds to account for:
                            // - First-run shader compilation (can take 5-10 seconds)
                            // - Long audio segments (10+ seconds of audio to process)
                            // - Model warm-up time on first inference
                            auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(15);
                            auto last_progress = std::chrono::steady_clock::now();
                            int wait_seconds = 0;
                            
                            while (!async_op->is_complete() && std::chrono::steady_clock::now() < timeout) {
                                // Show progress indicator every second if waiting
                                if (std::chrono::steady_clock::now() - last_progress > std::chrono::seconds(1)) {
                                    wait_seconds++;
                                    std::cout << "." << std::flush;
                                    last_progress = std::chrono::steady_clock::now();
                                }
                                
                                if (auto result = async_op->try_get_results()) {
                                    if (!result->text.empty()) {
                                        if (result->state == ExecutionState::DataPartial)
                                        {
                                            // INTERIM RESULT - overwrite current line
                                            printf("\33[2K\r");  // Clear line
                                            printf("%s", result->text.c_str());
                                            std::cout << std::flush;
                                            // Don't accumulate - this is interim
                                        }
                                        else if (result->state == ExecutionState::DataPending)
                                        {
                                            // FINALIZED PHRASE - print newline and accumulate
                                            printf("\33[2K\r");  // Clear interim line
                                            std::cout << result->text << std::endl;  // Print with newline
                                            accumulated_text += result->text;  // Accumulate finalized text
                                        }
                                    }
                                    
                                    // Done state signals stream completely ended
                                    if (result->state == ExecutionState::Done) {
                                        break;
                                    }
                                }
                                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            }
                            
                            if (!async_op->is_complete()) {
                                std::cerr << "\n\nWarning: Async operation did not complete within 15-second timeout!\n";
                                std::cerr << "This may indicate:\n";
                                std::cerr << "  - First-run shader compilation still in progress\n";
                                std::cerr << "  - Model is too slow for this amount of audio\n";
                                std::cerr << "  - System is under heavy load\n";
                                std::cerr << "Results may be incomplete.\n";
                            }
                        }
                        
                        // Explicitly close and finalize the WAV file
                        if (parser.has("save-wav") && wav_file.is_open()) {
                            wav_file.close();
                        }
                        
                        stop_thread.join();
                        std::cout << "\n\n[Streaming complete]\n";
                        std::cout << "Processed " << chunk_count << " chunks, total " 
                                  << audioData.size() << " bytes\n";
                        std::cout << "Final transcription: " << accumulated_text << "\n\n";
                        
                        // CRITICAL: Ensure background thread has fully stopped before next iteration
                        // Give the plugin time to clean up its state and exit the async loop
                        std::cout << "[Waiting for background cleanup to complete...]\n";
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        
                    } else {
                        std::cout << "Unknown command. Press enter to begin streaming or type 'exit'/'quit' to exit.\n";
                    }
                }
            } else {
                std::cout << "=== Complete Audio Mode ===\n";
                std::cout << "This mode processes complete audio clips after recording finishes.\n\n";

                std::cout << "Commands:\n";
                std::cout << "  Press enter (empty line) to start recording (5 seconds)\n";
                std::cout << "  Type 'quit' or 'exit' to quit\n\n";

                std::string input;
                while (true) {
                    std::cout << "> ";
                    std::getline(std::cin, input);

                    if (input == "quit" || input == "exit") {
                        break;
                    }

                    if (input == "") {
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
                        std::cout << "Unknown command. Press enter to record or type 'exit'/'quit' to quit.\n";
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

