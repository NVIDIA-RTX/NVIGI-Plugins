// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#define SDL_MAIN_HANDLED 

#ifdef GGML_USE_CUBLAS
#include "source/core/nvigi.api/nvigi_cuda.h"
#include "ggml-cuda.h"
#endif

#include "source/core/nvigi.log/log.h"
#include "external/asqflow.cpp/include/asqflow.h"
#include <mutex>
#include <queue>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <functional>

namespace nvigi
{
    namespace asqflow
    {
#ifdef GGML_USE_CUBLAS
        void setCudaMallocReportCallback(PFun_nvigiCudaReportCallback callback, void* userContext)
        {
            ggml_backend_cuda_set_malloc_report_callback(callback, userContext);
        }

        void setCudaFreeReportCallback(PFun_nvigiCudaReportCallback callback, void* userContext)
        {
            ggml_backend_cuda_set_free_report_callback(callback, userContext);
        }

        void setCudaMallocCallback(PFun_nvigiCudaMallocCallback callback, void* userContext)
        {
            ggml_backend_cuda_set_malloc_callback(callback, userContext);
        }

        void setCudaFreeCallback(PFun_nvigiCudaFreeCallback callback, void* userContext)
        {
            ggml_backend_cuda_set_free_callback(callback, userContext);
        }
#endif // GGML_USE_CUBLAS
        // Prompts buffer used when async mode is used
        class PromptsBuffer
        {
        public:
            PromptsBuffer()
            {
            }

            void clear()
            {
                std::scoped_lock lock(mtx_);
                while (!buffer_.empty())
                    buffer_.pop();
            }

            void write(const std::string prompt)
            {
                std::scoped_lock lock(mtx_);
                buffer_.push(prompt);
            }

            void read_and_pop(std::string& prompt)
            {
                std::scoped_lock lock(mtx_);
                if (buffer_.empty())
                {
                    NVIGI_LOG_ERROR("Buffer is empty");
                    return;
                }
                prompt = buffer_.front();
                buffer_.pop();
                return;
            }

            bool empty()
            {
                std::scoped_lock lock(mtx_);
                return buffer_.empty();
            }

            // Get the amount of data currently stored in the buffer
            size_t getDataSize()
            {
                std::scoped_lock lock(mtx_);
                return buffer_.size();
            }
  

        private:
            std::queue<std::string> buffer_; // The buffer of prompts
            std::mutex mtx_;
        };

        bool readSpectrogram(const std::string& path, std::vector<float>& spectrogram)
        {
            // Open the file in binary mode
            std::ifstream inputFile(path, std::ios::binary);

            // Check if the file is open
            if (!inputFile.is_open())
            {

                NVIGI_LOG_ERROR("Failed to open the spectrogram file: '%s'", path.c_str());
                return false;
            }

            // Go to the end of the file to determine its size
            inputFile.seekg(0, std::ios::end);
            std::streamsize file_size = inputFile.tellg();
            inputFile.seekg(0, std::ios::beg);

            // Calculate the number of float elements in the file
            std::size_t num_elements = file_size / sizeof(float);
            std::size_t sizeSpec = num_elements / 80;

            spectrogram.clear();
            spectrogram.reserve(num_elements);

            // Read the file data into a temporary buffer
            std::vector<float> tempBuffer(num_elements);
            if (!inputFile.read(reinterpret_cast<char*>(tempBuffer.data()), file_size))
            {
                NVIGI_LOG_ERROR("Error reading the spectrogram file: '%s'", path.c_str());
                return false;
            }

            // Rearrange the data from 1, 80, sizeSpec to sizeSpec, 80 since ggml uses the
            // inverted order
            spectrogram.resize(num_elements);
            for (std::size_t i = 0; i < sizeSpec; ++i)
            {
                for (std::size_t j = 0; j < 80; ++j)
                {
                    spectrogram[j * sizeSpec + i] = tempBuffer[j * sizeSpec + i];
                }
            }
            // Close the file
            inputFile.close();
            return true;
        }

        struct TranscriptInfo {
            std::string text;
            std::string language;
        };

        // NOTE: Supported languages are now read dynamically from model configuration files.
        // Use capabilities and requirements API to query the actual supported languages for your model.

        TranscriptInfo get_target_transcript(const std::string& spectrogram_path) {
            try {
                // Get the directory and filename from spectrogram path
                std::filesystem::path spec_path(spectrogram_path);
                std::filesystem::path spec_dir = spec_path.parent_path();
                std::string spec_filename = spec_path.filename().string();
                
                // Construct transcript.json path
                std::filesystem::path transcript_path = spec_dir / "transcripts.json";
                
                NVIGI_LOG_VERBOSE("Looking for transcript file: %s", transcript_path.string().c_str());
                
                // Check if transcript.json exists
                if (!std::filesystem::exists(transcript_path)) {
                    NVIGI_LOG_WARN("transcripts.json not found in %s", spec_dir.string().c_str());
                    return {"", ""};
                }
                
                // Read and parse the JSON file
                std::ifstream json_file(transcript_path);
                if (!json_file.is_open()) {
                    NVIGI_LOG_WARN("Cannot open transcripts.json file");
                    return {"", ""};
                }
                
                json transcript_json;
                json_file >> transcript_json;
                json_file.close();
                
                // Look for the transcript using the spectrogram filename as key
                std::string target_transcript = "";
                std::string target_language = "";
                std::string key = spec_path.stem().string(); // Remove file extension
                if (transcript_json.contains(key)) {
                    auto transcriptEntry = transcript_json[key];
                    if (transcriptEntry.is_object() && transcriptEntry.contains("text")) {
                        // New format with "text" and "language" fields
                        target_transcript = transcriptEntry["text"].get<std::string>();
                        if (transcriptEntry.contains("language")) {
                            target_language = transcriptEntry["language"].get<std::string>();                      
                        } else {
                            NVIGI_LOG_VERBOSE("Found transcript for key '%s': %s", key.c_str(), target_transcript.c_str());
                        }
                    } else if (transcriptEntry.is_string()) {
                        // Backwards compatibility for old format
                        target_transcript = transcriptEntry.get<std::string>();
                        NVIGI_LOG_VERBOSE("Found transcript for key '%s' (legacy format): %s", key.c_str(), target_transcript.c_str());
                    }
                }
                
                if (target_transcript.empty()) {
                    NVIGI_LOG_WARN("No transcript found for spectrogram file '%s'", spec_filename.c_str());
                    std::string available_keys = "Available keys in transcript.json: ";
                    for (auto it = transcript_json.begin(); it != transcript_json.end(); ++it) {
                        available_keys += "'" + it.key() + "' ";
                    }
                    NVIGI_LOG_WARN("%s", available_keys.c_str());
                }
                
                return {target_transcript, target_language};
                
            } catch (const std::exception& e) {
                NVIGI_LOG_ERROR("Error reading transcript.json: %s", e.what());
                return {"", ""};
            }
        }

        // Chunk processing callback data structure
        struct ChunkCallbackData
        {
            std::vector<int16_t>* outputAudio;
            nvigi::InferenceExecutionContext* execCtx;
            std::function<nvigi::InferenceExecutionState(const std::vector<int16_t>&, const std::string&, nvigi::InferenceExecutionState)>* callback;
            int totalChunks;
            int processedChunks;
            bool errorOccurred;
            
            ChunkCallbackData(std::vector<int16_t>* audio, nvigi::InferenceExecutionContext* ctx, 
                            std::function<nvigi::InferenceExecutionState(const std::vector<int16_t>&, const std::string&, nvigi::InferenceExecutionState)>* cb)
                : outputAudio(audio), execCtx(ctx), callback(cb), totalChunks(0), processedChunks(0), errorOccurred(false) {}
        };

        // Static callback function for chunk processing
        static void chunk_processing_callback(int chunk_index, int total_chunks, const char* chunk_text, 
                                            const float* audio_data, size_t audio_size, void* user_data)
        {
            ChunkCallbackData* callbackData = static_cast<ChunkCallbackData*>(user_data);
            if (!callbackData || callbackData->errorOccurred) return;

            try 
            {
                NVIGI_LOG_INFO("Processing chunk %d/%d: '%s' (%zu audio samples)", 
                             chunk_index + 1, total_chunks, chunk_text, audio_size);

                callbackData->totalChunks = total_chunks;
                callbackData->processedChunks = chunk_index + 1;


                // Convert float audio to int16 and append to output
                size_t currentSize = callbackData->outputAudio->size();
                callbackData->outputAudio->resize(currentSize + audio_size);
                    
                for (size_t i = 0; i < audio_size; ++i)
                {
                    float sample = std::clamp(audio_data[i], -1.0f, 1.0f);
                    (*callbackData->outputAudio)[currentSize + i] = static_cast<int16_t>(sample * 32767.0f);
                }

                // Call the inference callback for this chunk if provided
                if (callbackData->callback)
                {
                    std::vector<int16_t> chunkAudio(audio_size);
                    for (size_t i = 0; i < audio_size; ++i)
                    {
                        float sample = std::clamp(audio_data[i], -1.0f, 1.0f);
                        chunkAudio[i] = static_cast<int16_t>(sample * 32767.0f);
                    }

                    // Determine execution state
                    nvigi::InferenceExecutionState state = (chunk_index + 1 == total_chunks) 
                        ? nvigi::kInferenceExecutionStateDone 
                        : nvigi::kInferenceExecutionStateDataPending;

                    auto result = (*callbackData->callback)(chunkAudio, chunk_text, state);
                    if (result == nvigi::kInferenceExecutionStateInvalid)
                    {
                        callbackData->errorOccurred = true;
                        return;
                    }
                }
            }
            catch (const std::exception& e)
            {
                NVIGI_LOG_ERROR("Error in chunk processing callback: %s", e.what());
                callbackData->errorOccurred = true;
            }
        }

        // TTS inference function that uses pipeline with chunking support
        inline nvigi::Result performTTSInference(
            asqflow_pipeline_state* pipeline,
            const std::string& inputText,
            const std::string& spectrogramPath,
            float speechRate,
            std::vector<int16_t>& outputAudio,
            std::string& normalizedText,
            bool enableChunking = true,
            int minChunkSize = 100,
            int maxChunkSize = 200,
            std::function<nvigi::InferenceExecutionState(const std::vector<int16_t>&, const std::string&, nvigi::InferenceExecutionState)>* chunkCallback = nullptr,
            int nTimesteps = 16,
            int seed = -725171668,
            int sampler = 1,  // 0 = EULER, 1 = DPM_SOLVER_PLUS_PLUS
            int dpmpp_order = 2,
            bool use_flash_attention = true,
            const std::string& language = "en"  // Language parameter for inference
        )
        {
            if (!pipeline)
            {
                NVIGI_LOG_ERROR("Invalid pipeline provided");
                return kResultInvalidParameter;
            }

            // Log the language being used (validation is now done at runtime by the model)
            NVIGI_LOG_VERBOSE("Using language '%s' for TTS inference", language.c_str());

            try
            {
                // Get target transcript and language from the spectrogram path
                TranscriptInfo transcriptInfo = get_target_transcript(spectrogramPath);
                
                if (!transcriptInfo.text.empty())
                {
                    NVIGI_LOG_INFO("Using target transcript: '%s'", transcriptInfo.text.c_str());
                    if (!transcriptInfo.language.empty())
                    {
                        NVIGI_LOG_INFO("Target transcript language: '%s'", transcriptInfo.language.c_str());
                    }
                }
                else
                {
                    NVIGI_LOG_INFO("No target transcript found, proceeding without it");
                }
                
                // Clear output audio buffer
                outputAudio.clear();
                
                // Set up callback data for chunk processing
                ChunkCallbackData callbackData(&outputAudio, nullptr, chunkCallback);
                
                // Set up pipeline parameters for inference with chunking
                asqflow_pipeline_params params = asqflow_pipeline_default_params();
                params.input_text = inputText.c_str();
                params.target_transcript = transcriptInfo.text.c_str();
                params.spectrogram_path = spectrogramPath.c_str();
                params.speed = speechRate;
                params.n_timesteps = nTimesteps;
                params.seed = seed;
                params.sampler = static_cast<asqflow_sampler_type>(sampler);
                params.dpmpp_order = dpmpp_order;
                params.use_flash_attention = use_flash_attention;
                
                // Set language parameters
                params.language = language.c_str();
                if (!transcriptInfo.language.empty())
                {
                    params.target_transcript_language = transcriptInfo.language.c_str();
                }
                else
                {
                    params.target_transcript_language = language.c_str(); // Default to same as input language
                }
                
                // Configure chunking parameters
                params.enable_chunking = enableChunking;
                params.min_chunk_size = minChunkSize;
                params.max_chunk_size = maxChunkSize;
                params.chunk_callback = enableChunking ? chunk_processing_callback : nullptr;
                params.user_data = enableChunking ? &callbackData : nullptr;

                NVIGI_LOG_VERBOSE("TTS parameters: speed=%.2f, n_timesteps=%d, seed=%d, sampler=%d, dpmpp_order=%d, flash_attention=%s",
                                speechRate, nTimesteps, seed, sampler, dpmpp_order, use_flash_attention ? "true" : "false");

                if (enableChunking)
                {
                    NVIGI_LOG_INFO("Running pipeline with chunking enabled (min: %d, max: %d)", minChunkSize, maxChunkSize);
                    
                    // Run pipeline inference with chunking - audio will be accumulated via callback
                    float* audioFloat = nullptr;
                    size_t audioSize = 0;
                    
                    bool success = asqflow_pipeline_inference(pipeline, &params, &audioFloat, &audioSize);
                    
                    // Free any returned audio buffer (we're using chunked callback instead)
                    if (audioFloat)
                    {
                        asqflow_free_buffer(audioFloat);
                    }
                    
                    if (!success || callbackData.errorOccurred)
                    {
                        NVIGI_LOG_ERROR("Pipeline inference with chunking failed");
                        return kResultInvalidState;
                    }
                    
                    NVIGI_LOG_INFO("Pipeline chunked inference completed: %d chunks processed, %zu total audio samples", 
                                 callbackData.processedChunks, outputAudio.size());
                }
                else
                {
                    NVIGI_LOG_INFO("Running pipeline without chunking");
                    
                    // Run pipeline inference without chunking - get all audio at once
                    float* audioFloat = nullptr;
                    size_t audioSize = 0;
                    
                    bool success = asqflow_pipeline_inference(pipeline, &params, &audioFloat, &audioSize);
                    if (!success)
                    {
                        NVIGI_LOG_ERROR("Pipeline inference failed");
                        return kResultInvalidState;
                    }

                    if (!audioFloat || audioSize == 0)
                    {
                        NVIGI_LOG_ERROR("Pipeline returned no audio data");
                        return kResultInvalidState;
                    }

                    // Convert float audio to int16
                    outputAudio.resize(audioSize);
                    for (size_t i = 0; i < audioSize; ++i)
                    {
                        float sample = std::clamp(audioFloat[i], -1.0f, 1.0f);
                        outputAudio[i] = static_cast<int16_t>(sample * 32767.0f);
                    }

                    // Free the audio buffer allocated by asqflow_pipeline_inference
                    asqflow_free_buffer(audioFloat);
                    
                    NVIGI_LOG_INFO("Pipeline inference completed: %zu audio samples", outputAudio.size());
                }

                // Set normalized text to input text (pipeline handles normalization internally)
                normalizedText = inputText;

                return kResultOk;
            }
            catch (const std::exception& e)
            {
                NVIGI_LOG_ERROR("Pipeline inference failed: %s", e.what());
                return kResultInvalidState;
            }
        }

    } // namespace asqflow

} // namespace nvigi
