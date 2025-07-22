// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#define SDL_MAIN_HANDLED 

#ifdef GGML_USE_CUBLAS
#include "source/core/nvigi.api/nvigi_cuda.h"
#include "ggml-cuda.h"
#endif

#include <inttypes.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <regex>

#define M_PI 3.14159265358979323846   // pi

//#include "external/sdl/release/include/SDL2/SDL.h"
//#include "external/sdl/release/include/SDL2/SDL_audio.h"

constexpr uint32_t kCommonSampleRate = 16000;

namespace nvigi
{
namespace asr
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

struct whisper_params {
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t n_processors = 1;
    int32_t offset_t_ms = 0;
    int32_t offset_n = 0;
    int32_t duration_ms = 0;
    int32_t progress_step = 5;
    int32_t max_context = -1;
    int32_t max_len = 0;
    int32_t best_of = 1;
    int32_t beam_size = -1;
    int32_t max_tokens = 0;
    int32_t audio_ctx = 0;
    int32_t step_ms = 2000;
    int32_t length_ms = 20000;
    int32_t keep_ms = 200;
    
    float word_thold = 0.01f;
    float entropy_thold = 2.40f;
    float logprob_thold = -1.00f;
    float vad_thold = 0.6f;
    float freq_thold = 100.0f;

    bool speed_up = false;
    bool translate = false;
    bool detect_language = false;
    bool diarize = false;
    bool tinydiarize = false;
    bool split_on_word = false;
    bool no_fallback = false;
    bool output_txt = false;
    bool output_vtt = false;
    bool output_srt = false;
    bool output_wts = false;
    bool output_csv = false;
    bool output_jsn = false;
    bool output_lrc = false;
    bool print_special = false;
    bool print_colors = false;
    bool print_progress = false;
    bool no_timestamps = false;
    bool no_context = true;
    bool low_vram = false;

    std::string language = "en";
    std::string prompt;
    std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
    std::string model = "models/ggml-base.en.bin";

    // [TDRZ] speaker turn string
    std::string tdrz_speaker_turn = " [SPEAKER_TURN]"; // TODO: set from command line

    std::string openvino_encode_device = "CPU";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};
};

void high_pass_filter(std::vector<float>& data, float cutoff, float sample_rate) {
    const float rc = 1.0f / (2.0f * (float)M_PI * cutoff);
    const float dt = 1.0f / sample_rate;
    const float alpha = dt / (rc + dt);

    float y = data[0];

    for (size_t i = 1; i < data.size(); i++) {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

bool vad_simple(std::vector<float>& pcmf32, int sample_rate, int last_ms, float vad_thold, float freq_thold, bool verbose) {
    const int n_samples = (int)pcmf32.size();
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    if (n_samples_last >= n_samples) {
        // not enough samples - assume no speech
        return false;
    }

    if (freq_thold > 0.0f) {
        high_pass_filter(pcmf32, freq_thold, (float)sample_rate);
    }

    float energy_all = 0.0f;
    float energy_last = 0.0f;

    for (int i = 0; i < n_samples; i++) {
        energy_all += fabsf(pcmf32[i]);

        if (i >= n_samples - n_samples_last) {
            energy_last += fabsf(pcmf32[i]);
        }
    }

    energy_all /= n_samples;
    energy_last /= n_samples_last;

    if (verbose) {
        fprintf(stderr, "%s: energy_all: %f, energy_last: %f, vad_thold: %f, freq_thold: %f\n", __func__, energy_all, energy_last, vad_thold, freq_thold);
    }

    if (energy_last > vad_thold * energy_all) {
        return false;
    }

    return true;
}

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int)hr, (int)min, (int)sec, comma ? "," : ".", (int)msec);

    return std::string(buf);
}

class CircularBuffer {
public:
    CircularBuffer() {}

    void init(float ms)
    {
        size_t size = size_t(WHISPER_SAMPLE_RATE * sizeof(float) * ms * 0.001f);
        buffer_.resize(size, 0);
        bufferSize_ = size;
        writeIndex_ = readIndex_ = dataSize_ = 0;
    }

    void clear()
    {
        std::scoped_lock lock(mtx_);
        buffer_.clear();
        buffer_.resize(bufferSize_, 0);
        writeIndex_ = readIndex_ = dataSize_ = 0;
    }

    void write(const std::vector<float>& pcmf32) {
        write((uint8_t*)pcmf32.data(), pcmf32.size() * sizeof(float));
    }

    // Function to write arbitrary chunks of data to the buffer
    void write(const uint8_t* data, size_t dataSize) {
        if (dataSize == 0) {
            return;
        }

        std::scoped_lock lock(mtx_);

        size_t availableSpace = bufferSize_ - dataSize_;
        if (dataSize > availableSpace) {
            // If the data exceeds the available space, we overwrite the oldest data
            size_t excessData = dataSize - availableSpace;
            NVIGI_LOG_WARN("Overwriting %llu bytes of data due to buffer overflow.", excessData);
            readIndex_ = (readIndex_ + excessData) % bufferSize_;
            dataSize_ = bufferSize_;
        }
        else {
            dataSize_ += dataSize;
        }

        size_t firstPart = std::min(dataSize, bufferSize_ - writeIndex_);
        std::memcpy(&buffer_[writeIndex_], data, firstPart);
        writeIndex_ = (writeIndex_ + firstPart) % bufferSize_;

        if (dataSize > firstPart) {
            size_t secondPart = dataSize - firstPart;
            std::memcpy(&buffer_[0], data + firstPart, secondPart);
            writeIndex_ = secondPart;
        }
    }

    size_t read(std::vector<float>& pcmf32, int ms, bool& noMoreData)
    {
        // Everything here is measured in bytes
        auto available = getDataSize();
        if (!available)
        {
            noMoreData = (streamType == -1 || streamType == (int)StreamSignal::eStreamSignalStop);
            return 0;
        }
        size_t requested = ms == 0 ? available : size_t(WHISPER_SAMPLE_RATE * sizeof(float) * (float)ms * 0.001f);
        size_t bytesToCopy = available < requested ? available : requested;
        pcmf32.resize(bytesToCopy / sizeof(float), 0.0f);
        auto bytesRead = read((uint8_t*)pcmf32.data(), bytesToCopy);
        noMoreData = (streamType == -1 || streamType == (int)StreamSignal::eStreamSignalStop) && bytesRead == available;
        return bytesRead;
    }

    // Function to read arbitrary chunks of data from the buffer
    size_t read(uint8_t* outData, size_t dataSize) {
        if (dataSize_ == 0) {
            return 0; // Nothing to read
        }

        std::scoped_lock lock(mtx_);

        size_t readSize = std::min(dataSize, dataSize_);
        size_t firstPart = std::min(readSize, bufferSize_ - readIndex_);
        std::memcpy(outData, &buffer_[readIndex_], firstPart);
        readIndex_ = (readIndex_ + firstPart) % bufferSize_;

        if (readSize > firstPart) {
            size_t secondPart = readSize - firstPart;
            std::memcpy(outData + firstPart, &buffer_[0], secondPart);
            readIndex_ = secondPart;
        }

        dataSize_ -= readSize;
        return readSize;
    }

    // Get the amount of data currently stored in the buffer
    size_t getDataSize() {
        std::scoped_lock lock(mtx_);
        return dataSize_;
    }

    // Get the remaining space available in the buffer
    size_t getAvailableSpace() {
        std::scoped_lock lock(mtx_);
        return bufferSize_ - dataSize_;
    }

    void setStreamType(int t) { streamType.store(t); }
    int getStreamType() { return streamType.load(); }

private:
    std::vector<uint8_t> buffer_;  // The circular buffer
    size_t bufferSize_;            // Total size of the buffer
    size_t writeIndex_;            // Index for writing data
    size_t readIndex_;             // Index for reading data
    size_t dataSize_;              // Amount of data currently stored in the buffer
    std::mutex mtx_;
    std::atomic<int> streamType = -1;
};

}
}
