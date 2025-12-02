// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

// WAV file writer helper
namespace WavWriter {
    #pragma pack(push, 1)
        struct WavHeader {
            char riff[4] = {'R', 'I', 'F', 'F'};
            uint32_t chunk_size = 0;
            char wave[4] = {'W', 'A', 'V', 'E'};
            char fmt[4] = {'f', 'm', 't', ' '};
            uint32_t fmt_size = 16;
            uint16_t audio_format = 1;  // PCM
            uint16_t num_channels = 1;
            uint32_t sample_rate = 16000;
            uint32_t byte_rate = 32000;  // sample_rate * num_channels * bits_per_sample / 8
            uint16_t block_align = 2;    // num_channels * bits_per_sample / 8
            uint16_t bits_per_sample = 16;
            char data[4] = {'d', 'a', 't', 'a'};
            uint32_t data_size = 0;
        };
    #pragma pack(pop)
        
        static_assert(sizeof(WavHeader) == 44, "WAV header must be exactly 44 bytes");
    
        class WavFile {
        private:
            std::ofstream file;
            size_t data_written = 0;
            std::string filename;
    
        public:
            WavFile(const std::string& path) : filename(path) {
                file.open(path, std::ios::binary);
                if (file.is_open()) {
                    // Write initial header (will be updated later with correct sizes)
                    WavHeader header;
                    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
                }
            }
    
            ~WavFile() {
                close();
            }
    
            bool is_open() const {
                return file.is_open();
            }
    
            void write_samples(const uint8_t* data, size_t size) {
                if (file.is_open()) {
                    file.write(reinterpret_cast<const char*>(data), size);
                    data_written += size;
                    file.flush();
                }
            }
    
            void close() {
                if (file.is_open()) {
                    // Flush any pending writes
                    file.flush();
                    
                    // Update header with correct sizes
                    WavHeader header;
                    header.data_size = static_cast<uint32_t>(data_written);
                    header.chunk_size = static_cast<uint32_t>(36 + data_written);
                    
                    file.seekp(0, std::ios::beg);
                    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
                    file.flush();
                    file.close();
                    
                    std::cout << "\n[WAV file written: " << filename << " (" << data_written << " bytes, ";
                    std::cout << (data_written / 32000.0) << " seconds)]\n";
                }
            }
    
            size_t get_bytes_written() const {
                return data_written;
            }
        };
    }
    
    // Audio recording helper for Windows (SDL2-style)
    namespace AudioRecorder {
        // SDL2-like configuration: smaller buffers for lower latency
        // With 16kHz sampling rate, these settings give ~10ms latency per buffer
        constexpr int NUM_BUFFERS = 4;        // More buffers for smoother streaming (like SDL2's queue)
        constexpr int BUFFER_SIZE = 320;      // 320 bytes = 160 samples = 10ms at 16kHz 16-bit mono
        
        // SDL2-style audio callback signature
        // Called frequently with fixed-size buffers of audio data
        // userdata: user context pointer passed during initialization
        // stream: pointer to audio data buffer
        // len: length of audio data in bytes
        using AudioCallback = void(*)(void* userdata, const uint8_t* stream, int len);
    
        struct RecordingInfo {
            // SDL2-style: user callback instead of accumulation
            AudioCallback callback{nullptr};
            void* userdata{nullptr};
            
            // Optional: keep a rolling buffer for compatibility (can be disabled)
            std::vector<uint8_t> audioBuffer{};
            DWORD bytesWritten{0};
            bool accumulate{false};  // If false, only use callback (SDL2-pure mode)
            
            HWAVEIN hwi{nullptr};
            WAVEHDR headers[NUM_BUFFERS]{};
            WAVEFORMATEX waveFormat{};
            std::mutex mutex;  // Protect shared state
        };
    
        std::atomic<bool> isRecording{false};
    
        void CALLBACK waveInProc(HWAVEIN hwi, UINT uMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2) {
            if (uMsg == WIM_DATA) {
                if (dwInstance) {
                    RecordingInfo& info = *reinterpret_cast<RecordingInfo*>(dwInstance);
                    LPWAVEHDR waveHeader = reinterpret_cast<LPWAVEHDR>(dwParam1);
                    
                    // SDL2-style: call user callback immediately with audio data
                    // This provides real-time, low-latency access to audio samples
                    if (info.callback) {
                        // Call user callback directly (like SDL2 does)
                        info.callback(info.userdata, 
                                     reinterpret_cast<const uint8_t*>(waveHeader->lpData), 
                                     waveHeader->dwBytesRecorded);
                    }
                    
                    // Optional: accumulate for backward compatibility
                    if (info.accumulate) {
                        std::lock_guard<std::mutex> lock(info.mutex);
                        size_t oldSize = info.audioBuffer.size();
                        info.audioBuffer.resize(oldSize + waveHeader->dwBytesRecorded);
                        memcpy(info.audioBuffer.data() + oldSize, waveHeader->lpData, waveHeader->dwBytesRecorded);
                        info.bytesWritten += waveHeader->dwBytesRecorded;
                    }
    
                    // Immediately reuse the buffer (critical for low latency)
                    waveInUnprepareHeader(hwi, waveHeader, sizeof(WAVEHDR));
                    waveInPrepareHeader(hwi, waveHeader, sizeof(WAVEHDR));
                    waveInAddBuffer(hwi, waveHeader, sizeof(WAVEHDR));
                }
            }
        }
    
        // SDL2-style: Start recording with user callback (optional accumulation)
        // callback: function to call with each audio buffer (can be nullptr for accumulation-only mode)
        // userdata: user context pointer passed to callback
        // accumulate: if true, also accumulate data in internal buffer (for backward compatibility)
        RecordingInfo* StartRecording(AudioCallback callback = nullptr, void* userdata = nullptr, bool accumulate = true) {
            if (isRecording) return nullptr;
    
            auto* info = new RecordingInfo();
            info->callback = callback;
            info->userdata = userdata;
            info->accumulate = accumulate;
            
            // Setup WAVE format: 16kHz, 16-bit, mono (required for Whisper)
            info->waveFormat.wFormatTag = WAVE_FORMAT_PCM;
            info->waveFormat.nChannels = 1;
            info->waveFormat.nSamplesPerSec = 16000;
            info->waveFormat.wBitsPerSample = 16;
            info->waveFormat.cbSize = 0;
            info->waveFormat.nBlockAlign = (info->waveFormat.wBitsPerSample / 8) * info->waveFormat.nChannels;
            info->waveFormat.nAvgBytesPerSec = info->waveFormat.nSamplesPerSec * info->waveFormat.nBlockAlign;
    
            MMRESULT result = waveInOpen(&info->hwi, WAVE_MAPPER, &info->waveFormat, 
                                         (DWORD_PTR)waveInProc, (DWORD_PTR)info, CALLBACK_FUNCTION);
            if (result != MMSYSERR_NOERROR) {
                delete info;
                return nullptr;
            }
    
            info->bytesWritten = 0;
            info->audioBuffer.clear();
    
            // Prepare buffers (smaller buffers = lower latency, like SDL2)
            for (int i = 0; i < NUM_BUFFERS; i++) {
                info->headers[i].lpData = new char[BUFFER_SIZE];
                info->headers[i].dwBufferLength = BUFFER_SIZE;
                info->headers[i].dwBytesRecorded = 0;
                info->headers[i].dwUser = 0;
                info->headers[i].dwFlags = 0;
                info->headers[i].dwLoops = 0;
                
                result = waveInPrepareHeader(info->hwi, &info->headers[i], sizeof(WAVEHDR));
                if (result != MMSYSERR_NOERROR) {
                    delete info;
                    return nullptr;
                }
                
                result = waveInAddBuffer(info->hwi, &info->headers[i], sizeof(WAVEHDR));
                if (result != MMSYSERR_NOERROR) {
                    delete info;
                    return nullptr;
                }
            }
    
            result = waveInStart(info->hwi);
            if (result != MMSYSERR_NOERROR) {
                delete info;
                return nullptr;
            }
            
            isRecording = true;
            return info;
        }
    
        std::vector<uint8_t> StopRecording(RecordingInfo* info) {
            if (!info || !isRecording) return {};
    
            isRecording = false;
    
            waveInStop(info->hwi);
            
            // Give callbacks time to finish (SDL2 does this internally)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
            for (int i = 0; i < NUM_BUFFERS; i++) {
                waveInUnprepareHeader(info->hwi, &info->headers[i], sizeof(WAVEHDR));
                delete[] info->headers[i].lpData;
            }
    
            waveInClose(info->hwi);
    
            // Return accumulated buffer (if accumulation was enabled)
            std::vector<uint8_t> result;
            if (info->accumulate) {
                std::lock_guard<std::mutex> lock(info->mutex);
                result = std::move(info->audioBuffer);
            }
            
            delete info;
            
            return result;
        }
        
        // SDL2-style: Get current buffer size and frequency info
        struct AudioSpec {
            int freq;           // Sample rate (Hz)
            int channels;       // Number of channels
            int samples;        // Audio buffer size in sample frames
            int size;           // Audio buffer size in bytes
            float latency_ms;   // Expected latency in milliseconds
        };
        
        AudioSpec GetAudioSpec() {
            AudioSpec spec;
            spec.freq = 16000;
            spec.channels = 1;
            spec.samples = BUFFER_SIZE / 2;  // 16-bit samples
            spec.size = BUFFER_SIZE;
            spec.latency_ms = (float)spec.samples / (float)spec.freq * 1000.0f;
            return spec;
        }
    }