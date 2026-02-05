// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

// Standard library headers
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>

// Donut headers
#include <nvrhi/nvrhi.h>
#include <donut/app/DeviceManager.h>

// NVIGI headers
#include <nvigi.h>
#include <nvigi_types.h>
#include <nvigi_hwi_common.h>  // For SchedulingMode
#include <nvigi_stl_helpers.h>  // For InferenceDataTextSTLHelper, InferenceDataAudioSTLHelper

#include <d3d12.h>
#include <dxgi.h>
#include <dxgi1_5.h>

// Modern C++ wrappers
#include "cxx_wrappers/core.hpp"
#include "cxx_wrappers/gpt/gpt.hpp"
#include "cxx_wrappers/asr/asr.hpp"
#include "cxx_wrappers/tts/tts.hpp"
#include "cxx_wrappers/d3d12.hpp"
#include "cxx_wrappers/vulkan.hpp"

struct Parameters
{
    donut::app::DeviceCreationParameters deviceParams;
    std::string sceneName;
    bool checkSig = false;
    bool renderScene = true;
};

// NVIGI forward declarations
namespace nvigi {
    struct IHWICuda;
    struct IHWICommon;
};

namespace AudioRecorder {
    struct RecordingInfo;
};

using namespace std::chrono_literals;

struct SimpleTimer {
	SimpleTimer() : running(false), totalTime(std::chrono::high_resolution_clock::duration::zero()) {}
	void Start(bool reset = true) {
		if (reset) {
			totalTime = std::chrono::high_resolution_clock::duration::zero();
		}
		if (!running) {
			baseTime = std::chrono::high_resolution_clock::now();
			running = true;
		}
	}
	void Stop() {
		if (running) {
			totalTime += std::chrono::high_resolution_clock::now() - baseTime;
			running = false;
		}
	}
	void Reset() {
		totalTime = std::chrono::high_resolution_clock::duration::zero();
		if (running) {
			baseTime = std::chrono::high_resolution_clock::now();
		}
	}
	double GetElapsedMiliseconds() const {
		return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(running ? (std::chrono::high_resolution_clock::now() - baseTime) : totalTime).count();
	}

	bool running = false;
    std::chrono::high_resolution_clock::time_point baseTime;
    std::chrono::high_resolution_clock::duration totalTime;
};

struct NVIGIContext
{
    enum ModelStatus {
        AVAILABLE_LOCALLY,
        AVAILABLE_CLOUD,
        AVAILABLE_DOWNLOADER, // Not yet supported
        AVAILABLE_DOWNLOADING, // Not yet supported
        AVAILABLE_MANUAL_DOWNLOAD,
        UNAVAILABLE
    };
    struct PluginModelInfo
    {
        std::string m_modelName;
        std::string m_pluginName;
        std::string m_caption; // plugin AND model
        std::string m_guid;
        std::string m_modelRoot;
        std::string m_url;
        size_t m_vram;
        std::string m_backend;  // Backend string: "cuda", "d3d12", "vulkan", "cpu", "cloud", "trt"
        ModelStatus m_modelStatus;
    };

    // MODERNIZED: Use shared_ptr for clear ownership semantics and automatic cleanup
    using PluginModelInfoPtr = std::shared_ptr<PluginModelInfo>;

    struct PluginBackendChoices
    {
        std::string m_nvda;   // NVIDIA CUDA backend: "cuda"
        std::string m_gpu;    // Generic GPU backend: "d3d12" or "vulkan"
        std::string m_cloud;  // Cloud backend: "cloud"
        std::string m_cpu;    // CPU backend: "cpu"
    };

    struct StageInfo
    {
        PluginModelInfoPtr m_info{};
        // Model GUID to info maps (maps model GUIDs to a list of plugins that run it)
        std::map<std::string, std::vector<PluginModelInfoPtr>> m_pluginModelsMap{};
        PluginBackendChoices m_choices{};
        std::atomic<bool> m_ready = false;
        std::atomic<bool> m_running = false;
        size_t m_vramBudget{};
        bool m_automaticBackendSelection = false;
        std::mutex m_callbackMutex;
        std::condition_variable m_callbackCV;
    };

    // MODERNIZED: Specialized stage info for GPT with modern execution state
    struct GPTStageInfo : StageInfo
    {
        std::atomic<nvigi::gpt::ExecutionState> m_callbackState{nvigi::gpt::ExecutionState::Invalid};
    };

    // MODERNIZED: Specialized stage info for TTS with modern execution state
    struct TTSStageInfo : StageInfo
    {
        std::atomic<nvigi::tts::ExecutionState> m_callbackState{nvigi::tts::ExecutionState::DataPending};
    };

    NVIGIContext() {}
    virtual ~NVIGIContext() {}
    static NVIGIContext& Get();
    NVIGIContext(const NVIGIContext&) = delete;
    NVIGIContext(NVIGIContext&&) = delete;
    NVIGIContext& operator=(const NVIGIContext&) = delete;
    NVIGIContext& operator=(NVIGIContext&&) = delete;

    bool Initialize_preDeviceManager(nvrhi::GraphicsAPI api, int argc, const char* const* argv);
    bool Initialize_preDeviceCreate(donut::app::DeviceManager* deviceManager, donut::app::DeviceCreationParameters& params);
    bool Initialize_postDevice();
    void SetDevice_nvrhi(nvrhi::IDevice* device);
    void Shutdown();

    bool IsCIGEnabled() const { return m_useCiG; }

    bool AddGPTPlugin(const std::string& backend);
    bool AddGPTCloudPlugin();
    bool AddASRPlugin(const std::string& backend);
    bool AddTTSPlugin(const std::string& backend);
    
private:
    // Internal helper to check plugin compatibility by ID
    bool CheckPluginCompat(nvigi::PluginID id, const std::string& name);
    
public:

    void GetVRAMStats(size_t& current, size_t& budget);

    void LaunchASR();
    void LaunchGPT(std::string prompt);
    void AppendTTSText(std::string text, bool done);
    void LaunchTTS(std::string prompt);

    bool ModelsComboBox(const std::string& label, bool automatic,
        StageInfo& stage,
        PluginModelInfoPtr& value);
    bool SelectAutoPlugin(const StageInfo& stage, const std::vector<PluginModelInfoPtr>& options, PluginModelInfoPtr& model);
    bool BuildOptionsUI();
    void BuildModelsStatusUI();
    void BuildChatUI();
    void BuildUI();

    static void PresentEnd(donut::app::DeviceManager& manager, uint32_t i);

    void ReloadGPTModel(PluginModelInfoPtr newInfo);
    void ReloadASRModel(PluginModelInfoPtr newInfo);
    void ReloadTTSModel(PluginModelInfoPtr newInfo);
    void FlushInferenceThread();

    void FramerateLimit()
    {
        if (!m_framerateLimiting)
            return;

        if (m_framerateTimer.running)
        {
            m_framerateTimer.Stop();
            double leftoverTime = 1000.0 / (double)m_targetFramerate - m_framerateTimer.GetElapsedMiliseconds();
			if (leftoverTime > 0.0)
				std::this_thread::sleep_for(std::chrono::milliseconds((int)leftoverTime));
		}
        m_framerateTimer.Start();
    }
    bool m_framerateLimiting = false;
    int m_targetFramerate = 60;
	SimpleTimer m_framerateTimer;

    StageInfo m_asr;
    GPTStageInfo m_gpt;  // MODERNIZED: Use specialized type with modern execution state
    TTSStageInfo m_tts;  // MODERNIZED: Use specialized type with modern execution state

    std::string m_nvdaKey = "";
    std::string m_openAIKey = "";

    bool GetCloudModelAPIKey(const PluginModelInfoPtr& info, const char* & key, std::string& apiKeyName)
    {
        if (info->m_url.find("integrate.api.nvidia.com") != std::string::npos)
        {
            if (m_nvdaKey.empty())
            {
                const char* ckey = getenv("NVIDIA_INTEGRATE_KEY");
                if (ckey)
                {
                    m_nvdaKey = ckey;
                }
                else
                {
                    apiKeyName = "NVIDIA_INTEGRATE_KEY";
                    return false;
                }
            }
            key = m_nvdaKey.c_str();
            return true;
        }
        else if (info->m_url.find("openai.com") != std::string::npos)
        {
            if (m_openAIKey.empty())
            {
                const char* ckey = getenv("OPENAI_KEY");
                if (ckey)
                {
                    m_openAIKey = ckey;
                }
                else
                {
                    apiKeyName = "OPENAI_KEY";
                    return false;
                }
            }
            key = m_openAIKey.c_str();
            return true;
        }
        else
        {
            apiKeyName = "UNKNOWN SERVICE";
            donut::log::warning("Unknown cloud model URL (%s); cannot send authentication token", info->m_url.c_str());
        }

        return false;
    }

    // MODERNIZED: Simplified context for TTS - wrapper handles most complexity
    struct TTSInferenceContext {
        std::string m_selectedTargetVoice = "03_M-Tom_Sawyer_15s";
        std::queue<std::unique_ptr<std::thread>> playAudioThreads;
        std::mutex mtxPlayAudio;

        size_t posLastSpace = 0; // used to handle TTS input chunks
        size_t posLastPeriod = 0; // used to handle TTS input chunks
        size_t posLastComma = 0; // used to handle TTS input chunks

        ~TTSInferenceContext() {
            while (true) {
                std::unique_ptr<std::thread> thread;
                {
                    if (playAudioThreads.empty())
                        break;
                    thread = std::move(playAudioThreads.front());
                    playAudioThreads.pop();
                }

                if (thread->joinable()) {
                    thread->join();
                }
            }
        }
    };

    nvrhi::IDevice* m_Device = nullptr;
    nvrhi::RefCountPtr<ID3D12CommandQueue> m_D3D12Queue = nullptr;
    nvrhi::RefCountPtr<ID3D12CommandQueue> m_D3D12QueueCompute = nullptr;
    nvrhi::RefCountPtr<ID3D12CommandQueue> m_D3D12QueueCopy = nullptr;
    std::string m_appUtf8path = "";
    std::string m_shippedModelsPath = "";
    std::string m_modelASR;
    std::string m_LogFilename = "";
    std::string m_systemPromptGPT = "You are a helpful AI agent. Your goal is to provide information about queries.\
        Generate only medium size answers and avoid describing what you are doing physically.\
        Avoid using specific words that are not part of the dictionary.\n"; 
	TTSInferenceContext m_ttsInferenceCtx;
    bool m_useCiG = true;

    int m_adapter = -1;
    
    // MODERNIZED: Use Core wrapper instead of raw function pointers
    std::unique_ptr<nvigi::Core> m_core;

    // MODERNIZED: Use modern wrappers instead of raw interfaces
    std::unique_ptr<nvigi::gpt::Instance> m_gptInstance;
    std::optional<nvigi::gpt::Instance::Chat> m_gptChat;
    
    std::unique_ptr<nvigi::asr::Instance> m_asrInstance;
    std::optional<nvigi::asr::Instance::Stream> m_asrStream;
    std::unique_ptr<nvigi::tts::Instance> m_ttsInstance;
    
    // MODERNIZED: Use unique_ptr with custom deleters for HWI interfaces (auto cleanup)
    std::unique_ptr<nvigi::IHWICuda, std::function<void(nvigi::IHWICuda*)>> m_cig;
    std::unique_ptr<nvigi::IHWICommon, std::function<void(nvigi::IHWICommon*)>> m_hwiCommon;
    
    std::string m_ttsInput;

    bool m_newInferenceSequence = false;
    std::atomic<bool> m_recording = false;
    AudioRecorder::RecordingInfo* m_recordingData{};
    std::atomic<bool> m_gptInputReady = false;
    std::string m_a2t;
    std::string m_gptInput;
    std::mutex m_mtx;
    bool m_conversationInitialized = false;

    // MODERNIZED: Use smart pointers for threads
    std::unique_ptr<std::thread> m_inferThread;
    std::atomic<bool> m_inferThreadRunning = false;
    std::unique_ptr<std::thread> m_loadingThread;

    std::vector<int16_t> m_ttsOutputAudio;

#ifdef USE_DX12
    nvigi::d3d12::D3D12Config m_d3d12_config{};
    nvrhi::RefCountPtr<IDXGIAdapter3> m_targetAdapter;
#endif
#ifdef USE_VULKAN
    nvigi::vulkan::VulkanConfig m_vk_config{};
#endif
    nvrhi::GraphicsAPI m_api = nvrhi::GraphicsAPI::D3D12;
    size_t m_maxVRAM = 0;
    uint32_t m_schedulingMode = nvigi::SchedulingMode::kPrioritizeCompute;

	SimpleTimer m_asrTimer;
    SimpleTimer m_gptFirstTokenTimer;
    SimpleTimer m_ttsFirstAudioTimer;
};

struct cerr_redirect {
    cerr_redirect()
    {
        std::freopen("ggml.txt", "w", stderr);
    }

    ~cerr_redirect()
    {
        std::freopen("NUL", "w", stderr);
        std::ifstream t("ggml.txt");
        std::stringstream buffer;
        buffer << t.rdbuf();
    }

private:
    std::streambuf* old;
};

