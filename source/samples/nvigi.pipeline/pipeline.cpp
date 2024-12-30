// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#ifdef NVIGI_WINDOWS
#include <conio.h>
#else
#include <linux/limits.h>
#endif

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <regex>
#include <thread>

namespace fs = std::filesystem;

#if NVIGI_WINDOWS
#include <windows.h>
#endif

#include <nvigi.h>
#include "nvigi_aip.h"
#include "nvigi_asr_whisper.h"
#include "nvigi_gpt.h"
#include "nvigi_types.h"

#if NVIGI_LINUX
#include <unistd.h>
#include <dlfcn.h>
using HMODULE = void*;
#define GetProcAddress dlsym
#define FreeLibrary dlclose
#define LoadLibraryA(lib) dlopen(lib, RTLD_LAZY)
#define LoadLibraryW(lib) dlopen(nvigi::extra::toStr(lib).c_str(), RTLD_LAZY)

#define sscanf_s sscanf
#define strcpy_s(a,b,c) strcpy(a,c)
#define strcat_s(a,b,c) strcat(a,c)
#define memcpy_s(a,b,c,d) memcpy(a,c,d)
typedef struct __LUID
{
    unsigned long LowPart;
    long HighPart;
} 	LUID;
#endif

#define DECLARE_NVIGI_CORE_FUN(F) PFun_##F* ptr_##F
#define GET_NVIGI_CORE_FUN(F) ptr_##F = (PFun_##F*)GetProcAddress(lib, #F)
DECLARE_NVIGI_CORE_FUN(nvigiInit);
DECLARE_NVIGI_CORE_FUN(nvigiShutdown);
DECLARE_NVIGI_CORE_FUN(nvigiLoadInterface);
DECLARE_NVIGI_CORE_FUN(nvigiUnloadInterface);

inline std::vector<uint8_t> read(const char* fname)
{
    try
    {
        fs::path p(fname);
        size_t file_size = fs::file_size(p);
        std::vector<uint8_t> ret_buffer(file_size);
#ifdef NVIGI_LINUX
        std::fstream file(fname, std::ios::binary | std::ios::in);
#else
        std::fstream file(fname, std::ios::binary | std::ios::in);
#endif
        file.read((char*)ret_buffer.data(), file_size);
        return ret_buffer;
    }
    catch (...)
    {
    }
    return {};
}

inline std::string getExecutablePath()
{
#ifdef NVIGI_LINUX
    char exePath[PATH_MAX] = {};
    readlink("/proc/self/exe", exePath, sizeof(exePath));
    std::string searchPathW = exePath;
    searchPathW.erase(searchPathW.rfind('/'));
    return searchPathW + "/";
#else
    CHAR pathAbsW[MAX_PATH] = {};
    GetModuleFileNameA(GetModuleHandleA(NULL), pathAbsW, ARRAYSIZE(pathAbsW));
    std::string searchPathW = pathAbsW;
    searchPathW.erase(searchPathW.rfind('\\'));
    return searchPathW + "\\";
#endif
}

void loggingCallback(nvigi::LogType type, const char* msg)
{
#ifdef NVIGI_WINDOWS
    OutputDebugStringA(msg);
#endif
    std::cout << msg;
}

template<typename T>
bool unloadInterface(nvigi::PluginID feature, T*& _interface)
{
    if (_interface == nullptr)
        return false;

    nvigi::Result result = ptr_nvigiUnloadInterface(feature, _interface);
    if (result == nvigi::kResultOk)
    {
        _interface = nullptr;
    }
    else
    {
        loggingCallback(nvigi::LogType::eError, "Failed to unload interface");
        return false;
    }

    return true;
}

int main(int argc, char** argv)
{
#ifdef NVIGI_WINDOWS
    FILE* f{};
    freopen_s(&f, "NUL", "w", stderr);
#else
    freopen("dev/nul", "w", stderr);
#endif

    auto exePathUtf8 = getExecutablePath();
#ifdef NVIGI_WINDOWS
    auto libPath = exePathUtf8 + "/nvigi.core.framework.dll";
#else
    auto libPath = exePathUtf8 + "/nvigi.core.framework.so";
#endif
    HMODULE lib = LoadLibraryA(libPath.c_str());
    if (lib == nullptr)
    {
        loggingCallback(nvigi::LogType::eError, "Could not load NVIGI core library");
        return -1;
    }
    
    GET_NVIGI_CORE_FUN(nvigiInit);
    GET_NVIGI_CORE_FUN(nvigiShutdown);
    GET_NVIGI_CORE_FUN(nvigiLoadInterface);
    GET_NVIGI_CORE_FUN(nvigiUnloadInterface);

    if (ptr_nvigiInit == nullptr || ptr_nvigiShutdown == nullptr || 
        ptr_nvigiLoadInterface == nullptr || ptr_nvigiUnloadInterface == nullptr)
    {
        loggingCallback(nvigi::LogType::eError, "Could not load NVIGI core library");
        return -1;
    }

    const char* paths[] =
    {
        exePathUtf8.c_str()
    };

    if (argc != 3)
    {
        loggingCallback(nvigi::LogType::eError, "nvigi.basic <path to models> <path to wav file>");
        return -1;
    }
    std::string modelDir = argv[1];
    std::string audioFile = argv[2];

    //////////////////////////////////////////////////////////////////////////////
    //! Init NVIGI
    //! 

    nvigi::Preferences pref{};
    pref.logLevel = nvigi::LogLevel::eVerbose;
    pref.showConsole = true;
    pref.numPathsToPlugins = 1;
    pref.utf8PathsToPlugins = paths;
    pref.logMessageCallback = nullptr;
    pref.utf8PathToLogsAndData = exePathUtf8.c_str();

    uint32_t n_threads = 16;
    uint32_t vram = 1024 * 12;

    auto result = ptr_nvigiInit(pref, nullptr, nvigi::kSDKVersion);
    if (result != nvigi::kResultOk)
    {
        loggingCallback(nvigi::LogType::eError, "NVIGI init failed");
        return -1;
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Init Plugin Interfaces and Instances
    //! 

    //! AIP
    nvigi::aip::IAiPipeline* iaip{};
    if (NVIGI_FAILED(result, nvigiGetInterfaceDynamic(nvigi::plugin::ai::pipeline::kId, &iaip, ptr_nvigiLoadInterface)))
    {
        loggingCallback(nvigi::LogType::eError, "'nvigiGetInterface' failed");
        return -1;
    }

    //! ASR
    nvigi::ASRWhisperCreationParameters asrParams{};
    nvigi::CommonCreationParameters asrCommon{};
    {
        asrCommon.utf8PathToModels = modelDir.c_str();
        asrCommon.numThreads = n_threads;
        asrCommon.vramBudgetMB = vram;
        asrCommon.modelGUID = "{5CAD3A03-1272-4D43-9F3D-655417526170}";
        asrCommon.chain(asrParams);
    }

    //! GPT
    nvigi::GPTCreationParameters gptParams{};
    nvigi::CommonCreationParameters gptCommon{};
    {
        //! 
        //! Here we provide an example local vs cloud, same pattern applies to any other stage
        //! 

        gptCommon.utf8PathToModels = modelDir.c_str();
        gptCommon.numThreads = n_threads;
        gptCommon.vramBudgetMB = vram;

        //! Model is the same regardless of the backend
        gptCommon.modelGUID = "{01F43B70-CE23-42CA-9606-74E80C5ED0B6}";
        gptCommon.chain(gptParams);
    }

    std::vector<const nvigi::NVIGIParameter*> stageParams = { asrCommon, gptCommon };
    std::vector<nvigi::PluginID> stages =
    {
        nvigi::plugin::asr::ggml::cuda::kId,
        nvigi::plugin::gpt::ggml::cuda::kId
    };

    //! Creation parameters for the pipeline
    nvigi::aip::AiPipelineCreationParameters aipParams{};

    aipParams.numStages = stages.size();
    aipParams.stages = stages.data();
    aipParams.stageParams = stageParams.data();

    //! Create pipeline instance
    nvigi::InferenceInstance* instance{};
    iaip->createInstance(aipParams, &instance);

    //////////////////////////////////////////////////////////////////////////////
    //! Run inference
    //! 

    //! Prepare audio input slot
    auto wav = read(audioFile.c_str());
    if (wav.empty())
    {
        loggingCallback(nvigi::LogType::eError, "Could not load input WAV file");
        return -1;
    }

    //! Prepare audio input slot
    nvigi::CpuData _audio{ wav.size(), wav.data() };
    nvigi::InferenceDataAudio audio{ _audio };
    //! Prepare prompt
    std::string promptText("This is a conversation between John F. Kennedy (JFK), the late USA president and person named Bob. Bob's answers are short and on the point.\nJFK: ");
    nvigi::CpuData _prompt{ promptText.size(), promptText.data() };
    nvigi::InferenceDataText prompt{ _prompt };
    std::vector<nvigi::InferenceDataSlot> slots = { {nvigi::kASRWhisperDataSlotAudio, &audio}, {nvigi::kGPTDataSlotSystem, &prompt} };
    nvigi::InferenceDataSlotArray inputs{ slots.size(), slots.data() };

    struct AceCallbackCtx
    {
        nvigi::aip::AiPipelineCreationParameters aipParams{};
        std::string asrOutput{};
        std::string gptOutput{};
        std::string a2fOutput{};
        std::vector<nvigi::PluginID> stages{};
    };
    AceCallbackCtx aipCtx{};
    aipCtx.aipParams = aipParams;
    aipCtx.stages = stages;
    auto aipCallback = [](const nvigi::InferenceExecutionContext* ctx, nvigi::InferenceExecutionState state, void* userData)->nvigi::InferenceExecutionState
        {
            auto aipCtx = (AceCallbackCtx*)userData;
            if (ctx->instance->getFeatureId(ctx->instance->data) == aipCtx->stages[0])
            {
                // Outputs from ASR
                auto slots = ctx->outputs;
                const nvigi::InferenceDataText* text{};
                slots->findAndValidateSlot(nvigi::kASRWhisperDataSlotTranscribedText, &text);
                auto response = std::string(text->getUTF8Text());
                if (response.find("<JSON>") != std::string::npos)
                {
                    std::string text = "asr stats:" + response + "\n";
                    loggingCallback(nvigi::LogType::eInfo, text.c_str());
                }
                else
                {
                    aipCtx->asrOutput += response;
                }
                if (state == nvigi::kInferenceExecutionStateDone)
                {
                    std::string text = "asr output:" + aipCtx->asrOutput + "\n";
                    loggingCallback(nvigi::LogType::eInfo, text.c_str());
                }
            }
            else if (ctx->instance->getFeatureId(ctx->instance->data) == aipCtx->stages[1])
            {
                // Outputs from GPT
                auto slots = ctx->outputs;
                const nvigi::InferenceDataText* text{};
                slots->findAndValidateSlot(nvigi::kGPTDataSlotResponse, &text);
                auto response = std::string(text->getUTF8Text());
                aipCtx->gptOutput += response;
                if (state == nvigi::kInferenceExecutionStateDone)
                {
                    std::string text = "gpt output:" + aipCtx->gptOutput + "\n";
                    loggingCallback(nvigi::LogType::eInfo, text.c_str());
                }
            }

            return state;
        };

    nvigi::GPTRuntimeParameters gptRuntime{};
    gptRuntime.interactive = true;
    gptRuntime.reversePrompt = "JFK:";

    //! Setup execution context for the pipeline
    nvigi::InferenceExecutionContext ctx{};
    ctx.instance = instance;
    ctx.runtimeParameters = gptRuntime;
    ctx.callback = aipCallback;
    ctx.callbackUserData = &aipCtx;
    ctx.inputs = &inputs;
    instance->evaluate(&ctx);

    //////////////////////////////////////////////////////////////////////////////
    //! Shutdown NVIGI
    //! 

    iaip->destroyInstance(instance);

    unloadInterface(nvigi::plugin::ai::pipeline::kId, iaip);

    result = ptr_nvigiShutdown();
    if (result != nvigi::kResultOk)
    {
        loggingCallback(nvigi::LogType::eError, "Error in NVIGI shutdown");
        return -1;
    }

    FreeLibrary(lib);

    return 0;
}