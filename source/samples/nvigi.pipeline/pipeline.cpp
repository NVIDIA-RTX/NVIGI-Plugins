// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <nvigi_stl_helpers.h>

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
#define GET_NVIGI_CORE_FUN(lib, F) ptr_##F = (PFun_##F*)GetProcAddress(lib, #F)
DECLARE_NVIGI_CORE_FUN(nvigiInit);
DECLARE_NVIGI_CORE_FUN(nvigiShutdown);
DECLARE_NVIGI_CORE_FUN(nvigiLoadInterface);
DECLARE_NVIGI_CORE_FUN(nvigiUnloadInterface);

inline std::vector<int16_t> read(const char* fname)
{
    try
    {
        fs::path p(fname);
        if (!fs::exists(p) || fs::status(p).type() != fs::file_type::regular)
        {
            return {};
        }
        size_t file_size = fs::file_size(p);
        std::vector<int16_t> ret_buffer(file_size/2);
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

struct NVIGIAppCtx
{
    HMODULE coreLib{};
    nvigi::aip::IAiPipeline* iaip{};
    nvigi::InferenceInstance* pipelineInst{};

    std::vector<nvigi::PluginID> stages{};

    std::string asrOutput{};
    std::string gptOutput{};
    std::string a2fOutput{};
};

constexpr uint32_t n_threads = 16;

///////////////////////////////////////
//! NVIGI Init and Shutdown

int InitNVIGI(NVIGIAppCtx& nvigiCtx, const std::string& pathToSDKUtf8)
{
#ifdef NVIGI_WINDOWS
    auto libPath = pathToSDKUtf8 + "/nvigi.core.framework.dll";
#else
    auto libPath = pathToSDKUtf8 + "/nvigi.core.framework.so";
#endif
    nvigiCtx.coreLib = LoadLibraryA(libPath.c_str());
    if (nvigiCtx.coreLib == nullptr)
    {
        loggingCallback(nvigi::LogType::eError, "Could not load NVIGI core library");
        return -1;
    }

    GET_NVIGI_CORE_FUN(nvigiCtx.coreLib, nvigiInit);
    GET_NVIGI_CORE_FUN(nvigiCtx.coreLib, nvigiShutdown);
    GET_NVIGI_CORE_FUN(nvigiCtx.coreLib, nvigiLoadInterface);
    GET_NVIGI_CORE_FUN(nvigiCtx.coreLib, nvigiUnloadInterface);

    if (ptr_nvigiInit == nullptr || ptr_nvigiShutdown == nullptr ||
        ptr_nvigiLoadInterface == nullptr || ptr_nvigiUnloadInterface == nullptr)
    {
        loggingCallback(nvigi::LogType::eError, "Could not load NVIGI core library");
        return -1;
    }

    const char* paths[] =
    {
        pathToSDKUtf8.c_str()
    };

    nvigi::Preferences pref{};
#ifdef _DEBUG
    pref.logLevel = nvigi::LogLevel::eVerbose;
#else
    pref.logLevel = nvigi::LogLevel::eOff;
#endif
    pref.showConsole = true;
    pref.numPathsToPlugins = 1;
    pref.utf8PathsToPlugins = paths;
    pref.logMessageCallback = pref.showConsole ? (nvigi::PFun_LogMessageCallback*)nullptr : loggingCallback; // avoid duplicating logs in the console
    pref.utf8PathToLogsAndData = pathToSDKUtf8.c_str();

    if (NVIGI_FAILED(result, ptr_nvigiInit(pref, nullptr, nvigi::kSDKVersion)))
    {
        loggingCallback(nvigi::LogType::eError, "NVIGI init failed");
        return -1;
    }

    return 0;
}

int ShutdownNVIGI(NVIGIAppCtx& nvigiCtx)
{
    if (NVIGI_FAILED(result, ptr_nvigiShutdown()))
    {
        loggingCallback(nvigi::LogType::eError, "Error in 'nvigiShutdown'");
        return -1;
    }

    FreeLibrary(nvigiCtx.coreLib);

    return 0;
}

int CreatePipeline(NVIGIAppCtx& nvigiCtx, const std::string& modelDir, size_t vram)
{
    //! AIP
    if (NVIGI_FAILED(result, nvigiGetInterfaceDynamic(nvigi::plugin::ai::pipeline::kId, &nvigiCtx.iaip, ptr_nvigiLoadInterface)))
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
        if (NVIGI_FAILED(result, asrCommon.chain(asrParams)))
        {
            loggingCallback(nvigi::LogType::eError, "ASR param chaining failed");
            return -1;
        }
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
        if (NVIGI_FAILED(result, gptCommon.chain(gptParams)))
        {
            loggingCallback(nvigi::LogType::eError, "GPT param chaining failed");
            return -1;
        }
    }

    std::vector<const nvigi::NVIGIParameter*> stageParams = { asrCommon, gptCommon };
    nvigiCtx.stages =
    {
        nvigi::plugin::asr::ggml::cuda::kId,
        nvigi::plugin::gpt::ggml::cuda::kId
    };

    //! Creation parameters for the pipeline
    nvigi::aip::AiPipelineCreationParameters aipParams{};

    aipParams.numStages = nvigiCtx.stages.size();
    aipParams.stages = nvigiCtx.stages.data();
    aipParams.stageParams = stageParams.data();

    //! Create pipeline instance
    if (NVIGI_FAILED(result, nvigiCtx.iaip->createInstance(aipParams, &nvigiCtx.pipelineInst)))
    {
        loggingCallback(nvigi::LogType::eError, "Error creating pipeline plugin instance");
        return -1;
    }

    return 0;
}

int ReleasePipeline(NVIGIAppCtx& nvigiCtx)
{
    if (NVIGI_FAILED(result, nvigiCtx.iaip->destroyInstance(nvigiCtx.pipelineInst)))
    {
        loggingCallback(nvigi::LogType::eError, "Error destroying pipeline instance");
        return -1;
    }

    if (NVIGI_FAILED(result, ptr_nvigiUnloadInterface(nvigi::plugin::ai::pipeline::kId, nvigiCtx.iaip)))
    {
        loggingCallback(nvigi::LogType::eError, "Error unloading pipeline interface");
        return -1;
    }

    return 0;
}


nvigi::InferenceExecutionState ASRCallback(const nvigi::InferenceExecutionContext* ctx, nvigi::InferenceExecutionState state, NVIGIAppCtx& appCtx)
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
        appCtx.asrOutput += response;
    }
    if (state == nvigi::kInferenceExecutionStateDone)
    {
        std::string text = "asr output:" + appCtx.asrOutput + "\n";
        loggingCallback(nvigi::LogType::eInfo, text.c_str());
    }

    return state;
}

nvigi::InferenceExecutionState GPTCallback(const nvigi::InferenceExecutionContext* ctx, nvigi::InferenceExecutionState state, NVIGIAppCtx& appCtx)
{
    // Outputs from GPT
    auto slots = ctx->outputs;
    const nvigi::InferenceDataText* text{};
    slots->findAndValidateSlot(nvigi::kGPTDataSlotResponse, &text);
    auto response = std::string(text->getUTF8Text());
    appCtx.gptOutput += response;
    if (state == nvigi::kInferenceExecutionStateDone)
    {
        std::string text = "gpt output:" + appCtx.gptOutput + "\n";
        loggingCallback(nvigi::LogType::eInfo, text.c_str());
    }

    return state;
}

nvigi::InferenceExecutionState PipelineCallback(const nvigi::InferenceExecutionContext* ctx, nvigi::InferenceExecutionState state, void* userData)
{
    auto appCtx = (NVIGIAppCtx*)userData;
    if (!appCtx)
        return nvigi::kInferenceExecutionStateInvalid;

    if (ctx->instance->getFeatureId(ctx->instance->data) == appCtx->stages[0])
        return ASRCallback(ctx, state, *appCtx);
    else if (ctx->instance->getFeatureId(ctx->instance->data) == appCtx->stages[1])
        return GPTCallback(ctx, state, *appCtx);

    return state;
};

int RunInference(NVIGIAppCtx& nvigiCtx, const std::vector<int16_t>& wav, const std::string& promptText, const std::string& reversePromptText)
{
    //! Prepare audio input slot
    nvigi::InferenceDataAudioSTLHelper audio{ wav };

    //! Prepare prompt
    nvigi::InferenceDataTextSTLHelper prompt{ promptText };

    std::vector<nvigi::InferenceDataSlot> slots = { {nvigi::kASRWhisperDataSlotAudio, audio}, {nvigi::kGPTDataSlotSystem, prompt} };
    nvigi::InferenceDataSlotArray inputs{ slots.size(), slots.data() };

    nvigi::GPTRuntimeParameters gptRuntime{};
    gptRuntime.interactive = false;
    gptRuntime.reversePrompt = reversePromptText.c_str();

    //! Setup execution context for the pipeline
    nvigi::InferenceExecutionContext ctx{};
    ctx.instance = nvigiCtx.pipelineInst;
    ctx.runtimeParameters = gptRuntime;
    ctx.callback = PipelineCallback;
    ctx.callbackUserData = &nvigiCtx;
    ctx.inputs = &inputs;
    if (NVIGI_FAILED(result, nvigiCtx.pipelineInst->evaluate(&ctx)))
    {
        loggingCallback(nvigi::LogType::eError, "Error running pipeline inference");
        return -1;
    }

    return 0;
}

int main(int argc, char** argv)
{
    NVIGIAppCtx nvigiCtx;

#ifdef NVIGI_WINDOWS
    FILE* f{};
    freopen_s(&f, "NUL", "w", stderr);
#else
    freopen("dev/nul", "w", stderr);
#endif

    auto exePathUtf8 = getExecutablePath();

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
    if (InitNVIGI(nvigiCtx, exePathUtf8))
        return -1;

    uint32_t vram = 1024 * 12;

    //////////////////////////////////////////////////////////////////////////////
    //! Init Plugin Interfaces and Instances
    //! 
    if (CreatePipeline(nvigiCtx, modelDir, vram))
        return -1;

    //////////////////////////////////////////////////////////////////////////////
    //! Run inference
    //! 
    auto checkPath = [](std::string const& s)
        {
            return s.size() < 512
                && s.find("://") == s.npos;
        };
    if (!checkPath(audioFile))
    {
        loggingCallback(nvigi::LogType::eError, "Provided audio file path does not meet requirements. Provided path must be at most 512 char long and must not contain '://'");
        return -1;
    }
    auto wav = read(audioFile.c_str());
    if (wav.empty())
    {
        loggingCallback(nvigi::LogType::eError, "Could not load input WAV file");
        return -1;
    }

    std::string promptText("This is a conversation between John F. Kennedy (JFK), the late USA president and person named Bob. Bob's answers are short and on the point.\nJFK: ");

    std::string reversePromptText("JFK:");

    if (RunInference(nvigiCtx, wav, promptText, reversePromptText))
        return -1;

    //////////////////////////////////////////////////////////////////////////////
    //! Shutdown NVIGI
    //! 

    if (ReleasePipeline(nvigiCtx))
        return -1;

    if (ShutdownNVIGI(nvigiCtx))
        return -1;

    return 0;
}