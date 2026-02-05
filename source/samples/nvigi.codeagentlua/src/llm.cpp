// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "llm.h"


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
#include <sstream>
#include <regex>
#include <thread>
#include <format>

namespace fs = std::filesystem;

#if NVIGI_WINDOWS
#include <windows.h>
#endif

#include <nvigi.h>
#include "nvigi_gpt.h"
#include "nvigi_types.h"
#include <nvigi_stl_helpers.h>

extern void loggingPrint(nvigi::LogType type, const char* msg);

NVIGIAppCtx nvigiCtx;

std::string system_prompt;

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
}   LUID;
#endif

#define DECLARE_NVIGI_CORE_FUN(F) PFun_##F* ptr_##F
#define GET_NVIGI_CORE_FUN(lib, F) ptr_##F = (PFun_##F*)GetProcAddress(lib, #F)
DECLARE_NVIGI_CORE_FUN(nvigiInit);
DECLARE_NVIGI_CORE_FUN(nvigiShutdown);
DECLARE_NVIGI_CORE_FUN(nvigiLoadInterface);
DECLARE_NVIGI_CORE_FUN(nvigiUnloadInterface);

typedef std::vector< std::string > StringVec;
typedef std::vector< std::vector< float > > VectorStore;
typedef std::pair< size_t, float > IndexScore;
typedef std::vector< std::pair<size_t, float> > IndexScoreVec;

const uint32_t vram = 1024 * 12;    // maximum vram available in GB

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
        loggingPrint(nvigi::LogType::eError, "Failed to unload interface");
        return false;
    }

    return true;
}

///////////////////////////////////////
//! NVIGI Init and Shutdown

int InitNVIGI(NVIGIAppCtx& nvigiCtx, const std::string& pathToSDKUtf8)
{
    loggingPrint(nvigi::LogType::eInfo, "Initializing NVIGI\n");

#ifdef NVIGI_WINDOWS
    auto libPath = pathToSDKUtf8 + "/nvigi.core.framework.dll";
#else
    auto libPath = pathToSDKUtf8 + "/nvigi.core.framework.so";
#endif
    nvigiCtx.coreLib = LoadLibraryA(libPath.c_str());
    if (nvigiCtx.coreLib == nullptr)
    {
        loggingPrint(nvigi::LogType::eError, "Could not load NVIGI core library");
        return -1;
    }

    GET_NVIGI_CORE_FUN(nvigiCtx.coreLib, nvigiInit);
    GET_NVIGI_CORE_FUN(nvigiCtx.coreLib, nvigiShutdown);
    GET_NVIGI_CORE_FUN(nvigiCtx.coreLib, nvigiLoadInterface);
    GET_NVIGI_CORE_FUN(nvigiCtx.coreLib, nvigiUnloadInterface);

    if (ptr_nvigiInit == nullptr || ptr_nvigiShutdown == nullptr ||
        ptr_nvigiLoadInterface == nullptr || ptr_nvigiUnloadInterface == nullptr)
    {
        loggingPrint(nvigi::LogType::eError, "Could not load NVIGI core library");
        return -1;
    }

    const char* paths[] =
    {
        pathToSDKUtf8.c_str()
    };

    nvigi::Preferences pref{};
    pref.logLevel = nvigi::LogLevel::eVerbose;
    pref.showConsole = false;
    pref.numPathsToPlugins = 1;
    pref.utf8PathsToPlugins = paths;
    pref.logMessageCallback = nullptr; // loggingCallback; // avoid duplicating logs in the console
    pref.utf8PathToLogsAndData = pathToSDKUtf8.c_str();

    if (NVIGI_FAILED(result, ptr_nvigiInit(pref, nullptr, nvigi::kSDKVersion)))
    {
        loggingPrint(nvigi::LogType::eError, "NVIGI init failed");
        return -1;
    }

    loggingPrint(nvigi::LogType::eInfo, "Initializing NVIGI succeeded\n");

    return 0;
}

int ShutdownNVIGI(NVIGIAppCtx& nvigiCtx)
{
    if (NVIGI_FAILED(result, ptr_nvigiShutdown()))
    {
        loggingPrint(nvigi::LogType::eError, "Error in 'nvigiShutdown'");
        return -1;
    }

    FreeLibrary(nvigiCtx.coreLib);

    return 0;
}

int InitGPT(NVIGIAppCtx& nvigiCtx, const std::string& modelDir)
{
    loggingPrint(nvigi::LogType::eInfo, "Initializing GPT\n");

    //! GPT Interface
    if (NVIGI_FAILED(result, nvigiGetInterfaceDynamic(nvigi::plugin::gpt::ggml::cuda::kId, &nvigiCtx.igpt, ptr_nvigiLoadInterface)))
    {
        loggingPrint(nvigi::LogType::eError, "Could not query GPT interface");
        return -1;
    }

    nvigi::GPTCreationParameters gptParams{};
    nvigi::CommonCreationParameters gptCommon{};
    gptCommon.utf8PathToModels = modelDir.c_str();
    gptCommon.numThreads = 16;
    gptCommon.vramBudgetMB = vram;
    gptParams.contextSize = 32000;
    gptCommon.modelGUID = "{545F7EC2-4C29-499B-8FC8-61720DF3C626}"; // Qwen3-8B-Q4_K_M
    if (NVIGI_FAILED(result, gptCommon.chain(gptParams)))
    {
        loggingPrint(nvigi::LogType::eError, "GPT param chaining failed");
        return -1;
    }

    nvigi::CommonCapabilitiesAndRequirements* info{};
    getCapsAndRequirements(nvigiCtx.igpt, gptCommon, &info);
    if (info == nullptr)
        return -1;

    auto result = nvigiCtx.igpt->createInstance(gptCommon, &nvigiCtx.gptInst);

    loggingPrint(nvigi::LogType::eInfo, "Initializing GPT succeeded\n");

    return result;
}

int ReleaseGPT(NVIGIAppCtx& nvigiCtx)
{
    if (nvigiCtx.gptInst == nullptr)
        return -1;

    if (NVIGI_FAILED(result, nvigiCtx.igpt->destroyInstance(nvigiCtx.gptInst)))
    {
        loggingPrint(nvigi::LogType::eError, "Failed to destroy GPT instance");
        return -1;
    }

    if (!unloadInterface(nvigi::plugin::gpt::ggml::cuda::kId, nvigiCtx.igpt))
    {
        loggingPrint(nvigi::LogType::eError, "Failed to release GPT interface");
        return -1;
    }

    return 0;
}

void GetCompletion(NVIGIAppCtx& nvigiCtx, const std::string& system_prompt, const std::string& user_prompt, std::string& answer)
{
    answer = "";

    nvigi::InferenceDataTextSTLHelper system_data(system_prompt);
    nvigi::InferenceDataTextSTLHelper user_data(user_prompt);

    std::vector<nvigi::InferenceDataSlot> inSlots;
    if (!system_prompt.empty())
        inSlots.push_back({ nvigi::kGPTDataSlotSystem, system_data });
    inSlots.push_back({ nvigi::kGPTDataSlotUser, user_data });

    nvigi::InferenceDataSlotArray inputs = { inSlots.size(), inSlots.data() };

    nvigi::GPTRuntimeParameters runtime{};
    runtime.seed = -1;
    runtime.tokensToPredict = 15000;
    runtime.interactive = true;
    runtime.reversePrompt = "User: ";

    struct UserDataBlock
    {
        std::atomic<bool> done = false;
        std::string response;
        bool terminator_found = false;
    };
    UserDataBlock userData;

    nvigi::InferenceExecutionContext ctx{};
    ctx.instance = nvigiCtx.gptInst;
    ctx.callbackUserData = &userData;
    ctx.callback = [](const nvigi::InferenceExecutionContext* ctx, nvigi::InferenceExecutionState state, void* userData)->nvigi::InferenceExecutionState
        {
            UserDataBlock* userDataBlock = static_cast<UserDataBlock*>(userData);
            if (ctx)
            {
                auto slots = ctx->outputs;
                const nvigi::InferenceDataText* text{};
                slots->findAndValidateSlot(nvigi::kGPTDataSlotResponse, &text);
                auto response = std::string((const char*)text->getUTF8Text());

                if (response == "</s>")
                {
                    // For Nemotron, the </s> character denotes end of stream.  
                    // Must wait for state to be kInferenceEecutionStateDone before evaluation is finished though.
                    userDataBlock->terminator_found = true;
                }

                if (state == nvigi::kInferenceExecutionStateDone)
                    response += "\n\n";

                if (!userDataBlock->terminator_found)
                {
                    userDataBlock->response += response;
                }
            }
            userDataBlock->done.store(state == nvigi::kInferenceExecutionStateDone);
            return state;
        };

    ctx.inputs = &inputs;
    ctx.runtimeParameters = runtime;

    if (ctx.instance->evaluate(&ctx) != nvigi::kResultOk)
    {
        loggingPrint(nvigi::LogType::eError, "GPT evaluate failed");
    }
    // ctx is held in this scope, so we can't let it go out of scope while the LLM is evaluating.
    while (!userData.done);

    answer = userData.response;
}

int llmInit(const std::string& modelDir, const std::string& systemPromptPath)
{
    // Block the llama output, so it does not pollute the app's console output
#ifdef NVIGI_WINDOWS
    FILE* f{};
    freopen_s(&f, "NUL", "w", stderr);
#else
    freopen("/dev/null", "w", stderr);
#endif

    if (modelDir.empty())
    {
        loggingPrint(nvigi::LogType::eError, "nvigi.codeagentlua no model dir provided");
        return -1;
    }
    auto pathToSDKUtf8 = getExecutablePath();

    if (InitNVIGI(nvigiCtx, pathToSDKUtf8))
        return -1;

    //////////////////////////////////////////////////////////////////////////////
    //! Init Plugin Interfaces and Instances

    //! GPT Instance
    if (InitGPT(nvigiCtx, modelDir))
    {
        loggingPrint(nvigi::LogType::eError, "Could not create GPT instance");
        return -1;
    }

    if (std::filesystem::exists(systemPromptPath))
    {
        std::ifstream infile(systemPromptPath);
        std::stringstream buffer;
        buffer << infile.rdbuf();
        system_prompt = buffer.str();
    }
    else
    {
        return -1;
    }

    return 0;
}

inline void trim(std::string& s)
{
    // left
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
        [](unsigned char ch){ return !std::isspace(ch); }));

    // right
    s.erase(std::find_if(s.rbegin(), s.rend(),
        [](unsigned char ch){ return !std::isspace(ch); }).base(), s.end());
}

std::string sanitizeAIResponse( std::string& input )
{
    // For QWen reasoning models, remove all the thinking sections.
    std::regex pattern(R"(<think>[\s\S]*?</think>)", std::regex::icase);
    std::string cleaned_answer = std::regex_replace(input, pattern, "");

    // At times Qwen may only return </think> instead of <think>...</think>, especially in non-reasoning mode on smaller models.
    // This captures that case.
    if (cleaned_answer.find("</think>") != std::string::npos)
    {
        std::regex end_think_pattern(R"(</think>)", std::regex::icase);
        cleaned_answer = std::regex_replace(cleaned_answer, end_think_pattern, "");
    }

    // remove leading or trailing whitespace.
    trim( cleaned_answer );

    return cleaned_answer;
}

std::string llmCreateAIFunc(const std::string& prompt)
{
    std::string answer;
    GetCompletion(nvigiCtx, system_prompt, prompt, answer);
    answer = sanitizeAIResponse( answer );

    std::string filename = "ai_func_out.txt";
    std::ofstream outfile(filename); // overwrite mode (default)
    if( !outfile.is_open() )
    {
        loggingPrint(nvigi::LogType::eError, "llmCreateAIFunc: Unable to open file" );
        return answer;
    }
    
    outfile << answer;
    outfile.close();

    return answer;
}

int llmShutdown()
{
    if (ReleaseGPT(nvigiCtx))
        return -1;

    return ShutdownNVIGI(nvigiCtx);
}
