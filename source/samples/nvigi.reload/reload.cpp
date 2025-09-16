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
#include <iostream>
#include <thread>
#include <memory>
#include <queue>
#include <type_traits>
#include <mutex>
#include <atomic>
#include <chrono>


#include <map>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <conio.h>

#include "d3d12.h"

#ifdef NVIGI_WINDOWS
extern "C" __declspec(dllexport) UINT         D3D12SDKVersion = 615;
extern "C" __declspec(dllexport) const char* D3D12SDKPath = ".\\D3D12\\";
#endif

namespace fs = std::filesystem;

#if NVIGI_WINDOWS
#include <windows.h>
#endif

#include <nvigi.h>
#include <nvigi_d3d12.h>
#include "nvigi_gpt.h"
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

struct NVIGIAppCtx
{
    HMODULE coreLib{};
    nvigi::IGeneralPurposeTransformer* igpt{};
    nvigi::PluginID gptId{};
    struct GPTModel
    {
        nvigi::InferenceInstance* inst{};
        std::map< ID3D12Resource*, bool> resourceMap{};
    };
    std::map<std::string, GPTModel*> gptModels{};
    GPTModel* currentModel{};
};

///////////////////////////////////////
//! Full pipeline inference context

struct ReloadCallbackCtx
{
    std::mutex callbackMutex;
    std::condition_variable gptCallbackCV;
    std::atomic<nvigi::InferenceExecutionState> gptCallbackState = nvigi::kInferenceExecutionStateDataPending;
    std::string gptOutput;

    nvigi::InferenceInstance* gptInstance{};
    bool conversationInitialized = false;
};

static NVIGIAppCtx nvigiCtx;
constexpr uint32_t n_threads = 4;

class CommandLineParser {
public:
    struct Command {
        std::string short_name;
        std::string long_name;
        std::string description;
        std::string default_value;
        std::string value;
        bool required = false;
    };

    void add_command(const std::string& short_name, const std::string& long_name,
        const std::string& description, const std::string& default_value = "",
        bool required = false) {
        if (!short_name.empty())
        {
            commands[short_name] = { short_name, long_name, description, default_value, default_value, required };
        }
        commands[long_name] = { short_name, long_name, description, default_value, default_value, required };
    }

    void parse(int argc, char* argv[]) {
        std::vector<std::string> args(argv + 1, argv + argc);
        for (size_t i = 0; i < args.size(); ++i) {
            std::string arg = args[i];

            if (arg[0] == '-') {
                std::string key = (arg[1] == '-') ? arg.substr(2) : arg.substr(1); // Long or short name
                auto it = commands.find(key);
                if (it == commands.end()) {
                    throw std::invalid_argument("Unknown command: " + arg);
                }

                Command& cmd = it->second;
                if (i + 1 < args.size() && args[i + 1][0] != '-') {
                    cmd.value = args[++i]; // Take the next argument as the value
                    auto altKey = cmd.long_name == key ? cmd.short_name : cmd.long_name;
                    commands[altKey].value = cmd.value;
                }
                else if (cmd.default_value.empty()) {
                    throw std::invalid_argument("Missing value for command: " + arg);
                }
            }
            else {
                throw std::invalid_argument("Unexpected argument format: " + arg);
            }
        }

        // Check required commands
        for (const auto& [key, cmd] : commands) {
            if (cmd.required && cmd.value == cmd.default_value) {
                throw std::invalid_argument("Missing required command: --" + cmd.long_name);
            }
        }
    }

    std::string get(const std::string& name) const {
        auto it = commands.find(name);
        if (it == commands.end()) {
            throw std::invalid_argument("Unknown command: " + name);
        }
        return it->second.value;
    }

    bool has(const std::string& name) const {
        auto it = commands.find(name);
        if (it == commands.end()) {
            return false;
        }
        return !it->second.value.empty();
    }

    void print_help(const std::string& program_name) const {
        std::cout << "Usage: " << program_name << " [options]\n";
        for (const auto& [key, cmd] : commands) {
            if (key == cmd.long_name) { // Print each command only once
                std::string tmp;
                if (!cmd.short_name.empty())
                    tmp = "  -" + cmd.short_name + ", --" + cmd.long_name;
                else
                    tmp = "  --" + cmd.long_name;
                std::string spaces(std::max(0, 20 - (int)tmp.size()), ' ');
                std::cout << tmp << spaces << cmd.description << " (default: " << cmd.default_value << ")\n";
            }
        }
    }

private:
    std::map<std::string, Command> commands;
};

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

void LogVRAM(const char* prefix = nullptr)
{
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    nvigi::D3D12ContextInfo::GetActiveInstance()->adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info);
    char msg[1024];
    snprintf(msg, 1024, "\n%s: %0.2fMB\n", prefix ? prefix : "**** Current VRAM", info.CurrentUsage / 1000000.0f);
    loggingCallback(nvigi::LogType::eInfo, msg);
}

ID3D12Resource* createCommittedResource(
    ID3D12Device* device, const D3D12_HEAP_PROPERTIES* pHeapProperties,
    D3D12_HEAP_FLAGS HeapFlags, const D3D12_RESOURCE_DESC* pDesc,
    D3D12_RESOURCE_STATES InitialResourceState, const D3D12_CLEAR_VALUE* pOptimizedClearValue,
    void* userContext
)
{
	NVIGIAppCtx::GPTModel* model = nvigiCtx.currentModel;
    ID3D12Resource* resource = nullptr;
    HRESULT hr = device->CreateCommittedResource(pHeapProperties, HeapFlags, pDesc, InitialResourceState, pOptimizedClearValue, IID_PPV_ARGS(&resource));
    if (FAILED(hr))
    {
        // Handle error
        loggingCallback(nvigi::LogType::eError, "Resource could not be allocated!");
        return nullptr;
    }
    if (model)
    {
        if (model->resourceMap.find(resource) == model->resourceMap.end())
        {
            model->resourceMap[resource] = true; // Mark resource as resident
        }
        else
        {
            loggingCallback(nvigi::LogType::eError, "Resource already exists in the map, this should not happen!");
        }
    }
    return resource;
}

void destroyResource(ID3D12Resource* pResource, void* userContext)
{
    NVIGIAppCtx::GPTModel* model = nvigiCtx.currentModel;
    if (model)
    {
        if (model->resourceMap.find(pResource) == model->resourceMap.end())
        {
            loggingCallback(nvigi::LogType::eError, "Resource not found in the map, trying all maps!");

            model = nullptr;
            for (auto it : nvigiCtx.gptModels)
            {
                auto& resMap = it.second->resourceMap;
                if (resMap.find(pResource) != resMap.end())
                {
                    model = it.second; // Found the model
                    break;
                }
            }

            if (!model)
            {
                loggingCallback(nvigi::LogType::eError, "Resource not found in any model map, this should not happen!");
                pResource->Release();
                return; // Resource not found in any model's map
            }
        }

        model->resourceMap.erase(pResource); // Remove resource from the map
    }
    pResource->Release();
}

void SetResourceMapResidency(NVIGIAppCtx::GPTModel* model, bool resident)
{
    std::vector<ID3D12Pageable*> resourcesToUpdate;
    for (auto& it : model->resourceMap)
    {
        if (it.second != resident)
        {
            ID3D12Pageable* pageable = nullptr;
            it.first->QueryInterface(&pageable);
            resourcesToUpdate.push_back(pageable);
            it.second = resident; // Update the residency status - this does assume later success...
        }
    }

    if (resident)
    {
        // Make resources resident
        if (nvigi::D3D12ContextInfo::GetActiveInstance()->device->MakeResident(resourcesToUpdate.size(), resourcesToUpdate.data()) != S_OK)
        {
            loggingCallback(nvigi::LogType::eError, "Failed to make resources resident");
        }
    }
    else
    {
        // Evict resources
        if (nvigi::D3D12ContextInfo::GetActiveInstance()->device->Evict(resourcesToUpdate.size(), resourcesToUpdate.data()) != S_OK)
        {
            loggingCallback(nvigi::LogType::eError, "Failed to evict resources");
        }
    }
}

void SetResourceMapResidency(const std::string& name, bool resident)
{
    NVIGIAppCtx::GPTModel* model = nullptr;
    if (nvigiCtx.gptModels.find(name) != nvigiCtx.gptModels.end())
    {
        model = nvigiCtx.gptModels[name];
    }
    else
    {
        loggingCallback(nvigi::LogType::eError, "Model not found in the map");
        return;
    }
    SetResourceMapResidency(model, resident);
}


///////////////////////////////////////
//! NVIGI Init and Shutdown

int InitNVIGI(const std::string& pathToSDKUtf8)
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
    pref.logLevel = nvigi::LogLevel::eVerbose;
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

    //! GPT Interface and Instance
    if (!nvigiCtx.igpt)
    {
        nvigiCtx.gptId = nvigi::plugin::gpt::ggml::d3d12::kId;
        if (NVIGI_FAILED(result, nvigiGetInterfaceDynamic(nvigiCtx.gptId, &nvigiCtx.igpt, ptr_nvigiLoadInterface)))
        {
            loggingCallback(nvigi::LogType::eError, "Could not query GPT interface");
            return -1;
        }
    }

    return 0;
}

int ShutdownNVIGI()
{
    // Hard-coded to local
    if (NVIGI_FAILED(result, ptr_nvigiUnloadInterface(nvigiCtx.gptId, nvigiCtx.igpt)))
    {
        loggingCallback(nvigi::LogType::eError, "Error in 'nvigiUnloadInterface'");
        return -1;
    }

    if (NVIGI_FAILED(result, ptr_nvigiShutdown()))
    {
        loggingCallback(nvigi::LogType::eError, "Error in 'nvigiShutdown'");
        return -1;
    }

    FreeLibrary(nvigiCtx.coreLib);

    return 0;
}


///////////////////////////////////////
//! GPT Init and Release

int InitGPT(const std::string& modelDir, const std::string& modelName, const std::string& guidGPT, size_t vramBudgetMB)
{
    // GPT specific
    nvigi::GPTCreationParameters gptParams{};
    gptParams.contextSize = 4096;
    // Common
    nvigi::CommonCreationParameters gptCommon{};
    gptCommon.utf8PathToModels = modelDir.c_str();
    gptCommon.numThreads = n_threads;
    // Chain together specific and common
    if (NVIGI_FAILED(result, gptCommon.chain(gptParams)))
    {
        loggingCallback(nvigi::LogType::eError, "GPT param chaining failed");
        return -1;
    }

	nvigi::D3D12Parameters d3d12Params{};
    d3d12Params.device = nvigi::D3D12ContextInfo::GetActiveInstance()->device.Get();
	d3d12Params.queue = nvigi::D3D12ContextInfo::GetActiveInstance()->d3d_direct_queue.Get();
	d3d12Params.queueCompute = nvigi::D3D12ContextInfo::GetActiveInstance()->d3d_compute_queue.Get();
	d3d12Params.queueCopy = nvigi::D3D12ContextInfo::GetActiveInstance()->d3d_copy_queue.Get();
    d3d12Params.createCommittedResourceCallback = createCommittedResource;
    d3d12Params.destroyResourceCallback = destroyResource;
    if (NVIGI_FAILED(result, gptCommon.chain(d3d12Params)))
    {
        loggingCallback(nvigi::LogType::eError, "D3D12 param chaining failed");
        return -1;
    }

    //! Obtain capabilities and requirements for GPT model(s)
    //! 
    //! Few options here:
    //! 
    //! LOCAL
    //! 
    //! * provide specific model GUID and VRAM budget and check if that particular model can run within the budget
    //! * provide null model GUID and VRAM budget to get a list of models that can run within the budget
    //! * provide null model GUID and 'infinite' (SIZE_MAX) VRAM budget to get a list of ALL models
    //! 

    //! Here we are selection option #1 - specific model GUID and VRAM budget
    //! 
    //! To obtain all models we could do something like this:
    //! 
    //! gptCommon.modelGUID = nullptr;
    //! gptCommon.vramBudgetMB = SIZE_MAX;
    //! 
    gptCommon.modelGUID = guidGPT.c_str();
    gptCommon.vramBudgetMB = vramBudgetMB;

    nvigi::CommonCapabilitiesAndRequirements* caps{};
    if (NVIGI_FAILED(result, getCapsAndRequirements(nvigiCtx.igpt, gptCommon, &caps)))
    {
        loggingCallback(nvigi::LogType::eError, "'getCapsAndRequirements' failed");
        return -1;
    }

    //! We provided model GUID and VRAM budget so caps and requirements will contain just one model, assuming VRAM budget is sufficient or if cloud backend is selected!
    //! 
    //! NOTE: This will be >=1 if we provide null as modelGUID in common creation parameters
    if (caps->numSupportedModels != 1)
    {
        loggingCallback(nvigi::LogType::eError, "'getCapsAndRequirements' failed to find our model or model cannot run on system given the VRAM restrictions");
        return -1;
    }

    NVIGIAppCtx::GPTModel* model = new NVIGIAppCtx::GPTModel;
    nvigiCtx.currentModel = model;

    if (NVIGI_FAILED(result, nvigiCtx.igpt->createInstance(gptCommon, &(model->inst))))
    {
        loggingCallback(nvigi::LogType::eError, "Could not create GPT instance");
        return -1;
    }

    nvigiCtx.gptModels.insert_or_assign(modelName, model);
    return 0;
}

int DeleteModel(const std::string& modelName)
{
    if (nvigiCtx.gptModels.contains(modelName))
    {
        NVIGIAppCtx::GPTModel* deleteModel = nvigiCtx.gptModels[modelName];
        NVIGIAppCtx::GPTModel* prevModel = (nvigiCtx.currentModel == deleteModel) ? nullptr : nvigiCtx.currentModel;
        nvigiCtx.currentModel = deleteModel;

        if (nvigiCtx.currentModel != nullptr)
        {
            nvigiCtx.igpt->destroyInstance(nvigiCtx.currentModel->inst);
            nvigiCtx.currentModel->inst = nullptr;
        }
        delete deleteModel; // Clean up the model
        nvigiCtx.gptModels[modelName] = nullptr;
        nvigiCtx.gptModels.erase(modelName);

        nvigiCtx.currentModel = prevModel;
        return 0;
    }
    else
    {
        loggingCallback(nvigi::LogType::eError, "GPT model not found in the context");
        return -1;
    }
}

int ReleaseGPT(const std::string& modelName = "")
{
    if (modelName != "")
    {
        return DeleteModel(modelName);
    }
    else
    {
        // Loop over all models and destroy
        while (nvigiCtx.gptModels.size())
        {
            if (DeleteModel(nvigiCtx.gptModels.begin()->first))
            {
                loggingCallback(nvigi::LogType::eError, "Failed to delete model");
                return -1;
            }
        }
        nvigiCtx.currentModel = nullptr;
    }
    return 0;
}

///////////////////////////////////////
//! GPT inference

nvigi::InferenceExecutionState GPTInferenceDataCallback(const nvigi::InferenceExecutionContext* ctx, nvigi::InferenceExecutionState state, void* userData)
{
    auto cbkCtx = (ReloadCallbackCtx*)userData;
    std::scoped_lock lck(cbkCtx->callbackMutex);

    // Outputs from GPT
    auto slots = ctx->outputs;
    const nvigi::InferenceDataText* text{};
    slots->findAndValidateSlot(nvigi::kGPTDataSlotResponse, &text);
    auto response = std::string(text->getUTF8Text());
    if (cbkCtx->conversationInitialized)
    {
        cbkCtx->gptOutput += response;
        loggingCallback(nvigi::LogType::eInfo, response.c_str());
    }

    cbkCtx->gptCallbackState.store(state);
    cbkCtx->gptCallbackCV.notify_one();

    return state;
}

int GPTInference(const std::string& modelName, std::string& gptInputText, bool conversationInitialized)
{
	NVIGIAppCtx::GPTModel* model = nullptr;
    if (nvigiCtx.gptModels.contains(modelName))
	{
		model = nvigiCtx.gptModels[modelName];
	}
	else
	{
		loggingCallback(nvigi::LogType::eError, "GPT model not found in the context");
		return -1;
	}

	nvigiCtx.currentModel = model;
       
    ReloadCallbackCtx cbkCtx{};
    cbkCtx.conversationInitialized = conversationInitialized;
    cbkCtx.gptInstance = model->inst;

    SetResourceMapResidency(model, true);

    //! GPT
    nvigi::InferenceExecutionContext gptExecCtx{};
    gptExecCtx.instance = model->inst;
    gptExecCtx.callback = &GPTInferenceDataCallback;
    gptExecCtx.callbackUserData = &cbkCtx;
    cbkCtx.gptCallbackState.store(nvigi::kInferenceExecutionStateDataPending);
    cbkCtx.gptOutput = "";

    nvigi::GPTRuntimeParameters runtime{};
    runtime.seed = -1;
    runtime.tokensToPredict = 200;
    runtime.interactive = true;
    runtime.reversePrompt = "User: ";
    gptExecCtx.runtimeParameters = runtime;

    nvigi::InferenceDataTextSTLHelper text(gptInputText);
    std::vector<nvigi::InferenceDataSlot> slots = {
        {cbkCtx.conversationInitialized ? nvigi::kGPTDataSlotUser : nvigi::kGPTDataSlotSystem, text} };
    nvigi::InferenceDataSlotArray inputs = { slots.size(), slots.data() };
    gptExecCtx.inputs = &inputs;

    loggingCallback(nvigi::LogType::eInfo, "** Assistant:\n");
    cbkCtx.gptCallbackState.store(nvigi::kInferenceExecutionStateDataPending);
    std::thread infer([&gptExecCtx, &model]()
        {
            model->inst->evaluate(&gptExecCtx);
        });

    // Wait for the GPT to stop returning eDataPending in the callback
    {
        std::unique_lock lck(cbkCtx.callbackMutex);
        cbkCtx.gptCallbackCV.wait(lck, [&cbkCtx]() { return cbkCtx.gptCallbackState != nvigi::kInferenceExecutionStateDataPending; });
        if (cbkCtx.gptCallbackState != nvigi::kInferenceExecutionStateDone)
        {
            loggingCallback(nvigi::LogType::eError, "GPT Inference error!\n");
            return -1;
        }
    }
    infer.join();

    return 0;
}


int main(int argc, char** argv)
{
    CommandLineParser parser;
    parser.add_command("s", "sdk", " sdk location, if none provided assuming exe location", "");
    parser.add_command("m", "models", " model repo location", "", true);
    parser.add_command("", "vram", " the amount of vram to use in MB", "8192");

    try {
        parser.parse(argc, argv);
    }
    catch (std::exception& e)
    {
        printf("%s\n\n", e.what());
        parser.print_help("nvigi.reload");
        exit(1);
    }

    auto pathToSDKArgument = parser.get("sdk");
    auto pathToSDKUtf8 = pathToSDKArgument.empty() ? getExecutablePath() : pathToSDKArgument;

    // Mandatory so we know that they are provided
    std::string modelDir = parser.get("models");

    // Defaults
    size_t vramBudgetMB = (size_t)atoi(parser.get("vram").c_str());

    if (!nvigi::D3D12ContextInfo::GetActiveInstance())
        nvigi::D3D12ContextInfo::CreateD3D12Device();

    //////////////////////////////////////////////////////////////////////////////
    //! Init NVIGI
    if (InitNVIGI(pathToSDKUtf8))
        return -1;

    //////////////////////////////////////////////////////////////////////////////
    //! Init Plugin Interfaces and Instances
    //! 
    {
        LogVRAM("Pre-model loading");
        if (InitGPT(modelDir, "llama3", "{01F43B70-CE23-42CA-9606-74E80C5ED0B6}", vramBudgetMB))
            return -1;
        LogVRAM("Post-llama3 loading");
        if (InitGPT(modelDir, "nemotron", "{8E31808B-C182-4016-9ED8-64804FF5B40D}", vramBudgetMB))
            return -1;
        LogVRAM("Post nemotron loading");
    }

    for (auto& it : nvigiCtx.gptModels)
    {
        SetResourceMapResidency(it.second, false);
    }
    LogVRAM("Post eviction of all models");

    {
        //////////////////////////////////////////////////////////////////////////////
        //! Run inference
        //! 
        bool running = true;
        bool conversationInitialized = false;
		bool runInference = true;
		std::string gptModelName = "llama3";
        std::string gptInputText = "This is a transcript of a dialog between a user and a helpful AI assistant.\
 Generate only medium size answers and avoid describing what you are doing physically.\
 Avoid using specific words that are not part of the dictionary.\n";

        do
        {
            if (runInference)
            {
                if (GPTInference(gptModelName, gptInputText, conversationInitialized))
                    return -1;
            }
            else
            { 
                loggingCallback(nvigi::LogType::eInfo, "\nInference Skipped\n");
            }

            LogVRAM();

            runInference = false;

            if (!conversationInitialized)
            {
                loggingCallback(nvigi::LogType::eInfo, "\n** Please continue the converation ('q' or 'quit' to exit, any other text to type your query\n");
            }
            loggingCallback(nvigi::LogType::eInfo, "\n>:");

            conversationInitialized = true;

            std::getline(std::cin, gptInputText);
            if (gptInputText == "q" || gptInputText == "Q" || gptInputText == "quit")
            {
                loggingCallback(nvigi::LogType::eInfo, "Exiting - user request\n");
                running = false;
            }
            else if (gptInputText == "<unload>")
            {
                loggingCallback(nvigi::LogType::eInfo, "\n** Unloading all models\n");
                for (auto& it : nvigiCtx.gptModels)
                {
                    SetResourceMapResidency(it.second, false);
                }
            }
            else if (gptInputText == "<llama3>")
            {
                if (gptModelName != "llama3")
                {
                    loggingCallback(nvigi::LogType::eInfo, "\n** Loading llama3 model\n");
                    SetResourceMapResidency(gptModelName, false);
                    gptModelName = "llama3";
                    conversationInitialized = false;
                }
                else
                {
                    loggingCallback(nvigi::LogType::eInfo, "\n** Already using llama3 model\n");
				}
            }
            else if (gptInputText == "<nemotron>")
            {
                if (gptModelName != "nemotron")
                {
                    loggingCallback(nvigi::LogType::eInfo, "\n** Loading nemotron model\n");
                    SetResourceMapResidency(gptModelName, false);
                    gptModelName = "nemotron";
                    conversationInitialized = false;
                }
                else
                {
                    loggingCallback(nvigi::LogType::eInfo, "\n** Already using nemotron model\n");
				}
            }
            else if (gptInputText[0] == '<')
            {
                loggingCallback(nvigi::LogType::eInfo, "\n** Unknown \"<\" keyword.  This app supports <unload>, <llama3> and <nemotron>\n");
            }
            else
            {
                // Use the given getline result as the text
                runInference = true;
            }
        } while (running);
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Shutdown NVIGI
    //! 
    if (ReleaseGPT())
        return -1;

    nvigiCtx.currentModel = nullptr;

    if (ShutdownNVIGI())
        return -1;

    return 0;
}
