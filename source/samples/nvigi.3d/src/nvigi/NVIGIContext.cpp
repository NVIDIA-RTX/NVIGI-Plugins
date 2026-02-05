// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING 1

#include "NVIGIContext.h"

#include <donut/core/math/math.h>
#include <donut/core/math/basics.h>
#include <imgui.h>
#include <donut/app/imgui_renderer.h>

#if USE_DX12
#include <d3d12.h>
#include <nvrhi/d3d12.h>
#endif

#include "nvigi_vulkan.h"

// NVIGI headers (many come via wrappers in NVIGIContext.h)
#include <nvigi.h>
#include <nvigi_ai.h>               // For kModelFlagRequiresDownload
#include <nvigi_hwi_common.h>      // For IHWICommon interface
#include <nvigi_hwi_cuda.h>        // For CUDA plugin ID
#include <nvigi_security.h>        // For verifyEmbeddedSignature
#include <nvigi_stl_helpers.h>     // For CpuData, InferenceDataAudio
#include <source/utils/nvigi.dsound/player.h>  // For audio playback
#include "cxx_wrappers/asr/audio.hpp"

// Note: nvigi_gpt.h, nvigi_asr_whisper.h, nvigi_tts.h, nvigi_cloud.h are included via wrappers

// Standard library headers (some redundant with NVIGIContext.h but kept for clarity)
#include <assert.h>
#include <codecvt>       // For wstring_convert
#include <filesystem>    // For path operations
#include <regex>         // For regex operations

#ifndef _WIN32
#include <unistd.h>
#include <cstdio>
#include <climits>
#else
#define PATH_MAX MAX_PATH
#endif // _WIN32

extern "C" __declspec(dllexport) UINT        D3D12SDKVersion = 615;
extern "C" __declspec(dllexport) const char* D3D12SDKPath = ".\\D3D12\\";

namespace fs = std::filesystem;

struct Message {
    enum class Type {
        Question,
        Answer
    } type;
    std::string text;
};

static std::vector<Message> messages =
{
};

constexpr ImU32 TITLE_COL = IM_COL32(0, 255, 0, 255);

// Backend strings are used instead of plugin IDs for cleaner code
// Empty string "" represents "no backend available"

static std::wstring GetNVIGICoreDllLocation() {

    char path[PATH_MAX] = { 0 };
#ifdef _WIN32
    if (GetModuleFileNameA(nullptr, path, dim(path)) == 0)
        return std::wstring();
#else // _WIN32
    // /proc/self/exe is mostly linux-only, but can't hurt to try it elsewhere
    if (readlink("/proc/self/exe", path, std::size(path)) <= 0)
    {
        // portable but assumes executable dir == cwd
        if (!getcwd(path, std::size(path)))
            return ""; // failure
    }
#endif // _WIN32

    auto basePath = std::filesystem::path(path).parent_path();
    auto dllPath = basePath.wstring().append(L"\\nvigi.core.framework.dll");
    return dllPath;
}

static std::wstring GetNVIGICoreDllPath() {

    char path[PATH_MAX] = { 0 };
#ifdef _WIN32
    if (GetModuleFileNameA(nullptr, path, dim(path)) == 0)
        return std::wstring();
#else // _WIN32
    // /proc/self/exe is mostly linux-only, but can't hurt to try it elsewhere
    if (readlink("/proc/self/exe", path, std::size(path)) <= 0)
    {
        // portable but assumes executable dir == cwd
        if (!getcwd(path, std::size(path)))
            return ""; // failure
    }
#endif // _WIN32

    auto basePath = std::filesystem::path(path).parent_path();
    return basePath;
}

// Helper function to query plugin capabilities
inline std::expected<nvigi::ASRWhisperCapabilitiesAndRequirements*, nvigi::asr::Error> query_capabilities(
    nvigi::IAutoSpeechRecognition* iasr,
    const nvigi::asr::ModelConfig& config,
    std::source_location loc = std::source_location::current()
) {
    // Build creation parameters for capability query
    // Note: Convert string_view to string to get null-terminated c_str()
    std::string model_path_str{config.model_path};

    nvigi::CommonCreationParameters* common = new nvigi::CommonCreationParameters;
    common->numThreads = config.num_threads;
    common->vramBudgetMB = config.vram_budget_mb;
    common->utf8PathToModels = model_path_str.c_str();

    nvigi::ASRWhisperCreationParameters* params = new nvigi::ASRWhisperCreationParameters;

    // Chain common parameters
    if (auto res = params->chain(*common); res != nvigi::kResultOk) {
        delete common;
        delete params;
        return std::unexpected(nvigi::asr::Error(std::format("Failed to chain common parameters at {}:{}", loc.file_name(), loc.line())));
    }

    // Query capabilities
    nvigi::ASRWhisperCapabilitiesAndRequirements* caps{};
    nvigi::getCapsAndRequirements(iasr, *params, &caps);

    // Clean up parameters
    delete common;
    delete params;

    if (!caps) {
        return std::unexpected(nvigi::asr::Error(std::format("Failed to query capabilities at {}:{}", loc.file_name(), loc.line())));
    }

    return caps;
}

// Helper function to query plugin capabilities
inline std::expected<nvigi::CommonCapabilitiesAndRequirements*, nvigi::gpt::Error> query_capabilities(
    nvigi::IGeneralPurposeTransformer* igpt,
    const nvigi::gpt::ModelConfig& config,
    std::source_location loc = std::source_location::current()
) {
    // Build creation parameters for capability query
    // Note: Convert string_view to string to get null-terminated c_str()
    std::string model_path_str{config.model_path};
    std::string guid_str{config.guid};

    nvigi::CommonCreationParameters* common = new nvigi::CommonCreationParameters;
    common->numThreads = config.num_threads;
    common->vramBudgetMB = config.vram_budget_mb;
    common->utf8PathToModels = model_path_str.c_str();
    common->modelGUID = guid_str.empty() ? nullptr : guid_str.c_str();

    nvigi::GPTCreationParameters* params = new nvigi::GPTCreationParameters;

    // Chain common parameters
    if (auto res = params->chain(*common); res != nvigi::kResultOk) {
        delete common;
        delete params;
        return std::unexpected(nvigi::gpt::Error(std::format("Failed to chain common parameters at {}:{}", loc.file_name(), loc.line())));
    }

    params->seed = -1;
    params->maxNumTokensToPredict = 200;
    params->contextSize = config.context_size;

    // Query capabilities
    nvigi::CommonCapabilitiesAndRequirements* caps{};
    nvigi::getCapsAndRequirements(igpt, *params, &caps);

    // Clean up parameters
    delete common;
    delete params;

    if (!caps) {
        return std::unexpected(nvigi::gpt::Error(std::format("Failed to query capabilities at {}:{}", loc.file_name(), loc.line())));
    }

    return caps;
}

// Helper function to query plugin capabilities
inline std::expected<nvigi::TTSCapabilitiesAndRequirements*, nvigi::tts::Error> query_capabilities(
    nvigi::ITextToSpeech* itts,
    const nvigi::tts::ModelConfig& config,
    std::source_location loc = std::source_location::current()
) {
    // Build creation parameters for capability query
    // Note: Convert string_view to string to get null-terminated c_str()
    std::string model_path_str{config.model_path};

    nvigi::CommonCreationParameters* common = new nvigi::CommonCreationParameters;
    common->numThreads = config.num_threads;
    common->vramBudgetMB = config.vram_budget_mb;
    common->utf8PathToModels = model_path_str.c_str();

    nvigi::TTSCreationParameters* params = new nvigi::TTSCreationParameters;
    nvigi::TTSASqFlowCreationParameters* ttsASqFlowParams = new nvigi::TTSASqFlowCreationParameters;

    // Chain common parameters
    if (auto res = params->chain(*common); res != nvigi::kResultOk) {
        delete common;
        delete params;
        delete ttsASqFlowParams;
        return std::unexpected(nvigi::tts::Error(std::format("Failed to chain common parameters at {}:{}", loc.file_name(), loc.line())));
    }

    // Chain TTS ASqFlow parameters
    if (auto res = params->chain(*ttsASqFlowParams); res != nvigi::kResultOk) {
        delete common;
        delete params;
        delete ttsASqFlowParams;
        return std::unexpected(nvigi::tts::Error(std::format("Failed to chain TTS ASqFlow parameters at {}:{}", loc.file_name(), loc.line())));
    }

    // Query capabilities
    nvigi::TTSCapabilitiesAndRequirements* caps{};
    nvigi::getCapsAndRequirements(itts, *params, &caps);

    // Clean up parameters
    delete common;
    delete params;
    delete ttsASqFlowParams;

    if (!caps) {
        return std::unexpected(nvigi::tts::Error(std::format("Failed to query capabilities at {}:{}", loc.file_name(), loc.line())));
    }

    return caps;
}

std::vector<std::string> GetPossibleTargetVoices(const std::wstring& directory) {
    std::vector<std::string> binFiles;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".bin") {
            std::string filename = entry.path().filename().string();
            if (filename.size() > 7 && filename.substr(filename.size() - 7) == "_se.bin") {
                filename = filename.substr(0, filename.size() - 7);
            }
            binFiles.push_back(filename);
        }
    }
    return binFiles;
}

NVIGIContext& NVIGIContext::Get() {
    static NVIGIContext instance;
    return instance;
}

void NVIGIContext::PresentEnd(donut::app::DeviceManager& manager, uint32_t i)
{
    Get().FramerateLimit();
}

bool NVIGIContext::CheckPluginCompat(nvigi::PluginID id, const std::string& name)
{
    if (!m_core)
        return false;

    const auto& systemInfo = m_core->getSystemInfo();
    
    // Get adapter info if we have one selected
    nvigi::Adapter adapterInfo = (m_adapter >= 0 && static_cast<size_t>(m_adapter) < systemInfo.getNumAdapters()) ? 
        systemInfo.getAdapter(m_adapter) : nvigi::Adapter(nullptr);

    try {
        auto plugin = systemInfo.getPlugin(id);

        auto reqVendor = plugin.getRequiredAdapterVendor();
        if (reqVendor != nvigi::VendorId::eAny && reqVendor != nvigi::VendorId::eNone &&
            (m_adapter < 0 || reqVendor != adapterInfo.getVendor()))
        {
            donut::log::warning("Plugin %s could not be loaded on adapters from this GPU vendor (found %0x, requires %0x)", 
                name.c_str(), m_adapter >= 0 ? static_cast<uint32_t>(adapterInfo.getVendor()) : 0, static_cast<uint32_t>(reqVendor));
            return false;
        }

        if (reqVendor == nvigi::VendorId::eNVDA && m_adapter >= 0 && 
            plugin.getRequiredAdapterArchitecture() > adapterInfo.getArchitecture())
        {
            donut::log::warning("Plugin %s could not be loaded on this GPU architecture (found %d, requires %d)", 
                name.c_str(), adapterInfo.getArchitecture(), plugin.getRequiredAdapterArchitecture());
            return false;
        }

        if (reqVendor == nvigi::VendorId::eNVDA && m_adapter >= 0 && 
            plugin.getRequiredAdapterDriverVersion() > adapterInfo.getDriverVersion())
        {
            auto adapterDriver = adapterInfo.getDriverVersion();
            auto reqDriver = plugin.getRequiredAdapterDriverVersion();
            donut::log::warning("Plugin %s could not be loaded on this driver (found %d.%d, requires %d.%d)", 
                name.c_str(), adapterDriver.major, adapterDriver.minor, reqDriver.major, reqDriver.minor);
            return false;
        }

        return true;
    }
    catch (...) {
        // Not found
        donut::log::warning("Plugin %s could not be loaded", name.c_str());
        return false;
    }
}

bool NVIGIContext::AddGPTPlugin(const std::string& backend)
{
    // Convert backend string to plugin ID
    nvigi::PluginID id = nvigi::gpt::backend_to_plugin_id(backend);
    std::string name = "gpt." + backend;
    
    if (CheckPluginCompat(id, name))
    {
        nvigi::IGeneralPurposeTransformer* igpt{};
        nvigi::Result nvigiRes = nvigiGetInterfaceDynamic(id, &igpt, m_core->loadInterface());
        if (nvigiRes != nvigi::kResultOk)
            return false;

        // Query capabilities using modern wrapper
        nvigi::gpt::ModelConfig query_config{
            .backend = backend,
            .guid = "",
            .model_path = m_shippedModelsPath,
            .context_size = 4096,
            .num_threads = 1,
            .vram_budget_mb = m_gpt.m_vramBudget
        };

        auto caps_result = query_capabilities(igpt, query_config);
        if (!caps_result)
        {
            auto unloadFn = m_core->unloadInterface();
            unloadFn(id, igpt);
            donut::log::error("Failed to query GPT capabilities: %s", caps_result.error().what().c_str());
            return false;
        }

        nvigi::CommonCapabilitiesAndRequirements* models = *caps_result;

        for (uint32_t i = 0; i < models->numSupportedModels; i++)
        {
            auto info = std::make_shared<PluginModelInfo>();
            info->m_backend = backend;
            info->m_modelName = models->supportedModelNames[i];
            info->m_pluginName = name;
            info->m_caption = name + " : " + models->supportedModelNames[i];
            info->m_guid = models->supportedModelGUIDs[i];
            info->m_modelRoot = m_shippedModelsPath;
            info->m_vram = models->modelMemoryBudgetMB[i];
            info->m_modelStatus = (models->modelFlags[i] & nvigi::kModelFlagRequiresDownload)
                ? ModelStatus::AVAILABLE_MANUAL_DOWNLOAD : ModelStatus::AVAILABLE_LOCALLY;
            m_gpt.m_pluginModelsMap[info->m_guid].push_back(info);
        }

        auto unloadFn = m_core->unloadInterface();
        unloadFn(id, igpt);
        return true;
    }

    return false;
}

bool NVIGIContext::AddGPTCloudPlugin()
{
    const std::string backend = "cloud";
    nvigi::PluginID id = nvigi::gpt::backend_to_plugin_id(backend);
    const std::string name = "gpt.cloud";

    if (CheckPluginCompat(id, name))
    {
        nvigi::IGeneralPurposeTransformer* igpt{};
        nvigi::Result nvigiRes = nvigiGetInterfaceDynamic(id, &igpt, m_core->loadInterface());
        if (nvigiRes != nvigi::kResultOk)
            return false;

        // Query capabilities using modern wrapper to get list of models
        nvigi::gpt::ModelConfig query_config{
            .backend = backend,
            .guid = "",
            .model_path = m_shippedModelsPath,
            .context_size = 4096,
            .num_threads = 1,
            .vram_budget_mb = m_gpt.m_vramBudget
        };

        auto caps_result = query_capabilities(igpt, query_config);
        if (!caps_result)
        {
            auto unloadFn = m_core->unloadInterface();
            unloadFn(id, igpt);
            donut::log::error("Failed to query GPT cloud capabilities: %s", caps_result.error().what().c_str());
            return false;
        }

        nvigi::CommonCapabilitiesAndRequirements* models = *caps_result;
        std::vector<std::tuple<std::string, std::string>> cloudItems;

        for (uint32_t i = 0; i < models->numSupportedModels; i++)
            cloudItems.push_back({ models->supportedModelGUIDs[i], models->supportedModelNames[i] });

        // Query capabilities for each specific model
        for (auto& item : cloudItems)
        {
            auto guid = std::get<0>(item);
            auto modelName = std::get<1>(item);
            query_config.guid = guid;
            
            auto model_caps_result = query_capabilities(igpt, query_config);
            if (!model_caps_result) continue;
            
            models = *model_caps_result;
            auto cloudCaps = nvigi::findStruct<nvigi::CloudCapabilities>(*models);

            auto info = std::make_shared<PluginModelInfo>();
            info->m_backend = backend;
            info->m_modelName = modelName;
            info->m_pluginName = name;
            info->m_caption = name + " : " + modelName;
            info->m_guid = guid;
            info->m_modelRoot = m_shippedModelsPath;
            info->m_vram = 0;
            info->m_modelStatus = ModelStatus::AVAILABLE_CLOUD;
            info->m_url = cloudCaps->url;
            m_gpt.m_pluginModelsMap[info->m_guid].push_back(info);

        }

        auto unloadFn = m_core->unloadInterface();
        unloadFn(id, igpt);
        return true;
    }

    return false;
}


bool NVIGIContext::AddASRPlugin(const std::string& backend)
{
    // Convert backend string to plugin ID
    nvigi::PluginID id = nvigi::asr::backend_to_plugin_id(backend);
    std::string name = "asr." + backend;
    
    if (CheckPluginCompat(id, name))
    {
        nvigi::IAutoSpeechRecognition* iasr{};
        nvigi::Result nvigiRes = nvigiGetInterfaceDynamic(id, &iasr, m_core->loadInterface());
        if (nvigiRes != nvigi::kResultOk)
            return false;

        // Query capabilities using modern wrapper
        nvigi::asr::ModelConfig query_config{
            .backend = backend,
            .guid = "",
            .model_path = m_shippedModelsPath,
            .num_threads = 4,
            .vram_budget_mb = m_asr.m_vramBudget
        };

        auto caps_result = query_capabilities(iasr, query_config);
        if (!caps_result)
        {
            auto unloadFn = m_core->unloadInterface();
            unloadFn(id, iasr);
            donut::log::error("Failed to query ASR capabilities: %s", caps_result.error().what().c_str());
            return false;
        }

        nvigi::ASRWhisperCapabilitiesAndRequirements* caps = *caps_result;
        nvigi::CommonCapabilitiesAndRequirements& models = *(caps->common);
        for (uint32_t i = 0; i < models.numSupportedModels; i++)
        {
            auto info = std::make_shared<PluginModelInfo>();
            info->m_backend = backend;
            info->m_modelName = models.supportedModelNames[i];
            info->m_pluginName = name;
            info->m_caption = name + " : " + models.supportedModelNames[i];
            info->m_guid = models.supportedModelGUIDs[i];
            info->m_modelRoot = m_shippedModelsPath;
            info->m_vram = models.modelMemoryBudgetMB[i];
            info->m_modelStatus = (models.modelFlags[i] & nvigi::kModelFlagRequiresDownload)
                ? ModelStatus::AVAILABLE_MANUAL_DOWNLOAD : ModelStatus::AVAILABLE_LOCALLY;
            m_asr.m_pluginModelsMap[info->m_guid].push_back(info);
        }

        auto unloadFn = m_core->unloadInterface();
        unloadFn(id, iasr);
        return true;
    }

    return false;
}

bool NVIGIContext::AddTTSPlugin(const std::string& backend)
{
    // Convert backend string to plugin ID
    nvigi::PluginID id = nvigi::tts::backend_to_plugin_id(backend);
    std::string name = "tts." + backend;
    
    if (CheckPluginCompat(id, name))
    {
        nvigi::ITextToSpeech* itts{};
        nvigi::Result nvigiRes = nvigiGetInterfaceDynamic(id, &itts, m_core->loadInterface());
        if (nvigiRes != nvigi::kResultOk)
            return false;

        // Query capabilities using modern wrapper
        nvigi::tts::ModelConfig query_config{
            .backend = backend,
            .guid = "",
            .model_path = m_shippedModelsPath,
            .num_threads = 4,
            .vram_budget_mb = m_tts.m_vramBudget
        };

        auto caps_result = query_capabilities(itts, query_config);
        if (!caps_result)
        {
            auto unloadFn = m_core->unloadInterface();
            unloadFn(id, itts);
            donut::log::error("Failed to query TTS capabilities: %s", caps_result.error().what().c_str());
            return false;
        }

        nvigi::TTSCapabilitiesAndRequirements* caps = *caps_result;
        nvigi::CommonCapabilitiesAndRequirements& models = *(caps->common);
        for (uint32_t i = 0; i < models.numSupportedModels; i++)
        {
            auto info = std::make_shared<PluginModelInfo>();
            info->m_backend = backend;
            info->m_modelName = models.supportedModelNames[i];
            info->m_pluginName = name;
            info->m_caption = name + " : " + models.supportedModelNames[i];
            info->m_guid = models.supportedModelGUIDs[i];
            info->m_modelRoot = m_shippedModelsPath;
            info->m_vram = models.modelMemoryBudgetMB[i];
            info->m_modelStatus = (models.modelFlags[i] & nvigi::kModelFlagRequiresDownload)
                ? ModelStatus::AVAILABLE_MANUAL_DOWNLOAD : ModelStatus::AVAILABLE_LOCALLY;
            m_tts.m_pluginModelsMap[info->m_guid].push_back(info);
        }

        auto unloadFn = m_core->unloadInterface();
        unloadFn(id, itts);
        return true;
    }

    return false;
}

bool NVIGIContext::Initialize_preDeviceManager(nvrhi::GraphicsAPI api, int argc, const char* const* argv)
{
    m_api = api;

    // Hack for now, as we don't really want to check the sigs
#ifdef NVIGI_PRODUCTION
    bool checkSig = true;
#else
    bool checkSig = false;
#endif
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-pathToModels"))
        {
            m_shippedModelsPath = argv[++i];
        }
        else if (!strcmp(argv[i], "-systemPromptGPT")) {
            m_systemPromptGPT = argv[++i];
        }
        else if (!strcmp(argv[i], "-noSigCheck"))
        {
            checkSig = false;
        }
        else if (!strcmp(argv[i], "-logToFile"))
        {
            m_LogFilename = argv[++i];
        }
        else if (!strcmp(argv[i], "-noCiG") || !strcmp(argv[i], "-noCIG"))
        {
            m_useCiG = false;
        }
    }

    // Get executable path for SDK location
    wchar_t path[PATH_MAX] = { 0 };
    GetModuleFileNameW(nullptr, path, dim(path));
    auto basePath = std::filesystem::path(path).parent_path();
    static std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> convert;
    m_appUtf8path = convert.to_bytes(basePath);

    if (checkSig) {
        auto pathNVIGIDll = basePath / "nvigi.core.framework.dll";
        donut::log::info("Checking NVIGI core DLL signature");
        if (!nvigi::security::verifyEmbeddedSignature(pathNVIGIDll.wstring().c_str())) {
            donut::log::error("NVIGI core DLL is not signed - disable signature checking with -noSigCheck or use a signed NVIGI core DLL");
            return false;
        }
    }

    // Find models path if not specified
    if (m_shippedModelsPath.empty())
    {
        using convert_type = std::codecvt_utf8<wchar_t>;
        std::wstring_convert<convert_type, wchar_t> converter;

        std::wstring wpath = basePath;
        for (int i = 0; i < 6; i++)
        {
            wpath = wpath + L"\\..";
            if (std::filesystem::exists(wpath + L"\\data\\nvigi.models\\nvigi.plugin.gpt.ggml"))
            {
                m_shippedModelsPath = converter.to_bytes(wpath) + "\\data\\nvigi.models";
                break;
            }
            if (i == 5)
            {
                donut::log::error("Unable to find shipped models path.  Please specify it with -pathToModels <path>");
                return false;
			}
        }
    }

    try {
        // Determine log path
        std::string logPath;
        if (m_LogFilename.empty())
        {
            logPath = m_appUtf8path + "\\";
        }
        else
        {
            logPath = m_LogFilename;
        }

        m_core = std::make_unique<nvigi::Core>(nvigi::Core::Config{
            .sdkPath = m_appUtf8path,
            .logLevel = nvigi::LogLevel::eVerbose,
            .showConsole = true,
            .pathToLogsAndData = logPath
        });

        donut::log::info("NVIGI core initialized successfully");

    } catch (const std::exception& e) {
        donut::log::error("Failed to initialize NVIGI: %s", e.what());
        return false;
    }

    const auto& systemInfo = m_core->getSystemInfo();
    uint32_t nvdaArch = 0;
    for (size_t i = 0; i < systemInfo.getNumAdapters(); i++)
    {
        auto adapter = systemInfo.getAdapter(i);
        if (adapter.getVendor() == nvigi::VendorId::eNVDA && nvdaArch < adapter.getArchitecture())
        {
            nvdaArch = adapter.getArchitecture();
            m_adapter = static_cast<int>(i);
        }
    }

    if (m_adapter < 0)
    {
        donut::log::warning("No NVIDIA adapters found.  GPU plugins will not be available\n");
        if (systemInfo.getNumAdapters() > 0)
            m_adapter = 0;
    }

    m_gpt.m_vramBudget = 8500;

    // Setup backend choices using simple strings
    m_gpt.m_choices = {
        .m_nvda = "cuda",   // NVIDIA CUDA
        .m_gpu = "d3d12",   // Generic GPU (will be updated based on API)
        .m_cloud = "cloud", // Cloud
        .m_cpu = ""         // CPU (not supported for GPT)
    };

    AddGPTPlugin("cuda");
    if (m_api == nvrhi::GraphicsAPI::D3D12)
    {
        m_gpt.m_choices.m_gpu = "d3d12";
        AddGPTPlugin("d3d12");
    }
    else if (m_api == nvrhi::GraphicsAPI::VULKAN)
    {
        m_gpt.m_choices.m_gpu = "vulkan";
        AddGPTPlugin("vulkan");
    }
    AddGPTCloudPlugin();

    {
		// Select initial plugin m_gpt.m_info...  We have a preferred one, but if it is not available, we select the first one
        m_gpt.m_info = nullptr;
        const std::string preferredModelGUID = "{D5E8DEB3-28C2-4B9E-9412-B9A012B23584}"; // Llama 3
        for (auto& m : m_gpt.m_pluginModelsMap)
        {
            for (auto& info : m.second)
            {
                ModelStatus status = info->m_modelStatus;
                if (status == ModelStatus::AVAILABLE_LOCALLY)
                {
                    // Match by backend string instead of plugin ID
                    if ((m_gpt.m_choices.m_nvda == info->m_backend) ||
                        (m_gpt.m_info == nullptr && m_gpt.m_choices.m_gpu == info->m_backend))
                    {
                        m_gpt.m_info = info;
                        if (m_gpt.m_info->m_guid == preferredModelGUID)
                            break;
                    }
                }
            }
            if (m_gpt.m_info && m_gpt.m_info->m_guid == preferredModelGUID)
                break;
        }
    }

    m_asr.m_vramBudget = 3000;

    // Setup backend choices using simple strings
    m_asr.m_choices = {
        .m_nvda = "cuda",   // NVIDIA CUDA
        .m_gpu = "",        // Generic GPU (will be updated based on API)
        .m_cloud = "",      // Cloud (not supported for ASR)
        .m_cpu = "cpu"      // CPU fallback
    };

    AddASRPlugin("cuda");
    if (m_api == nvrhi::GraphicsAPI::D3D12)
    {
        m_asr.m_choices.m_gpu = "d3d12";
        AddASRPlugin("d3d12");
    }
    else if (m_api == nvrhi::GraphicsAPI::VULKAN)
    {
        m_asr.m_choices.m_gpu = "vulkan";
        AddASRPlugin("vulkan");
    }
    AddASRPlugin("cpu");

    {
        // Select initial plugin m_asr.m_info...  Or we set it to null?
        m_asr.m_info = nullptr;
        for (auto& m : m_asr.m_pluginModelsMap)
        {
            for (auto& info : m.second)
            {
                ModelStatus status = info->m_modelStatus;
                if (status == ModelStatus::AVAILABLE_LOCALLY)
                {
                    // Match by backend string instead of plugin ID
                    if ((m_asr.m_choices.m_nvda == info->m_backend) ||
                        (m_asr.m_info == nullptr && !m_asr.m_choices.m_gpu.empty() && m_asr.m_choices.m_gpu == info->m_backend))
                        m_asr.m_info = info;
                }
            }
            if (m_asr.m_info)
                break;
        }
        if (!m_asr.m_info)
        {
            for (auto& m : m_asr.m_pluginModelsMap)
            {
                for (auto& info : m.second)
                {
                    ModelStatus status = info->m_modelStatus;
                    if (status == ModelStatus::AVAILABLE_LOCALLY)
                    {
                        // Match CPU backend string
                        if (!m_asr.m_choices.m_cpu.empty() && m_asr.m_choices.m_cpu == info->m_backend)
                            m_asr.m_info = info;
                    }
                }
                if (m_asr.m_info)
                    break;
            }
        }
    }

    m_tts.m_vramBudget = 8500;

    // Setup backend choices using simple strings
    m_tts.m_choices = {
        .m_nvda = "trt",    // TensorRT (NVIDIA)
        .m_gpu = "",        // Generic GPU (will be updated based on API)
        .m_cloud = "",      // Cloud (not supported for TTS)
        .m_cpu = ""         // CPU (not supported for TTS)
    };

    AddTTSPlugin("trt");
    if (m_api == nvrhi::GraphicsAPI::D3D12)
    {
        m_tts.m_choices.m_gpu = "d3d12";
        AddTTSPlugin("d3d12");
    }
    else if (m_api == nvrhi::GraphicsAPI::VULKAN)
    {
        m_tts.m_choices.m_gpu = "vulkan";
        AddTTSPlugin("vulkan");
    }
    AddTTSPlugin("cuda");

    {
        // Set the TTS to unselected initially, as not everyone will want to use it
        m_tts.m_info = nullptr;
        for (auto& m : m_tts.m_pluginModelsMap)
        {
            for (auto& info : m.second)
            {
                ModelStatus status = info->m_modelStatus;
                if (status == ModelStatus::AVAILABLE_LOCALLY)
                {
                    // Match by backend string instead of plugin ID
                    if ((m_tts.m_choices.m_nvda == info->m_backend) ||
                        (m_tts.m_info == nullptr && !m_tts.m_choices.m_gpu.empty() && m_tts.m_choices.m_gpu == info->m_backend))
                        m_tts.m_info = info;
                }
            }
            if (m_tts.m_info)
                break;
        }
    }

    m_gpt.m_callbackState.store(nvigi::gpt::ExecutionState::Invalid);

    messages.push_back({ Message::Type::Answer, "Type a query or record audio to interact!" });


    return true;
}

bool NVIGIContext::Initialize_preDeviceCreate(donut::app::DeviceManager* deviceManager, donut::app::DeviceCreationParameters& params)
{
    nvrhi::RefCountPtr<IDXGIAdapter> dxgiAdapter;
    uint32_t index = 0;

    if (m_api == nvrhi::GraphicsAPI::D3D11 || m_api == nvrhi::GraphicsAPI::D3D12)
    {
        donut::app::InstanceParameters instParams{};
#ifdef _DEBUG
        instParams.enableDebugRuntime = true;
#endif

        params.enableComputeQueue = true;
        params.enableCopyQueue = true;

        if (!deviceManager->CreateInstance(instParams))
            return false;

        std::vector<donut::app::AdapterInfo> outAdapters;
        if (!deviceManager->EnumerateAdapters(outAdapters))
            return false;

        for (auto& adapterDesc : outAdapters)
        {
            if (adapterDesc.vendorID == 4318)
            {
                dxgiAdapter = adapterDesc.dxgiAdapter;
                params.adapterIndex = index;
                break;
            }
            index++;
        }
    }
    else if (m_api == nvrhi::GraphicsAPI::VULKAN)
    {
        IDXGIFactory4* factory;
        if (SUCCEEDED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory)))
        {
            IDXGIAdapter3* adapter;
            while (factory->EnumAdapters(index, reinterpret_cast<IDXGIAdapter**>(&adapter)) != DXGI_ERROR_NOT_FOUND)
            {
                DXGI_ADAPTER_DESC desc;
                if (SUCCEEDED(adapter->GetDesc(&desc)))
                {
                    if (desc.VendorId == 4318 /* NVDA */ || desc.VendorId == 4098 /* AMD */)
                    {
                        dxgiAdapter = adapter;
                        params.adapterIndex = index;
                        break;
                    }
                    else
                    {
                        adapter->Release();
                    }
                }
                index++;
            }
        }
    }

    if (dxgiAdapter)
        if (S_OK != dxgiAdapter->QueryInterface(__uuidof(IDXGIAdapter3), (void**)m_targetAdapter.GetAddressOf()))
            return false;

    return true;
}

bool NVIGIContext::Initialize_postDevice()
{
    auto readFile = [](const char* fname)->std::vector<uint8_t>
        {
            fs::path p(fname);
            size_t file_size = fs::file_size(p);
            std::vector<uint8_t> ret_buffer(file_size);
            std::fstream file(fname, std::ios::binary | std::ios::in);
            file.read((char*)ret_buffer.data(), file_size);
            return ret_buffer;
        };

    // Setup CiG
    if (m_useCiG)
    {
        // Because of a current bug, we can't create and
        // destroy the CIG context many times in one app (yet). The CIG context is owned by
        // the HWI.cuda plugin, so we need to keep it alive between tests by 
        // getting it here.
        nvigi::IHWICuda* cig_raw = nullptr;
        nvigi::IHWICommon* hwi_raw = nullptr;
        
        nvigiGetInterfaceDynamic(nvigi::plugin::hwi::cuda::kId, &cig_raw, m_core->loadInterface());
        nvigiGetInterfaceDynamic(nvigi::plugin::hwi::common::kId, &hwi_raw, m_core->loadInterface());
        
        // Wrap in unique_ptr with custom deleters that unload the interfaces
        if (cig_raw) {
            m_cig = std::unique_ptr<nvigi::IHWICuda, std::function<void(nvigi::IHWICuda*)>>(
                cig_raw,
                [this](nvigi::IHWICuda* ptr) {
                    if (ptr && m_core) {
                        auto unloadFn = m_core->unloadInterface();
                        unloadFn(nvigi::plugin::hwi::cuda::kId, ptr);
                    }
                }
            );
        }
        
        if (hwi_raw) {
            m_hwiCommon = std::unique_ptr<nvigi::IHWICommon, std::function<void(nvigi::IHWICommon*)>>(
                hwi_raw,
                [this](nvigi::IHWICommon* ptr) {
                    if (ptr && m_core) {
                        auto unloadFn = m_core->unloadInterface();
                        unloadFn(nvigi::plugin::hwi::common::kId, ptr);
                    }
                }
            );
        }
    }
    else
    {
        donut::log::info("Not using a shared CUDA context - CiG disabled");
    }

    if (m_api == nvrhi::GraphicsAPI::D3D11 || m_api == nvrhi::GraphicsAPI::D3D12)
    {
        m_d3d12_config.device = m_Device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        m_d3d12_config.direct_queue = m_D3D12Queue;
        m_d3d12_config.compute_queue = m_D3D12QueueCompute;
        m_d3d12_config.transfer_queue = m_D3D12QueueCopy;
    }
    else if (m_api == nvrhi::GraphicsAPI::VULKAN)
    {
        m_vk_config.instance = m_Device->getNativeObject(nvrhi::ObjectTypes::VK_Instance);
        m_vk_config.device = m_Device->getNativeObject(nvrhi::ObjectTypes::VK_Device);
        m_vk_config.physical_device = m_Device->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice);
        m_vk_config.compute_queue = m_Device->getNativeQueue(nvrhi::ObjectTypes::VK_Queue, nvrhi::CommandQueue::Compute);
        m_vk_config.transfer_queue = m_Device->getNativeQueue(nvrhi::ObjectTypes::VK_Queue, nvrhi::CommandQueue::Copy);
    }

    size_t currentVRAM;
    GetVRAMStats(currentVRAM, m_maxVRAM);
    m_maxVRAM /= (1024 * 1024);

    //! Load A2T script    
    auto loadASR = [this]()->HRESULT
        {
            PluginModelInfoPtr gptInfo = m_gpt.m_info;

            if (gptInfo)
            {
                // Build ModelConfig
                nvigi::gpt::ModelConfig model_config{
                    .backend = gptInfo->m_backend,
                    .guid = gptInfo->m_guid,
                    .model_path = gptInfo->m_modelRoot.empty() ? m_shippedModelsPath : gptInfo->m_modelRoot,
                    .context_size = 4096,
                    .num_threads = 1,
                    .vram_budget_mb = m_gpt.m_vramBudget,
                    .flash_attention = true,
                    .cache_type = "fp16"
                };

                nvigi::gpt::CloudConfig cloud_config{};
                if (gptInfo->m_backend == "cloud")
                {
                    const char* key = nullptr;
                    std::string apiKeyName = "";
                    if (!GetCloudModelAPIKey(gptInfo, key, apiKeyName))
                    {
                        std::string text = "CLOUD API key not found at " + apiKeyName + " cloud model will not be available";
                        donut::log::warning(text.c_str());
                    }

                    cloud_config.url = gptInfo->m_url;
                    cloud_config.token = key;
					cloud_config.verbose = true;
				}

                // Create GPT instance using modern wrapper
                auto result = nvigi::gpt::Instance::create(
                    model_config,
                    m_d3d12_config,
                    m_vk_config,
                    cloud_config,
                    {}, // io_config (default file system)
                    m_core->loadInterface(),
                    m_core->unloadInterface(),
                    {} // plugin_path (empty = use default SDK location)
                );

                if (result) {
                    m_gptInstance = std::move(*result);
                    // Create chat interface with default settings using emplace
                    m_gptChat.emplace(m_gptInstance->create_chat({
                        .tokens_to_predict = 200,
                        .batch_size = 2048,
                        .temperature = 0.3f,
                        .top_p = 0.8f,
                        .interactive = true,
                        .reverse_prompt = "User: "
                    }));
                    m_gpt.m_ready.store(true);
                } else {
                    donut::log::error("Unable to create GPT instance/model: %s", result.error().what().c_str());
                    m_gpt.m_ready.store(false);
                }
            }
            else
            {
                m_gpt.m_ready.store(false);
            }

            PluginModelInfoPtr asrInfo = m_asr.m_info;

            if (asrInfo)
            {
                // Build ModelConfig
                nvigi::asr::ModelConfig model_config{
                    .backend = asrInfo->m_backend,
                    .guid = asrInfo->m_guid,
                    .model_path = asrInfo->m_modelRoot.empty() ? m_shippedModelsPath : asrInfo->m_modelRoot,
                    .num_threads = 1,
                    .vram_budget_mb = m_asr.m_vramBudget,
                    .flash_attention = true,
                    .language = "en",
                    .translate = false,
                    .detect_language = false
                };

                // Create ASR instance using modern wrapper
                auto result = nvigi::asr::Instance::create(
                    model_config,
                    m_d3d12_config,
                    m_vk_config,
                    {}, // io_config (default file system)
                    m_core->loadInterface(),
                    m_core->unloadInterface(),
                    {} // plugin_path (empty = use default SDK location)
                );

                if (result) {
                    m_asrInstance = std::move(*result);
                    m_asr.m_ready.store(true);
                } else {
                    donut::log::error("Unable to create ASR instance/model: %s", result.error().what().c_str());
                    m_asr.m_ready.store(false);
                }
            }
            else
            {
                m_asr.m_ready.store(false);
            }

            PluginModelInfoPtr ttsInfo = m_tts.m_info;

            std::vector<std::string> targetVoices = GetPossibleTargetVoices(GetNVIGICoreDllPath());
            // The initial value of the selected voice, if non-empty, was the voice we'd prefer if it is available
            if (std::find(targetVoices.begin(), targetVoices.end(), m_ttsInferenceCtx.m_selectedTargetVoice) == targetVoices.end())
            {
                // We could not find that voice, so pick the first valid one...
                if (!targetVoices.empty())
                    m_ttsInferenceCtx.m_selectedTargetVoice = targetVoices[0];
            }

            if (ttsInfo)
            {
                // Build ModelConfig
                nvigi::tts::ModelConfig model_config{
                    .backend = ttsInfo->m_backend,
                    .guid = ttsInfo->m_guid,
                    .model_path = ttsInfo->m_modelRoot.empty() ? m_shippedModelsPath : ttsInfo->m_modelRoot,
                    .num_threads = 1,
                    .vram_budget_mb = m_tts.m_vramBudget,
                    .warm_up_models = true
                };

                nvigi::vulkan::VulkanConfig vk_config{};
                if (model_config.backend != "vulkan")
                    vk_config = m_vk_config;

                // Create TTS instance using modern wrapper
                auto result = nvigi::tts::Instance::create(
                    model_config,
                    m_d3d12_config,
                    vk_config,
                    m_core->loadInterface(),
                    m_core->unloadInterface(),
                    {} // plugin_path (empty = use default SDK location)
                );

                if (result) {
                    m_ttsInstance = std::move(*result);
                    m_tts.m_ready.store(true);
                } else {
                    donut::log::error("Unable to create TTS instance/model: %s", result.error().what().c_str());
                    m_tts.m_ready.store(false);
                }
            }
            else
            {
                m_tts.m_ready.store(false);
            }

            return S_OK;
        };
    m_loadingThread = std::make_unique<std::thread>(loadASR);

    return true;
}

void NVIGIContext::SetDevice_nvrhi(nvrhi::IDevice* device)
{
    m_Device = device;
    if (m_Device)
    {
        m_D3D12Queue = m_Device->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, nvrhi::CommandQueue::Graphics);
        m_D3D12QueueCompute = m_Device->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, nvrhi::CommandQueue::Compute);
        m_D3D12QueueCopy = m_Device->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, nvrhi::CommandQueue::Copy);
    }
}

void NVIGIContext::Shutdown()
{
    if (m_inferThread && m_inferThread->joinable())
        m_inferThread->join();
    m_inferThread.reset();

    if (m_loadingThread && m_loadingThread->joinable())
        m_loadingThread->join();
    m_loadingThread.reset();

    m_hwiCommon.reset();
    m_cig.reset();

	m_ttsInstance.reset();
	m_gptChat.reset();
	m_gptInstance.reset();
	m_asrInstance.reset();
    
    m_core.reset();
}

void NVIGIContext::ReloadGPTModel(PluginModelInfoPtr newGptInfo)
{
    if (m_loadingThread && m_loadingThread->joinable())
        m_loadingThread->join();
    m_loadingThread.reset();

    m_conversationInitialized = false;

    PluginModelInfoPtr prevGptInfo = m_gpt.m_info;

    m_gpt.m_info = newGptInfo;

    if (!newGptInfo || (newGptInfo->m_modelStatus == ModelStatus::AVAILABLE_MANUAL_DOWNLOAD))
    {
        m_gpt.m_info = prevGptInfo;
        return;
    }

    m_gpt.m_ready.store(false);

    m_gptChat.reset();
    m_gptInstance.reset();

    if (!newGptInfo)
    {
        m_gpt.m_ready.store(false);
        return;
    }

    auto loadModel = [this, prevGptInfo, newGptInfo]()->void
        {
            // Build ModelConfig
            nvigi::gpt::ModelConfig model_config{
                .backend = newGptInfo->m_backend,
                .guid = newGptInfo->m_guid,
                .model_path = newGptInfo->m_modelRoot.empty() ? m_shippedModelsPath : newGptInfo->m_modelRoot,
                .context_size = 4096,
                .num_threads = 1,
                .vram_budget_mb = m_gpt.m_vramBudget,
                .flash_attention = true,
                .cache_type = "fp16"
            };

            cerr_redirect ggmlLog;
            
            nvigi::gpt::CloudConfig cloud_config{};
            if (newGptInfo->m_backend == "cloud")
            {
                const char* key = nullptr;
                std::string apiKeyName = "";
                if (!GetCloudModelAPIKey(newGptInfo, key, apiKeyName))
                {
                    std::string text = "CLOUD API key not found at " + apiKeyName + " cloud model will not be available";
                    donut::log::warning(text.c_str());
                }

                cloud_config.url = newGptInfo->m_url;
                cloud_config.token = key;
                cloud_config.verbose = true;
            }

            // Create GPT instance using modern wrapper
            auto result = nvigi::gpt::Instance::create(
                model_config,
                m_d3d12_config,
                m_vk_config,
                cloud_config,
                {}, // io_config (default file system)
                m_core->loadInterface(),
                m_core->unloadInterface(),
                {} // plugin_path (empty = use default SDK location)
            );

            if (result) {
                m_gptInstance = std::move(*result);
                // Create chat interface with default settings using emplace
                m_gptChat.emplace(m_gptInstance->create_chat({
                    .tokens_to_predict = 200,
                    .batch_size = 2048,
                    .temperature = 0.3f,
                    .top_p = 0.8f,
                    .interactive = true,
                    .reverse_prompt = "User: "
                }));
                m_gpt.m_ready.store(true);
            } else {
                // Fallback to previous model
                donut::log::error("Unable to create GPT instance/model: %s. Reverting to previous model", result.error().what().c_str());
                m_gpt.m_info = prevGptInfo;
                
                if (prevGptInfo) {
                    // Try previous model
                    model_config.backend = prevGptInfo->m_backend;
                    model_config.guid = prevGptInfo->m_guid;
                    model_config.model_path = prevGptInfo->m_modelRoot.empty() ? m_shippedModelsPath : prevGptInfo->m_modelRoot;

                    nvigi::gpt::CloudConfig cloud_config{};
                    if (prevGptInfo->m_backend == "cloud")
                    {
                        const char* key = nullptr;
                        std::string apiKeyName = "";
                        if (!GetCloudModelAPIKey(prevGptInfo, key, apiKeyName))
                        {
                            std::string text = "CLOUD API key not found at " + apiKeyName + " cloud model will not be available";
                            donut::log::warning(text.c_str());
                        }

                        cloud_config.url = prevGptInfo->m_url;
                        cloud_config.token = key;
                        cloud_config.verbose = true;
                    }

                    auto fallback_result = nvigi::gpt::Instance::create(
                        model_config,
                        m_d3d12_config,
                        m_vk_config,
                        cloud_config,
                        {},
                        m_core->loadInterface(),
                        m_core->unloadInterface(),
                        {} // plugin_path
                    );

                    if (fallback_result) {
                        m_gptInstance = std::move(*fallback_result);
                        m_gptChat.emplace(m_gptInstance->create_chat({
                            .tokens_to_predict = 200,
                            .batch_size = 2048,
                            .temperature = 0.3f,
                            .top_p = 0.8f,
                            .interactive = true,
                            .reverse_prompt = "User: "
                        }));
                        m_gpt.m_ready.store(true);
                    } else {
                        donut::log::error("Unable to create GPT instance/model and cannot revert to previous model");
                        m_gpt.m_ready.store(false);
                    }
                } else {
                    m_gpt.m_ready.store(false);
                }
            }
        };
    m_loadingThread = std::make_unique<std::thread>(loadModel);
}

void NVIGIContext::ReloadASRModel(PluginModelInfoPtr newAsrInfo)
{
    if (m_loadingThread && m_loadingThread->joinable())
    {
        m_loadingThread->join();
    }
    m_loadingThread.reset();
    m_asr.m_ready.store(false);

    m_asr.m_info = newAsrInfo;

    m_asrInstance.reset();

    m_asrStream.reset();

    if (!newAsrInfo)
    {
        m_asr.m_ready.store(false);
        return;
    }

    auto loadModel = [this, newAsrInfo]()->void
        {
            cerr_redirect ggmlLog;

            // Build ModelConfig
            nvigi::asr::ModelConfig model_config{
                .backend = newAsrInfo->m_backend,
                .guid = newAsrInfo->m_guid,
                .model_path = newAsrInfo->m_modelRoot.empty() ? m_shippedModelsPath : newAsrInfo->m_modelRoot,
                .num_threads = 1,
                .vram_budget_mb = m_asr.m_vramBudget,
                .flash_attention = true,
                .language = "en",
                .translate = false,
                .detect_language = false
            };

            // Create ASR instance using modern wrapper
            auto result = nvigi::asr::Instance::create(
                model_config,
                m_d3d12_config,
                m_vk_config,
                {},
                m_core->loadInterface(),
                m_core->unloadInterface(),
                {}
            );

            if (result) {
                m_asrInstance = std::move(*result);
                m_asr.m_ready.store(true);
            } else {
                donut::log::error("Unable to create ASR instance/model: %s", result.error().what().c_str());
                m_asr.m_ready.store(false);
            }
        };
    m_loadingThread = std::make_unique<std::thread>(loadModel);
}

void NVIGIContext::ReloadTTSModel(PluginModelInfoPtr newTtsInfo)
{
    if (m_loadingThread && m_loadingThread->joinable())
    {
        m_loadingThread->join();
    }
    m_loadingThread.reset();
    m_tts.m_ready.store(false);

    m_tts.m_info = newTtsInfo;
    m_ttsInput = "";

    m_ttsInstance.reset();

    if (!newTtsInfo)
    {
        m_tts.m_ready.store(false);
        return;
    }

    auto loadModel = [this, newTtsInfo]()->void
        {
            // Build ModelConfig
            nvigi::tts::ModelConfig model_config{
                .backend = newTtsInfo->m_backend,
                .guid = newTtsInfo->m_guid,
                .model_path = newTtsInfo->m_modelRoot.empty() ? m_shippedModelsPath : newTtsInfo->m_modelRoot,
                .num_threads = 1,
                .vram_budget_mb = m_tts.m_vramBudget,
                .warm_up_models = true
            };

            nvigi::vulkan::VulkanConfig vk_config{};
            if (model_config.backend != "vulkan")
                vk_config = m_vk_config;

            // Create TTS instance using modern wrapper
            auto result = nvigi::tts::Instance::create(
                model_config,
                m_d3d12_config,
                vk_config, // TODO send this down
                m_core->loadInterface(),
                m_core->unloadInterface(),
                {} // plugin_path
            );

            if (result) {
                m_ttsInstance = std::move(*result);
                m_tts.m_ready.store(true);
            } else {
                donut::log::error("Unable to create TTS instance/model: %s", result.error().what().c_str());
                m_tts.m_ready.store(false);
            }
        };
    m_loadingThread = std::make_unique<std::thread>(loadModel);
}

void NVIGIContext::LaunchASR()
{
    m_newInferenceSequence = true;

    if (!m_asr.m_ready)
    {
        donut::log::warning("Skipping Speech to Text as it is still loading or failed to load");
        return;
    }

    // Current approach: accumulation mode (backward compatible)
    m_recordingData = AudioRecorder::StartRecording();
    if (!m_recordingData) {
        donut::log::error("Failed to start recording!\n"
            "Please check your microphone settings in Windows.");
        return;
    }

    auto l = [this]()->void
        {
            m_inferThreadRunning = true;
            
            if (m_hwiCommon)
                m_hwiCommon->SetGpuInferenceSchedulingMode(m_schedulingMode);

            {
                // Create a stream for audio processing
                if (!m_asrStream)
                    m_asrStream.emplace(m_asrInstance->create_stream(
                    nvigi::asr::RuntimeConfig{}
                        .set_sampling(nvigi::asr::SamplingStrategy::Greedy)
                        .set_temperature(0.0f)
                        .set_best_of(2)
                    ));

                m_asr.m_running.store(true);
                bool first_audio_submitted = false;
                bool first_result_received = false;

                // Audio format: 16000 Hz, 16-bit, mono = 32000 bytes/second
                const size_t chunk_size = size_t(0.2f * 32000); // 200ms

                size_t chunk_count = 0;
                size_t bytes_processed = 0;
                std::optional< nvigi::asr::Instance::AsyncOperation> async_op;
                std::string partial_text = "";
                bool streaming = true;
                bool is_recording = true;
                while (streaming) {
					// Handle the transition from recording to not recording based on the UI button
                    //  We are STILL streaming after recording is done, as we need to empty the pipeline
                    bool recording_is_stopping = is_recording && !m_recording;
                    if (!m_recording)
                        is_recording = false;

                    // Poll stream for results (manages operations internally!)
                    if (async_op.has_value()) {
                        if (auto result = async_op->try_get_results())
                        {
                            std::scoped_lock lock(m_mtx);
                               
                            if (messages.size() == 0 || messages.back().type != Message::Type::Question)
                                messages.push_back({ Message::Type::Question, "" });

                            if (!first_result_received && !result->text.empty() && !result->text.contains("[BLANK_AUDIO]")) {
                                first_result_received = true;
                                m_asrTimer.Stop();
                            }

                            std::string redacted = std::regex_replace(result->text, std::regex("\\[BLANK\\_AUDIO\\]"), "");
                            if (!result->text.empty()) {
                                if (result->state == nvigi::asr::ExecutionState::DataPartial)
                                {
                                    // Partial
                                    partial_text = redacted;
                                }

                                if (result->state != nvigi::asr::ExecutionState::DataPartial) {
                                    // Non-partial
                                    partial_text = "";
                                    m_a2t.append(redacted);
                                    m_gptInput.append(redacted);
                                }
                            }
                            messages.back().text = m_gptInput + partial_text;

                            // Look for Done messages only after (or as) we end recording
                            // // This ignores any spurious "Done" messages at the start of things before
                            // real data has come in from recording
                            if (!is_recording && result->state == nvigi::asr::ExecutionState::Done)
                                streaming = false;
                        }
                    }

                    // Get new audio chunk if available
                    std::vector<uint8_t> chunk_data;

                    {
                        // Thread-safe: Read current size and copy chunk data
                        std::lock_guard<std::mutex> lock(m_recordingData->mutex);
                        size_t current_size = m_recordingData->bytesWritten;

                        // Check if we have new data
                        if (current_size > bytes_processed) {
                            size_t available = current_size - bytes_processed;

                            // Only process if we have at least chunk_size of new data
                            if (available >= chunk_size) {
                                // Copy the new chunk while holding the lock
                                chunk_data.resize(chunk_size);
                                memcpy(chunk_data.data(),
                                    m_recordingData->audioBuffer.data() + bytes_processed,
                                    chunk_size);
                                bytes_processed += chunk_size;
                            }
                        }
                    }
                    // Audio buffer lock released

                    // Send chunk of data to ASR Stream; send "last block" if we are stopping the recording
                    if (!chunk_data.empty() && is_recording && !recording_is_stopping) {
                        bool is_first = (chunk_count == 0);

                        if (!first_audio_submitted) {
                            first_audio_submitted = true;
                            m_asrTimer.Start();
                        }

                        // Send chunk to ASR (non-blocking, managed internally!)
                        async_op = m_asrStream->send_audio_async(
                            chunk_data.data(),
                            chunk_data.size(),
                            is_first,
                            false
                        ).value();

                        chunk_count++;
                    }
                    else if (recording_is_stopping)
                    {
                        // Send chunk to ASR (non-blocking, managed internally!)
                        async_op = m_asrStream->send_audio_async(
                            chunk_data.data(),
                            chunk_data.size(),
                            false,
                            true
                        ).value();
                    }

                    // Small sleep to avoid busy-wait
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }

            // If GPT is not available, we give output of ASR directly to TTS.
            if (!m_gpt.m_ready && m_tts.m_ready)
            {
                AppendTTSText(m_a2t, true);
            }

            m_gptInputReady = true;
            m_asr.m_running.store(false);
            m_inferThreadRunning = false;
        };
    m_inferThread = std::make_unique<std::thread>(l);
}

void NVIGIContext::LaunchGPT(std::string prompt)
{
    m_newInferenceSequence = true;

    auto l = [this, prompt]()->void
        {
            m_inferThreadRunning = true;

            {
                std::scoped_lock lock(m_mtx);
                messages.push_back({ Message::Type::Answer, "" });
            }

            if (!m_gptChat) {
                donut::log::error("GPT Chat interface not initialized");
                m_inferThreadRunning = false;
                return;
            }

            if (m_hwiCommon)
                m_hwiCommon->SetGpuInferenceSchedulingMode(m_schedulingMode);

            // Initialize conversation with system prompt if not already done
            if (!m_conversationInitialized)
            {
                auto init_result = m_gptChat->send_message(
                    { .role = nvigi::gpt::Instance::Chat::Message::Role::System,
                      .content = m_systemPromptGPT
                    }
                );
                if (!init_result) {
                    donut::log::error("Failed to initialize GPT conversation: %s", init_result.error().what().c_str());
                    m_inferThreadRunning = false;
                    return;
                }
                m_conversationInitialized = true;
            }

            m_gpt.m_running.store(true);
            m_gptFirstTokenTimer.Start();

            // Send user message with callback for streaming
            auto result = m_gptChat->send_message(
                { .role = nvigi::gpt::Instance::Chat::Message::Role::User,
                  .content = prompt
                },
                [this](std::string_view response, nvigi::gpt::ExecutionState state) -> nvigi::gpt::ExecutionState {
                    std::string str(response);
                    
                    // Stop timer on first token
                    if (!str.empty() && m_gptFirstTokenTimer.running) {
                        m_gptFirstTokenTimer.Stop();
                    }

                    // Clean up response text (remove common tags)
                    if (str.find("<JSON>") == std::string::npos) {
                        // Remove the most common tags that may appear in the output
                        str = std::regex_replace(str, std::regex("</?THINK>", std::regex::icase), "");
                        str = std::regex_replace(str, std::regex("</?USER>", std::regex::icase), "");
                        str = std::regex_replace(str, std::regex("</?AGENT>", std::regex::icase), "");
                        
                        // Append to messages
                        if (!messages.empty()) {
                            std::scoped_lock lock(m_mtx);
                            messages.back().text.append(str);
                        }
                    }

                    // Send to TTS for audio output
                    bool done = (state == nvigi::gpt::ExecutionState::Done);
                    if (m_tts.m_ready) {
                        AppendTTSText(str, done);
                    }

                    // Wait for TTS to complete if we're done
                    if (done && m_tts.m_ready) {
                        std::unique_lock lck(m_tts.m_callbackMutex);
                        m_tts.m_callbackCV.wait(lck, [this]() { 
                            return m_tts.m_callbackState != nvigi::tts::ExecutionState::DataPending; 
                        });
                        m_ttsOutputAudio.clear();
                    }

                    return state; // Continue with current state
                }
            );

            if (!result) {
                donut::log::error("GPT inference failed: %s", result.error().what().c_str());
            }

            m_gpt.m_running.store(false);
            m_inferThreadRunning = false;
        };
    m_inferThread = std::make_unique<std::thread>(l);
}

void NVIGIContext::AppendTTSText(std::string text, bool done)
{
    m_ttsInput += text;

    if (m_ttsInput != "") {

        // We try to process chunks between 64 and 128 chracters maximum and avoid cutting sentences
        bool isLastCharacterPeriod = (m_ttsInput.back() == '\n' || m_ttsInput.back() == '.' ||
            m_ttsInput.back() == '!' || m_ttsInput.back() == '?' || m_ttsInput.back() == '"');

        if (isLastCharacterPeriod) {
            m_ttsInferenceCtx.posLastPeriod = m_ttsInput.size() - 1;
        }
        else if (m_ttsInput.back() == ' ') {
            m_ttsInferenceCtx.posLastSpace = m_ttsInput.size() - 1;
        }
        else if (m_ttsInput.back() == ',') {
            m_ttsInferenceCtx.posLastComma = m_ttsInput.size() - 1;
        }

        if ((isLastCharacterPeriod && (m_ttsInput.size() >= 64)) ||
            (m_ttsInput.size() > 128) || done) {

            std::string chunkToProcess;
            if (done || isLastCharacterPeriod ||
                (m_ttsInferenceCtx.posLastPeriod == 0
                    && m_ttsInferenceCtx.posLastSpace == 0
                    && m_ttsInferenceCtx.posLastComma == 0)) {
                chunkToProcess = m_ttsInput;
                m_ttsInput = "";
            }
            else if (m_ttsInferenceCtx.posLastPeriod != 0) {
                chunkToProcess = m_ttsInput.substr(0, m_ttsInferenceCtx.posLastPeriod + 1);
                m_ttsInput = m_ttsInput.substr(m_ttsInferenceCtx.posLastPeriod + 1);
            }
            else if (m_ttsInferenceCtx.posLastComma != 0) {
                chunkToProcess = m_ttsInput.substr(0, m_ttsInferenceCtx.posLastComma + 1);
                m_ttsInput = m_ttsInput.substr(m_ttsInferenceCtx.posLastComma + 1);
            }
            else if (m_ttsInferenceCtx.posLastSpace != 0) {
                chunkToProcess = m_ttsInput.substr(0, m_ttsInferenceCtx.posLastSpace + 1);
                m_ttsInput = m_ttsInput.substr(m_ttsInferenceCtx.posLastSpace + 1);

            }

            m_ttsInferenceCtx.posLastPeriod = 0;
            m_ttsInferenceCtx.posLastSpace = 0;
            m_ttsInferenceCtx.posLastComma = 0;

            // Synchronous TTS inference. We wait for TTS to finish before resuming GPT
            if (m_tts.m_ready)
                LaunchTTS(chunkToProcess);
        }
    }
}

void NVIGIContext::LaunchTTS(std::string prompt)
{
    if (m_newInferenceSequence)
    { 
        m_ttsFirstAudioTimer.Start();
        m_newInferenceSequence = false;
    }
    
    m_inferThreadRunning = true;

    // Remove non-UTF-8 characters inside a string
    auto removeNonUTF8 = [](const std::string& input)->std::string
        {
            std::string output;
            int countNonUtf8 = 0;
            for (char ch : input) {
                if (ch >= 0 && ch <= 127) { // ASCII range (valid UTF-8 single byte)
                    output += ch;
                }
                else {
                    countNonUtf8++;
                }
            }
            return output;
        };
    
    auto preprocessPrompt = [](std::string prompt)->std::string
        {
            // GPT answers can produce a lot of asterisks, and TTS will read them as a word.
            // We need to remove them except when they're between numbers (e.g., keep "3*5" but remove standalone *)

            // Step 1: Temporarily replace number*number patterns with a placeholder
            std::string result = std::regex_replace(prompt, std::regex(R"((\d)\*(\d))"), "$1MULT$2");
            
            // Step 2: Remove all remaining asterisks
            result = std::regex_replace(result, std::regex(R"(\*)"), "");
            
            // Step 3: Restore the multiplication patterns
            result = std::regex_replace(result, std::regex(R"(MULT)"), "*");
            
            return result;
        };

    std::string promptNonUTF8 = removeNonUTF8(preprocessPrompt(prompt));
    static std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> convert;
    std::string targetPathSpectrogram = convert.to_bytes(GetNVIGICoreDllPath().c_str()) + "/" + m_ttsInferenceCtx.m_selectedTargetVoice + "_se.bin";

    m_tts.m_callbackState.store(nvigi::tts::ExecutionState::DataPending);

    if (m_hwiCommon)
        m_hwiCommon->SetGpuInferenceSchedulingMode(m_schedulingMode);

    m_tts.m_running.store(true);

    // Setup runtime config
    nvigi::tts::RuntimeConfig config;
    config.n_timesteps = 16;

    // Modern callback signature: (const int16_t* audio, size_t samples, ExecutionState state)
    auto result = m_ttsInstance->generate(
        prompt,
        targetPathSpectrogram,
        config,
        [this](const int16_t* audio, size_t samples, nvigi::tts::ExecutionState state) -> nvigi::tts::ExecutionState {
            // Stop timer on first audio
            if (state == nvigi::tts::ExecutionState::DataPending || state == nvigi::tts::ExecutionState::Done) {
                m_ttsFirstAudioTimer.Stop();
            }

            // Process audio chunks
            if (audio && samples > 0) {
                std::scoped_lock lck(m_tts.m_callbackMutex);

                // Accumulate audio
                std::vector<int16_t> tempChunkAudio;
                for (size_t i = 0; i < samples; i++) {
                    m_ttsOutputAudio.push_back(audio[i]);
                    tempChunkAudio.push_back(audio[i]);
                }

                // Play audio chunk in a separate thread
                auto playAudio = [](const std::vector<int16_t>& audio_data_int16, std::mutex& mtxPlayAudio) -> void {
                    constexpr int bytesPerSample = 16;
                    constexpr int sampling_rate = 22050;

                    mtxPlayAudio.lock();
                    nvigi::utils::Player player(bytesPerSample, sampling_rate);
                    nvigi::utils::Buffer buffer(player,
                        audio_data_int16.data(),
                        (DWORD)(audio_data_int16.size() * sizeof(int16_t)));
                    buffer.Play();
                    buffer.Wait();
                    mtxPlayAudio.unlock();
                    };

                m_ttsInferenceCtx.playAudioThreads.push(
                    std::make_unique<std::thread>(playAudio, tempChunkAudio, std::ref(m_ttsInferenceCtx.mtxPlayAudio)));
            }

            {
                std::unique_lock lck(m_tts.m_callbackMutex);
                m_tts.m_callbackState.store(state);
                m_tts.m_callbackCV.notify_one();
            }

            return state;
        }
    );

    if (!result)
    {
        m_tts.m_callbackState.store(nvigi::tts::ExecutionState::Invalid);
        m_tts.m_callbackCV.notify_one();
    }

    m_tts.m_running.store(false);

    m_inferThreadRunning = false;
}

void NVIGIContext::FlushInferenceThread()
{
    if (m_inferThread && m_inferThread->joinable())
    {
        m_inferThread->join();
    }
    m_inferThread.reset();
}

bool NVIGIContext::ModelsComboBox(const std::string& label, bool automatic, StageInfo& stage, PluginModelInfoPtr& value)
{
    const std::map<std::string, std::vector<PluginModelInfoPtr>>& models = stage.m_pluginModelsMap;
    PluginModelInfoPtr info = value;
    bool changed = false;
    if (automatic)
    {
        int newVram = (int)stage.m_vramBudget;

        const std::string uniqueLabel = "MB ##" + label;
        ImGui::Text("VRAM Budget");
        if (ImGui::InputInt(uniqueLabel.c_str(), &newVram, 100, 500, ImGuiInputTextFlags_EnterReturnsTrue))
        {
            if (newVram < 0)
                newVram = 0;
            if (m_maxVRAM && newVram > (int)m_maxVRAM)
                newVram = (int)m_maxVRAM;
            stage.m_vramBudget = newVram;
        }

        ImGui::Text("Model Name");
        PluginModelInfoPtr newInfo = nullptr;
        if (ImGui::BeginCombo(label.c_str(), (value == nullptr) ? "No Selection" : value->m_modelName.c_str()))
        {
            if (ImGui::Selectable("No Selection", (value == nullptr)))
            {
                info = nullptr;
            }

            for (auto m : stage.m_pluginModelsMap)
            {
                PluginModelInfoPtr newInfo = nullptr;
                if (SelectAutoPlugin(stage, m.second, newInfo))
                {
                    bool is_selected_guid = info && newInfo && newInfo->m_guid == info->m_guid;
                    if (ImGui::Selectable(newInfo->m_modelName.c_str(), is_selected_guid) || is_selected_guid)
                    {
                        info = newInfo;
                    }
                }
            }
            ImGui::EndCombo();
        }
        else if (info)
        {
            // This will be hit when we move from manual to auto or adjust the vram values.
            for (auto m : stage.m_pluginModelsMap)
            {
                PluginModelInfoPtr newInfo = nullptr;
                if (SelectAutoPlugin(stage, m.second, newInfo))
                {
                    if (newInfo && newInfo->m_guid == info->m_guid)
                    {
                        info = newInfo;
                        break;
                    }
                }
            }
        }

        changed = (value != info);
    }
    else
    {
        ImGui::Text("Model Name (with Backend)");
        if (ImGui::BeginCombo(label.c_str(), (info == nullptr) ? "No Selection" : info->m_caption.c_str()))
        {
            if (ImGui::Selectable("No Selection", (info == nullptr)))
            {
                info = nullptr;
                changed = true;
            }

            // Available models
            for (auto m : models)
            {
                for (auto it : m.second)
                {
                    PluginModelInfoPtr newInfo = it;
                    bool is_selected = newInfo == info;
                    const char* key = nullptr;
                    if (newInfo)
                    {
                        std::string apiKeyName = "";
                        bool cloudNotAvailable = (newInfo->m_modelStatus == ModelStatus::AVAILABLE_CLOUD) && !GetCloudModelAPIKey(newInfo, key, apiKeyName);
                        if (cloudNotAvailable)
                        {
                            // skip
                        }
                        else if (newInfo->m_modelStatus == ModelStatus::AVAILABLE_LOCALLY || newInfo->m_modelStatus == ModelStatus::AVAILABLE_CLOUD)
                        {
                            if (ImGui::Selectable(newInfo->m_caption.c_str(), is_selected))
                            {
                                changed = !is_selected;
                                info = newInfo;
                            }
                        }
                        if (is_selected) ImGui::SetItemDefaultFocus();
                    }
                }
            }
            // Unavailable models
            for (auto m : models)
            {
                for (auto it : m.second)
                {
                    PluginModelInfoPtr newInfo = it;
                    bool is_selected = newInfo == info;
                    const char* key = nullptr;
                    if (newInfo)
                    {
                        std::string apiKeyName = "";
                        bool cloudNotAvailable = (newInfo->m_modelStatus == ModelStatus::AVAILABLE_CLOUD) && !GetCloudModelAPIKey(newInfo, key, apiKeyName);
                        if (cloudNotAvailable)
                        {
                            ImGui::TextDisabled((newInfo->m_pluginName + ": No " + apiKeyName + " API KEY " + newInfo->m_modelName).c_str());
                        }
                        else if (newInfo->m_modelStatus == ModelStatus::AVAILABLE_MANUAL_DOWNLOAD)
                        {
                            ImGui::TextDisabled((newInfo->m_caption + ": MANUAL DOWNLOAD").c_str());
                        }
                    }
                }
            }
            ImGui::EndCombo();
        }
    }

    value = info; // shared_ptr assignment

    return changed;
}

bool NVIGIContext::SelectAutoPlugin(const StageInfo& stage, const std::vector<PluginModelInfoPtr>& options, PluginModelInfoPtr& model)
{
    // Helper to find option by backend string
    auto findOption = [&options](const std::string& backend) -> PluginModelInfoPtr {
        if (backend.empty())
            return nullptr;
        for (auto info : options)
            if (info->m_backend == backend)
                return info;
        return nullptr;
    };

    // First, can we use the NV-specific plugin?
    if (auto info = findOption(stage.m_choices.m_nvda))
    {
        // Only use if we have enough VRAM budgetted
        if (info->m_modelStatus == ModelStatus::AVAILABLE_LOCALLY && stage.m_vramBudget >= info->m_vram)
        {
            model = info;
            return true;
        }
    }

    // Can we use a generic GPU plugin?
    if (auto info = findOption(stage.m_choices.m_gpu))
    {
        // Only use if we have enough VRAM budgetted
        if (info->m_modelStatus == ModelStatus::AVAILABLE_LOCALLY && stage.m_vramBudget >= info->m_vram)
        {
            model = info;
            return true;
        }
    }

    // Is cloud an option?
    if (auto info = findOption(stage.m_choices.m_cloud))
    {
        const char* key = nullptr;
        std::string apiKeyName = "";
        if (GetCloudModelAPIKey(info, key, apiKeyName))
        {
            model = info;
            return true;
        }
    }

    // What about CPU?
    if (auto info = findOption(stage.m_choices.m_cpu))
    {
        if (info->m_modelStatus == ModelStatus::AVAILABLE_LOCALLY)
        {
            model = info;
            return true;
        }
    }

    // No viable options...
    return false;
}

bool NVIGIContext::BuildOptionsUI()
{
    bool isOpen = false;

    if (ImGui::Button("Options..."))
    {
        ImGui::OpenPopup("OptionsPopup");
    }
    if (ImGui::BeginPopup("OptionsPopup"))
    {
        ImGui::BeginChild("PopupContent", ImVec2(350, 250), false);
        isOpen = true;
        if (ImGui::BeginTabBar("Settings..."))
        {
            if (ImGui::BeginTabItem("App"))
            {
                const static std::map<std::string, uint32_t> schedulingModes = {
                    { "Prioritize Graphics", (uint32_t)nvigi::SchedulingMode::kPrioritizeGraphics },
                    { "Prioritize Inference", (uint32_t)nvigi::SchedulingMode::kPrioritizeCompute },
                    { "Balanced", (uint32_t)nvigi::SchedulingMode::kBalance}
                };
                std::string value = "Not Selected";
                for (auto& iter : schedulingModes)
                    if (iter.second == m_schedulingMode)
                    {
                        value = iter.first;
                        break;
                    }

                ImGui::Text("GPU Scheduling Priority");
                if (ImGui::BeginCombo("Mode", value.c_str()))
                {
                    for (auto m : schedulingModes)
                    {
                        bool is_selected = m_schedulingMode == m.second;
                        if (ImGui::Selectable(m.first.c_str(), &is_selected) || is_selected)
                        {
                            m_schedulingMode = m.second;
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::Checkbox("Frame Rate Limiter", &m_framerateLimiting);
                if (m_framerateLimiting)
                {
                    ImGui::Separator();
                    const int maxFPS = 300;
                    if (ImGui::InputInt("Target FPS", &m_targetFramerate, 1, maxFPS, ImGuiInputTextFlags_EnterReturnsTrue))
                    {
                        if (m_targetFramerate < 1)
                            m_targetFramerate = 1;
                        if (m_targetFramerate > maxFPS)
                            m_targetFramerate = maxFPS;
                    }
                }
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("ASR"))
            {
                ImGui::BeginDisabled(m_recording || m_asr.m_running);
                ImGui::PushStyleColor(ImGuiCol_Text, TITLE_COL);
                ImGui::Text("Automatic Speech Recognition");
                ImGui::PopStyleColor();

                ImGui::Checkbox("Automatic Backend Selection", &m_asr.m_automaticBackendSelection);
                {
                    PluginModelInfoPtr newInfo = m_asr.m_info;
                    if (ModelsComboBox("##ASR", m_asr.m_automaticBackendSelection, m_asr, newInfo))
                        ReloadASRModel(newInfo);
                    ImGui::EndDisabled();
                }
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("GPT"))
            {
                ImGui::BeginDisabled(m_gpt.m_running);
                ImGui::PushStyleColor(ImGuiCol_Text, TITLE_COL);
                ImGui::Text("GPT");
                ImGui::PopStyleColor();

                ImGui::Checkbox("Automatic Backend Selection", &m_gpt.m_automaticBackendSelection);
                {
                    PluginModelInfoPtr newInfo = m_gpt.m_info;
                    if (ModelsComboBox("##GPT", m_gpt.m_automaticBackendSelection, m_gpt, newInfo))
                        ReloadGPTModel(newInfo);
                    ImGui::EndDisabled();
                }
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("TTS"))
            {
                ImGui::BeginDisabled(m_tts.m_running);
                ImGui::PushStyleColor(ImGuiCol_Text, TITLE_COL);
                ImGui::Text("TTS");
                ImGui::PopStyleColor();

                ImGui::Checkbox("Automatic Backend Selection", &m_tts.m_automaticBackendSelection);
                {
                    PluginModelInfoPtr newInfo = m_tts.m_info;
                    if (ModelsComboBox("##TTS", m_tts.m_automaticBackendSelection, m_tts, newInfo))
                        ReloadTTSModel(newInfo);

                    // Add comboBox for target voices files
                    ImGui::Text("Voice Name");
                    std::vector<std::string> targetVoices = GetPossibleTargetVoices(GetNVIGICoreDllPath());
                    if (ImGui::BeginCombo("##TargetVoices", m_ttsInferenceCtx.m_selectedTargetVoice.empty() ? "Select a target voice" : m_ttsInferenceCtx.m_selectedTargetVoice.c_str())) {
                        for (const auto& file : targetVoices) {
                            bool isSelected = (m_ttsInferenceCtx.m_selectedTargetVoice == file);
                            if (ImGui::Selectable(file.c_str(), isSelected)) {
                                m_ttsInferenceCtx.m_selectedTargetVoice = file;
                            }
                            if (isSelected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::EndDisabled();
                }
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
		ImGui::EndChild();
        ImGui::EndPopup();
    }
    return isOpen;
}

void NVIGIContext::BuildModelsStatusUI()
{
    ImGui::Separator();

    if (m_asr.m_ready)
    {
        std::string asr = "ASR: " + m_asr.m_info->m_caption;
        ImGui::Text(asr.c_str());
    }
    else
    {
        if (m_asr.m_info != nullptr)
            ImGui::Text("ASR: Loading model please wait...");
        else
            ImGui::Text("ASR: No model selected ...");
    }

    if (m_gpt.m_ready)
    {
        std::string gpt = "GPT: " + m_gpt.m_info->m_caption;
        ImGui::Text(gpt.c_str());
    }
    else
    {
        if (m_gpt.m_info != nullptr)
            ImGui::Text("GPT: Loading model please wait...");
        else
            ImGui::Text("GPT: No model selected ...");
    }

    if (m_tts.m_ready)
    {
        std::string tts = "TTS: " + m_tts.m_info->m_caption;
        ImGui::Text(tts.c_str());

        std::string tts_voice = "TTS Voice: " + m_ttsInferenceCtx.m_selectedTargetVoice;
        ImGui::Text(tts_voice.c_str());
    }
    else
    {
        if (m_tts.m_info != nullptr)
            ImGui::Text("TTS: Loading model please wait...");
        else
            ImGui::Text("TTS: No model selected ...");
    }
}

void NVIGIContext::BuildChatUI()
{
    if (m_gpt.m_ready || m_tts.m_ready)
    {
        std::scoped_lock lock(m_mtx);

        static char inputBuffer[512] = {};

        static bool setFocusOnPromptInput = false;
        static float lastScrollMaxY = 0.0f;
        static size_t lastMsgCount = 0;

        // Shorten text box if settings are visible
        auto child_size = ImVec2(ImGui::GetContentRegionAvail().x, 800);
        if (ImGui::BeginChild("Chat UI", child_size, false))
        {
            if (m_gpt.m_ready || m_asr.m_ready || m_tts.m_ready)
            {
                // Create a child window with a scrollbar for messages
                if (ImGui::BeginChild("Messages", ImVec2(0, -5 * ImGui::GetFrameHeightWithSpacing()), true))
                {
                    ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + child_size.x - 30);  // Wrapping text before the edge, added a small offset for aesthetics

                    for (const auto& message : messages)
                    {
                        if (message.type == Message::Type::Question)
                            ImGui::TextColored(ImVec4(1, 1, 0, 1), "Q: %s", message.text.c_str());
                        else
                            ImGui::TextColored(ImVec4(0, 1, 0, 1), "A: %s", message.text.c_str());
                    }

                    ImGui::PopTextWrapPos();  // Reset wrapping position

                    // Scroll to the bottom when a new message is added, if we were previously at the bottom
                    float newScrollMaxY = ImGui::GetScrollMaxY();
                    if (newScrollMaxY > lastScrollMaxY)
                    {
                        if (ImGui::GetScrollY() == lastScrollMaxY)
                        {
                            ImGui::SetScrollHereY(1.0f);
                        }
                        // We need to reacquire focus on the prompt input
                        setFocusOnPromptInput = true;
                    }
                    else if ((newScrollMaxY == 0.0f) && (lastMsgCount != messages.size()))
                    {
                        // New text was added, but there's not enough text to require a scrollbar yet, but we still need to set focus to the prompt input
                        setFocusOnPromptInput = true;
                    }

                    // Update maximum scroll position and message count to check against next frame
                    lastScrollMaxY = newScrollMaxY;
                    lastMsgCount = messages.size();
                }
                ImGui::EndChild();
            }
        }

        // Do not show Send when ASR or GPT is running, or when we're recording audio
        if (!m_gpt.m_running &&!m_tts.m_running)
        {
            if (!m_asr.m_running)
            {
                ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
                // Input text box and button to send messages
                if (ImGui::InputText("##Input", inputBuffer, sizeof(inputBuffer), ImGuiInputTextFlags_EnterReturnsTrue))
                {
                    if (m_gpt.m_ready)
                    {
                        m_gptInput = inputBuffer;
                        m_gptInputReady = true;
                        inputBuffer[0] = '\0';  // Clear the buffer
                        messages.push_back({ Message::Type::Question, m_gptInput });
                    }
                    else if (m_tts.m_ready)
                    {
                        m_newInferenceSequence = true;

                        auto inferTTS = [this]()->void
                            {
                                AppendTTSText(inputBuffer, true);
                                inputBuffer[0] = '\0';  // Clear the buffer
                            };
                        m_inferThread = std::make_unique<std::thread>(inferTTS);
                    }
                    // Focus is lost when we hit enter, so we need to reacquire it to type another prompt
                    setFocusOnPromptInput = true;
                }
                // If we have lost focus and need to re-acquire it, do so now and clear the flag so we don't keep setting it.
                // Also make sure we don't try to steal it away from another active element
                if (setFocusOnPromptInput && !ImGui::IsAnyItemActive())
                {
                    ImGui::SetKeyboardFocusHere(-1);
                    setFocusOnPromptInput = false;
                }
                ImGui::PopItemWidth();
            }

            if (m_asr.m_ready)
            {
                if (m_recording)
                {
                    if (ImGui::Button("Stop"))
                    {
                        m_recording = false;
                    }
                } // Do not show Record button when ASR or GPT is running
                else if (!m_gpt.m_running && !m_asr.m_running && ImGui::Button("Record"))
                {
                    FlushInferenceThread();
                    m_recording = true;

                    m_a2t = "";
                    m_gptInput = "";
                    LaunchASR();
                }
            }

            if (!m_recording && m_gpt.m_ready)
            {
                ImGui::SameLine();
                if (ImGui::Button("Reset Chat"))
                {
                    m_conversationInitialized = false;
                    messages.clear();
                    messages.push_back({ Message::Type::Answer, "Conversation Reset: I'm here to chat - type a query or record audio to interact!" });
                }
            }
        }
        if (ImGui::BeginChild("Performance"))
        {
            if (m_asr.m_ready)
                ImGui::Text("ASR First Text: %.2f ms", m_asrTimer.GetElapsedMiliseconds());
            if (m_gpt.m_ready)
                ImGui::Text("GPT First Token: %.2f ms", m_gptFirstTokenTimer.GetElapsedMiliseconds());
            if (m_tts.m_ready)
                ImGui::Text("TTS First Audio: %.2f ms", m_ttsFirstAudioTimer.GetElapsedMiliseconds());
        }
        ImGui::EndChild();
        ImGui::EndChild();
    }
    else
    {
        if (m_gpt.m_info == nullptr || m_asr.m_info == nullptr)
            ImGui::Text("Loading models please wait ...");
        else
            ImGui::Text("No models selected ...");
    }
}

void NVIGIContext::BuildUI()
{

    if (m_gptInputReady && !m_asr.m_running)
    {
        m_gptInputReady = false;

        if (m_gpt.m_ready) {
            FlushInferenceThread();
            if (m_recordingData)
            {
                auto audioData = AudioRecorder::StopRecording(m_recordingData);
                m_recordingData = nullptr;
            }

            LaunchGPT(m_gptInput);
        }
    }

    BuildModelsStatusUI();
    BuildOptionsUI();
    BuildChatUI();
}

void NVIGIContext::GetVRAMStats(size_t& current, size_t& budget)
{
    DXGI_QUERY_VIDEO_MEMORY_INFO videoMemoryInfo{};
    if (m_targetAdapter)
    {
        m_targetAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &videoMemoryInfo);
    }
    current = videoMemoryInfo.CurrentUsage;
    budget = videoMemoryInfo.Budget;
}
