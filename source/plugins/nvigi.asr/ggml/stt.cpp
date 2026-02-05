// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <future>

// premake determines the include path
#include <whisper.h>

#include "source/core/nvigi.api/nvigi.h"
#include "source/core/nvigi.log/log.h"
#include "source/core/nvigi.exception/exception.h"
#include "source/core/nvigi.plugin/plugin.h"
#include "source/core/nvigi.file/file.h"
#include "source/core/nvigi.extra/extra.h"
#include "source/core/nvigi.thread/thread.h"
#include "source/plugins/nvigi.asr/ggml/versions.h"
#include "source/plugins/nvigi.asr/nvigi_asr_whisper.h"
#include "source/utils/nvigi.ai/ai.h"
#include "source/utils/nvigi.poll/poll.h"
#include "_artifacts/gitVersion.h"
#include "external/json/source/nlohmann/json.hpp"
#include "source/plugins/nvigi.asr/ggml/stt.h"
#include "source/utils/nvigi.ai/ai_data_helpers.h"
#include "source/utils/nvigi.hwi/cuda/runtime_context_scope.h"


using json = nlohmann::json;

#if GGML_USE_CUBLAS
#include "source/core/nvigi.api/nvigi_cuda.h"
#include "source/utils/nvigi.hwi/cuda/push_poppable_cuda_context.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvtx3/nvToolsExt.h"
#elif defined(GGML_USE_D3D12)
#include "external/agility-sdk/build/native/include/d3d12.h"
#include "source/core/nvigi.api/nvigi_d3d12.h"
#include "source/plugins/nvigi.hwi/d3d12/nvigi_hwi_d3d12.h"
#include "source/utils/nvigi.d3d12/d3d12_helpers.h"
#if PROFILE_D3D
#include "nvtx3/nvToolsExt.h"
#endif
namespace nvigi {
    // this should match the PFun_commandListAction defined in WhisperCPP - redefining as it's at a depth/scope in WhisperCPP not exposed to IGI
    using PFun_commandListAction = void(ID3D12CommandList* pCommandList, int action);
}
struct ggml_d3d12_ctx
{
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* cmd_queue_direct = nullptr;
    ID3D12CommandQueue* cmd_queue_compute = nullptr;
    ID3D12CommandQueue* cmd_queue_copy = nullptr;
    nvigi::PFun_createCommittedResource* createCommittedResource = nullptr;
    nvigi::PFun_destroyResource* destroyResource = nullptr;
    nvigi::PFun_commandListAction* commandListAction = nullptr;
    void* userContextCreate = nullptr;
    void* userContextDestroy = nullptr;
    bool allowReBAR = true;
    // IMPORTANT: Some kernels can read/write beyond the actual buffer size so we fix this with padding.
    // 
    // Why? Because fixing this in kernels with if/else is expensive and using descriptor tables adds overhead (root uav/srv with no bound checks are fastest).
    uint64_t bufferPadBytes = 0;
    uint32_t maxDescriptorSets = 32 * 1024;
    uint32_t constantBufferSize = 1024 * 256;
    uint32_t coreCount = 0;
    uint32_t architecture = 0;
};
extern void ggml_d3d12_set_params(const ggml_d3d12_ctx& ctx);
#elif defined GGML_USE_VULKAN
#include "external/vulkanSDK/include/vulkan/vulkan.h"
#include "source/core/nvigi.api/nvigi_vulkan.h"
extern void ggml_set_vk_params(VkPhysicalDevice physicalDevice, VkDevice device, VkQueue cmd_queue_direct, VkQueue cmd_queue_compute, VkQueue cmd_queue_copy,
    nvigi::PFun_allocateMemoryCallback* allocateMemory, nvigi::PFun_freeMemoryCallback* freeMemory,
    void* userContextAlloc, void* userContextFree);
#endif

#if defined GGML_USE_VULKAN || defined(GGML_USE_D3D12)
extern void ggml_vk_backend_free();
#endif


#define NUM_BUFFERS 2
#define BUFFER_SIZE 4096

namespace nvigi
{
namespace asr
{

constexpr StreamSignal kStreamTypeNone = (StreamSignal)-1;
constexpr int kAvailableAudioBufferSize = 0;

#define NUM_BUFFERS 2
#define BUFFER_SIZE 4096

static uint32_t replace_all(std::string& str, const std::string& search_str, const std::string& replace_str)
{
    uint32_t num_replaced = 0;
    size_t start_pos = str.find(search_str);
    while (start_pos != std::string::npos && start_pos < str.length())
    {
        str.replace(start_pos, search_str.length(), replace_str);
        num_replaced++;
        start_pos += replace_str.length(); // Move past the replacement
        start_pos = str.find(search_str, start_pos);
    }

    return num_replaced;
}

static void whisperLogCallback(ggml_log_level level, const char* text, void* user_data) 
{
    std::string msg(text);
#if defined(GGML_USE_D3D12)
    // Running Vulkan on DX12 but ggml is unaware, replace Vulkan with D3D12 to avoid confusion
    replace_all(msg, "Vulkan0", "D3D12(0)");
    replace_all(msg, "Vulkan1", "D3D12(1)");
    replace_all(msg, "vulkan", "d3d12");
    replace_all(msg, "Vulkan", "D3D12");
#endif
    if (level == GGML_LOG_LEVEL_WARN)
    {
        NVIGI_LOG("whisper", LogType::eWarn, nvigi::log::YELLOW, "%s", msg.c_str());
    }
    else if (level == GGML_LOG_LEVEL_ERROR)
    {
        NVIGI_LOG("whisper", LogType::eError, nvigi::log::RED, "%s", msg.c_str());
    }
    else
    {
        NVIGI_LOG("whisper", LogType::eInfo, nvigi::log::WHITE, "%s", msg.c_str());
    }
}

struct InferenceContext
{
    InferenceContext(const nvigi::NVIGIParameter* params) : cudaContext(params) {}

    poll::PollContext<nvigi::InferenceExecutionState> pollCtx;
#ifndef NVIGI_ASR_GFN_NVCF
    whisper_context* model{};
#endif
    whisper_params params{};
    CircularBuffer audio;
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32;
    std::vector<float> pcmf32_new;

    std::mutex mtx;
    std::future<Result> job;
    std::atomic<bool> running = true;
    std::atomic<bool> cancelled = false;  // Flag to request early cancellation
    int n_iter = 0;  // Iteration counter for sliding window mode (matches whisper.cpp stream.cpp)

#if GGML_USE_CUBLAS
    // Used to set the relative priority of GPU inference and graphics
    std::vector<cudaStream_t> cuda_streams;
#endif

#ifdef GGML_USE_D3D12
    // Used to set the relative priority of GPU inference and graphics
    ID3D12Device* device{};
#endif

#if GGML_USE_CUBLAS
    // Use PushPoppableCudaContext defined in push_poppable_cuda_context.h
#else
    // Use dummy PushPoppableCudaContext to avoid depending on CUDA
    struct PushPoppableCudaContext
    {    
        bool constructorSucceeded = true;
        PushPoppableCudaContext(const nvigi::NVIGIParameter* params) {}
        void pushRuntimeContext() {}
        void popRuntimeContext() {}
    };    
#endif
    PushPoppableCudaContext cudaContext;
};

PluginID getFeatureId(InferenceInstanceData* data)
{
#if GGML_USE_CUBLAS
    return plugin::asr::ggml::cuda::kId;
#elif GGML_USE_VULKAN
    return plugin::asr::ggml::vulkan::kId;
#elif defined(GGML_USE_D3D12)
    return plugin::asr::ggml::d3d12::kId;
#else
    return plugin::asr::ggml::cpu::kId;
#endif
}

const nvigi::InferenceDataDescriptorArray* getInputSignature(InferenceInstanceData* data)
{
    static std::vector<InferenceDataDescriptor> slots = { {nvigi::kASRWhisperDataSlotAudio,InferenceDataAudio::s_type, false } };
    static InferenceDataDescriptorArray s_desc = { slots.size(), slots.data() };
    return &s_desc;
}

const nvigi::InferenceDataDescriptorArray* getOutputSignature(InferenceInstanceData* data)
{
    static std::vector<InferenceDataDescriptor> slots = { {nvigi::kASRWhisperDataSlotTranscribedText,InferenceDataText::s_type, false } };
    static InferenceDataDescriptorArray s_desc = { slots.size(), slots.data() };
    return &s_desc;
}

struct ASRContext
{
    NVIGI_PLUGIN_CONTEXT_CREATE_DESTROY(ASRContext);

    void onCreateContext() {};
    void onDestroyContext() {};

    IAutoSpeechRecognition api{};
    IPolledInferenceInterface polledApi{};
    
    PluginID feature{};

    // Caps and requirements
    json modelInfo;
    ai::CommonCapsData capsData;

#ifdef GGML_USE_CUBLAS
    nvigi::IHWICuda* icig{};
#elif defined(GGML_USE_D3D12)
    nvigi::IHWID3D12* iscg{};
#endif
    nvigi::system::ISystem* isystem{};
};

nvigi::Result evaluate(nvigi::InferenceExecutionContext* execCtx);
nvigi::Result evaluateAsync(nvigi::InferenceExecutionContext* execCtx);
nvigi::Result cancelAsyncEvaluation(nvigi::InferenceExecutionContext* execCtx);

}

//! Define our plugin, make sure to update version numbers in versions.h
NVIGI_PLUGIN_DEFINE("nvigi.plugin.asr", Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH), Version(API_MAJOR, API_MINOR, API_PATCH), asr, ASRContext)

nvigi::Result flushAndTerminate(nvigi::asr::InferenceContext* instance)
{
    auto result = nvigi::kResultOk;
    if (instance->job.valid())
    {
        // Signal the async job to stop
        instance->running.store(false);
        instance->cancelled.store(true);

        // Keep draining results while waiting for job to complete
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < timeout)
        {
            // Release any pending results to unblock the background thread
            if (instance->pollCtx.checkResultPending())
            {
                instance->pollCtx.releaseResults(nvigi::kInferenceExecutionStateDone);
            }

            // Check if job is done
            auto futureResult = instance->job.wait_for(std::chrono::milliseconds(10));
            if (futureResult == std::future_status::ready)
            {
                // Job completed, get the result
                if (NVIGI_FAILED(result, instance->job.get()))
                {
                    return result;
                }
                return kResultOk;
            }
        }

        // Timed out
        NVIGI_LOG_WARN("Async job timed out.");
        return kResultTimedOut;
    }
    return result;
}

nvigi::Result whisperDestroyInstance(const nvigi::InferenceInstance* instance)
{
    if (instance)
    {
        auto sttInstance = (nvigi::asr::InferenceContext*)(instance->data);
        if (NVIGI_FAILED(result, flushAndTerminate(sttInstance)))
        {
            return result;
        }

        {
#if GGML_USE_CUBLAS
            nvigi::RuntimeContextScope scope(*sttInstance);
#endif
            whisper_free(sttInstance->model);
        }

        delete sttInstance;
        delete instance;
    }
    return nvigi::kResultOk;
}

nvigi::Result whisperCreateInstance(const nvigi::NVIGIParameter* _params, nvigi::InferenceInstance** _instance)
{
    auto common = findStruct<CommonCreationParameters>(_params);
    auto creationParams = findStruct<ASRWhisperCreationParameters>(_params);
    if (!creationParams || !common) return nvigi::kResultInvalidParameter;

    auto& params = *creationParams;
    if (!_instance || !common->utf8PathToModels || !common->modelGUID) return nvigi::kResultInvalidParameter;

#ifdef GGML_USE_CUBLAS
    auto cudaParams = findStruct<CudaParameters>(_params);
    if (cudaParams && cudaParams->getVersion() >= kStructVersion2)
    {
        if (cudaParams->cudaMallocReportCallback)
            nvigi::asr::setCudaMallocReportCallback(cudaParams->cudaMallocReportCallback, cudaParams->cudaMallocReportUserContext);
        if (cudaParams->cudaFreeReportCallback)
            nvigi::asr::setCudaFreeReportCallback(cudaParams->cudaFreeReportCallback, cudaParams->cudaFreeReportUserContext);
        if (cudaParams->cudaMallocCallback)
            nvigi::asr::setCudaMallocCallback(cudaParams->cudaMallocCallback, cudaParams->cudaMallocUserContext);
        if (cudaParams->cudaFreeCallback)
            nvigi::asr::setCudaFreeCallback(cudaParams->cudaFreeCallback, cudaParams->cudaFreeUserContext);
    }
#endif

    using namespace nvigi::asr;
    auto& ctx = (*asr::getContext());

    *_instance = nullptr;

    std::string pathToModel{};

    if (ctx.modelInfo.empty())
    {
        if (!ai::findModels(common, { "gguf" }, ctx.modelInfo))
        {
            return kResultInvalidParameter;
        }
    }

    std::vector<std::string> files;
    try
    {
        files = ctx.modelInfo[common->modelGUID]["gguf"];

#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
        size_t neededVRAM = ctx.modelInfo[common->modelGUID]["vram"];
        if (common->vramBudgetMB < neededVRAM)
        {
            NVIGI_LOG_WARN("Provided VRAM %uMB is insufficient, required VRAM is %uMB", common->vramBudgetMB, neededVRAM);
            return kResultInsufficientResources;
        }
#endif
    }
    catch (const std::exception& e)
    {
        NVIGI_LOG_ERROR("Exception %s", e.what());
        return kResultJSONException;
    }

    if (files.empty())
    {
        NVIGI_LOG_ERROR("Failed to find model in the expected directory '%s'", common->utf8PathToModels);
        return kResultInvalidParameter;
    }

    auto instanceData = new nvigi::asr::InferenceContext(_params);
    if (!instanceData->cudaContext.constructorSucceeded) return kResultInvalidState;

#if defined(GGML_USE_CUBLAS)
    if (common->numThreads > 1)
    {
        NVIGI_LOG_WARN("For optimal performance when using CUDA only one CPU thread is used");
    }
    instanceData->params.n_threads = 1;
#else
    instanceData->params.n_threads = common->numThreads;
#endif


    instanceData->params.language = params.language ? params.language : "en";

#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
#ifndef NVIGI_PRODUCTION
    size_t currentUsageMB{};
    extra::ScopedTasks vram([&currentUsageMB]() {
        system::VRAMUsage* usage;
        system::getInterface()->getVRAMStats(0, &usage);
        currentUsageMB = usage->currentUsageMB;
        },
        [&currentUsageMB]() {
            system::VRAMUsage* usage;
            system::getInterface()->getVRAMStats(0, &usage);
            currentUsageMB = usage->currentUsageMB - currentUsageMB;
            NVIGI_LOG_INFO("New instance using %lluMB budget %lluMB", currentUsageMB, usage->budgetMB);
        }
    );
#endif
#endif
    {
        pathToModel = files[0];

        NVIGI_LOG_INFO("Loading model '%s'", pathToModel.c_str());

        {
#if GGML_USE_CUBLAS
            // Multi GPU / CIG support:
            // ggml_backend_cuda_reg checks whether it has been called before, 
            // and if not iterates over all CUDA devices and initializes them.
            // For each iteration, cudart calls cuDevicePrimaryCtxRetain, which
            // sets the context to the primary context, which is non-CIG. 
            // To be able to support CIG we need to set the context and not have
            // anyone else change it.
            // So we call ggml_backend_cuda_reg here outside the 
            // RuntimeContextScope. That way it will never be called again, and
            // we have full control of the context inside the scope.
            ggml_backend_reg_t cuda_reg = ggml_backend_cuda_reg();
            nvigi::RuntimeContextScope scope(*instanceData);
            cudaStream_t stream{};
            cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
#endif
#ifdef GGML_USE_VULKAN
            auto vkParams = findStruct<VulkanParameters>(_params);
            if (vkParams)
            {
                ggml_set_vk_params(vkParams->physicalDevice, vkParams->device, vkParams->queue, vkParams->queueCompute, vkParams->queueTransfer,
                    vkParams->allocateMemoryCallback, vkParams->freeMemoryCallback,
                    vkParams->allocateMemoryCallbackUserContext, vkParams->freeMemoryCallbackUserContext);
            }
#elif defined(GGML_USE_D3D12)
            auto d3d12Params = findStruct<D3D12Parameters>(_params);
            if (NVIGI_FAILED(res, d3d12::validateParameters(d3d12Params)))
            {
                return res;
            }

            system::Adapter* adapter{};
            if (NVIGI_FAILED(res, d3d12::getDeviceAdapter(d3d12Params, ctx.isystem, &adapter)))
            {
                return res;
            }
            if (adapter->vendor == VendorId::eNVDA)
            {
                if (!ctx.iscg && !framework::getInterface(plugin::getContext()->framework, plugin::hwi::d3d12::kId, &ctx.iscg))
                {
                    NVIGI_LOG_ERROR("Missing interface from 'nvigi.plugin.hwi.d3d12'");
                    return kResultInvalidState;
                }

                if (NVIGI_FAILED(res, d3d12::applyNVDASpecificSettings(d3d12Params, ctx.iscg)))
                {
                    return res;
                }
            }

            auto applySchedulingModeCallback = [](ID3D12CommandList* pCommandList, int action)
                {
                    // Don't check iscg and version every call, because we do it once when callback is registered
                    IHWID3D12* iscg = asr::getContext()->iscg;
                    ID3D12GraphicsCommandList* pGraphicsCommandList = nullptr;
                    if (SUCCEEDED(pCommandList->QueryInterface<ID3D12GraphicsCommandList>(&pGraphicsCommandList)))
                    {
                        iscg->d3d12ApplyGlobalGpuInferenceSchedulingModeToCommandList(pGraphicsCommandList);
                    }
                };

            // Do iscg and version checks now to avoid doing them inside every callback       
            PFun_commandListAction* CLresetCallback = nullptr;
            IHWID3D12* iscg = asr::getContext()->iscg;
            if (iscg)
            {
                if (iscg->getVersion() >= 4)
                {
                    CLresetCallback = applySchedulingModeCallback;
                }
                else
                {
                    NVIGI_LOG_WARN_ONCE("We need version 4 of hwi.d3d12 in order to set D3D12 compute scheduling mode, and version available is %d. Performance may be suboptimal.\n", iscg->getVersion());
                }
            }
            else
            {
                NVIGI_LOG_WARN_ONCE("hwi.d3d12 was not loaded, so couldn't set D3D12 compute scheduling mode. Performance may be suboptimal.\n");
            }

            bool allowReBAR = d3d12Params->getVersion() < 3 || !(d3d12Params->flags & nvigi::D3D12ParametersFlags::eDisableReBAR);
            ggml_d3d12_ctx d3d12GGMLParams = {
                d3d12Params->device,
                d3d12Params->queue,
                d3d12Params->queueCompute,
                d3d12Params->queueCopy,
                d3d12Params->createCommittedResourceCallback,
                d3d12Params->destroyResourceCallback,
                CLresetCallback,
                d3d12Params->createCommitResourceUserContext,
                d3d12Params->destroyResourceUserContext,
                allowReBAR,
                0, // bufferPadBytes
                32 * 1024, // maxDescriptorSets
                256 * 1024, // constantBufferSize
                adapter->coreCount,
                adapter->architecture
            };
            ggml_d3d12_set_params(d3d12GGMLParams);

#endif
            struct whisper_context_params cparams = whisper_context_default_params();
            if (params.getVersion() >= kStructVersion2)
            {
                cparams.flash_attn = params.flashAtt;
            }
            if (params.getVersion() >= kStructVersion3)
            {
                instanceData->params.translate = params.translate;
                instanceData->params.detect_language = params.detectLanguage;
                instanceData->params.length_ms = params.lengthMs;
                instanceData->params.step_ms = params.stepMs;
                instanceData->params.keep_ms = params.keepMs;
            }
#if GGML_USE_CUBLAS
            cparams.use_gpu = true;
            auto cudaParams = findStruct<CudaParameters>(_params);
            auto d3dParams = findStruct<D3D12Parameters>(_params);
            
            // If using CIG/D3D12, the CIG context is already current (pushed by RuntimeContextScope)
            // Query which device it's on so whisper uses the correct device
            if (d3dParams && d3dParams->queue && instanceData->cudaContext.cudaCtx)
            {
                // CIG context is already current, just query which device
                CUdevice cuDevice;
                CUresult cuerr = cuCtxGetDevice(&cuDevice);
                if (cuerr == CUDA_SUCCESS)
                {
                    cparams.gpu_device = (int)cuDevice;
                    NVIGI_LOG_INFO("CIG context is already active on device %d, whisper will use this device", 
                                  cparams.gpu_device);
                }
                else
                {
                    cparams.gpu_device = 0;
                    NVIGI_LOG_WARN("Failed to query current device, defaulting to device 0");
                }
            }
            else if (cudaParams)
            {
                cparams.gpu_device = cudaParams->device;
                NVIGI_LOG_INFO("User specified CUDA device %d in CudaParameters struct", cparams.gpu_device);
            }
            else
            {
                cparams.gpu_device = 0;
                NVIGI_LOG_INFO("No device specified, using default CUDA device 0");
            }
#endif
            try
            {
                instanceData->model = whisper_init_from_file_with_params(pathToModel.c_str(), cparams);
            }
            catch (std::exception& e)
            {
                NVIGI_LOG_ERROR("%s", e.what());
                delete instanceData;
                return kResultInvalidState;
            }

            // Check before doing anything else
            if (!instanceData->model)
            {
                NVIGI_LOG_ERROR("Call to 'whisper_init_from_file_with_params' failed");
                delete instanceData;
                return kResultInvalidState;
            }
#if GGML_USE_CUBLAS
            // We ask whisper for all the cuda_streams it is going to use and 
            // store them to enable us to change their priorities dynamically
            size_t stream_count = whisper_get_cuda_stream_count(instanceData->model);
            instanceData->cuda_streams.resize(stream_count);
            whisper_get_cuda_streams(instanceData->model, (void**)instanceData->cuda_streams.data(), stream_count);

            // Apply the global priority to all streams
            if (instanceData->cudaContext.usingCiG && ctx.icig->getVersion() >= 2)
            {
                nvigi::Result cuerr = ctx.icig->cudaApplyGlobalGpuInferenceSchedulingMode(instanceData->cuda_streams.data(), instanceData->cuda_streams.size());
                if (cuerr != kResultOk)
                {
                    if (cuerr == kResultDriverOutOfDate)
                    {
                        NVIGI_LOG_WARN_ONCE("Could not set relative priority of CUDA compute and graphics because the driver is out of date\n");
                    }
                    else
                    {
                        NVIGI_LOG_ERROR("Could not set relative priority of CUDA compute and graphics because a CUDA error occurred.\n");
                    }
                }
            }
#endif
        }
        
#if GGML_USE_CUBLAS
        auto platform = "ggml.cuda";
#elif GGML_USE_VULKAN
        auto platform = "ggml.vulkan";
#elif defined(GGML_USE_D3D12)
        auto platform = "ggml.d3d12";
#else
        auto platform = "ggml.cpu";
#endif
        NVIGI_LOG_VERBOSE("Created instance for backend '%s' - threads %d", platform, instanceData->params.n_threads);

        instanceData->audio.init(instanceData->params.length_ms);
    }
    
    auto instance = new InferenceInstance();
    instance->data = instanceData;
    instance->getFeatureId = asr::getFeatureId;
    instance->getInputSignature = asr::getInputSignature;
    instance->getOutputSignature = asr::getOutputSignature;
    instance->evaluate = asr::evaluate;
    instance->evaluateAsync = asr::evaluateAsync;
    instance->cancelAsyncEvaluation = asr::cancelAsyncEvaluation;
    
    *_instance = instance;

    return kResultOk;
}

nvigi::Result whisperGetCapsAndRequirements(nvigi::NVIGIParameter** _info, const nvigi::NVIGIParameter* _params)
{
    auto common = findStruct<CommonCreationParameters>(_params);
    auto params = findStruct<ASRWhisperCreationParameters>(_params);
    if (!common || !params) return nvigi::kResultInvalidParameter;

    static CommonCapabilitiesAndRequirements s_capsCommon{};
    static ASRWhisperCapabilitiesAndRequirements s_caps{};
    s_caps.common = &s_capsCommon;
    auto info = &s_caps;
    *_info = s_caps;

    auto& ctx = (*asr::getContext());
    if (!ai::findModels(common, { "gguf" }, ctx.modelInfo))
    {
        return kResultInvalidParameter;
    }

    // Supported languages, must be null terminated since we don't provide a size
    static const char* s_languages[] = { "auto", nullptr };
    info->supportedLanguages = s_languages;

    // CUDA or CPU backend
#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
    info->common->supportedBackends = nvigi::InferenceBackendLocations::eGPU | nvigi::InferenceBackendLocations::eCPU;
#else
    info->common->supportedBackends = nvigi::InferenceBackendLocations::eCPU;
#endif

    //! Must be called after we set supported backends to correctly filter models
    ai::populateCommonCapsAndRequirements(ctx.capsData, *common, *info->common, ctx.modelInfo);

    return kResultOk;
}

nvigi::Result whisperEvaluate(nvigi::InferenceExecutionContext* execCtx, bool async)
{
#if GGML_USE_CUBLAS || (defined(GGML_USE_D3D12) && PROFILE_D3D)
    nvtxRangePushA("whisperEvaluate");
#endif

    auto& ctx = (*asr::getContext());

    // Validate all inputs first

    if (!execCtx)
    {
        NVIGI_LOG_ERROR("Unable to find 'InferenceExecutionContext' structure in the inputs");
        return kResultInvalidParameter;
    }

    // In async mode callback is optional since we can poll for results
    if (!execCtx->callback && !async)
    {
        NVIGI_LOG_ERROR("ASR inference callback not provided");
        return kResultInvalidParameter;
    }

    if (!execCtx->instance)
    {
        NVIGI_LOG_ERROR("ASR inference instance not provided");
        return kResultInvalidParameter;
    }

    if (ctx.feature != execCtx->instance->getFeatureId(execCtx->instance->data))
    {
        NVIGI_LOG_ERROR("Invalid inference instance - expecting ASR %u got %u", ctx.feature, execCtx->instance->getFeatureId(execCtx->instance->data));
        return kResultInvalidParameter;
    }

    // Now we are good to go!
    using namespace nvigi::asr;

    const nvigi::InferenceDataAudio* audioInput{};
    if (!execCtx->inputs->findAndValidateSlot(kASRWhisperDataSlotAudio,&audioInput))
    {
        NVIGI_LOG_ERROR("Expecting single inference input of type 'nvigi::InferenceDataAudio'");
        return kResultInvalidParameter;
    }
    if (audioInput->samplingRate != 16000 || audioInput->channels != 1)
    {
        NVIGI_LOG_ERROR("ASR requires audio input with the sampling rate of 16000 and a single channel");
        return kResultInvalidParameter;
    }

    auto triggerCallback = [](nvigi::InferenceExecutionContext* execCtx, const std::string& content, nvigi::InferenceExecutionState state)->nvigi::InferenceExecutionState
    {
        auto res = nvigi::kInferenceExecutionStateInvalid;
        const nvigi::InferenceDataText* output{};
        if (execCtx->outputs && execCtx->outputs->findAndValidateSlot(kASRWhisperDataSlotTranscribedText, &output))
        {
            auto cpuBuffer = castTo<CpuData>(output->utf8Text);
            if (cpuBuffer->buffer && cpuBuffer->sizeInBytes >= content.size())
            {
                strcpy_s((char*)cpuBuffer->buffer, cpuBuffer->sizeInBytes, content.c_str());
                if (execCtx->callback)
                {
                    res = execCtx->callback(execCtx, state, execCtx->callbackUserData);
                }
                else
                {
                    // Using polled results
                    auto instance = (nvigi::asr::InferenceContext*)(execCtx->instance->data);
                    res = instance->pollCtx.triggerCallback(state);
                }
            }
        }
        else
        {
            //! Temporary outputs for the callback since host did not provide any
            auto text = nvigi::CpuData(content.length()+1, (const void*)content.c_str());
            auto data = nvigi::InferenceDataText(text);
            std::vector<nvigi::InferenceDataSlot> slots = { {kASRWhisperDataSlotTranscribedText, data} };
            nvigi::InferenceDataSlotArray outputs = { slots.size(), slots.data() };
            execCtx->outputs = &outputs;
            if (execCtx->callback)
            {
                res = execCtx->callback(execCtx, state, execCtx->callbackUserData);
            }
            else
            {
                // Using polled results
                auto instance = (nvigi::asr::InferenceContext*)(execCtx->instance->data);
                res = instance->pollCtx.triggerCallback(state);
            }
            //! Clear outputs since these are all local variables
            execCtx->outputs = {};
        }
        return res;
    };

    auto instance = (nvigi::asr::InferenceContext*)(execCtx->instance->data);

#if GGML_USE_CUBLAS
    // We apply the global scheduling mode to our streams at every evaluate() 
    // to reflect any changes the user made to the global mode between calls
    if (instance->cudaContext.usingCiG && ctx.icig->getVersion() >= 2)
    {
        nvigi::Result err = ctx.icig->cudaApplyGlobalGpuInferenceSchedulingMode(instance->cuda_streams.data(), instance->cuda_streams.size());
        if (err != kResultOk)
        {
            if (err == kResultDriverOutOfDate)
            {
                NVIGI_LOG_WARN_ONCE("Could not set relative priority of CUDA compute and graphics because the driver is out of date\n");
            }
            else
            {
                NVIGI_LOG_ERROR("Could not set relative priority of CUDA compute and graphics because a CUDA error occurred.\n");
            }
        }
    }
#endif

    auto& params = instance->params;
    auto strategy = WHISPER_SAMPLING_GREEDY;
    auto runtime = findStruct<ASRWhisperRuntimeParameters>(execCtx->runtimeParameters);
    auto streaming = findStruct<StreamingParameters>(execCtx->runtimeParameters);
    if (runtime)
    {
        if (runtime->sampling == ASRWhisperSamplingStrategy::eBeamSearch)
        {
            strategy = WHISPER_SAMPLING_BEAM_SEARCH;
            params.beam_size = runtime->beamSize;
        }
        params.best_of = runtime->bestOf;
        if (runtime->getVersion() >= 2)
        {
            params.prompt = runtime->prompt ? runtime->prompt : "";
            params.temperature = runtime->temperature;
            params.entropy_thold = runtime->entropyThold;
            params.logprob_thold = runtime->logprobThold;
            params.suppressBlank = runtime->suppressBlank;
            params.suppressNonSpeechTokens = runtime->suppressNonSpeechTokens;
            params.noSpeechThold = runtime->noSpeechThold;
            params.no_context = runtime->noContext;
        }
    }
    StreamSignal streamType = streaming && async ? streaming->signal : kStreamTypeNone;

    if (streamType == kStreamTypeNone || streamType == StreamSignal::eStreamSignalStart)
    {
        if (async && streamType == kStreamTypeNone)
        {
            NVIGI_LOG_ERROR("In async mode, stream signal cannot be 'kStreamTypeNone' and must be provided in StreamingParameters");
            return kResultInvalidState;
        }
        static const char* s_strategy[] = { "ASRSamplingStrategy::eGreedy","ASRSamplingStrategy::eBeamSearch" };
        NVIGI_LOG_VERBOSE("Processing audio on %u threads - strategy '%s' - best of %d - beam size %d", params.n_threads, s_strategy[strategy], params.best_of, params.beam_size);
        
        // Reset iteration counter when stream starts
        instance->n_iter = 0;        
    }

    // Convert to fp32
    std::vector<float> pcmf32;
    ai::InferenceDataAudioHelper audioHelper(audioInput);
    if (!audioHelper.getFloat(pcmf32))
    {
        NVIGI_LOG_ERROR("Failed to convert input audio to fp32");
        return kResultInvalidParameter;
    }

    // Write to our audio circular buffer
    instance->audio.setStreamType((int)streamType);
    instance->audio.write(pcmf32);
    
    auto runInference = [execCtx, triggerCallback, async, instance, strategy]()->nvigi::Result
    {
        auto& params = instance->params;

        // IMPORTANT: Read streamType fresh each iteration to pick up changes made by new chunks
        auto streamType = (StreamSignal)instance->audio.getStreamType();
        bool noMoreData = false;
        bool finalSegment = false;
        auto bufferSize = instance->audio.getDataSize();
       
        
        // If no data in buffer and still streaming, wait longer before checking again
        // This reduces spinning when chunks arrive slowly
        if (bufferSize == 0 && streamType != StreamSignal::eStreamSignalStop && streamType != kStreamTypeNone)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return kResultOk;
        }
        
        if (streamType != kStreamTypeNone)
        {
            bool hasData = instance->audio.read(instance->pcmf32_new, params.step_ms, noMoreData);
            if (hasData)
            {
                // Calculate how much old audio to keep (mimicking whisper.cpp stream.cpp logic)
                // This prevents audio from growing unbounded and improves accuracy
                const int n_samples_new = (int)instance->pcmf32_new.size();
                const int n_samples_keep = (params.keep_ms * WHISPER_SAMPLE_RATE) / 1000;
                const int n_samples_len = (params.length_ms * WHISPER_SAMPLE_RATE) / 1000;
                
                // Take up to params.length_ms audio from previous iteration
                const int n_samples_take = std::min(
                    (int)instance->pcmf32_old.size(), 
                    std::max(0, n_samples_keep + n_samples_len - n_samples_new)
                );
                
                
                // Build the audio buffer: old tail + new data
                instance->pcmf32.clear();
                instance->pcmf32.reserve(n_samples_take + n_samples_new);
                
                if (n_samples_take > 0) {
                    // Take only the tail of old audio (for context continuity)
                    instance->pcmf32.insert(instance->pcmf32.end(), 
                        instance->pcmf32_old.end() - n_samples_take, 
                        instance->pcmf32_old.end());
                }
                
                instance->pcmf32.insert(instance->pcmf32.end(), 
                    instance->pcmf32_new.begin(), 
                    instance->pcmf32_new.end());

                // Sliding window mode 
                // NO VAD! Just process every step_ms worth of audio
                // Results are interim (DataPartial) until we decide to finalize them
                
                // Check minimum audio length - wait until we have at least step_ms worth
                const int n_samples_step = (params.step_ms * WHISPER_SAMPLE_RATE) / 1000;
                if ((int)instance->pcmf32.size() < n_samples_step && !noMoreData)
                {
                    // Not enough audio yet and not final - accumulate more
                    instance->pcmf32_old.insert(instance->pcmf32_old.end(), instance->pcmf32_new.begin(), instance->pcmf32_new.end());
                    return kResultOk;
                }
                
                // We have enough audio to process
                // finalSegment is ONLY true when stream actually ends (noMoreData)
                finalSegment = noMoreData;                
            }
            else if (!noMoreData)
            {
                // No data available and not at end - this should rarely happen now
                // because we check buffer size upfront and sleep 50ms if empty
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                return kResultOk;
            }
            else
            {
                // hasData=false and noMoreData=true
                // Check if we have accumulated audio that needs processing
                if (!instance->pcmf32.empty() || !instance->pcmf32_old.empty())
                {
                    // We have accumulated audio from previous iteration(s) that hasn't been processed yet
                    // This can happen when chunks arrived but weren't enough for minimum processing
                    // Merge any remaining old data and process it as the final segment
                    if (!instance->pcmf32_old.empty() && instance->pcmf32.empty())
                    {
                        instance->pcmf32 = std::move(instance->pcmf32_old);
                    }
                    else if (!instance->pcmf32_old.empty())
                    {
                        instance->pcmf32.insert(instance->pcmf32.end(), 
                            instance->pcmf32_old.begin(), instance->pcmf32_old.end());
                        instance->pcmf32_old.clear();
                    }
                    
                    finalSegment = true;
                    // Continue to process below
                }
                else
                {
                    // No data to process, stream already finalized in previous iteration
                    
                    // Trigger final Done callback with empty text to signal completion
                    triggerCallback(execCtx, "", nvigi::kInferenceExecutionStateDone);
                    
                    // Clean up and stop the loop
                    instance->audio.clear();
                    instance->n_iter = 0;  // Reset iteration counter
                    instance->running.store(false);
                    return kResultOk;
                }
            }
        }
        else
        {
            // Read everything in non-async mode
            instance->audio.read(instance->pcmf32, kAvailableAudioBufferSize, noMoreData);
            finalSegment = true;
        }

        if (!instance->pcmf32.empty())
        {
            whisper_full_params wparams = whisper_full_default_params(strategy);

            wparams.single_segment = async;

            wparams.translate = params.translate;
            wparams.detect_language = params.detect_language;
            wparams.temperature = params.temperature;
            wparams.suppress_blank = params.suppressBlank;
            wparams.suppress_nst = params.suppressNonSpeechTokens;
            wparams.no_speech_thold = params.noSpeechThold;
            wparams.logprob_thold = params.logprob_thold;
            wparams.entropy_thold = params.entropy_thold;

            // let whisper.cpp handle the prompt and tokenization
            wparams.no_context = params.no_context;
            wparams.initial_prompt = params.prompt.c_str();

            wparams.debug_mode = false;
            wparams.print_realtime = false;
            wparams.print_progress = false;
            wparams.print_timestamps = false;
            wparams.print_special = false;

            wparams.translate = params.translate;
            wparams.language = params.language.c_str();
            wparams.detect_language = params.detect_language;
            wparams.n_threads = params.n_threads;
            wparams.n_max_text_ctx = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
            wparams.offset_ms = params.offset_t_ms;
            wparams.duration_ms = params.duration_ms;

            wparams.token_timestamps = params.max_len > 0;
            wparams.thold_pt = params.word_thold;
            wparams.max_len = params.max_len;
            wparams.split_on_word = params.split_on_word;

            wparams.greedy.best_of = params.best_of;
            wparams.beam_search.beam_size = params.beam_size;

            wparams.temperature_inc = params.no_fallback ? 0.0f : wparams.temperature_inc;
            wparams.entropy_thold = params.entropy_thold;
            wparams.logprob_thold = params.logprob_thold;

#if GGML_USE_CUBLAS
            nvigi::RuntimeContextScope scope(*instance);
#endif

#ifndef NVIGI_PRODUCTION
            whisper_reset_timings(instance->model);
#endif
            //! IMPORTANT: Avoid warnings from whisper.cpp when input sample is shorter than 1sec by padding with silence
            size_t safeSize = size_t(WHISPER_SAMPLE_RATE * 1.1f);
            if (instance->pcmf32.size() < safeSize)
            {
                instance->pcmf32.resize(safeSize, 0.0f);
            }

            if (whisper_full(instance->model, wparams, instance->pcmf32.data(), (int)instance->pcmf32.size()) != 0)
            {
                NVIGI_LOG_ERROR("Call to 'whisper_full' failed");
                triggerCallback(execCtx, "", nvigi::kInferenceExecutionStateInvalid);
                instance->audio.clear();
                instance->pcmf32.clear();
                instance->pcmf32_new.clear();
                instance->pcmf32_old.clear();
                return kResultInvalidState;
            }

            const int n_segments = whisper_full_n_segments(instance->model);
            
            // Increment iteration counter (matches whisper.cpp line 406)
            instance->n_iter++;
            
            // Calculate n_new_line (matches whisper.cpp line 139)
            // This determines how often to "finalize" a line (print \n instead of \r)
            const int n_new_line = std::max(1, params.length_ms / params.step_ms - 1);
            
            // Check if this iteration should finalize the line
            // In whisper.cpp: if ((n_iter % n_new_line) == 0) printf("\n");
            bool shouldFinalizeLine = (instance->n_iter % n_new_line) == 0;
            
            for (int i = 0; i < n_segments; ++i)
            {
                // Check for cancellation request
                if (instance->cancelled.load())
                {
                    NVIGI_LOG_VERBOSE("ASR segment processing cancelled by user request");
                    triggerCallback(execCtx, "", nvigi::kInferenceExecutionStateCancel);
                    return kResultOk;
                }
                
                const char* text = whisper_full_get_segment_text(instance->model, i);
                
                // State logic for sliding window mode (like whisper.cpp !use_vad, line 349-370, 408):
                // 
                // - DataPartial: Interim results (whisper.cpp prints with \r to overwrite)
                //   Used for all iterations EXCEPT when finalizing
                // 
                // - DataPending: Finalized phrase (whisper.cpp prints \n for new line)
                //   Used when (n_iter % n_new_line) == 0 OR when stream ends
                // 
                // - Done: Stream completely finished (noMoreData)
                //   Sent separately after all segments
                nvigi::InferenceExecutionState state;
                
                if (noMoreData)
                {
                    // Stream ended completely - finalize with DataPending
                    // (Done state sent separately in cleanup code)
                    state = nvigi::kInferenceExecutionStateDataPending;                    
                }
                else if (shouldFinalizeLine)
                {
                    // Every n_new_line iterations - finalize the current line
                    state = nvigi::kInferenceExecutionStateDataPending;
                }
                else
                {
                    // Interim result - will be overwritten next iteration
                    state = nvigi::kInferenceExecutionStateDataPartial;
                }
                
                triggerCallback(execCtx, text, state);
            }
            
            // This creates overlapping windows normally, then "jumps forward" on finalize            
            if (!noMoreData)
            {
                // STEP 1: Always save full processed audio (matches whisper.cpp line 291)
                instance->pcmf32_old = instance->pcmf32;
                
                // STEP 2: On finalize iterations, REPLACE with tail only (matches line 412)
                if (shouldFinalizeLine)
                {
                    const int n_samples_keep = (params.keep_ms * WHISPER_SAMPLE_RATE) / 1000;
                    if ((int)instance->pcmf32.size() > n_samples_keep)
                    {
                        instance->pcmf32_old.assign(
                            instance->pcmf32.end() - n_samples_keep, 
                            instance->pcmf32.end()
                        );                
                    }
                }
            }
        }
        else
        {
            NVIGI_LOG_VERBOSE("==> [INFERENCE] No audio to process (empty buffer)");
        }
        
        // Stop loop when stream ends
        if (noMoreData)
        {
            instance->audio.clear();
            instance->pcmf32_old.clear();
            instance->pcmf32.clear();
            instance->pcmf32_new.clear();
            instance->n_iter = 0;  // Reset iteration counter for next stream
            
            triggerCallback(execCtx, "", nvigi::kInferenceExecutionStateDone);

#ifndef NVIGI_PRODUCTION
            // Log timing information in non-production builds only
            auto timings = whisper_get_timings(instance->model);

            NVIGI_LOG_INFO("timings:sample %.2fms", timings->sample_ms);
            NVIGI_LOG_INFO("timings:encode %.2fms", timings->encode_ms);
            NVIGI_LOG_INFO("timings:decode %.2fms", timings->decode_ms);
            NVIGI_LOG_INFO("timings:prompt %.2fms", timings->prompt_ms);
            NVIGI_LOG_INFO("timings:batchd  %.2fms", timings->batchd_ms);

            delete timings;
#endif
            
            // Stop the async loop after processing the complete stream
            // The loop will restart when a new stream begins
            instance->running.store(false);            
        }
        return kResultOk;
    };

    nvigi::Result result = kResultOk;
    if (async)
    {
        // Non-blocking, schedule work
        if (!instance->job.valid())
        {
            instance->running.store(true);
            instance->cancelled.store(false);  // Reset cancellation flag for new evaluation
            instance->job = std::async(std::launch::async, [instance, triggerCallback, runInference]()->Result
            {
                auto res = kResultOk;
                while (instance->running.load() && !instance->cancelled.load() && res == kResultOk)
                {
                    res = runInference();
                }
                // If cancelled, exit cleanly
                if (instance->cancelled.load())
                {
                    NVIGI_LOG_VERBOSE("ASR evaluation cancelled by user request");
                    return kResultOk;
                }
                return res;
            });
        }
        else
        {
            if (instance->job.wait_for(std::chrono::microseconds(10)) == std::future_status::ready)
            {
                // Our task finished, make sure we report any errors
                if (NVIGI_FAILED(result, instance->job.get()))
                {
                    NVIGI_LOG_ERROR("whisper runtime returned early with result %u", result);
                    return result;
                }
                // No errors, start new job
                instance->running.store(true);
                instance->cancelled.store(false);  // Reset cancellation flag for new evaluation
                instance->job = std::async(std::launch::async, [instance, triggerCallback, runInference]()->Result
                {
                    auto res = kResultOk;
                    while (instance->running.load() && !instance->cancelled.load() && res == kResultOk)
                    {
                        res = runInference();
                    }
                    // If cancelled, exit cleanly
                    if (instance->cancelled.load())
                    {
                        NVIGI_LOG_VERBOSE("ASR evaluation cancelled by user request");
                        return kResultOk;
                    }
                    return res;
                });
            }
            else
            {
                // Previous job is still running - this shouldn't happen in normal operation
                // This indicates the previous stream didn't send the final chunk with is_last=true
                // or there's a thread leak. We need to handle this gracefully.
                // Check if this is a new stream starting (eStreamSignalStart)
                assert(streamType != StreamSignal::eStreamSignalStart);
                if (streamType == StreamSignal::eStreamSignalStart)
                {
                    NVIGI_LOG_ERROR("Previous job NOT READY (still running) but new stream started, signal='StreamSignal::eStreamSignalStart', running=%s, cancelled=%s", 
                        instance->running.load() ? "true" : "false",
                        instance->cancelled.load() ? "true" : "false");
                    return kResultInvalidState;
                }
            }
        }
    }
    else
    {
        // Blocking, run and return

        // First make sure any async jobs are done
        if (instance->job.valid())
        {
            NVIGI_LOG_WARN("'evaluateAsync' task not finished, interrupting before running blocking 'evaluate' ...");
            instance->running.store(false);
            instance->job.get();
        }
        result = runInference();
    }

#if GGML_USE_CUBLAS || (defined(GGML_USE_D3D12) && PROFILE_D3D)
    nvtxRangePop();
#endif

    return result;
}

// Add new functions for polling support:
Result asrGetResults(InferenceExecutionContext* execCtx, bool wait, InferenceExecutionState* state) {
    if (!execCtx || !execCtx->instance) {
        return kResultInvalidParameter;
    }

    auto instance = static_cast<asr::InferenceContext*>(execCtx->instance->data);
    return instance->pollCtx.getResults(wait, state);
}

Result asrReleaseResults(InferenceExecutionContext* execCtx, InferenceExecutionState state) {
    if (!execCtx || !execCtx->instance) {
        return kResultInvalidParameter;
    }

    auto instance = static_cast<asr::InferenceContext*>(execCtx->instance->data);
    return instance->pollCtx.releaseResults(state);
}

Result asrCancelAsyncEvaluation(InferenceExecutionContext* execCtx) {
    if (!execCtx || !execCtx->instance) {
        return kResultInvalidParameter;
    }

    auto instance = static_cast<asr::InferenceContext*>(execCtx->instance->data);
    
    // Check if async job is actually running
    if (!instance->job.valid())
    {
        NVIGI_LOG_WARN("cancelAsyncEvaluation called but no async evaluation is running");
        return kResultNoImplementation;
    }
    
    // Set cancellation flag to interrupt the evaluation loop as early as possible
    instance->cancelled.store(true);
    
    if (NVIGI_FAILED(result, flushAndTerminate(instance)))
    {
        return result;
    }
    return kResultOk;
}

//! Exception handling wrappers
//! 
//! Note that we export these via our interface
//! 
namespace asr
{
nvigi::Result createInstance(const nvigi::NVIGIParameter* params, nvigi::InferenceInstance** instance)
{
    NVIGI_CATCH_EXCEPTION(whisperCreateInstance(params,instance));
}
nvigi::Result destroyInstance(const nvigi::InferenceInstance* instance)
{
    NVIGI_CATCH_EXCEPTION(whisperDestroyInstance(instance));
}
nvigi::Result getCapsAndRequirements(nvigi::NVIGIParameter** modelInfo, const nvigi::NVIGIParameter* params)
{
    NVIGI_CATCH_EXCEPTION(whisperGetCapsAndRequirements(modelInfo, params));
}
nvigi::Result evaluate(nvigi::InferenceExecutionContext* execCtx)
{
    NVIGI_CATCH_EXCEPTION(whisperEvaluate(execCtx, false));
}
nvigi::Result evaluateAsync(nvigi::InferenceExecutionContext* execCtx)
{
    NVIGI_CATCH_EXCEPTION(whisperEvaluate(execCtx, true));
}
nvigi::Result getResults(nvigi::InferenceExecutionContext* execCtx, bool wait, nvigi::InferenceExecutionState* state)
{
    NVIGI_CATCH_EXCEPTION(asrGetResults(execCtx, wait, state));
}
nvigi::Result releaseResults(nvigi::InferenceExecutionContext* execCtx, nvigi::InferenceExecutionState state)
{
    NVIGI_CATCH_EXCEPTION(asrReleaseResults(execCtx, state));
}
nvigi::Result cancelAsyncEvaluation(nvigi::InferenceExecutionContext* execCtx)
{
    NVIGI_CATCH_EXCEPTION(asrCancelAsyncEvaluation(execCtx));
}
} // asr

//! Main entry point - get information about our plugin
//! 
Result nvigiPluginGetInfo(nvigi::framework::IFramework* framework, nvigi::plugin::PluginInfo** _info)
{
    auto& ctx = (*asr::getContext());

    if (!plugin::internalPluginSetup(framework)) return kResultInvalidState;

    // Internal API, we know that incoming pointer is always valid
    auto& info = plugin::getContext()->info;
    *_info = &info;

    info.id = asr::getFeatureId(nullptr);
    info.description = "ggml backend implementation for the 'asr' inference";
    info.author = "NVIDIA";
    info.build = GIT_BRANCH_AND_LAST_COMMIT;
    info.interfaces = { plugin::getInterfaceInfo<IAutoSpeechRecognition>()};

    // Always the same OS requirements
    info.minOS = { NVIGI_DEF_MIN_OS_MAJOR, NVIGI_DEF_MIN_OS_MINOR, NVIGI_DEF_MIN_OS_BUILD };

    // Default to no GPU requirements for now
    info.minGPUArch = {};
    info.minDriver = {};

    if (!framework::getInterface(framework, nvigi::core::framework::kId, &ctx.isystem))
    {
        NVIGI_LOG_ERROR("Missing interface from 'nvigi.plugin.hwi.cuda'");
        return kResultInvalidState;
    }

    // Check if we have an NV adapter available and if so, set the minimum GPU architecture and driver version
    bool hasNVAdapter = false;
    const nvigi::system::SystemCaps* caps = ctx.isystem->getSystemCaps();
    for (uint32_t i = 0; i < caps->adapterCount; i++)
    {
        const nvigi::system::Adapter* adapter = caps->adapters[i];
        if (adapter->vendor == VendorId::eNVDA)
        {
            hasNVAdapter = true;
            break;
        }
        // Later, we could add min driver and arch for AMD or other vendors
    }

#ifdef GGML_USE_CUBLAS
    info.minDriver = { NVIGI_CUDA_MIN_DRIVER_MAJOR, NVIGI_CUDA_MIN_DRIVER_MINOR, NVIGI_CUDA_MIN_DRIVER_BUILD };
    info.minGPUArch = { NVIGI_CUDA_MIN_GPU_ARCH };
    info.requiredVendor = VendorId::eNVDA;

    nvigi::system::ISystem* isystem{};
    if (!framework::getInterface(framework, nvigi::core::framework::kId, &isystem))
    {
        NVIGI_LOG_ERROR("Missing interface from 'nvigi.plugin.hwi.cuda'");
        return kResultInvalidState;
    }

    if (hasNVAdapter && caps->driverVersion.major < 580)
    {
        NVIGI_LOG_WARN_ONCE("CUDA backend recommends driver version 580 or higher, current version is %d.%d.%d - see documentation for details", caps->driverVersion.major, caps->driverVersion.minor, caps->driverVersion.build);
	}
#elif defined(GGML_USE_D3D12)
    // Default to any adapter and no driver restrictions
    info.requiredVendor = VendorId::eAny;

    if (hasNVAdapter)
    {
        info.minGPUArch = { NVIGI_CUDA_MIN_GPU_ARCH };
        info.minDriver = { NVIGI_D3D12_MIN_DRIVER_MAJOR, NVIGI_D3D12_MIN_DRIVER_MINOR, NVIGI_D3D12_MIN_DRIVER_BUILD };
        info.requiredVendor = VendorId::eNVDA;
    }
    // Later, we could add min driver and arch for AMD or other vendors
#elif defined(GGML_USE_VULKAN)
    // Requires SOME GPU with Vulkan support, no specific vendor or driver version
    info.requiredVendor = VendorId::eAny;
#else
    // No requirements for CPU backend
    info.requiredVendor = nvigi::VendorId::eNone;
#endif

    // Must release, as we may not have the chance to release it later if we are not registered
    framework::releaseInterface(plugin::getContext()->framework, nvigi::core::framework::kId, ctx.isystem);
    ctx.isystem = nullptr;

    return kResultOk;
}

//! Main entry point - starting our plugin
//! 
//! IMPORTANT: Plugins are started based on their priority.
//!
Result nvigiPluginRegister(framework::IFramework* framework)
{
    if (!plugin::internalPluginSetup(framework)) return kResultInvalidState;

    auto& ctx = (*asr::getContext());

    ctx.feature = asr::getFeatureId(nullptr);

#if GGML_USE_CUBLAS
    if (!framework::getInterface(plugin::getContext()->framework, plugin::hwi::cuda::kId, &ctx.icig))
    {
        NVIGI_LOG_ERROR("Missing interface from 'nvigi.plugin.hwi.cuda'");
        return kResultMissingInterface;
    }
#elif defined(GGML_USE_D3D12)
    if (!framework::getInterface(framework, nvigi::core::framework::kId, &ctx.isystem))
    {
        NVIGI_LOG_ERROR("Missing core interface 'nvigi::system::ISystem'");
        return kResultMissingInterface;
    }
#endif

    ctx.api.createInstance = asr::createInstance;
    ctx.api.destroyInstance = asr::destroyInstance;
    ctx.api.getCapsAndRequirements = asr::getCapsAndRequirements;

    framework->addInterface(ctx.feature, &ctx.api, 0);

    // Add polled interface
    ctx.polledApi.getResults = asr::getResults;
    ctx.polledApi.releaseResults = asr::releaseResults;
    framework->addInterface(ctx.feature, &ctx.polledApi, 0);

    whisper_log_set(nvigi::asr::whisperLogCallback, nullptr);

    return kResultOk;
}

//! Main exit point - shutting down our plugin
//! 
Result nvigiPluginDeregister()
{
    auto& ctx = (*asr::getContext());

    ai::freeCommonCapsAndRequirements(ctx.capsData);

#if GGML_USE_CUBLAS
    if (ctx.icig && plugin::getContext() && plugin::getContext()->framework)
    {
        framework::releaseInterface(plugin::getContext()->framework, plugin::hwi::cuda::kId, ctx.icig);
        ctx.icig = nullptr;
    }
#elif defined(GGML_USE_D3D12)
    if (ctx.iscg && plugin::getContext() && plugin::getContext()->framework)
    {
        framework::releaseInterface(plugin::getContext()->framework, plugin::hwi::d3d12::kId, ctx.iscg);
        ctx.iscg = nullptr;
    }
    if (ctx.isystem && plugin::getContext() && plugin::getContext()->framework)
    {
        framework::releaseInterface(plugin::getContext()->framework, nvigi::core::framework::kId, ctx.isystem);
        ctx.isystem = nullptr;
    }
#endif

    whisper_log_set(nullptr, nullptr);

#if defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
    // llama.cpp does not provide a way to free backends explicitly hence the hack
    ggml_vk_backend_free();
#endif

    // cpu 
    // Reference:  https://ofekshilon.com/2017/11/03/on-omp_wait_policy/
    // WhisperCPP uses OpenMP in CPU mode.  On shutdown, OpenMP threads can spin wait for 200ms-2000ms (varying reports)
    // if you unload the dll before these threads have exited, you get a crash in PartialBarrierN::Block from ggml.c threadpool (where it's coming from won't be obvious)
    // Potential alternative solution: 
    // Prior to loading the dll, you can set environment variables to change how OpenMP behaves.
        //#ifdef _WIN32
        //	SetEnvironmentVariable(L"OMP_WAIT_POLICY", L"passive");
        //#else
        //	setenv("OMP_WAIT_POLICY", "passive", 1);  // The '1' means overwrite if it exists
        //#endif
    // but setting an environment variable might be a security concern.  Also, passive means the OpenMP threads will sleep faster, and you might get more latency
    // as they have to wake up more frequently for work.

    // alternatively, next time we upgrade WhisperCPP, we could build CPU WIHTOUT OpenMP, but it will be slower.

    // So this seems to be the least of evils - wait for the OpenMP threads to spinwait stop and then proceed with shutdown.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    return kResultOk;
}

//! The only exported function - gateway to all functionality
NVIGI_EXPORT void* nvigiPluginGetFunction(const char* functionName)
{
    //! Core API
    NVIGI_EXPORT_FUNCTION(nvigiPluginGetInfo);
    NVIGI_EXPORT_FUNCTION(nvigiPluginRegister);
    NVIGI_EXPORT_FUNCTION(nvigiPluginDeregister);

    return nullptr;
}

}