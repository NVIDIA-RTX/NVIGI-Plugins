// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <future>

#include "source/core/nvigi.api/nvigi.h"
#include "source/core/nvigi.api/internal.h"
#include "source/core/nvigi.log/log.h"
#include "source/core/nvigi.exception/exception.h"
#include "source/core/nvigi.exception/exception.h"
#include "source/core/nvigi.plugin/plugin.h"
#include "source/core/nvigi.file/file.h"
#include "source/core/nvigi.extra/extra.h"
#include "source/plugins/nvigi.gpt/ggml/versions.h"
#include "source/utils/nvigi.ai/ai.h"
#include "source/utils/nvigi.hwi/cuda/runtime_context_scope.h"
#include "source/utils/nvigi.poll/poll.h"
#include "source/plugins/nvigi.gpt/nvigi_gpt.h"
#include "_artifacts/gitVersion.h"

#include "source/plugins/nvigi.gpt/ggml/gpt.h"

#include "log.h"

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
    // this should match the PFun_commandListAction defined in LlamaCPP - redefining as it's at a depth/scope in LlamaCPP not exposed to IGI
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

namespace nvigi
{
namespace gpt
{

uint32_t replace_all(std::string& str, const std::string& search_str, const std::string& replace_str);

static void llamaLogCallback(ggml_log_level level, const char* text, void* user_data)
{
    //! Special case, llama prints progress bars with a stream of consecutive "." so we want to ignore that
    if (strcmp(text, ".") == 0) return;

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
        NVIGI_LOG("llama", LogType::eWarn, nvigi::log::YELLOW, "%s", msg.c_str());
    }
    else if (level == GGML_LOG_LEVEL_ERROR)
    {
        NVIGI_LOG("llama", LogType::eError, nvigi::log::RED, "%s", msg.c_str());
    }
    else
    {
        NVIGI_LOG("llama", LogType::eInfo, nvigi::log::WHITE, "%s", msg.c_str());
    }
}

// BEGIN VLM Addition
#if COMPILE_VLM
static bool replace_first(std::string& str, const std::string& search_str, const std::string& replace_str)
{
    size_t start_pos = str.find(search_str);
    if (start_pos != std::string::npos)
    {
        str.replace(start_pos, search_str.length(), replace_str);
        return true;
    }

    return false;
}
#endif // COMPILE_VLM
// End VLM Addition

uint32_t replace_all(std::string& str, const std::string& search_str, const std::string& replace_str)
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

ggml_type to_ggml_cache_type(int32_t type)
{
    if (type < 0 || type >= GGML_TYPE_COUNT)
    {
        NVIGI_LOG_WARN("Invalid cache type [%d] provided, using default GGML_TYPE_F16", type);
        return GGML_TYPE_F16; // default
    }
    return (ggml_type)type;
}

struct InferenceContext
{
    InferenceContext(const nvigi::NVIGIParameter* params) : cudaContext(params) {}

    poll::PollContext<nvigi::InferenceExecutionState> pollCtx;
    common_params params{};
    int32_t instanceBatchSize = 2048;

    InferenceContextSync sync{};

    json modelInfo;

#ifndef NVIGI_GPT_GFN_NVCF
    common_init_result llamaContext{};
    mtmd_context* clipContext{};

    // IGI manages lora adapters itself, similiarly to the way LlamaCPP does.
    // This added complexity allows us to switch scales of loras per evaluate call
    // which LlamaCPP typically does not provide a mechanism for.
    struct igi_adapter_lora_info
    {
        std::string loraPath;
        llama_adapter_lora_ptr ptr;
    };
    std::unordered_map<std::string, igi_adapter_lora_info> lora_adapters;
#endif

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
    std::atomic<Result> state;
};

PluginID getFeatureId(InferenceInstanceData* data)
{
#ifdef GGML_USE_CUBLAS
    return plugin::gpt::ggml::cuda::kId;
#elif defined GGML_USE_VULKAN
    return plugin::gpt::ggml::vulkan::kId;
#elif defined(GGML_USE_D3D12)
    return plugin::gpt::ggml::d3d12::kId;
#else
    return plugin::gpt::ggml::cpu::kId;
#endif
}

constexpr const char* kChatTemplate = "chat_template";

void igiProcessChatTemplate(const GPTRuntimeParameters* runtime, nvigi::gpt::InferenceContext* instance, const std::string& system, const std::string& user, const std::string& assistant)
{
    //! High level logic:
    //! 
    //! * No runtime or runtime version <3, we will setup template from JSON, if not present resort to defaults in llama.cpp
    //! * Runtime >= 3, if pretemplatized, we will not process template, just use system and user as is
    //! * Runtime >= 3 and not pretemplatized, we will process template if we have one in JSON, otherwise use internal llama.cpp templates
    //! * Runtime >= 5, if chatTemplate is provided, we will use it as is, otherwise we will process template if we have one in JSON, otherwise use internal llama.cpp templates
    //! * Runtime >= 5, useJinja will be used to determine if jinja formatting is used or not
    
    // Set defaults first
    instance->params.prompt = user;
    instance->sync.assistant = assistant;
    instance->params.enable_chat_template = false; 
    instance->params.use_jinja = false;
    instance->params.conversation_mode = COMMON_CONVERSATION_MODE_AUTO;
    if (runtime && !runtime->interactive && system.empty())
    {
        // Running in instruct mode but host did not provide system prompt, so we will provide a default one to prevent llama.cpp from logging warnings
        instance->params.system_prompt = "You are a helpful assistant. Please answer user's questions or queries as best as you can and remain professional and friendly.";
    }
    else
    {
        instance->params.system_prompt = system;
    }

    // Handle legacy case, if user provided pretemplatized prompt, we will not attempt to process it
    bool shouldProcessChatTemplate = !runtime || (runtime->getVersion() >= 3 && !runtime->promptPretemplatized);
    if (runtime && runtime->getVersion() >= 5)
    {
        // Newest version, so we can use the new chatTemplate and useJinja parameters
        instance->params.use_jinja = runtime->useJinja;
        if (runtime->chatTemplate)
        {
            // user provided a custom template, so we will use it
            instance->params.chat_template = runtime->chatTemplate;
            instance->params.enable_chat_template = true;
            // we are not going to process the chat template ourselves
            shouldProcessChatTemplate = false;
        }
    }
    if (shouldProcessChatTemplate)
    {
        // user wants us to process it - so we need to determine if we have IGI of LlamaCPP do it.
        bool igiShouldProcessChatTemplate = instance->modelInfo.contains(kChatTemplate);
        if (igiShouldProcessChatTemplate)
        {
            // Template in JSON is using jinja format, note that for readability we break it into multiple lines in JSON, but we need to concatenate it into a single string for llama.cpp
            std::string chatTemplate;
            for (auto& s : instance->modelInfo[kChatTemplate])
            {
                chatTemplate += s;
            }
            instance->params.chat_template = chatTemplate;
            instance->params.use_jinja = true;
            // If there is no template in JSON llama.cpp will use internal templates, regardless chat template must be enabled
            instance->params.enable_chat_template = true;
        }
        else if (instance->modelInfo.contains(ai::kPromptTemplate))
        {
            // legacy template, not using jinja, system prompt always empty
            instance->params.system_prompt.clear();
            instance->params.prompt = ai::generatePrompt(instance->modelInfo, system, user, assistant);
        }
        else
        {
            // If there is no template in JSON llama.cpp will use internal templates, regardless chat template must be enabled
            instance->params.enable_chat_template = true;
        }
    }
}

const nvigi::InferenceDataDescriptorArray* getInputSignature(InferenceInstanceData* data)
{
    static std::vector<InferenceDataDescriptor> slots = 
    { 
        {nvigi::kGPTDataSlotSystem,InferenceDataText::s_type, true }, 
        {nvigi::kGPTDataSlotUser,InferenceDataText::s_type, false },
        {nvigi::kGPTDataSlotAssistant,InferenceDataText::s_type, true},
    };
    static InferenceDataDescriptorArray s_desc = { slots.size(), slots.data() };
    return &s_desc;
}

const nvigi::InferenceDataDescriptorArray* getOutputSignature(InferenceInstanceData* data)
{
    static std::vector<InferenceDataDescriptor> slots = { {nvigi::kGPTDataSlotResponse,InferenceDataText::s_type, false } };
    static InferenceDataDescriptorArray s_desc = { slots.size(), slots.data() };
    return &s_desc;
}

struct GPTContext
{
    NVIGI_PLUGIN_CONTEXT_CREATE_DESTROY(GPTContext);

    void onCreateContext() {};
    void onDestroyContext() {};

    std::atomic<bool> initialized = false;

    nvigi::PluginID feature{};

    IGeneralPurposeTransformer api{};
    IPolledInferenceInterface polledApi{};

    // Caps and requirements
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
NVIGI_PLUGIN_DEFINE("nvigi.plugin.gpt", Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH), Version(API_MAJOR, API_MINOR, API_PATCH), gpt, GPTContext)

nvigi::Result ggmlEvaluate(nvigi::InferenceExecutionContext* execCtx);

nvigi::Result flushAndTerminate(nvigi::gpt::InferenceContext* instance, gpt::InferenceContextSync& sync)
{
    auto result = nvigi::kResultOk;
    if (sync.job.valid())
    {
        // Make sure to signal cancellation so token generation finishes early
        sync.cancelled.store(true);

        if (sync.runningChat.load())
        {
            //NVIGI_LOG_VERBOSE("Flushing running chat ...");
            {
                std::scoped_lock lock(sync.mtx);
                sync.runningChat.store(false);
                sync.newInput.store(true); // set to true to ensure that any woken up old threads are able to continue properly.
            }
            sync.cvInput.notify_all();
        }
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
            auto futureResult = sync.job.wait_for(std::chrono::milliseconds(10));
            if (futureResult == std::future_status::ready)
            {
                sync.newInput.store(false); // set to false to ensure state of generate is consistent when it is first entered.
                // Job completed, get the result
                if (NVIGI_FAILED(result, sync.job.get()))
                {
                    return result;
                }
                return kResultOk;
            }
        }

        // if we timeout, we still need newInput to be false for consistency.
        sync.newInput.store(false);

        // Timed out
        NVIGI_LOG_WARN("Async job timed out.");
        return kResultTimedOut;
    }
    return result;
}

nvigi::Result ggmlDestroyInstance(const nvigi::InferenceInstance* instance)
{
    if (instance)
    {
        auto gptInstance = static_cast<nvigi::gpt::InferenceContext*>(instance->data);
        if (NVIGI_FAILED(result, flushAndTerminate(gptInstance,gptInstance->sync)))
        {
            return result;
        }

        {
#ifdef GGML_USE_CUBLAS
            nvigi::RuntimeContextScope scope(*gptInstance);
#endif
            // Begin VLM Addition
#if COMPILE_VLM
            if (gptInstance->clipContext != nullptr)
            {
                mtmd_free( gptInstance->clipContext );
                gptInstance->clipContext = nullptr;
            }
#endif // COMPILE_VLM
            // End VLM Addition

            gptInstance->llamaContext.model.reset();
            gptInstance->llamaContext.context.reset();
            gptInstance->llamaContext.lora.clear();
        }

        delete gptInstance->sync.tgLimiter;
        delete gptInstance;
        delete instance;
    }
    return nvigi::kResultOk;
}

nvigi::Result ggmlCreateInstance(const nvigi::NVIGIParameter* _params, nvigi::InferenceInstance** _instance)
{
    auto common = findStruct<CommonCreationParameters>(_params);
    auto creationParams = findStruct<GPTCreationParameters>(_params);
    if (!common || !creationParams) return nvigi::kResultInvalidParameter;
    
    if (!_instance || !common->utf8PathToModels || !common->modelGUID) return nvigi::kResultInvalidParameter;

    if (!ai::isGuid(common->modelGUID))
    {
        NVIGI_LOG_ERROR("Provided model GUID '%s' is invalid", common->modelGUID);
        return kResultInvalidParameter;
    }

#ifdef GGML_USE_CUBLAS
    auto cudaParams = findStruct<CudaParameters>(_params);
    if (cudaParams && cudaParams->getVersion() >= kStructVersion2 )
    {
        if (cudaParams->cudaMallocReportCallback)
            nvigi::gpt::setCudaMallocReportCallback(cudaParams->cudaMallocReportCallback, cudaParams->cudaMallocReportUserContext);
        if (cudaParams->cudaFreeReportCallback)
            nvigi::gpt::setCudaFreeReportCallback(cudaParams->cudaFreeReportCallback, cudaParams->cudaFreeReportUserContext);
        if (cudaParams->cudaMallocCallback)
            nvigi::gpt::setCudaMallocCallback(cudaParams->cudaMallocCallback, cudaParams->cudaMallocUserContext);
        if (cudaParams->cudaFreeCallback)
            nvigi::gpt::setCudaFreeCallback(cudaParams->cudaFreeCallback, cudaParams->cudaFreeUserContext);
    }
#endif

    *_instance = nullptr;

    auto instanceData = new nvigi::gpt::InferenceContext(_params);
    if (!instanceData->cudaContext.constructorSucceeded) return kResultInvalidState;

    auto& ctx = (*gpt::getContext());

    using namespace nvigi::gpt;

#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
#ifndef NVIGI_PRODUCTION
    size_t currentUsageMB{};
    extra::ScopedTasks vram(
        [&currentUsageMB]() {
        system::VRAMUsage* usage;
        system::getInterface()->getVRAMStats(0, &usage);
        currentUsageMB = usage->currentUsageMB;
        },
        [&currentUsageMB, instanceData, common]() {
            system::VRAMUsage* usage;
            system::getInterface()->getVRAMStats(0, &usage);
            currentUsageMB = usage->currentUsageMB - currentUsageMB;
            auto n_layers = llama_model_n_layer(instanceData->llamaContext.model.get());
            NVIGI_LOG_INFO("New instance vram : %lluMB [budget : %lluMB, total : %lluMB] - n_layers : %d", currentUsageMB, common->vramBudgetMB, usage->budgetMB, n_layers);
        }
    );
#endif     
#else
    instanceData->params.n_gpu_layers = 0;
#endif

#if defined(GGML_USE_CUBLAS)
    if (common->numThreads > 1)
    {
        NVIGI_LOG_WARN("For optimal performance when using CUDA only one CPU thread is used");
    }
    instanceData->params.cpuparams.n_threads = 1;
#else
    instanceData->params.cpuparams.n_threads = common->numThreads;
#endif

    // Optional
    auto sampler = findStruct<GPTSamplerParameters>(_params);
    if (sampler)
    {
        instanceData->params.sampling.n_prev = sampler->numPrev;
        instanceData->params.sampling.n_probs = sampler->numProbs;
        instanceData->params.sampling.min_keep = sampler->minKeep;
        instanceData->params.sampling.top_k = sampler->topK;
        instanceData->params.sampling.min_p = sampler->minP;
        instanceData->params.sampling.xtc_probability = sampler->xtcProbability;
        instanceData->params.sampling.xtc_threshold = sampler->xtcThreshold;
        //instanceData->params.sampling.tfs_z = sampler->tfsZ;
        instanceData->params.sampling.typ_p = sampler->typP;
        instanceData->params.sampling.dynatemp_range = sampler->dynatempRange;
        instanceData->params.sampling.dynatemp_exponent = sampler->dynatempExponent;
        instanceData->params.sampling.penalty_last_n = sampler->penaltyLastN;
        instanceData->params.sampling.penalty_repeat = sampler->penaltyRepeat;
        instanceData->params.sampling.penalty_freq = sampler->penaltyFreq;
        instanceData->params.sampling.penalty_present = sampler->penaltyPresent;
        instanceData->params.sampling.mirostat = sampler->mirostat;
        instanceData->params.sampling.mirostat_tau = sampler->mirostatTAU;
        instanceData->params.sampling.mirostat_eta = sampler->mirostatETA;
        //instanceData->params.sampling.penalize_nl = sampler->penalizeNewLine;
        instanceData->params.sampling.ignore_eos = sampler->ignoreEOS;
    }

    // Optional
    auto runtime = findStruct<GPTRuntimeParameters>(_params);
    if (runtime)
    {
        instanceData->params.sampling.seed = runtime->seed;
        instanceData->params.n_predict = runtime->tokensToPredict;
        instanceData->params.n_keep = runtime->tokensToKeep;
        instanceData->params.n_batch = runtime->batchSize;
        instanceData->params.n_chunks = runtime->numChunks;
        instanceData->params.n_parallel = runtime->numParallel;
        instanceData->params.n_sequences = runtime->numSequences;
        instanceData->params.interactive = runtime->interactive;
        instanceData->params.sampling.temp = runtime->temperature;
        instanceData->params.sampling.top_p = runtime->topP;
    }

    {
        instanceData->params.sampling.seed = creationParams->seed;
        instanceData->params.n_ctx = creationParams->contextSize;
        instanceData->params.n_predict = creationParams->maxNumTokensToPredict;
        if (creationParams->getVersion() >= 2)
        {
            instanceData->params.n_batch = creationParams->batchSize;
            instanceData->params.n_ubatch = creationParams->physicalBatchSize;
            instanceData->params.flash_attn_type = creationParams->flashAttention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED; // BROWLETT - Default is LLAMA_FLASH_ATTN_TYPE_AUTO, but not sure how to set that with current setup
            instanceData->params.cache_type_k = to_ggml_cache_type(creationParams->cacheTypeK);
            instanceData->params.cache_type_v = to_ggml_cache_type(creationParams->cacheTypeV);
        }
        // Store for later so we can make sure it is not increased
        instanceData->instanceBatchSize = instanceData->params.n_batch;

        instanceData->modelInfo.clear();
        if (!ai::findModels(common, { "gguf" }, instanceData->modelInfo))
        {
            return kResultInvalidParameter;
        }
        
        size_t neededVRAM = 0;
        int n_layers = 0;

        try
        {
            // Trim down to our GUID for this instance
            instanceData->modelInfo = instanceData->modelInfo[common->modelGUID];

          /*  std::string tmp = instanceData->modelInfo.dump(1, ' ', false, json::error_handler_t::replace);
            NVIGI_LOG_VERBOSE("%s", tmp.c_str());*/

            n_layers = instanceData->modelInfo["n_layers"];
#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
            // Allow offloading to CPU as needed
            neededVRAM = instanceData->modelInfo["vram"];            
            auto gpuRatio = std::min(1.0, static_cast<double>(common->vramBudgetMB) / static_cast<double>(neededVRAM));
            instanceData->params.n_gpu_layers = static_cast<int>(std::floor(gpuRatio * n_layers));
#endif
            std::vector<std::string> files = instanceData->modelInfo["gguf"];
            if (files.empty())
            {
                NVIGI_LOG_ERROR("Failed to find model in the expected directory '%s'", common->utf8PathToModels);
                return kResultInvalidParameter;
            }
            instanceData->params.model.path = files[0];
            // Begin VLM Addition
#if COMPILE_VLM
            if (instanceData->modelInfo.contains("weights") &&
                instanceData->modelInfo.contains("mmproj_weights"))
            {
                std::string weights_suffix = instanceData->modelInfo["weights"];
                std::string mmproj_model_suffix = instanceData->modelInfo["mmproj_weights"];

                auto ends_with = [](const std::string& str, const std::string suffix) {
                    return suffix.size() <= str.size() &&
                        str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
                    };

                for (auto const& model_file : files)
                {
                    if (ends_with(model_file, weights_suffix))
                        instanceData->params.model.path = model_file;
                    if (ends_with(model_file, mmproj_model_suffix))
                        instanceData->params.mmproj.path = model_file;
                }
            }
#endif // COMPILE_VLM
            // End VLM Addition
        }
        catch (std::exception& e)
        {
            NVIGI_LOG_ERROR("%s", e.what());
            return kResultInvalidState;
        }

        NVIGI_LOG_INFO("Loading model '%s'", instanceData->params.model.path.c_str());
#if GGML_USE_CUBLAS
        auto platform = "ggml.cuda";
#elif defined GGML_USE_VULKAN
        auto platform = "ggml.vulkan";
#elif defined(GGML_USE_D3D12)
        auto platform = "ggml.d3d12";
#else
        auto platform = "ggml.cpu";
#endif
        // Log everything together since llama.cpp in verbose mode can clog the log
        NVIGI_LOG_VERBOSE("# backend '%s'", platform);
        NVIGI_LOG_VERBOSE("# threads %d", instanceData->params.cpuparams.n_threads);
        NVIGI_LOG_VERBOSE("# GPU layers %d[%d]", instanceData->params.n_gpu_layers, n_layers);
        NVIGI_LOG_VERBOSE("# VRAM budget : %lluMB - needed %lluMB", common->vramBudgetMB, neededVRAM);
        NVIGI_LOG_VERBOSE("# context size %d", instanceData->params.n_ctx);
        NVIGI_LOG_VERBOSE("# predicting %d", instanceData->params.n_predict);
        NVIGI_LOG_VERBOSE("# batch %d", instanceData->params.n_batch);

#ifdef GGML_USE_VULKAN
        auto vkParams = findStruct<VulkanParameters>(_params);
        if (vkParams)
        {
            // Unlike D3D12 no validation possible with VK, we just assume that devs are passing correct bits
            ggml_set_vk_params(vkParams->physicalDevice, vkParams->device, vkParams->queue, vkParams->queueCompute, vkParams->queueTransfer,
                vkParams->allocateMemoryCallback, vkParams->freeMemoryCallback,
                vkParams->allocateMemoryCallbackUserContext, vkParams->freeMemoryCallbackUserContext);
        }
#elif defined(GGML_USE_D3D12)
        auto d3d12Params = findStruct<D3D12Parameters>(_params);
        if(NVIGI_FAILED(res, d3d12::validateParameters(d3d12Params)))
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
            IHWID3D12* iscg = gpt::getContext()->iscg;
            ID3D12GraphicsCommandList* pGraphicsCommandList = nullptr;
            if (SUCCEEDED(pCommandList->QueryInterface<ID3D12GraphicsCommandList>(&pGraphicsCommandList)))
            {
                iscg->d3d12ApplyGlobalGpuInferenceSchedulingModeToCommandList(pGraphicsCommandList);
            }
        };

        // Do iscg and version checks now to avoid doing them inside every callback       
        PFun_commandListAction* CLresetCallback = nullptr;
        IHWID3D12* iscg = gpt::getContext()->iscg;
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
            0,          // bufferPadBytes
            32 * 1024,  // maxDescriptorSets
            256 * 1024, // constantBufferSize
            adapter->coreCount,
            adapter->architecture
        };
        ggml_d3d12_set_params(d3d12GGMLParams);
#endif
        {
#if GGML_USE_CUBLAS
            ggml_backend_reg_t cuda_reg = ggml_backend_cuda_reg();

            nvigi::RuntimeContextScope scope(*instanceData);

            // Query the CIG-active CUDA device (if CIG is enabled) or use the device from CudaParameters
            int targetDevice = 0;

            auto cudaParams = findStruct<CudaParameters>(_params);
            auto d3dParams = findStruct<D3D12Parameters>(_params);

            // If using CIG/D3D12, the CIG context is already current (pushed by RuntimeContextScope)
            // Query which device it's on so embed uses the correct device
            if (d3dParams && d3dParams->queue && instanceData->cudaContext.cudaCtx)
            {
                // CIG context is already current, just query which device
                CUdevice cuDevice;
                CUresult cuerr = cuCtxGetDevice(&cuDevice);
                if (cuerr == CUDA_SUCCESS)
                {
                    targetDevice = (int)cuDevice;
                    NVIGI_LOG_INFO("CIG context is already active on device %d, whisper will use this device",
                        targetDevice);
                }
                else
                {
                    targetDevice = 0;
                    NVIGI_LOG_WARN("Failed to query current device, defaulting to device 0");
                }
            }
            else if (cudaParams)
            {
                targetDevice = cudaParams->device;
                NVIGI_LOG_INFO("User specified CUDA device %d in CudaParameters struct", targetDevice);
            }
            else
            {
                targetDevice = 0;
                NVIGI_LOG_INFO("No device specified, using default CUDA device 0");
            }
                
            // Pre-specify the device to prevent GGML from enumerating all devices (which changes context)
            // This must be done before llama_backend_init() and common_init_from_params()
            // IMPORTANT: The devices vector must be NULL-terminated (llama.cpp expects a NULL-terminated array)
            ggml_backend_dev_t cuda_dev = ggml_backend_reg_dev_get(cuda_reg, targetDevice);
            instanceData->params.devices.clear();
            instanceData->params.devices.push_back(cuda_dev);
            instanceData->params.devices.push_back(nullptr);  // NULL terminator
                
            // When devices vector is specified, main_gpu is an INDEX into that vector, not a device ordinal
            // Since we only have one device in the vector, main_gpu should always be 0
            instanceData->params.main_gpu = 0;                
#endif

#if GGML_USE_CUBLAS || GGML_USE_VULKAN || GGML_USE_D3D12
            // Force single GPU mode for all GPU backends (CUDA, Vulkan, D3D12) to allow graphics to run at full speed on the other GPU
            // Note that llama is still free to run layers on CPU when it reaches the vram limit specified by the user
            instanceData->params.split_mode = LLAMA_SPLIT_MODE_NONE;
#endif

            try
            {
                if (!ctx.initialized)
                {
                    // NOTE: Each instance can potentially change device, queue etc but llama.cpp has not backend free function.
                    // 
                    // As a result, we cannot have instances with different devices or queues due to llama.cpp limitations
                    llama_backend_init();
                    llama_numa_init(instanceData->params.numa);
                    ctx.initialized = true;
                }

                instanceData->llamaContext = common_init_from_params(instanceData->params);

                // Load Loras
                if (creationParams->getVersion() >= kStructVersion3)
                {

                    std::vector<common_adapter_lora_info> lora_info_vec;
                    for (int loraIndex = 0; loraIndex != creationParams->numLoras; ++loraIndex)
                    {
                        std::string loraName(creationParams->loraNames[loraIndex]);

                        if (!instanceData->modelInfo.contains("loras"))
                        {
                            NVIGI_LOG_ERROR("Config for GUID '%s' has no loras.  Unable to load Lora.", common->modelGUID);
                            break;
                        }

                        const json& loras = instanceData->modelInfo["loras"];
                        for (const auto& lora : loras)
                        {
                            if (loraName == lora["name"])
                            {
                                std::string pluginDir = plugin::getContext()->framework->getModelDirectoryForPlugin(plugin::getContext()->info.id).c_str();
                                auto directory = std::string((common->utf8PathToModels + std::string("/") + pluginDir + std::string("/") + common->modelGUID).c_str());
                                auto loraPath = std::string(directory + std::string("/") + std::string(lora["filename"]));

                                std::ifstream file_exists(loraPath);
                                if (!file_exists.good())
                                {
                                    NVIGI_LOG_WARN("Unable to find %s.  Attempting direct search...", loraPath);
                                    // our first attempt to locate the filename relative to the models folder was unsuccessful.  Now see if we can't just locate it without modification.
                                    loraPath = std::string(lora["filename"]);
                                    std::ifstream file_exists(loraPath);
                                    if (!file_exists.good())
                                    {
                                        // if the file still can't be found, then report an error
                                        NVIGI_LOG_ERROR("Unable to find %s", loraPath);
                                        break;
                                    }
                                }

                                neededVRAM += lora["vram"].get<size_t>();

                                // This has LlamaCPP manage the loras
                                llama_adapter_lora_ptr lora;
                                lora.reset(llama_adapter_lora_init(instanceData->llamaContext.model.get(), loraPath.c_str()));
                                if (lora == nullptr)
                                    NVIGI_LOG_WARN("### NVIGI GPT SH -> Load Lora Adapter Failed: %s", loraName.c_str());
                                else {
                                    //instanceData->lora_adapters[loraName] = std::make_pair(loraPath, std::move(lora));
                                    instanceData->lora_adapters[loraName] = InferenceContext::igi_adapter_lora_info{ loraPath, std::move(lora) };
                                    NVIGI_LOG_INFO("### NVIGI GPT SH -> Load Lora Adapter: %s", loraName.c_str());
                                }

                                // If scales have been set, load the loras immediately.  If not, then they must be set during runtime before
                                // they activate.
                                if (creationParams->loraScales != nullptr)
                                {
                                    float loraScale = creationParams->loraScales[loraIndex];
                                    if (loraScale < 0.0f || loraScale > 1.0f)
                                    {
                                        NVIGI_LOG_ERROR("Provided Lora Scale '%f' is out of bounds", loraScale);
                                        break;
                                    }

                                    // BROWLETT - These are new additions as of Oct 2025...not clear if we need to pipe them through or not.
                                    std::string task_name("");
                                    std::string prompt_prefix("");
                                    lora_info_vec.push_back({ loraPath, creationParams->loraScales[loraIndex], task_name, prompt_prefix, instanceData->lora_adapters[loraName].ptr.get() });
                                }

                                break;
                            }
                        }
                    }

                    if (!lora_info_vec.empty())
                    {
                        common_set_adapter_lora(instanceData->llamaContext.context.get(), lora_info_vec);
                    }

                    if (instanceData->lora_adapters.size() != creationParams->numLoras)
                    {
                        NVIGI_LOG_WARN("Check logs, not all loras were loaded");
                    }
                }

            }
            catch (std::exception& e)
            {
                NVIGI_LOG_ERROR("%s", e.what());
                delete instanceData;
                return kResultInvalidState;
            }
            
            if (!instanceData->llamaContext.model)
            {
                delete instanceData;
                return kResultInvalidState;
            }

#if GGML_USE_CUBLAS
            // We ask llama for all the cuda_streams it is going to use and 
            // store them to enable us to change their priorities dynamically
            size_t stream_count = llama_get_cuda_stream_count(instanceData->llamaContext.context.get());
            instanceData->cuda_streams.resize(stream_count);
            llama_get_cuda_streams(instanceData->llamaContext.context.get(), (void**)instanceData->cuda_streams.data(), stream_count);


            if (instanceData->cudaContext.usingCiG && ctx.icig->getVersion() >= 2)
            {
                // Apply the global priority to all streams
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
            // Begin VLM Addition
#if COMPILE_VLM
            if (!instanceData->params.mmproj.path.empty())
            {
                const char* clip_path = instanceData->params.mmproj.path.c_str();

                nvigi::log::ILog* log = nvigi::log::getInterface();
                nvigi::LogLevel logLevel = log->getLogLevel();

                mtmd_context_params mparams = mtmd_context_params_default();

#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
                mparams.use_gpu   = true;
#else
                mparams.use_gpu   = false;
#endif
                mparams.verbosity = logLevel >= nvigi::LogLevel::eVerbose ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_NONE;
                instanceData->clipContext = mtmd_init_from_file(clip_path, instanceData->llamaContext.model.get(),mparams);
            }

#endif // COMPILE_VLM
            // End VLM Addition

            // At this point we are OK
            instanceData->state.store(nvigi::kResultOk);
        }
    }

    auto instance = new InferenceInstance();
    instance->data = instanceData;
    instance->getFeatureId = gpt::getFeatureId;
    instance->getInputSignature = gpt::getInputSignature;
    instance->getOutputSignature = gpt::getOutputSignature;
    instance->evaluate = gpt::evaluate;
    instance->evaluateAsync = gpt::evaluateAsync;
    instance->cancelAsyncEvaluation = gpt::cancelAsyncEvaluation;

    *_instance = instance;

    return kResultOk;
}

nvigi::Result ggmlGetCapsAndRequirements(nvigi::NVIGIParameter** _info, const nvigi::NVIGIParameter* _params)
{
    auto common = findStruct<CommonCreationParameters>(_params);
    if (!common) return nvigi::kResultInvalidParameter;

    static CommonCapabilitiesAndRequirements s_caps{};
    static GPTSamplerParameters s_sampler{}; // sets defaults
    auto info = &s_caps;
    *_info = s_caps;

    if (!findStruct<GPTSamplerParameters>(s_caps))
    {
        // Chaining this struct indicates that we support sampler parameters
        s_caps.chain(s_sampler);
    }

    auto& ctx = (*gpt::getContext());
    json modelInfo;
    if (!ai::findModels(common, { "gguf" }, modelInfo))
    {
        return kResultInvalidParameter;
    }
    

#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
    info->supportedBackends = nvigi::InferenceBackendLocations::eGPU;
#else
    info->supportedBackends = nvigi::InferenceBackendLocations::eCPU;
#endif

    //! Must be called after we set supported backends to correctly filter models
    ai::populateCommonCapsAndRequirements(ctx.capsData, *common, *info, modelInfo);

    return kResultOk;
}

nvigi::Result ggmlEvaluate(nvigi::InferenceExecutionContext* execCtx, bool async)
{
#if GGML_USE_CUBLAS || (defined(GGML_USE_D3D12) && PROFILE_D3D)
    nvtxRangePushA("ggmlEvaluate");
#endif

    auto& ctx = (*gpt::getContext());

    // Validate all inputs first
    
    if (!execCtx)
    {
        NVIGI_LOG_ERROR("Unable to find 'InferenceExecutionContext' structure in the inputs");
        return kResultInvalidParameter;
    }

    // OK not to have callback in async mode with polled results
    if (!execCtx->callback && !async)
    {
        NVIGI_LOG_ERROR("GPT inference callback not provided");
        return kResultInvalidParameter;
    }

    if (!execCtx->instance)
    {
        NVIGI_LOG_ERROR("GPT inference instance not provided");
        return kResultInvalidParameter;
    }
    
    if (ctx.feature != execCtx->instance->getFeatureId(execCtx->instance->data))
    {
        NVIGI_LOG_ERROR("Invalid inference instance - expecting GPT %u got %u", ctx.feature, execCtx->instance->getFeatureId(execCtx->instance->data));
        return kResultInvalidParameter;
    }

    // Now we are good to go!
    using namespace nvigi::gpt;

    auto responseCallback = [](nvigi::InferenceExecutionContext* execCtx, int32_t token, const std::string& response, nvigi::InferenceExecutionState state = nvigi::kInferenceExecutionStateDataPending)->nvigi::InferenceExecutionState
    {
        // check for error
        if (token < 0)
        {
            return nvigi::kInferenceExecutionStateInvalid;
        }
        // ignore special tokens
        if (response.find("<|") != std::string::npos) return nvigi::kInferenceExecutionStateDataPending;

        nvigi::InferenceExecutionState res{};

        const nvigi::InferenceDataText* output{};
        if (execCtx->outputs && execCtx->outputs->findAndValidateSlot(kGPTDataSlotResponse, &output))
        {
            auto cpuBuffer = castTo<CpuData>(output->utf8Text);
            if (!cpuBuffer || !cpuBuffer->buffer || cpuBuffer->sizeInBytes < response.size())
            {
                return nvigi::kInferenceExecutionStateInvalid;
            }
            strcpy_s((char*)cpuBuffer->buffer, cpuBuffer->sizeInBytes, response.c_str());
            if (execCtx->callback)
            {
                res = execCtx->callback(execCtx, state, execCtx->callbackUserData);
            }
            else
            {
                auto instance = (nvigi::gpt::InferenceContext*)(execCtx->instance->data);
                res = instance->pollCtx.triggerCallback(state);
            }
        }
        else
        {
            //! Temporary outputs for the callback since host did not provide any
            nvigi::CpuData text{response.length() + 1, (const void*)response.c_str()};
            nvigi::InferenceDataText data(text);            
            std::vector<nvigi::InferenceDataSlot> slots = { {kGPTDataSlotResponse, data} };
            nvigi::InferenceDataSlotArray outputs = { slots.size(), slots.data() };
            execCtx->outputs = &outputs;
            if (execCtx->callback)
            {
                res = execCtx->callback(execCtx, state, execCtx->callbackUserData);
            }
            else
            {
                auto instance = (nvigi::gpt::InferenceContext*)(execCtx->instance->data);
                res = instance->pollCtx.triggerCallback(state);
            }
            //! Clear outputs since these are all local variables
            execCtx->outputs = {};
        }
        return res;
    };

    
    auto instance = (nvigi::gpt::InferenceContext*)(execCtx->instance->data);

    if (NVIGI_FAILED(result, instance->state.load()))
    {
        NVIGI_LOG_ERROR("Instance is in invalid state and it must be destroyed and recreated, check previous evaluate calls or result callbacks for errors");
        return result;
    }

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

    //NVIGI_LOG_VERBOSE("Processing input '%s' on %u threads...", input, ctx.params.n_threads);

    const nvigi::InferenceDataText* systemSlot{};
    const nvigi::InferenceDataText* userSlot{};
    const nvigi::InferenceDataText* assistantSlot{};
    execCtx->inputs->findAndValidateSlot(kGPTDataSlotSystem, &systemSlot);
    execCtx->inputs->findAndValidateSlot(kGPTDataSlotUser, &userSlot);
    execCtx->inputs->findAndValidateSlot(kGPTDataSlotAssistant, &assistantSlot);
    if (!userSlot && !systemSlot && !assistantSlot)
    {
        NVIGI_LOG_ERROR("Expecting inference input(s) of type 'nvigi::InferenceDataText' - either system, user and/or assistant inputs should be provided");
        return kResultInvalidParameter;
    }

    // Optional
    auto runtime = findStruct<GPTRuntimeParameters>(execCtx->runtimeParameters);
    if (runtime)
    {
        if (runtime->batchSize > instance->instanceBatchSize)
        {
            NVIGI_LOG_ERROR("Batch size cannot be increased after instance creation");
            return kResultInvalidParameter;
        }

        // Make sure async job is not running when we modify parameters
        std::lock_guard<std::mutex> lock(instance->sync.mtx);

        instance->params.sampling.seed = runtime->seed;
        instance->params.n_predict = runtime->tokensToPredict;
        instance->params.n_keep = runtime->tokensToKeep;
        instance->params.n_batch = runtime->batchSize;
        instance->params.n_chunks = runtime->numChunks;
        instance->params.n_parallel = runtime->numParallel;
        instance->params.n_sequences = runtime->numSequences;
        instance->params.interactive = runtime->interactive;
        instance->params.sampling.temp = runtime->temperature;
        instance->params.sampling.top_p = runtime->topP;
        if (runtime->reversePrompt)
        {
            if (instance->params.antiprompt.empty() || instance->params.antiprompt.back() != runtime->reversePrompt)
            {
                instance->params.antiprompt.clear();
                instance->params.antiprompt.emplace_back(runtime->reversePrompt);
            }
        }
        else
        {
            if (runtime->interactive)
            {
                NVIGI_LOG_ERROR("'GPTRuntimeParameters::reversePrompt' runtime parameter must be specified when in interactive mode");
                return kResultInvalidParameter;
            }
            instance->params.antiprompt.clear();
        }
        instance->params.input_prefix = runtime->prefix ? runtime->prefix : "";
        instance->params.input_suffix = runtime->suffix ? runtime->suffix : "";

        // Adjust/Reload Loras if scales have changed.
        if (runtime->getVersion() >= kStructVersion4)
        {
            std::vector<common_adapter_lora_info> lora_info_vec;
            for (int i = 0; i < runtime->numLoras; ++i) {
                if (!instance->lora_adapters.contains(runtime->loraNames[i])) {
                    NVIGI_LOG_WARN("### NVIGI GPT SH -> Lora Adapter Not Found: %s", runtime->loraNames[i]);
                    continue;
                }

                // BROWLETT - These are new additions as of Oct 2025...not clear if we need to pipe them through or not.
                std::string task_name("");
                std::string prompt_prefix("");

                // need to store the path when I found it during create so I can look it up again to reset it.
                lora_info_vec.push_back({ instance->lora_adapters[runtime->loraNames[i]].loraPath, runtime->loraScales[i], task_name, prompt_prefix, instance->lora_adapters[runtime->loraNames[i]].ptr.get() });
            }
            
            // If we have an update to the lora info vec, then update it.
            if (!lora_info_vec.empty())
            {
                common_set_adapter_lora(instance->llamaContext.context.get(), lora_info_vec);
            }
        }

        if (runtime->getVersion() >= 2 && runtime->targetTokensPerSecond > 0 && runtime->frameTimeMs > 0)
        {
            if (!instance->sync.tgLimiter)
            {
                instance->sync.tgLimiter = new TokenGenLimiter();
            }
            instance->sync.tgLimiter->update(1000.0f / std::max(0.01f, runtime->frameTimeMs), runtime->targetTokensPerSecond);
        }
    }

    // Optional
    auto sampler = findStruct<GPTSamplerParameters>(execCtx->runtimeParameters);
    if (sampler)
    {
        // Make sure async job is not running when we modify parameters
        std::lock_guard<std::mutex> lock(instance->sync.mtx);

        instance->params.sampling.n_prev = sampler->numPrev;
        instance->params.sampling.n_probs = sampler->numProbs;
        instance->params.sampling.min_keep = sampler->minKeep;
        instance->params.sampling.top_k = sampler->topK;
        instance->params.sampling.min_p = sampler->minP;
        instance->params.sampling.xtc_probability = sampler->xtcProbability;
        instance->params.sampling.xtc_threshold = sampler->xtcThreshold;
        //instance->params.sampling.tfs_z = sampler->tfsZ;
        instance->params.sampling.typ_p = sampler->typP;
        instance->params.sampling.dynatemp_range = sampler->dynatempRange;
        instance->params.sampling.dynatemp_exponent = sampler->dynatempExponent;
        instance->params.sampling.penalty_last_n = sampler->penaltyLastN;
        instance->params.sampling.penalty_repeat = sampler->penaltyRepeat;
        instance->params.sampling.penalty_freq = sampler->penaltyFreq;
        instance->params.sampling.penalty_present = sampler->penaltyPresent;
        instance->params.sampling.mirostat = sampler->mirostat;
        instance->params.sampling.mirostat_tau = sampler->mirostatTAU;
        instance->params.sampling.mirostat_eta = sampler->mirostatETA;
        //instance->params.sampling.penalize_nl = sampler->penalizeNewLine;
        instance->params.sampling.ignore_eos = sampler->ignoreEOS;
        // New parameters so always make sure we have the latest version
        if (sampler->getVersion() >= 2)
        {
            instance->params.sampling.grammar = sampler->grammar ? sampler->grammar : "";
            instance->sync.persistentKVCache.store(sampler->persistentKVCache);
            instance->params.path_prompt_cache = sampler->utf8PathToSessionCache ? sampler->utf8PathToSessionCache : "";
        }
    }

    std::string system = systemSlot ? systemSlot->getUTF8Text() : "";
    std::string user = userSlot ? userSlot->getUTF8Text() : "";
    std::string assistant = assistantSlot ? assistantSlot->getUTF8Text() : "";

    // Begin VLM Addition
#if COMPILE_VLM
    const unsigned char* rgb_data = nullptr;
    int rgb_width = 0;
    int rgb_height = 0;
    const nvigi::InferenceDataImage* imageSlot{};
    execCtx->inputs->findAndValidateSlot(kGPTDataSlotImage, &imageSlot);
    if (imageSlot != nullptr)
    {
        CpuData* cpu_image = castTo<CpuData>(imageSlot->bytes);

        rgb_width = imageSlot->w;
        rgb_height = imageSlot->h;
        rgb_data = static_cast<const unsigned char*>(cpu_image->buffer);

        bool data_valid = true;
        if (rgb_width <= 0 || rgb_height <= 0)
        {
            NVIGI_LOG_ERROR("InferenceDataImage width or height is not positive. Image ignored.");
            data_valid = false;
        }

        if (rgb_data == nullptr)
        {
            NVIGI_LOG_ERROR("InferenceDataImage rgb_data is null. Image ignored.");
            data_valid = false;
        }

        if (!data_valid)
        {
            rgb_data = nullptr;
            rgb_width = 0;
            rgb_height = 0;
        }

        if (!user.empty() && rgb_data != nullptr )
        {
            std::string search_str = "<image>";
            std::string nvigi_img_marker = "NVIGI_IMG";
            // Note - it is critical that the spaces exist on each side of the replace string to ensure that NVIGI_IMG gets tokenized properly 
            std::string replace_str = " " + nvigi_img_marker + " ";

            // Don't allow the user to have any "NVIGI_IMG" markers in their prompt.  Replace with nothing.
            replace_all(user, nvigi_img_marker, "");

            size_t start_pos = user.find(search_str);
            if (start_pos == std::string::npos)
            {
                // if the user hasn't specified where the image should go, 
                // add a marker that tokenization can find later on where to place the image.
                // Nemovision was trained with the image at the beginning or end of the user prompt, so choosing end for now.
                user += replace_str;
            }
            else
            {
                // replace the first <image> with " NVIGI_IMG " 
                replace_first(user, search_str, replace_str);
                // replace all other <image> with nothing, as we only support 1 image per prompt right now.
                replace_all(user, search_str, "");
            }
        }
    }
#endif // COMPILE_VLM
    // End VLM Addition

    if (instance->params.interactive)
    {
        // Interactive (chat) mode
        if (instance->sync.job.valid())
        {
            // Chat is active, check if new system prompt is provided
            if (systemSlot)
            {
                // Starting new conversation, flush and clear cache
                if (NVIGI_FAILED(result, flushAndTerminate(instance, instance->sync)))
                {
                    return result;
                }
                instance->sync.runningChat.store(true);
                instance->sync.cancelled.store(false);  // Reset cancellation flag for new evaluation
                instance->sync.execCtx = execCtx;
                
                if(!instance->sync.persistentKVCache.load())
                {
#ifdef GGML_USE_CUBLAS
                    nvigi::RuntimeContextScope scope(*instance);
#endif
                    llama_memory_clear(instance->llamaContext.context->get_memory(), true);
                }

                // Prepare prompt based on model's template
                igiProcessChatTemplate(runtime, instance, system, user, assistant);
                instance->sync.runningWithMutexLockedPromise = std::promise<void>();
                instance->sync.silenceOutput.store(false);
                // Begin VLM Addition
#if COMPILE_VLM
                instance->sync.rgb_data = rgb_data;
                instance->sync.rgb_width = rgb_width;
                instance->sync.rgb_height = rgb_height;
#endif 
                // End VLM Addition

                std::future<void> runningWithMtxLocked = instance->sync.runningWithMutexLockedPromise.get_future();
                instance->sync.job = std::async(std::launch::async, [instance, responseCallback]()->nvigi::Result
                {
#ifdef GGML_USE_CUBLAS
                    nvigi::RuntimeContextScope scope(*instance);
#endif
                    // Begin VLM Modification - VLM added instance->clipContext
                    if (generate(&instance->sync, instance->llamaContext.model.get(), instance->llamaContext.context.get(), instance->clipContext, instance->params, responseCallback))
                    {
                        instance->state.store(nvigi::kInferenceExecutionStateInvalid);
                        responseCallback(instance->sync.execCtx, -1, "", nvigi::kInferenceExecutionStateInvalid);
                        return kResultInvalidState;
                    }
                    return kResultOk;
                });
                if (!async)
                {
                    runningWithMtxLocked.wait(); // Wait for the signal
                    //! At this point we know 100% that job started and locked on mutex
                    //! 
                    //! Since eval is a blocking call we need to wait here until mutex is released (waiting for user to take turn)
                    std::unique_lock<std::mutex> lock(instance->sync.mtx);
                }
            }
            else
            {
                // No system input, process this as a turn in an active chat

                // First let's check to make sure our thread did not die (hence tiny delay, we just care if thread is dead or not)
                if (instance->sync.job.wait_for(std::chrono::microseconds(10)) == std::future_status::ready)
                {
                    // Unexpected termination here, job should be running at this point so must be an error
                    auto result = instance->sync.job.get();
                    NVIGI_LOG_ERROR("gpt.ggml generate returned early with result %u", result);
                    instance->state.store(result);
                    return result;
                }
                {
                    // Lock and wait until thread is ready for user input
                    std::unique_lock<std::mutex> lock(instance->sync.mtx);
                    // Prepare for our turn based on model's template, note that system prompt is NOT used here
                    if (instance->modelInfo.contains(ai::kTurnTemplate))
                    {
                        // legacy support for old turn templates
                        instance->sync.input = ai::generateTurn(instance->modelInfo, user, assistant);
                    }
                    else
                    {
                        instance->sync.input = user;
                    }
                    // Begin VLM Addition
#if COMPILE_VLM
                    instance->sync.rgb_data = rgb_data;
                    instance->sync.rgb_width = rgb_width;
                    instance->sync.rgb_height = rgb_height;
#endif
                    // End VLM Addition
                    instance->sync.execCtx = execCtx;
                    instance->sync.cancelled.store(false);  // Reset cancellation flag for new turn
                    instance->sync.silenceOutput.store(false);
                    instance->sync.newInput.store(true);
                }
                // Generate new promise
                instance->sync.runningWithMutexLockedPromise = std::promise<void>();
                std::future<void> runningWithMtxLocked = instance->sync.runningWithMutexLockedPromise.get_future();
                // Notify that user input is ready so thread can continue
                instance->sync.cvInput.notify_one();
                if (!async)
                {
                    runningWithMtxLocked.wait(); // Wait for the signal
                    //! At this point we know 100% that job started and locked on mutex
                    //! 
                    //! Since eval is a blocking call we need to wait here until mutex is released (waiting for user to take turn)
                    std::unique_lock<std::mutex> lock(instance->sync.mtx);
                }
            }
        }
        else
        {
            // Starting new conversation first time, prepare prompt based on model's template
            igiProcessChatTemplate(runtime, instance, system, user, assistant);
            instance->sync.silenceOutput.store(false);
            instance->sync.runningChat.store(true);
            instance->sync.cancelled.store(false);  // Reset cancellation flag for new evaluation
            instance->sync.execCtx = execCtx;
            // Begin VLM Addition
#if COMPILE_VLM
            instance->sync.rgb_data = rgb_data;
            instance->sync.rgb_width = rgb_width;
            instance->sync.rgb_height = rgb_height;
#endif
            // End VLM Addition
            // Generate new promise
            instance->sync.runningWithMutexLockedPromise = std::promise<void>();
            std::future<void> runningWithMtxLocked = instance->sync.runningWithMutexLockedPromise.get_future();
            // First time initializing our job
            instance->sync.job = std::async(std::launch::async, [instance, responseCallback]()->nvigi::Result
            {
#ifdef GGML_USE_CUBLAS
                nvigi::RuntimeContextScope scope(*instance);
#endif
                // Begin VLM Modification - VLM added instance->clipContext
                if (generate(&instance->sync, instance->llamaContext.model.get(), instance->llamaContext.context.get(), instance->clipContext, instance->params, responseCallback))
                {
                    instance->state.store(nvigi::kInferenceExecutionStateInvalid);
                    responseCallback(instance->sync.execCtx, -1, "", nvigi::kInferenceExecutionStateInvalid);
                    return kResultInvalidState;
                }
                return kResultOk;
            });
            if (!async)
            {
                runningWithMtxLocked.wait(); // Wait for the signal
                //! At this point we know 100% that job started and locked on mutex
                //! 
                //! Since eval is a blocking call we need to wait here until mutex is released (waiting for user to take turn)
                std::unique_lock<std::mutex> lock(instance->sync.mtx);
            }
        }
    }
    else
    {
        // Instruct mode, prepare prompt based on model's template
        if (async)
        {
            if (NVIGI_FAILED(result, flushAndTerminate(instance, instance->sync)))
            {
                return result;
            }

            igiProcessChatTemplate(runtime, instance, system, user, assistant);
            // Begin VLM Addition
#if COMPILE_VLM
            instance->sync.rgb_data = rgb_data;
            instance->sync.rgb_width = rgb_width;
            instance->sync.rgb_height = rgb_height;
#endif // COMPILE_VLM
            // End VLM Addition
            instance->sync.execCtx = execCtx;
            instance->sync.cancelled.store(false);  // Reset cancellation flag for new evaluation
            instance->sync.silenceOutput.store(false);

            instance->sync.job = std::async(std::launch::async, [instance, responseCallback]()->nvigi::Result
            {
#ifdef GGML_USE_CUBLAS
                nvigi::RuntimeContextScope scope(*instance);
#endif
                // Begin VLM Modification - VLM added instance->clipContext
                if (generate(&instance->sync, instance->llamaContext.model.get(), instance->llamaContext.context.get(), instance->clipContext, instance->params, responseCallback))
                {
                    instance->state.store(nvigi::kInferenceExecutionStateInvalid);
                    responseCallback(instance->sync.execCtx, -1, "", nvigi::kInferenceExecutionStateInvalid);
                    return kResultInvalidState;
                }
                return kResultOk;
            });
        }
        else
        {
            igiProcessChatTemplate(runtime, instance, system, user, assistant);
            instance->sync.execCtx = execCtx;
            instance->sync.cancelled.store(false);  // Reset cancellation flag for new evaluation
            instance->sync.silenceOutput.store(false);
            instance->sync.runningChat.store(false);
#if COMPILE_VLM
            // Begin VLM Addition
            instance->sync.rgb_data = rgb_data;
            instance->sync.rgb_width = rgb_width;
            instance->sync.rgb_height = rgb_height;
            // End VLM Addition
#endif

#ifdef GGML_USE_CUBLAS
            nvigi::RuntimeContextScope scope(*instance);
#endif
            if (generate(&instance->sync, instance->llamaContext.model.get(), instance->llamaContext.context.get(), instance->clipContext, instance->params, responseCallback))
            {
                return kResultInvalidState;
            }
        }
    }

#if GGML_USE_CUBLAS || (defined(GGML_USE_D3D12) && PROFILE_D3D)
    nvtxRangePop();
#endif

    return kResultOk;
}

nvigi::Result gptGetResults(nvigi::InferenceExecutionContext* execCtx, bool wait, nvigi::InferenceExecutionState* state)
{
    if (!execCtx || !execCtx->instance)
        return kResultInvalidParameter;

    auto instance = static_cast<gpt::InferenceContext*>(execCtx->instance->data);
    return instance->pollCtx.getResults(wait, state);
}

nvigi::Result gptReleaseResults(nvigi::InferenceExecutionContext* execCtx, nvigi::InferenceExecutionState state)
{
    if (!execCtx || !execCtx->instance)
        return kResultInvalidParameter;

    auto instance = static_cast<gpt::InferenceContext*>(execCtx->instance->data);
    return instance->pollCtx.releaseResults(state);
}

nvigi::Result gptCancelAsyncEvaluation(nvigi::InferenceExecutionContext* execCtx)
{
    if (!execCtx || !execCtx->instance)
        return kResultInvalidParameter;

    auto instance = static_cast<gpt::InferenceContext*>(execCtx->instance->data);
    
    // Check if async job is actually running
    if (!instance->sync.job.valid())
    {
        NVIGI_LOG_WARN("cancelAsyncEvaluation called but no async evaluation is running");
        return kResultNoImplementation;
    }
    
    // Set cancellation flag to interrupt the generate loop as early as possible
    instance->sync.cancelled.store(true);
    
    if (NVIGI_FAILED(result, flushAndTerminate(instance, instance->sync)))
    {
        return result;
    }
    
    return kResultOk;
}

//! Exception handling wrappers
//! 
//! Note that we export these via our interface
//! 
namespace gpt
{
nvigi::Result createInstance(const nvigi::NVIGIParameter* params, nvigi::InferenceInstance** instance)
{
    NVIGI_CATCH_EXCEPTION(ggmlCreateInstance(params, instance));
}
nvigi::Result destroyInstance(const nvigi::InferenceInstance* instance)
{
    NVIGI_CATCH_EXCEPTION(ggmlDestroyInstance(instance));
}
nvigi::Result getCapsAndRequirements(nvigi::NVIGIParameter** modelInfo, const nvigi::NVIGIParameter* params)
{
    NVIGI_CATCH_EXCEPTION(ggmlGetCapsAndRequirements(modelInfo, params));
}
nvigi::Result evaluate(nvigi::InferenceExecutionContext* execCtx)
{
    NVIGI_CATCH_EXCEPTION(ggmlEvaluate(execCtx, false));
}
nvigi::Result evaluateAsync(nvigi::InferenceExecutionContext* execCtx)
{
    NVIGI_CATCH_EXCEPTION(ggmlEvaluate(execCtx, true));
}
nvigi::Result getResults(nvigi::InferenceExecutionContext* execCtx, bool wait, nvigi::InferenceExecutionState* state)
{
    NVIGI_CATCH_EXCEPTION(gptGetResults(execCtx, wait, state));
}

nvigi::Result releaseResults(nvigi::InferenceExecutionContext* execCtx, nvigi::InferenceExecutionState state)
{
    NVIGI_CATCH_EXCEPTION(gptReleaseResults(execCtx, state));
}

nvigi::Result cancelAsyncEvaluation(nvigi::InferenceExecutionContext* execCtx)
{
    NVIGI_CATCH_EXCEPTION(gptCancelAsyncEvaluation(execCtx));
}
} // gpt

//! Main entry point - get information about our plugin
//! 
Result nvigiPluginGetInfo(framework::IFramework* framework, nvigi::plugin::PluginInfo** _info)
{
    auto& ctx = (*gpt::getContext());
    
    if (!plugin::internalPluginSetup(framework)) return kResultInvalidState;

    // Internal API, we know that incoming pointer is always valid
    auto& info = plugin::getContext()->info;
    *_info = &info;

    info.id = gpt::getFeatureId(nullptr);
    info.description = "ggml based backend for LLM";
    info.author = "NVIDIA";
    info.build = GIT_BRANCH_AND_LAST_COMMIT;
    info.interfaces = { plugin::getInterfaceInfo<IGeneralPurposeTransformer>()};

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

    auto& ctx = (*gpt::getContext());

    ctx.feature = gpt::getFeatureId(nullptr);

#if GGML_USE_CUBLAS
    if (!framework::getInterface(plugin::getContext()->framework, plugin::hwi::cuda::kId, &ctx.icig))
    {
        NVIGI_LOG_ERROR("Missing interface from 'nvigi.plugin.hwi.cuda'");
        return kResultInvalidState;
    }
#elif defined(GGML_USE_D3D12)
    if (!framework::getInterface(framework, nvigi::core::framework::kId, &ctx.isystem))
    {
        NVIGI_LOG_ERROR("Missing core interface 'nvigi::system::ISystem'");
        return kResultMissingInterface;
    }
#endif

    ctx.api.createInstance = gpt::createInstance;
    ctx.api.destroyInstance = gpt::destroyInstance;
    ctx.api.getCapsAndRequirements = gpt::getCapsAndRequirements;
 
    framework->addInterface(ctx.feature, &ctx.api, 0);

    // Add polled interface
    ctx.polledApi.getResults = gpt::getResults;
    ctx.polledApi.releaseResults = gpt::releaseResults;
    framework->addInterface(ctx.feature, &ctx.polledApi, 0);

    // Begin Llama Logging Setup
    // Make sure there is NO llama logging, it all gets redirected to our log
    llama_log_set(nvigi::gpt::llamaLogCallback, NULL);
    common_log* llama_log = common_log_main();
    common_log_pause(llama_log);
    // End Llama Logging Setup

    return kResultOk;
}

//! Main exit point - shutting down our plugin
//! 
Result nvigiPluginDeregister()
{
    auto& ctx = (*gpt::getContext());

    ai::freeCommonCapsAndRequirements(ctx.capsData);

#if GGML_USE_CUBLAS
    framework::releaseInterface(plugin::getContext()->framework, plugin::hwi::cuda::kId, ctx.icig);
    ctx.icig = nullptr;
#elif defined(GGML_USE_D3D12)
    framework::releaseInterface(plugin::getContext()->framework, plugin::hwi::d3d12::kId, ctx.iscg);
    ctx.iscg = nullptr;
    framework::releaseInterface(plugin::getContext()->framework, nvigi::core::framework::kId, ctx.isystem);
    ctx.isystem = nullptr;
#endif

    // call this to flush any threads waiting to write to disk
    common_log_pause(common_log_main());

    llama_backend_free();

#if defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
    // llama.cpp does not provide a way to free backends explicitly hence the hack
    ggml_vk_backend_free();
#endif
    // cpu 
    // Note, GPT hasn't shown the same crash as Embed, but could be susceptible to it as well.
    // Reference:  https://ofekshilon.com/2017/11/03/on-omp_wait_policy/
    // LlamaCPP uses OpenMP in CPU mode.  On shutdown, OpenMP threads can spin wait for 200ms-2000ms (varying reports)
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

    // alternatively, next time we upgrade LlamaCPP, we could build CPU WIHTOUT OpenMP, but it will be slower.

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