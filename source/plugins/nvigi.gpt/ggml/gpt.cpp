// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "source/plugins/nvigi.gpt/nvigi_gpt.h"
#include "_artifacts/gitVersion.h"

#include "source/plugins/nvigi.gpt/ggml/gpt.h"
//#include "source/plugins/nvigi.imgui/imgui.h"

#include "log.h"

#if GGML_USE_CUBLAS
#include "source/core/nvigi.api/nvigi_cuda.h"
#include "source/utils/nvigi.hwi/cuda/push_poppable_cuda_context.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include "nvtx3/nvToolsExt.h"
#endif

namespace nvigi
{
namespace gpt
{

static void llamaLogCallback(ggml_log_level level, const char* text, void* user_data)
{
    //! Special case, llama prints progress bars with a stream of consecutive "." so we want to ignore that
    if (strcmp(text, ".") == 0) return;

    if (level == GGML_LOG_LEVEL_WARN)
    {
        NVIGI_LOG("llama", LogType::eWarn, nvigi::log::YELLOW, "%s", text);
    }
    else if (level == GGML_LOG_LEVEL_ERROR)
    {
        NVIGI_LOG("llama", LogType::eError, nvigi::log::RED, "%s", text);
    }
    else
    {
        NVIGI_LOG("llama", LogType::eInfo, nvigi::log::WHITE, "%s", text);
    }
}

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

struct InferenceContext
{
    InferenceContext(const nvigi::NVIGIParameter* params) : cudaContext(params) {}

    common_params params{};

    InferenceContextSync sync{};

    json modelInfo;

#ifndef NVIGI_GPT_GFN_NVCF
    common_init_result llamaContext{};
    clip_ctx* clipContext{};
#endif

#if GGML_USE_CUBLAS
    // Used to set the relative priority of GPU inference and graphics
    std::vector<cudaStream_t> cuda_streams;
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
#else
    return plugin::gpt::ggml::cpu::kId;
#endif
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

    // Caps and requirements
    ai::CommonCapsData capsData;

#ifdef GGML_USE_CUBLAS
    nvigi::IHWICuda* icig{};
#endif
};

nvigi::Result evaluate(nvigi::InferenceExecutionContext* execCtx);
nvigi::Result evaluateAsync(nvigi::InferenceExecutionContext* execCtx);
}

//! Define our plugin, make sure to update version numbers in versions.h
NVIGI_PLUGIN_DEFINE("nvigi.plugin.gpt", Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH), Version(API_MAJOR, API_MINOR, API_PATCH), gpt, GPTContext)

nvigi::Result ggmlEvaluate(nvigi::InferenceExecutionContext* execCtx);

nvigi::Result flushAndTerminate(gpt::InferenceContextSync& sync)
{
    auto result = nvigi::kResultOk;
    if (sync.job.valid())
    {
        if (sync.runningChat.load())
        {
            //NVIGI_LOG_VERBOSE("Flushing running chat ...");
            {
                std::scoped_lock lock(sync.mtx);
                sync.runningChat.store(false);
                sync.newInput.store(true);
            }
            sync.cvInput.notify_all();
        }
        result = sync.job.get();
    }
    return result;
}

nvigi::Result ggmlDestroyInstance(const nvigi::InferenceInstance* instance)
{
    if (instance)
    {
        auto gptInstance = static_cast<nvigi::gpt::InferenceContext*>(instance->data);
        if (NVIGI_FAILED(result, flushAndTerminate(gptInstance->sync)))
        {
            return result;
        }

        {
#ifdef GGML_USE_CUBLAS
            nvigi::RuntimeContextScope scope(*gptInstance);
#endif
			if (gptInstance->clipContext != nullptr)
            {
                clip_free(gptInstance->clipContext);
                gptInstance->clipContext = nullptr;
            }
            gptInstance->llamaContext.model.reset();
            gptInstance->llamaContext.context.reset();
            gptInstance->llamaContext.lora.clear();
        }

        delete gptInstance;
        delete instance;
    }
    return nvigi::kResultOk;
}

nvigi::Result ggmlCreateInstance(const nvigi::NVIGIParameter* _params, nvigi::InferenceInstance** _instance)
{
    auto common = findStruct<CommonCreationParameters>(_params);
    auto paramsGPT = findStruct<GPTCreationParameters>(_params);
    if (!common || !paramsGPT) return nvigi::kResultInvalidParameter;
    auto& params = *paramsGPT;

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

#if defined(GGML_USE_CUBLAS)
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

    if (common->numThreads > 1)
    {
        NVIGI_LOG_WARN("For optimal performance when using CUDA only one CPU thread is used");
}
    instanceData->params.cpuparams.n_threads = 1;
#else
    instanceData->params.n_gpu_layers = 0;
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
        instanceData->params.n_ctx = params.contextSize;
        instanceData->params.n_predict = params.maxNumTokensToPredict;

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
            n_layers = instanceData->modelInfo["n_layers"];
#if defined(GGML_USE_CUBLAS)
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
            instanceData->params.model = files[0];
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
                        instanceData->params.model = model_file;
                    if (ends_with(model_file, mmproj_model_suffix))
                        instanceData->params.mmproj = model_file;
                }
            }
        }
        catch (std::exception& e)
        {
            NVIGI_LOG_ERROR("%s", e.what());
            return kResultInvalidState;
        }

        NVIGI_LOG_INFO("Loading model '%s'", instanceData->params.model.c_str());
#if GGML_USE_CUBLAS
        auto platform = "ggml.cuda";
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

        {
#ifdef GGML_USE_CUBLAS
            nvigi::RuntimeContextScope scope(*instanceData);
#endif
            if (!ctx.initialized)
            {
                llama_backend_init();
                llama_numa_init(instanceData->params.numa);
                ctx.initialized = true;
            }

            instanceData->llamaContext = common_init_from_params(instanceData->params);
            
#if GGML_USE_CUBLAS
            // We ask llama for all the cuda_streams it is going to use and 
            // store them to enable us to change their priorities dynamically
            size_t stream_count = llama_get_cuda_stream_count(instanceData->llamaContext.context.get());
            instanceData->cuda_streams.resize(stream_count);
            llama_get_cuda_streams(instanceData->llamaContext.context.get(), (void**)instanceData->cuda_streams.data(), stream_count);


            if (ctx.icig->getVersion() >= 2)
            {
                // Apply the global priority to all streams
                nvigi::Result cuerr = ctx.icig->cudaApplyGlobalGpuInferenceSchedulingMode(instanceData->cuda_streams.data(), instanceData->cuda_streams.size());
                if (cuerr != kResultOk)
                {
                    NVIGI_LOG_WARN("Could not set relative priority of compute and graphics. Please use 575 driver or higher\n");
                }
            }
#endif

            if (!instanceData->params.mmproj.empty())
            {
                const char* clip_path = instanceData->params.mmproj.c_str();
                instanceData->clipContext = clip_model_load(clip_path, /*verbosity=*/ 1);
            }

            if (!instanceData->llamaContext.model)
            {
                delete instanceData;
                return kResultInvalidState;
            }
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
    

#if defined(GGML_USE_CUBLAS)
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
#if GGML_USE_CUBLAS
    nvtxRangePushA("ggmlEvaluate");
#endif

    auto& ctx = (*gpt::getContext());

    // Validate all inputs first
    
    if (!execCtx)
    {
        NVIGI_LOG_ERROR("Unable to find 'InferenceExecutionContext' structure in the inputs");
        return kResultInvalidParameter;
    }

    if (!execCtx->callback)
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
            res = execCtx->callback(execCtx, state, execCtx->callbackUserData);
        }
        else
        {
            //! Temporary outputs for the callback since host did not provide any
            nvigi::CpuData text{response.length() + 1, (const void*)response.c_str()};
            nvigi::InferenceDataText data(text);            
            std::vector<nvigi::InferenceDataSlot> slots = { {kGPTDataSlotResponse, data} };
            nvigi::InferenceDataSlotArray outputs = { slots.size(), slots.data() };
            execCtx->outputs = &outputs;
            res = execCtx->callback(execCtx, state, execCtx->callbackUserData);
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

    if (ctx.icig->getVersion() >= 2)
    {
        nvigi::Result err = ctx.icig->cudaApplyGlobalGpuInferenceSchedulingMode(instance->cuda_streams.data(), instance->cuda_streams.size());
        if (err != kResultOk)
        {
            NVIGI_LOG_WARN_ONCE("Could not set relative priority of compute and graphics, insufficient driver\n");
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

    if (instance->params.interactive)
    {
        instance->params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
        // Interactive (chat) mode
        if (instance->sync.job.valid())
        {
            // Chat is active, check if new system prompt is provided
            if (systemSlot)
            {
                // Starting new conversation, flush and clear cache
                if (NVIGI_FAILED(result, flushAndTerminate(instance->sync)))
                {
                    return result;
                }
                instance->sync.runningChat.store(true);
                instance->sync.execCtx = execCtx;
                
                if(!instance->sync.persistentKVCache.load())
                {
#ifdef GGML_USE_CUBLAS
                    nvigi::RuntimeContextScope scope(*instance);
#endif
                    llama_kv_cache_clear(instance->llamaContext.context.get());
                }

                // Prepare prompt based on model's template
                instance->params.prompt = ai::generatePrompt(instance->modelInfo, system, user, assistant);
                instance->sync.runningWithMutexLockedPromise = std::promise<void>();
                instance->sync.silenceOutput.store(false);
                instance->sync.rgb_data = rgb_data;
                instance->sync.rgb_width = rgb_width;
                instance->sync.rgb_height = rgb_height;

                std::future<void> runningWithMtxLocked = instance->sync.runningWithMutexLockedPromise.get_future();
                instance->sync.job = std::async(std::launch::async, [instance, responseCallback]()->nvigi::Result
                {
#ifdef GGML_USE_CUBLAS
                    nvigi::RuntimeContextScope scope(*instance);
#endif
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
                    instance->sync.input = ai::generateTurn(instance->modelInfo, user, assistant);
                    instance->sync.rgb_data = rgb_data;
                    instance->sync.rgb_width = rgb_width;
                    instance->sync.rgb_height = rgb_height;
                    instance->sync.execCtx = execCtx;
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
            instance->params.prompt = ai::generatePrompt(instance->modelInfo, system, user, assistant);
            instance->sync.silenceOutput.store(false);
            instance->sync.runningChat.store(true);
            instance->sync.execCtx = execCtx;
            instance->sync.rgb_data = rgb_data;
            instance->sync.rgb_width = rgb_width;
            instance->sync.rgb_height = rgb_height;
            std::future<void> runningWithMtxLocked = instance->sync.runningWithMutexLockedPromise.get_future();
            // First time initializing our job
            instance->sync.job = std::async(std::launch::async, [instance, responseCallback]()->nvigi::Result
            {
#ifdef GGML_USE_CUBLAS
                nvigi::RuntimeContextScope scope(*instance);
#endif
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
        instance->params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        if (async)
        {
            if (NVIGI_FAILED(result, flushAndTerminate(instance->sync)))
            {
                return result;
            }

            instance->params.prompt = ai::generatePrompt(instance->modelInfo, system, user, assistant);
            instance->sync.rgb_data = rgb_data;
            instance->sync.rgb_width = rgb_width;
            instance->sync.rgb_height = rgb_height;
            instance->sync.execCtx = execCtx;
            instance->sync.silenceOutput.store(false);

            instance->sync.job = std::async(std::launch::async, [instance, responseCallback]()->nvigi::Result
            {
#ifdef GGML_USE_CUBLAS
                nvigi::RuntimeContextScope scope(*instance);
#endif
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
            instance->params.prompt = ai::generatePrompt(instance->modelInfo, system, user, assistant);
            instance->sync.execCtx = execCtx;
            instance->sync.silenceOutput.store(false);
            instance->sync.runningChat.store(false);
            instance->sync.rgb_data = rgb_data;
            instance->sync.rgb_width = rgb_width;
            instance->sync.rgb_height = rgb_height;

#ifdef GGML_USE_CUBLAS
            nvigi::RuntimeContextScope scope(*instance);
#endif
            if (generate(&instance->sync, instance->llamaContext.model.get(), instance->llamaContext.context.get(), instance->clipContext, instance->params, responseCallback))
            {
                return kResultInvalidState;
            }
        }
    }

#if GGML_USE_CUBLAS
    nvtxRangePop();
#endif

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

#ifdef GGML_USE_CUBLAS
    info.minDriver = { NVIGI_CUDA_MIN_DRIVER_MAJOR, NVIGI_CUDA_MIN_DRIVER_MINOR, NVIGI_CUDA_MIN_DRIVER_BUILD };
    info.minOS = { NVIGI_DEF_MIN_OS_MAJOR, NVIGI_DEF_MIN_OS_MINOR, NVIGI_DEF_MIN_OS_BUILD };
    info.minGPUArch = { NVIGI_CUDA_MIN_GPU_ARCH };
    info.requiredVendor = VendorId::eNVDA;
#else
    //! Defaults indicate no restrictions - plugin can run on any system, even without any adapter
    info.requiredVendor = nvigi::VendorId::eNone;
    info.minDriver = {};
    info.minOS = { NVIGI_DEF_MIN_OS_MAJOR, NVIGI_DEF_MIN_OS_MINOR, NVIGI_DEF_MIN_OS_BUILD };
    info.minGPUArch = {};
#endif
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
#endif

    ctx.api.createInstance = gpt::createInstance;
    ctx.api.destroyInstance = gpt::destroyInstance;
    ctx.api.getCapsAndRequirements = gpt::getCapsAndRequirements;
 
    framework->addInterface(ctx.feature, &ctx.api, 0);

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
#endif

    // call this to flush any threads waiting to write to disk
    common_log_pause(common_log_main());

    llama_backend_free();

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