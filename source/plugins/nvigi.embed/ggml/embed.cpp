// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <future>

#include "source/core/nvigi.api/nvigi.h"
#include "source/core/nvigi.api/internal.h"
#include "source/core/nvigi.log/log.h"
#include "source/core/nvigi.exception/exception.h"
#include "source/core/nvigi.exception/exception.h"
#include "source/core/nvigi.plugin/plugin.h"
#include "source/core/nvigi.file/file.h"
#include "source/core/nvigi.extra/extra.h"
#include "source/plugins/nvigi.embed/ggml/versions.h"
#include "source/utils/nvigi.ai/ai.h"
#include "source/utils/nvigi.ai/ai_data_helpers.h"
#include "source/utils/nvigi.hwi/cuda/runtime_context_scope.h"
#include "source/plugins/nvigi.embed/nvigi_embed.h"
#include "_artifacts/gitVersion.h"

#include "arg.h"
#include "llama.h"
#include "common.h"
#include "source/plugins/nvigi.embed/ggml/embed.h"
#include "log.h"

#if GGML_USE_CUBLAS
#include "source/core/nvigi.api/nvigi_cuda.h"
#include "source/utils/nvigi.hwi/cuda/push_poppable_cuda_context.h"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace nvigi
{
// forward declaration
nvigi::Result ggmlGetCapsAndRequirements(nvigi::NVIGIParameter** _info, const nvigi::NVIGIParameter* _params);

namespace embed
{
struct InferenceContext
{
    InferenceContext(const nvigi::NVIGIParameter* params) : cudaContext(params) {}

    common_params params{};
    json modelInfo;

    common_init_result llama_init;

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
#ifdef GGML_USE_CUBLAS
    return plugin::embed::ggml::cuda::kId;
#else
    return plugin::embed::ggml::cpu::kId;
#endif
}

const nvigi::InferenceDataDescriptorArray* getInputSignature(InferenceInstanceData* data)
{
    static std::vector<InferenceDataDescriptor> slots = { {nvigi::kEmbedDataSlotInText,InferenceDataText::s_type, false } };
    static InferenceDataDescriptorArray s_desc = { slots.size(), slots.data() };
    return &s_desc;
}

const nvigi::InferenceDataDescriptorArray* getOutputSignature(InferenceInstanceData* data)
{
    static std::vector<InferenceDataDescriptor> slots = { {nvigi::kEmbedDataSlotOutEmbedding,InferenceDataText::s_type, false } };
    static InferenceDataDescriptorArray s_desc = { slots.size(), slots.data() };
    return &s_desc;
}

struct EmbedContext
{
    NVIGI_PLUGIN_CONTEXT_CREATE_DESTROY(EmbedContext);

    void onCreateContext() {};
    void onDestroyContext() {};

    std::thread* worker{};

    std::atomic<bool> initialized = false;

    nvigi::PluginID feature{};

    IEmbed api{};

    // Caps and requirements
    json modelInfo;
    ai::CommonCapsData capsData;
    std::vector<size_t> embedding_sizes{};
    std::vector<int> max_position_embeddings{};
    std::vector<common_params> llamacpp_params{};
};

nvigi::Result evaluate(nvigi::InferenceExecutionContext* execCtx);
} // embed

//! Define our plugin, make sure to update version numbers in versions.h
NVIGI_PLUGIN_DEFINE("nvigi.plugin.embed", Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH), Version(API_MAJOR, API_MINOR, API_PATCH), embed, EmbedContext)

nvigi::InferenceExecutionState ggmlReturnOutputEmbedding(nvigi::InferenceExecutionContext* execCtx, const std::vector<float>& embedding)
{
    // check for error
    if (embedding.empty())
    {
        return nvigi::kInferenceExecutionStateInvalid;
    }

    nvigi::InferenceExecutionState res{};

    const nvigi::InferenceDataByteArray* output{};
    if (execCtx->outputs && execCtx->outputs->findAndValidateSlot(kEmbedDataSlotOutEmbedding, &output))
    {
        CpuData* cpuBuffer = castTo<CpuData>(output->bytes);
        if (cpuBuffer->buffer == nullptr || cpuBuffer->sizeInBytes < embedding.size() * sizeof(float))
        {
            return nvigi::kInferenceExecutionStateInvalid;
        }

        memcpy_s((float*)(cpuBuffer->buffer), cpuBuffer->sizeInBytes, &(embedding[0]), embedding.size() * sizeof(float));
    }
    else
    {
        //! Temporary outputs for the callback since host did not provide any
        nvigi::ai::InferenceDataByteArrayHelper responseSlot((const uint8_t*)embedding.data(), embedding.size() * sizeof(float));
        std::vector<nvigi::InferenceDataSlot> slots = { {kEmbedDataSlotOutEmbedding, responseSlot}};
        nvigi::InferenceDataSlotArray outputs = { slots.size(), slots.data() };
        execCtx->outputs = &outputs;
    }
    res = execCtx->callback(execCtx, nvigi::kInferenceExecutionStateDone, execCtx->callbackUserData);

    //! Clear outputs since these are all local variables
    execCtx->outputs = {};

    return res;
}

nvigi::Result ggmlDestroyInstance(const nvigi::InferenceInstance* instance)
{
    if (instance)
    {
        auto embedInstance = static_cast<nvigi::embed::InferenceContext*>(instance->data);

        {
            nvigi::RuntimeContextScope scope(*embedInstance);
            embedInstance->llama_init.model.reset();
            embedInstance->llama_init.context.reset();
            embedInstance->llama_init.lora.clear();
        }

        delete embedInstance;
        embedInstance = nullptr;
        delete instance;
    }
    return nvigi::kResultOk;
}

nvigi::Result ggmlCreateInstance(const nvigi::NVIGIParameter* _params, nvigi::InferenceInstance** _instance)
{
    auto common = findStruct<CommonCreationParameters>(_params);
    auto creationParams = findStruct<EmbedCreationParameters>(_params);
    if (!creationParams || !common) return nvigi::kResultInvalidParameter;
    auto& params = *creationParams;

    if (!_instance || !common->utf8PathToModels || !common->modelGUID) return nvigi::kResultInvalidParameter;

    if (!ai::isGuid(common->modelGUID))
    {
        NVIGI_LOG_ERROR("Provided model GUID '%s' is invalid", common->modelGUID);
        return kResultInvalidParameter;
    }

    *_instance = nullptr;

    auto instanceData = new nvigi::embed::InferenceContext(_params);
    if (!instanceData->cudaContext.constructorSucceeded) return kResultInvalidState;

    auto& ctx = (*embed::getContext());

    // get any gpt_params as set by the config.json for this model.
    nvigi::NVIGIParameter* _info;
    nvigi::ggmlGetCapsAndRequirements(&_info, _params);
    EmbedCapabilitiesAndRequirements* capsAndReqs = nvigi::castTo<EmbedCapabilitiesAndRequirements>(_info);
    if (_info != nullptr)
    {
        // we specify model GUID so we always get just one model in common caps
        assert(capsAndReqs->common->numSupportedModels == 1);
        if (capsAndReqs->common->numSupportedModels != 1)
        {
            return kResultInvalidState;
        }
        instanceData->params = ctx.llamacpp_params[0];
    }
    else
    {
        return kResultInvalidState;
    }

    using namespace nvigi::embed;

#if GGML_USE_CUBLAS
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

    instanceData->params.n_gpu_layers = INT_MAX;
    if (common->numThreads > 1)
    {
        NVIGI_LOG_WARN("For optimal performance when using CUDA only one CPU thread is used");
    }
    instanceData->params.cpuparams.n_threads = 1;
#else
    instanceData->params.n_gpu_layers = 0;
    instanceData->params.cpuparams.n_threads = common->numThreads;
#endif

    {

        instanceData->modelInfo.clear();
        {
            if (!ai::findModels(common, { "gguf" }, instanceData->modelInfo))
            {
                return kResultInvalidParameter;
            }
        }
        try
        {
            // Trim down to our GUID for this instance
            instanceData->modelInfo = instanceData->modelInfo[common->modelGUID];
            if (instanceData->params.n_gpu_layers > 0)
            {
                size_t neededVRAM = instanceData->modelInfo["vram"];
                if (common->vramBudgetMB < neededVRAM)
                {
                    NVIGI_LOG_WARN("Provided VRAM %uMB is insufficient, required VRAM is %uMB", common->vramBudgetMB, neededVRAM);
                    return kResultInsufficientResources;
                }
            }
            std::vector<std::string> files = instanceData->modelInfo["gguf"];
            if (files.empty())
            {
                NVIGI_LOG_ERROR("Failed to find model in the expected directory '%s'", common->utf8PathToModels);
                return kResultInvalidParameter;
            }
            instanceData->params.model = files[0];
        }
        catch (std::exception& e)
        {
            NVIGI_LOG_ERROR("%s", e.what());
            return kResultInvalidState;
        }


        {
            NVIGI_LOG_INFO("Loading model '%s'", instanceData->params.model.c_str());
            nvigi::RuntimeContextScope scope(*instanceData);
            // load the model and apply lora adapter, if any

            if (!ctx.initialized)
            {
                llama_backend_init();
                llama_numa_init(instanceData->params.numa);
                ctx.initialized = true;
            }

            instanceData->params.embedding = true;
            instanceData->params.embd_sep = prompts_sep;
            // We set the maximum batch size as the maximum position embeddings for the model. Since the model is theoritically not capable of extracting good quality embedding beyond that.
            instanceData->params.n_batch = instanceData->modelInfo.contains("max_position_embeddings") ? (int)instanceData->modelInfo["max_position_embeddings"] : nvigi::default_max_position_embeddings;
            // For non-causal models, batch size must be equal to ubatch size
            instanceData->params.n_ubatch = instanceData->params.n_batch;
            instanceData->params.n_ctx = instanceData->params.n_batch; // context size must be smaller or equal to batch size with the new llama.cpp update
            instanceData->params.warmup = false;
            instanceData->llama_init = common_init_from_params(instanceData->params);            
            if (!instanceData->llama_init.model)
            {
                delete instanceData;
                return kResultInvalidState;
            }
            else if (get_embed_size(instanceData->llama_init.model.get()) <= 0) {
                NVIGI_LOG_ERROR("Embedding size should be at least 1");
                return kResultInvalidState;
            }
        }

#if GGML_USE_CUBLAS
        auto platform = "ggml.cuda";
#else
        auto platform = "ggml.cpu";
#endif

        NVIGI_LOG_VERBOSE("Created instance for backend '%s' - threads %d - GPU layers '%s' - context size %d - predicting %d - batch %d", platform, instanceData->params.cpuparams.n_threads, instanceData->params.n_gpu_layers > 0 ? "all" : "none", instanceData->params.n_ctx, instanceData->params.n_predict, instanceData->params.n_batch);
    }

    // Explicitly declaring that we are implementing v1
    auto instance = new InferenceInstance(kStructVersion1);
    instance->data = instanceData;
    instance->getFeatureId = embed::getFeatureId;
    instance->getInputSignature = embed::getInputSignature;
    instance->getOutputSignature = embed::getOutputSignature;
    instance->evaluate = embed::evaluate;

    *_instance = instance;

    return kResultOk;
}

nvigi::Result ggmlGetCapsAndRequirements(nvigi::NVIGIParameter** _info, const nvigi::NVIGIParameter* _params)
{
    // population of the CapabilitiesAndRequirements is doing a data oriented population of 
    // ALL models supported by this plugin.  It is encumbent on the user to iterate over these vectors
    // find the model they are interested in and the corresponding caps and requirements for that model.

    auto common = findStruct<CommonCreationParameters>(_params);
    if (!common)
    {
        return nvigi::kResultInvalidParameter;
    }

    auto params = castTo<EmbedCreationParameters>(_params);
    if (!_info)
        return nvigi::kResultInvalidParameter;

    static CommonCapabilitiesAndRequirements s_capsCommon{};
    static EmbedCapabilitiesAndRequirements s_caps{};
    s_caps.common = &s_capsCommon;
    auto info = &s_caps;
    *_info = s_caps;

    auto& ctx = (*embed::getContext());
    if (!ai::findModels(common, { "gguf" }, ctx.modelInfo))
    {
        return kResultInvalidParameter;
    }

    // then populate the plugin specific caps and requirements
    ctx.embedding_sizes.clear();
    ctx.max_position_embeddings.clear();
    ctx.llamacpp_params.clear();
    for (auto& model : ctx.modelInfo) try
    {
        // Filter by GUID if one provided
        std::string guid = model["guid"];
        if (common->modelGUID && guid != common->modelGUID)
        {
            continue;
        }
        size_t embedding_size = model["embedding_size"];
        ctx.embedding_sizes.push_back(embedding_size);

        if (model.contains("max_position_embeddings")) {
            ctx.max_position_embeddings.push_back(model["max_position_embeddings"]);
        }
        else {
            ctx.max_position_embeddings.push_back(default_max_position_embeddings);
        }

        // all key/values in here should match up with command line arguments as set forth in llamacpp executable.
        // a fake command line will be constructed to parameterize a gpt_params struct.
        // this allows us more flexibility in exposing llama_cpp params without additional coding burden in nvigi.
        if (model.contains("llama_cpp_params"))
        {
            std::string json_llama_cpp_cl("");
            std::vector< std::string > params_vec_str;
            // name here meant to represent the "executable" we are executing.
            params_vec_str.push_back(std::string("llama_embedding.exe"));
            // build up a command line from this dictionary...the parameters should match _exactly_ what the CL that LlamaCPP would want

            std::string cl;
            for (auto& [key, value] : model["llama_cpp_params"].items()) {
                cl += "--" + key + " " + value.get<std::string>() + " ";
                params_vec_str.push_back(std::string("--") + key);
                params_vec_str.push_back(value.get<std::string>());
            };

            common_params json_llama_cpp_params;
            std::vector< char* > params_vec_cstr;
            for (auto& str : params_vec_str)
            {
                params_vec_cstr.push_back(const_cast<char*>(str.c_str()));
            }

            if (!common_params_parse(static_cast<int>(params_vec_str.size()), &(params_vec_cstr[0]), json_llama_cpp_params, LLAMA_EXAMPLE_EMBEDDING)) {
                return 1;
            }

            ctx.llamacpp_params.push_back(json_llama_cpp_params);
        }
        else
        {
            ctx.llamacpp_params.push_back(common_params());
        }

    }
    catch (std::exception&)
    {
        return nvigi::kResultInvalidParameter;
    }

    info->embedding_numel = ctx.embedding_sizes.data();
    info->max_position_embeddings = ctx.max_position_embeddings.data();

#if GGML_USE_CUBLAS
    info->common->supportedBackends = nvigi::InferenceBackendLocations::eGPU;
#else
    info->common->supportedBackends = nvigi::InferenceBackendLocations::eCPU;
#endif

    //! Must be called after we set supported backends to correctly filter models
    ai::populateCommonCapsAndRequirements(ctx.capsData, *common, *info->common, ctx.modelInfo);

    return kResultOk;
}

nvigi::Result ggmlEvaluate(nvigi::InferenceExecutionContext* execCtx)
{
    auto& ctx = (*embed::getContext());

    // Validate all inputs first

    if (!execCtx)
    {
        NVIGI_LOG_ERROR("Unable to find 'InferenceExecutionContext' structure in the inputs");
        return kResultInvalidParameter;
    }

    if (!execCtx->callback)
    {
        NVIGI_LOG_ERROR("Embed inference callback not provided");
        return kResultInvalidParameter;
    }

    if (!execCtx->instance)
    {
        NVIGI_LOG_ERROR("Embed inference instance not provided");
        return kResultInvalidParameter;
    }

    if (ctx.feature != execCtx->instance->getFeatureId(execCtx->instance->data))
    {
        NVIGI_LOG_ERROR("Invalid inference instance - expecting Embed %u got %u", ctx.feature, execCtx->instance->getFeatureId(execCtx->instance->data));
        return kResultInvalidParameter;
    }

    // Now we are good to go!
    using namespace nvigi::embed;

    auto instance = (nvigi::embed::InferenceContext*)(execCtx->instance->data);

    nvigi::RuntimeContextScope scope(*instance);

    const nvigi::InferenceDataText* data{};
    if (!execCtx->inputs->findAndValidateSlot(kEmbedDataSlotInText, &data))
    {
        NVIGI_LOG_ERROR("Expecting single inference input of type 'sl::InferenceDataText'");
        return kResultInvalidParameter;
    }

    instance->params.prompt = (const char*)data->getUTF8Text();

#ifndef NVIGI_EMBED_GFN_NVCF
    // Local inference
    std::vector<float> embedding;
    nvigi::Result res = nvigi::embed::embed(instance->llama_init.context.get(), instance->llama_init.model.get(), instance->params, embedding);
    if (res != kResultOk)
        return res;

    res = kResultOk;
    if (ggmlReturnOutputEmbedding(execCtx, embedding) != nvigi::kInferenceExecutionStateDone)
        res = kResultInvalidState;
    return res;
#else
    return kResultMissingInterface;
#endif 
}

//! Exception handling wrappers
//! 
//! Note that we export these via our interface
//! 
namespace embed
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
    NVIGI_CATCH_EXCEPTION(ggmlEvaluate(execCtx));
}
} // embed

//! Main entry point - get information about our plugin
//! 
Result nvigiPluginGetInfo(framework::IFramework* framework, nvigi::plugin::PluginInfo** _info)
{
    auto& ctx = (*embed::getContext());

    if (!plugin::internalPluginSetup(framework)) return kResultInvalidState;

    // Internal API, we know that incoming pointer is always valid
    auto& info = plugin::getContext()->info;
    *_info = &info;

    info.id = embed::getFeatureId(nullptr);
    info.description = "ggml backend for embedding model";
    info.author = "NVIDIA";
    info.build = GIT_BRANCH_AND_LAST_COMMIT;
    info.interfaces = { plugin::getInterfaceInfo<IEmbed>() };

    //! We can run on any driver or GPU sku
    info.minDriver = {};
    info.minOS = { NVIGI_DEF_MIN_OS_MAJOR, NVIGI_DEF_MIN_OS_MINOR, NVIGI_DEF_MIN_OS_BUILD };
    info.minGPUArch = {};
    return kResultOk;
}

//! Main entry point - starting our plugin
//! 
//! IMPORTANT: Plugins are started based on their priority.
//!
Result nvigiPluginRegister(framework::IFramework* framework)
{
    if (!plugin::internalPluginSetup(framework)) return kResultInvalidState;

    auto& ctx = (*embed::getContext());

    ctx.feature = embed::getFeatureId(nullptr);

    ctx.api.createInstance = embed::createInstance;
    ctx.api.destroyInstance = embed::destroyInstance;
    ctx.api.getCapsAndRequirements = embed::getCapsAndRequirements;

    framework->addInterface(ctx.feature, &ctx.api, 0);

    // Begin Llama Logging Setup
    // Make sure there is NO llama logging, it all gets redirected to our log
    common_log* llama_log = common_log_main();
    common_log_pause(llama_log);
    // End Llama Logging Setup

    return kResultOk;
}

//! Main exit point - shutting down our plugin
//! 
Result nvigiPluginDeregister()
{
    auto& ctx = (*embed::getContext());

    ai::freeCommonCapsAndRequirements(ctx.capsData);

    if (ctx.worker)
    {
        ctx.worker->join();
        delete ctx.worker;
        ctx.worker = {};
    }

    // call this to flush any threads waiting to write to disk
    common_log_pause(common_log_main());

    llama_backend_free();

    // cpu 
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

} // nvigi