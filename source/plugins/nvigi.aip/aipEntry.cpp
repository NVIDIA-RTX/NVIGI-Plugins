// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "source/core/nvigi.api/nvigi.h"
#include "source/core/nvigi.api/internal.h"
#include "source/core/nvigi.log/log.h"
#include "source/core/nvigi.plugin/plugin.h"
#include "source/core/nvigi.exception/exception.h"
#include "source/plugins/nvigi.aip/versions.h"
#include "source/utils/nvigi.ai/ai.h"
#include "_artifacts/gitVersion.h"
#include "external/json/source/nlohmann/json.hpp"
#include "source/plugins/nvigi.aip/nvigi_aip.h"

using json = nlohmann::json;

namespace nvigi
{
namespace aip
{

struct AiPipelineContext
{
    NVIGI_PLUGIN_CONTEXT_CREATE_DESTROY(AiPipelineContext);

    void onCreateContext() {};
    void onDestroyContext() {};

    IAiPipeline api{};

    std::string modelPath;
};

struct InferenceContext
{
    //! Interfaces and instances
    std::vector<std::tuple<InferenceInterface*, InferenceInstance*>> interfaceInstancePairs;
    std::vector<std::map<std::string, std::tuple<UID,std::vector<uint8_t>>>> outputs;

    //! Synchronization
    std::mutex mtx;
    std::condition_variable cv;
    uint32_t stageIndex = 0;
    types::vector<nvigi::InferenceExecutionState> stageStates;

    std::string processedPrompt{};
};

//! Forward declarations

//! Instance
PluginID getFeatureId(InferenceInstanceData* data);
const nvigi::InferenceDataDescriptorArray* getInputSignature(InferenceInstanceData* data);
const nvigi::InferenceDataDescriptorArray* getOutputSignature(InferenceInstanceData* data);
nvigi::Result evaluate(nvigi::InferenceExecutionContext* execCtx);
//! Interface
nvigi::Result createInstance(const AiPipelineCreationParameters& params, nvigi::InferenceInstance** instance);
nvigi::Result destroyInstance(const nvigi::InferenceInstance* instance);

} // ai

//! Define our plugin, make sure to update version numbers in versions.h
NVIGI_PLUGIN_DEFINE("nvigi.plugin.ai.pipeline", Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH), Version(API_MAJOR, API_MINOR, API_PATCH), aip, AiPipelineContext)

//! Implementation
//! 
//! 
namespace aip
{

//! Instance

PluginID getFeatureId(InferenceInstanceData* data)
{
    return nvigi::plugin::ai::pipeline::kId;
}

const nvigi::InferenceDataDescriptorArray* getInputSignature(InferenceInstanceData* data)
{
    auto ctx = (nvigi::aip::InferenceContext*)(data);
    auto& [inter, inst] = ctx->interfaceInstancePairs.front();
    return inst->getInputSignature(inst->data);
}

const nvigi::InferenceDataDescriptorArray* getOutputSignature(InferenceInstanceData* data)
{
    auto ctx = (nvigi::aip::InferenceContext*)(data);
    auto& [inter, inst] = ctx->interfaceInstancePairs.back();
    return inst->getOutputSignature(inst->data);
}

nvigi::Result aipEvaluate(nvigi::InferenceExecutionContext* execCtx)
{
    //! Validate inputs
    
    if (!execCtx)
    {
        NVIGI_LOG_ERROR("Unable to find 'InferenceExecutionContext' structure in the inputs");
        return kResultInvalidParameter;
    }

    if (!execCtx->callback)
    {
        NVIGI_LOG_ERROR("ASR inference callback not provided");
        return kResultInvalidParameter;
    }

    if (!execCtx->instance)
    {
        NVIGI_LOG_ERROR("ASR inference instance not provided");
        return kResultInvalidParameter;
    }

    auto instanceId = execCtx->instance->getFeatureId(execCtx->instance->data);
    if (instanceId != plugin::ai::pipeline::kId)
    {
        NVIGI_LOG_ERROR("Invalid inference instance - expecting instance created with interface '%s' got '%s'", extra::guidToString(plugin::ai::pipeline::kId).c_str(), extra::guidToString(instanceId).c_str());
        return kResultInvalidParameter;
    }

    auto data = (nvigi::aip::InferenceContext*)(execCtx->instance->data);

    struct AiCallbackCtx
    {
        InferenceExecutionContext* userExecCtx{};
        aip::InferenceContext* data{};
    };
    AiCallbackCtx cbCtx{};
    cbCtx.data = data;
    cbCtx.userExecCtx = execCtx;

    auto aceCallback = [](const nvigi::InferenceExecutionContext* ctx, nvigi::InferenceExecutionState state, void* userData)->nvigi::InferenceExecutionState
    {
        auto cbCtx = (AiCallbackCtx*)userData;

        for (auto& [inter, inst] : cbCtx->data->interfaceInstancePairs)
        {
            if (ctx->instance == inst)
            {
                //! IMPORTANT - assuming all data on CPU for now
                //! 
                //! TODO: change this to support various data backends but warn when copies are expensive
                //! 
                auto& outputs = cbCtx->data->outputs[cbCtx->data->stageIndex];
                for (size_t i = 0; i < ctx->outputs->count; i++)
                {
                    auto& slot = outputs[ctx->outputs->items[i].key];
                    auto& [id, data] = slot;
                    auto param = ((NVIGIParameter*)ctx->outputs->items[i].data);
                    id = param->type;
                    if (param->type == InferenceDataText::s_type)
                    {
                        auto text = castTo<const nvigi::InferenceDataText>(param);
                        auto utf8 = text->getUTF8Text();
                        assert(utf8);
                        // Check and ignore internal perf data returned in form of JSON string
                        auto response = std::string(utf8);
                        if (response.find("<JSON>") == std::string::npos)
                        {
                            std::vector<uint8_t> newData = { utf8,utf8 + strlen(utf8) };
                            data.insert(data.end(), newData.begin(), newData.end());
                        }
                    }
                    else if (param->type == InferenceDataAudio::s_type)
                    {
                        auto audio = castTo<const nvigi::InferenceDataAudio>(param);
                        auto cpuBuffer = castTo<CpuData>(audio->audio);
                        assert(cpuBuffer);
                        std::vector<uint8_t> newData = { (uint8_t*)cpuBuffer->buffer,(uint8_t*)cpuBuffer->buffer + cpuBuffer->sizeInBytes };
                        data.insert(data.end(), newData.begin(), newData.end());
                    }
                    else if (param->type == InferenceDataByteArray::s_type)
                    {
                        auto byteArray = castTo<const nvigi::InferenceDataByteArray>(param);
                        auto cpuBuffer = castTo<CpuData>(byteArray->bytes);
                        assert(cpuBuffer);
                        std::vector<uint8_t> newData = {(uint8_t*)cpuBuffer->buffer,(uint8_t*)cpuBuffer->buffer + cpuBuffer->sizeInBytes};
                        data.insert(data.end(), newData.begin(), newData.end());
                    }
                    else
                    {
                        assert(!"unknown data type");
                    }
                }
            }
        }

        //! Check if we are done and notify waiting thread (if running asynchronously)
        {
            std::lock_guard<std::mutex> lock(cbCtx->data->mtx);
            cbCtx->data->stageStates[cbCtx->data->stageIndex] = state;
        }

        auto callbackReturn = cbCtx->userExecCtx->callback(ctx, state, cbCtx->userExecCtx->callbackUserData);
        if (cbCtx->data->stageStates[cbCtx->data->stageIndex] != kInferenceExecutionStateDataPending)
        {
            cbCtx->data->cv.notify_one();
        }
        return callbackReturn;
    };

    //! Wait for stage to finish, if async
    auto waitForStage = [](aip::InferenceContext* data)->Result
    {
        std::unique_lock<std::mutex> lock(data->mtx);
        auto done = data->cv.wait_for(lock, std::chrono::seconds(30), [data] {return data->stageStates[data->stageIndex] == kInferenceExecutionStateDone || data->stageStates[data->stageIndex] == kInferenceExecutionStateCancel; });
        lock.unlock();
        return done ? kResultOk : kResultInvalidState;
    };

    //! First stage, inputs coming from the host
    {
        auto& [inter, inst] = data->interfaceInstancePairs.front();
        data->stageIndex = 0;
        data->stageStates[data->stageIndex] = kInferenceExecutionStateInvalid;
        nvigi::InferenceExecutionContext stageExecCtx{};
        stageExecCtx.instance = inst;
        stageExecCtx.runtimeParameters = execCtx->runtimeParameters;
        stageExecCtx.callback = aceCallback;
        stageExecCtx.callbackUserData = (void*)&cbCtx;
        stageExecCtx.inputs = execCtx->inputs;
        NVIGI_CHECK(inst->evaluate(&stageExecCtx));
        NVIGI_CHECK(waitForStage(data));
    }

    //! The rest of the pipeline
    for (size_t i = 1; i < data->stageStates.size(); i++)
    {
        auto& [inter, inst] = data->interfaceInstancePairs[i];
        data->stageIndex = (uint32_t)i;
        data->stageStates[data->stageIndex] = kInferenceExecutionStateInvalid;
        nvigi::InferenceExecutionContext stageExecCtx{};
        stageExecCtx.instance = inst;
        stageExecCtx.runtimeParameters = execCtx->runtimeParameters;
        stageExecCtx.callback = aceCallback;
        stageExecCtx.callbackUserData = (void*)&cbCtx;

        //! Outputs from the previous stage become inputs for this stage
        //! 
        //! IMPORTANT - assuming all data on CPU for now
        //! 
        //! TODO: change this to support various data backends but warn when copies are expensive
        //! 
        std::vector<CpuData*> cpuBuffers;
        std::vector<InferenceDataText*> textData;
        std::vector<InferenceDataAudio*> audioData;
        std::vector<InferenceDataByteArray*> byteData;
        std::vector<nvigi::InferenceDataSlot> inSlots;
        auto& outputs = data->outputs[i - 1];
        for (auto& [key, uid_data] : outputs)
        {
            auto& [id, bytes] = uid_data;
            // Check and make sure all text is 0 terminated
            if (id == InferenceDataText::s_type && bytes.back() != 0)
            {
                bytes.push_back(0);
            }
            cpuBuffers.push_back(new CpuData{ bytes.size(), bytes.data() });
            if (id == InferenceDataText::s_type)
            {
                textData.push_back(new InferenceDataText(*cpuBuffers.back()));
                inSlots.push_back({ key.c_str(), *textData.back() });
            }
            else if (id == InferenceDataAudio::s_type)
            {
                audioData.push_back(new InferenceDataAudio(*cpuBuffers.back()));
                inSlots.push_back({ key.c_str(), *audioData.back() });
            }
            else if (id == InferenceDataByteArray::s_type)
            {
                byteData.push_back(new InferenceDataByteArray(*cpuBuffers.back()));
                inSlots.push_back({ key.c_str(), *byteData.back() });
            }
            else
            {
                assert(!"unknown data type");
            }
        }
        //! Important: Include all original inputs from the host as well
        //! 
        //! Sometimes host can provide input(s) for the other stages not just the first one
        //! 
        for (size_t i = 0; i < execCtx->inputs->count; i++)
        {
            inSlots.push_back(execCtx->inputs->items[i]);
        }
        // We have all the inputs, let's evaluate this stage
        InferenceDataSlotArray inputs = { inSlots.size(), inSlots.data() };
        stageExecCtx.inputs = &inputs;
        NVIGI_CHECK(inst->evaluate(&stageExecCtx));
        NVIGI_CHECK(waitForStage(data));

        for (auto ptr : cpuBuffers)
        {
            delete ptr;
        }
        for (auto ptr : textData)
        {
            delete ptr;
        }
        for (auto ptr : audioData)
        {
            delete ptr;
        }
        for (auto ptr : byteData)
        {
            delete ptr;
        }

        for (auto& values : data->outputs)
        {
            values.clear();
        }
    }

    return kResultOk;
}

//! Interface

nvigi::Result aipCreateInstance(const nvigi::NVIGIParameter* _params, nvigi::InferenceInstance** _instance)
{
    if (!isOfType<AiPipelineCreationParameters>(_params)) return nvigi::kResultInvalidParameter;
    auto& params = *castTo<AiPipelineCreationParameters>(_params);

    if (!_instance) return nvigi::kResultInvalidParameter;

    *_instance = nullptr;

    auto data = new InferenceContext();

    for (size_t i = 0; i < params.numStages; i++)
    {
        InferenceInterface* _inter{};
        InferenceInstance* _inst{};
        if (!framework::getInterface(plugin::getContext()->framework, params.stages[i], &_inter))
        {
            NVIGI_LOG_ERROR("Failed to obtain interface {%s} for stage %llu", extra::guidToString(params.stages[i]).c_str(), i);
            return kResultInvalidState;
        }
        if (NVIGI_FAILED(error, _inter->createInstance(params.stageParams[i], &_inst)))
        {
            NVIGI_LOG_ERROR("Failed to create instance for stage %llu using interface {%s}", i, extra::guidToString(params.stages[i]).c_str());
            return error;
        }
        data->interfaceInstancePairs.push_back({ _inter, _inst });
    }
    data->stageStates.resize(params.numStages);
    data->outputs.resize(params.numStages);

    // Explicitly declaring that we are implementing v1
    auto instance = new InferenceInstance(kStructVersion1);
    instance->data = data;
    instance->getFeatureId = aip::getFeatureId;
    instance->getInputSignature = aip::getInputSignature;
    instance->getOutputSignature = aip::getOutputSignature;
    instance->evaluate = aip::evaluate;

    *_instance = instance;

    return kResultOk;
}
nvigi::Result aipDestroyInstance(const nvigi::InferenceInstance* instance)
{
    if (instance)
    {
        auto data = (nvigi::aip::InferenceContext*)(instance->data);
        assert(data);
        if (!data) return kResultInvalidState;

        for (auto& [inter, inst] : data->interfaceInstancePairs)
        {
            nvigi::PluginID feature = inst->getFeatureId(nullptr);
            NVIGI_CHECK(inter->destroyInstance(inst));
            framework::releaseInterface(plugin::getContext()->framework, feature, inter);
        }
        delete data;
        delete instance;
    }

    return kResultOk;
}

nvigi::Result evaluate(nvigi::InferenceExecutionContext* execCtx)
{
    NVIGI_CATCH_EXCEPTION(aipEvaluate(execCtx));
}
nvigi::Result createInstance(const NVIGIParameter* params, nvigi::InferenceInstance** instance)
{
    NVIGI_CATCH_EXCEPTION(aipCreateInstance(params, instance));
}
nvigi::Result destroyInstance(const nvigi::InferenceInstance* instance)
{
    NVIGI_CATCH_EXCEPTION(aipDestroyInstance(instance));
}

} // ai

//! Core API

//! Main entry point - get information about our plugin
//! 
Result nvigiPluginGetInfo(nvigi::framework::IFramework* framework, nvigi::plugin::PluginInfo** _info)
{
    if (!plugin::internalPluginSetup(framework)) return kResultInvalidState;
    
    // Internal API, we know that incoming pointer is always valid
    auto& info = plugin::getContext()->info;
    *_info = &info;

    info.id = plugin::ai::pipeline::kId;
    info.description = "Generic AI pipeline implementation plugin";
    info.author = "NVIDIA";
    info.build = GIT_BRANCH_AND_LAST_COMMIT;
    info.interfaces = { plugin::getInterfaceInfo<aip::IAiPipeline>()};

    //! Defaults indicate no restrictions - plugin can run on any system, even without any adapter
    info.requiredVendor = nvigi::VendorId::eNone;
    info.minDriver = {};
    info.minOS = { NVIGI_DEF_MIN_OS_MAJOR, NVIGI_DEF_MIN_OS_MINOR, NVIGI_DEF_MIN_OS_BUILD };
    info.minGPUArch = {};

    return kResultOk;
}

//! Main entry point - starting our plugin
//! 
Result nvigiPluginRegister(framework::IFramework* framework)
{
    if (!plugin::internalPluginSetup(framework)) return kResultInvalidState;

    auto& ctx = (*aip::getContext());

    ctx.api.createInstance = aip::createInstance;
    ctx.api.destroyInstance = aip::destroyInstance;
    
    framework->addInterface(plugin::ai::pipeline::kId, &ctx.api, 0);

    return kResultOk;
}

//! Main exit point - shutting down our plugin
//! 
Result nvigiPluginDeregister()
{
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