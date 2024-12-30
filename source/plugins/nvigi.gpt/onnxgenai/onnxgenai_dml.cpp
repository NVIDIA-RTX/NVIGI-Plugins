// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <string>
#include <vector>

#include "onnxgenai_dml.h"

#include <wrl.h>

using Microsoft::WRL::ComPtr;

void CheckResult(OgaResult* result) {
    if (result) {
        std::string string = OgaResultGetError(result);
        OgaDestroyResult(result);
        throw std::runtime_error(string);
    }
}

namespace nvigi
{
namespace onnxgenai
{
//! Returns feature Id
PluginID getFeatureId(InferenceInstanceData* data)
{
    return plugin::gpt::onnxgenai::dml::kId;
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

nvigi::Result evaluate(nvigi::InferenceExecutionContext* execCtx);

};
//! REQUIRED INTERFACE
NVIGI_PLUGIN_DEFINE("nvigi.plugin.gpt.onnxgenai", Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH), Version(API_MAJOR, API_MINOR, API_PATCH), onnxgenai, onnxgenaiContext)

//! Exported plugin specific API
//!
nvigi::Result OnnxgenaiEvaluate(nvigi::InferenceExecutionContext* execCtx)
{
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

    if (!execCtx->instance->data)
    {
        NVIGI_LOG_ERROR("GPT inference instance not correct");
        return kResultInvalidParameter;
    }

    auto& ctx = *(static_cast<nvigi::onnxgenai::InferenceContext*>(execCtx->instance->data));

    if (onnxgenai::getContext()->feature != execCtx->instance->getFeatureId(execCtx->instance->data))
    {
        NVIGI_LOG_ERROR("Invalid inference instance - expecting onnxgenai %u got %u", onnxgenai::getContext()->feature,
            execCtx->instance->getFeatureId(execCtx->instance->data));
        return kResultInvalidParameter;
    }

    // Now we are good to go!

    auto instance = (nvigi::onnxgenai::InferenceContext*)(execCtx->instance->data);

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

    // TODO validate and use these runtime parameters
    auto runtime = findStruct<GPTRuntimeParameters>(execCtx->runtimeParameters);
    if (runtime)
    {
        instance->seed = runtime->seed;
        instance->n_predict = runtime->tokensToPredict;
        instance->n_keep = runtime->tokensToKeep;
        instance->n_draft = runtime->tokensToDraft;
        instance->n_batch = runtime->batchSize;
        instance->n_chunks = runtime->numChunks;
        instance->n_parallel = runtime->numParallel;
        instance->n_sequences = runtime->numSequences;
        
        // set history max tokens allowed to be max(1024, runtime->tokensToPredict)
        // if no runtime provided, history's default is 1024 too
        size_t historyMaxTokens = runtime->tokensToPredict > 1024 ? runtime->tokensToPredict : 1024;
        instance->history.setMaxTokens(historyMaxTokens);
    }

    auto responseCallback = [&execCtx](nvigi::onnxgenai::InferenceContext* instance, const std::string& response)->nvigi::InferenceExecutionState
        {
            nvigi::InferenceExecutionState res{};

            const nvigi::InferenceDataText* output{};
            if (execCtx->outputs && execCtx->outputs->findAndValidateSlot(kGPTDataSlotResponse, &output))
            {
                auto cpuBuffer = castTo<CpuData>(output->utf8Text);
                if (!cpuBuffer || !cpuBuffer->buffer || cpuBuffer->sizeInBytes < response.size())
                {
                    return kInferenceExecutionStateInvalid;
                }
                strcpy_s((char*)cpuBuffer->buffer, cpuBuffer->sizeInBytes, response.c_str());
                res = execCtx->callback(execCtx, nvigi::kInferenceExecutionStateDone, execCtx->callbackUserData);
            }
            else
            {
                //! Temporary outputs for the callback since host did not provide any
                nvigi::CpuData text{ response.length() + 1, (void*)response.c_str() };
                nvigi::InferenceDataText data(text);
                std::vector<nvigi::InferenceDataSlot> slots = { {kGPTDataSlotResponse, &data} };
                nvigi::InferenceDataSlotArray outputs = { slots.size(), slots.data() };
                execCtx->outputs = &outputs;
                res = execCtx->callback(execCtx, nvigi::kInferenceExecutionStateDone, execCtx->callbackUserData);
                //! Clear outputs since these are all local variables
                execCtx->outputs = {};
            }

            return res;
        };

    std::string system = systemSlot ? systemSlot->getUTF8Text() : "";
    std::string user = userSlot ? userSlot->getUTF8Text() : "";
    std::string assistant = assistantSlot ? assistantSlot->getUTF8Text() : "";
    // all fields might have content
    std::string curr_prompt = system;
    if (user != "")
    {
        if (curr_prompt != "") curr_prompt += "\n";
        curr_prompt += user;
    }
    if (assistant != "")
    {
        if (curr_prompt != "") curr_prompt += "\n";
        curr_prompt += assistant;
    }
    // system prompt only, we will reset history and return empty
    bool system_mode = (curr_prompt == system);
    // chat history
    bool use_history = runtime ? runtime->interactive : false;

    // reset history in system mode
    if (!use_history || system_mode) {
        instance->history.clear();
    }
    
    // current prompt for record
    OgaSequences* curr_sequences;
    CheckResult(OgaCreateSequences(&curr_sequences));
    CheckResult(OgaTokenizerEncode(instance->tokenizer, curr_prompt.c_str(), curr_sequences));
    size_t curr_sequence_length = OgaSequencesGetSequenceCount(curr_sequences, 0);
    OgaDestroySequences(curr_sequences);

    // add system prompt to history and return empty
    if (system_mode)
    {
        ChatData currChat;
        currChat.query = system;
        currChat.n_queryTokens = curr_sequence_length;
        currChat.response = "";
        currChat.n_responseTokens = 0;
        instance->history.addChat(currChat);
        responseCallback(instance, "");
        return kResultOk;
    }

    // history prompt
    if (use_history) 
    {
        // # of tokens template elements will take, hardcoded 
        // Mistral prompt template measured to take 31 tokens 
        constexpr size_t n_templateTokens = 40; 
        size_t n_initialTokens = n_templateTokens + n_newlineTokens + curr_sequence_length;
        std::string history_prompt = instance->history.getHistoryPrompt(n_initialTokens);
        if (system != "")
        {
            system = history_prompt + "\n" + system;
        }
        else
        {
            system = history_prompt;
        }
    }
    // apply prompt template
    instance->prompt = ai::generatePrompt(instance->modelInfo, system, user, assistant);

    // TODO: execute this in async thread
    OgaSequences* input_sequences;
    CheckResult(OgaCreateSequences(&input_sequences));
    CheckResult(OgaTokenizerEncode(instance->tokenizer, instance->prompt.c_str(), input_sequences));
    size_t in_sequence_length = OgaSequencesGetSequenceCount(input_sequences, 0);
    CheckResult(OgaGeneratorParamsSetInputSequences(instance->params, input_sequences));

    // ort output sequence will also contain input in it, which taking up `max_length` spaces
    // set "max_length = in_sequence_length + tokensToPredict" to account for input 
    if (runtime)
    {
        CheckResult(OgaGeneratorParamsSetSearchNumber(instance->params, "max_length", in_sequence_length + runtime->tokensToPredict));
    }

    OgaSequences* output_sequences;
    CheckResult(OgaGenerate(instance->model, instance->params, &output_sequences));
    size_t out_sequence_length = OgaSequencesGetSequenceCount(output_sequences, 0);
    const int32_t* sequence = OgaSequencesGetSequenceData(output_sequences, 0);

    const char* out_string;
    CheckResult(OgaTokenizerDecode(instance->tokenizer, sequence, out_sequence_length, &out_string));
    std::string response = out_string;
    size_t pos = response.find(instance->prompt);

    size_t response_sequence_length =  out_sequence_length;
    if (pos != std::string::npos) 
    {
        // Erase the prompt from the output
        response.erase(pos, instance->prompt.length());
        response_sequence_length -= in_sequence_length;
    }

    // add chat to history
    if (use_history) 
    {
        ChatData currChat;
        currChat.query = curr_prompt;
        currChat.n_queryTokens = curr_sequence_length;
        currChat.response = response;
        currChat.n_responseTokens = response_sequence_length;
        instance->history.addChat(currChat);
    }
    
    responseCallback(instance, response);

    OgaDestroyString(out_string);
    OgaDestroySequences(output_sequences);
    OgaDestroySequences(input_sequences);
    
    return kResultOk;
}

nvigi::Result OnnxgenaiDestroyInstance(const nvigi::InferenceInstance* instance)
{
    if (instance)
    {
        auto gptInstance = static_cast<nvigi::onnxgenai::InferenceContext*>(instance->data);
        OgaDestroyGeneratorParams(gptInstance->params);
        OgaDestroyTokenizer(gptInstance->tokenizer);
        OgaDestroyModel(gptInstance->model);
        delete gptInstance;
        delete instance;
    }

    return nvigi::kResultOk;
}

nvigi::Result OnnxGenAICreateInstance(const nvigi::NVIGIParameter* _params, nvigi::InferenceInstance** _instance)
{
    auto common = findStruct<CommonCreationParameters>(_params);
    auto paramsGPT = findStruct<GPTCreationParameters>(_params);
    if (!common || !paramsGPT) return nvigi::kResultInvalidParameter;
    auto& params = *paramsGPT;

    if (!_instance || !common->modelGUID) return nvigi::kResultInvalidParameter;

    if (!ai::isGuid(common->modelGUID))
    {
        NVIGI_LOG_ERROR("Provided model GUID '%s' is invalid", common->modelGUID);
        return false;
    }

    *_instance = nullptr;

    auto instanceData = new nvigi::onnxgenai::InferenceContext();

    auto& ctx = (*onnxgenai::getContext());

    //TODO use background mode flags here
    auto dml_params = nvigi::findStruct<nvigi::GPTOnnxgenaiCreationParameters>((const void**)&params, 1);
    if (!dml_params)
    {
        NVIGI_LOG_ERROR("Unable to find onnxgenai creation parameters");
    }

    instanceData->GPT_params.contextSize = params.contextSize;
    instanceData->GPT_params.maxNumTokensToPredict = params.maxNumTokensToPredict;

    instanceData->modelInfo.clear();
    if (!ai::findModels(common, { "onnx", "onnx.data", "json"}, instanceData->modelInfo))
    {
        NVIGI_LOG_ERROR("unable to find models");
        return kResultInvalidParameter;
    }

    auto guidInfo = instanceData->modelInfo[common->modelGUID];
    std::string onnxpath;
    if (!ai::findFilePath(guidInfo, "model.onnx", onnxpath))
    {
        NVIGI_LOG_ERROR("Failed to find file(s) required to create onnxgenai instance");
        return kResultInvalidParameter;
    }

    // Trim down to our GUID for this instance
    if (!guidInfo.contains(nvigi::ai::kPromptTemplate))
    {
        NVIGI_LOG_ERROR("Missing `prompt_templates` in model config JSON, GUID '%s'", common->modelGUID);
        return kResultInvalidParameter;
    }
    instanceData->modelInfo = guidInfo;

    fs::path path(onnxpath);
    fs::path parentDir = path.parent_path();
    std::string parentPathStr = parentDir.string();
    const char* modelPath = parentPathStr.c_str();

    OgaCreateModel(modelPath, &instanceData->model);
    CheckResult(OgaCreateTokenizer(instanceData->model, &instanceData->tokenizer));
    CheckResult(OgaCreateGeneratorParams(instanceData->model, &instanceData->params));

    CheckResult(OgaGeneratorParamsSetSearchNumber(instanceData->params, "max_length", params.maxNumTokensToPredict));
    CheckResult(OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize(instanceData->params, 1));

    // chat history: set history maxTokens to be max_length, can be updated at runtime
    // ort generator always include input prompt in its output, so set maxTokens the same as max_length 
    instanceData->history.setMaxTokens(params.maxNumTokensToPredict);

    // Explicitly declaring that we are implementing v1
    auto instance = new InferenceInstance(kStructVersion1);
    instance->data = instanceData;
    instance->getFeatureId = onnxgenai::getFeatureId;
    instance->getInputSignature = onnxgenai::getInputSignature;
    instance->getOutputSignature = onnxgenai::getOutputSignature;
    instance->evaluate = onnxgenai::evaluate;

    *_instance = instance;

    return nvigi::kResultOk;
}

nvigi::Result OnnxgenaiGetCapsAndRequirements(nvigi::NVIGIParameter** _info, const nvigi::NVIGIParameter* _params)
{
    auto common = findStruct<CommonCreationParameters>(_params);
    auto params = findStruct<GPTCreationParameters>(_params);
    if (!common || !params) return nvigi::kResultInvalidParameter;

    static CommonCapabilitiesAndRequirements s_caps{};
    auto info = &s_caps;
    *_info = s_caps;

    auto& ctx = (*onnxgenai::getContext());
    json modelInfo;
    {
        if (!ai::findModels(common, { "onnx" }, modelInfo))
        {
            return kResultInvalidParameter;
        }
    }
    info->supportedBackends = nvigi::InferenceBackendLocations::eGPU;

    //! Must be called after we set supported backends to correctly filter models
    ai::populateCommonCapsAndRequirements(ctx.capsData, *common, *info, modelInfo);
    
    return nvigi::kResultOk;
}

//! Exception handling wrappers
//! 
//! Note that we export these via our interface
//! 
namespace onnxgenai
{
    nvigi::Result createInstance(const nvigi::NVIGIParameter* params, nvigi::InferenceInstance** instance)
    {
        NVIGI_CATCH_EXCEPTION(OnnxGenAICreateInstance(params, instance));
    }
    nvigi::Result destroyInstance(const nvigi::InferenceInstance* instance)
    {
        NVIGI_CATCH_EXCEPTION(OnnxgenaiDestroyInstance(instance));
    }
    nvigi::Result getCapsAndRequirements(nvigi::NVIGIParameter** modelInfo, const nvigi::NVIGIParameter* params)
    {
        NVIGI_CATCH_EXCEPTION(OnnxgenaiGetCapsAndRequirements(modelInfo, params));
    }
    nvigi::Result evaluate(nvigi::InferenceExecutionContext* execCtx)
    {
        NVIGI_CATCH_EXCEPTION(OnnxgenaiEvaluate(execCtx));
    }
}

Result nvigiPluginGetInfo(nvigi::framework::IFramework* framework, nvigi::plugin::PluginInfo** _info)
{
    if (!plugin::internalPluginSetup(framework)) return kResultInvalidState;

    // Internal API, we know that incoming pointer is always valid
    auto& info = plugin::getContext()->info;
    *_info = &info;

    info.id = plugin::gpt::onnxgenai::dml::kId;

    info.description = "dml backend for GPT model";
    info.author = "NVIDIA";
    info.build = GIT_BRANCH_AND_LAST_COMMIT;
    info.interfaces = { plugin::getInterfaceInfo<IGeneralPurposeTransformer>() };

    //! We can run on any OS, driver or GPU sku
    info.minDriver = {};
    info.minOS = {};
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

    auto& ctx = (*onnxgenai::getContext());

    ctx.feature = plugin::gpt::onnxgenai::dml::kId;

    ctx.api.createInstance = onnxgenai::createInstance;
    ctx.api.destroyInstance = onnxgenai::destroyInstance;
    ctx.api.getCapsAndRequirements = onnxgenai::getCapsAndRequirements;

    framework->addInterface(ctx.feature, &ctx.api, 0);

    return kResultOk;
}

//! Main exit point - shutting down our plugin
//! 
Result nvigiPluginDeregister()
{
    auto& ctx = (*onnxgenai::getContext());

    ai::freeCommonCapsAndRequirements(ctx.capsData);
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
