// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <future>

#include "source/core/nvigi.api/nvigi.h"
#include "source/core/nvigi.api/nvigi_cloud.h"
#include "source/core/nvigi.api/internal.h"
#include "source/core/nvigi.log/log.h"
#include "source/core/nvigi.exception/exception.h"
#include "source/core/nvigi.exception/exception.h"
#include "source/core/nvigi.plugin/plugin.h"
#include "source/core/nvigi.file/file.h"
#include "source/core/nvigi.extra/extra.h"
#include "source/plugins/nvigi.gpt/rest/versions.h"
#include "source/utils/nvigi.ai/ai.h"
#include "source/utils/nvigi.ai/ai_data_helpers.h"
#include "source/utils/nvigi.poll/poll.h"
#include "source/plugins/nvigi.net/net.h"
#include "source/plugins/nvigi.gpt/nvigi_gpt.h"
#include "_artifacts/gitVersion.h"

#include "source/plugins/nvigi.gpt/rest/gpt.h"

namespace nvigi
{
namespace gpt
{

struct InferenceContext
{
    poll::PollContext<nvigi::InferenceExecutionState> pollCtx;
    std::string prompt{};
    std::string token{};
    std::string url{};
    bool verboseMode{};
    bool stream{};
    json modelInfo;
    std::future<nvigi::Result> asyncJob;
};

PluginID getFeatureId(InferenceInstanceData* data)
{
    return plugin::gpt::cloud::rest::kId;
}

const nvigi::InferenceDataDescriptorArray* getInputSignature(InferenceInstanceData* data)
{
    static std::vector<InferenceDataDescriptor> slots = 
    { 
        // Mandatory
        {nvigi::kGPTDataSlotJSON,InferenceDataText::s_type, false },
        {nvigi::kGPTDataSlotUser,InferenceDataText::s_type, false },
        // Optional
        {nvigi::kGPTDataSlotSystem,InferenceDataText::s_type, true },
        {nvigi::kGPTDataSlotAssistant,InferenceDataText::s_type, true},
    };
    static InferenceDataDescriptorArray s_desc = { slots.size(), slots.data() };
    return &s_desc;
}

const nvigi::InferenceDataDescriptorArray* getOutputSignature(InferenceInstanceData* data)
{
    static std::vector<InferenceDataDescriptor> slots = 
    { 
        {nvigi::kGPTDataSlotResponse,InferenceDataText::s_type, false },
        {nvigi::kGPTDataSlotJSON,InferenceDataText::s_type, true },
    };
    static InferenceDataDescriptorArray s_desc = { slots.size(), slots.data() };
    return &s_desc;
}

struct GPTContext
{
    NVIGI_PLUGIN_CONTEXT_CREATE_DESTROY(GPTContext);

    void onCreateContext() {};
    void onDestroyContext() {};

    nvigi::net::INet* net{};

    nvigi::PluginID feature{};

    IGeneralPurposeTransformer api{};
    IPolledInferenceInterface polledApi{};

    // Caps and requirements (allocations destroyed on unregister plugin)
    json modelInfo;
    ai::CommonCapsData capsData;
    std::map<std::string, std::string*> urls;
    std::map<std::string, std::string*> jsonRequestBodies;
};

nvigi::Result evaluate(nvigi::InferenceExecutionContext* execCtx);
nvigi::Result evaluateAsync(nvigi::InferenceExecutionContext* execCtx);
}

//! Define our plugin, make sure to update version numbers in versions.h
NVIGI_PLUGIN_DEFINE("nvigi.plugin.gpt", Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH), Version(API_MAJOR, API_MINOR, API_PATCH), gpt, GPTContext)

nvigi::Result restEvaluate(nvigi::InferenceExecutionContext* execCtx);

nvigi::Result restCreateInstance(const nvigi::NVIGIParameter* _params, nvigi::InferenceInstance** _instance)
{
    auto common = findStruct<CommonCreationParameters>(_params);
    if (!common) return nvigi::kResultInvalidParameter;

    if (!_instance || !common->modelGUID) return nvigi::kResultInvalidParameter;

    *_instance = nullptr;

    auto& ctx = (*gpt::getContext());

    using namespace nvigi::gpt;

    auto restParams = findStruct<RESTParameters>(_params);
    if (!restParams)
    {
        NVIGI_LOG_ERROR("Unable to find rest parameters, please make sure to provide them");
        return kResultInvalidParameter;
    }

    if (!ctx.net)
    {
        NVIGI_LOG_ERROR("Unable to find networking interface, please make sure that 'nvigi.net' is loaded and enabled");
        return kResultInvalidState;
    }
    
    json modelInfo;
    if (!ai::findModels(common, { "json" }, modelInfo))
    {
        return kResultInvalidParameter;
    }

    if (!modelInfo[common->modelGUID].contains("request_body"))
    {
        NVIGI_LOG_ERROR("Model with GUID %s does not contain valid 'nvigi.model.config.json' - missing 'request_body' block", common->modelGUID);
        return kResultInvalidState;
    }

    std::string model = modelInfo[common->modelGUID]["request_body"]["model"];
    NVIGI_LOG_INFO("Created instance for the remote model '%s', stream %s, verbose %s", model.c_str(), restParams->useStreaming ? "yes" : "no", restParams->verboseMode ? "yes" : "no");

    auto instanceData = new nvigi::gpt::InferenceContext();
    instanceData->token = restParams->authenticationToken;
    instanceData->url = restParams->url;
    instanceData->verboseMode = restParams->verboseMode;
    instanceData->stream = restParams->getVersion() >= 2 ? restParams->useStreaming : false;

    // Explicitly declaring that we are implementing v1
    auto instance = new InferenceInstance(kStructVersion1);
    instance->data = instanceData;
    instance->getFeatureId = gpt::getFeatureId;
    instance->getInputSignature = gpt::getInputSignature;
    instance->getOutputSignature = gpt::getOutputSignature;
    instance->evaluate = gpt::evaluate;
    instance->evaluateAsync = gpt::evaluateAsync;

    *_instance = instance;

    instanceData->modelInfo = modelInfo[common->modelGUID];

    return kResultOk;
}

nvigi::Result restGetCapsAndRequirements(nvigi::NVIGIParameter** _info, const nvigi::NVIGIParameter* _params)
{
    auto common = findStruct<CommonCreationParameters>(_params);
    if (!_info || !common) return nvigi::kResultInvalidParameter;

    static CommonCapabilitiesAndRequirements s_caps{};
    static CloudCapabilities s_cloudCaps{};

    // Make sure we don't chain the same thing multiple times
    if (!findStruct< CloudCapabilities>(s_caps))
    {
        s_caps.chain(s_cloudCaps);
    }

    auto info = &s_caps;
    *_info = s_caps;

    auto& ctx = (*gpt::getContext());
    // Always find model since host can change GUID etc.
    if (!ai::findModels(common, { "json" }, ctx.modelInfo))
    {
        return kResultInvalidParameter;
    }

    // This will be set ONLY if common GUID is valid
    s_cloudCaps.url = nullptr;
    s_cloudCaps.jsonRequestBody = nullptr;

    // Model card can still be corrupted so 
    if(common->modelGUID != nullptr) try
    {
        auto modelCard = ctx.modelInfo[common->modelGUID];
        // Check if we already processed this GUID, data will be the same so don't create memory leak
        if (ctx.urls.find(common->modelGUID) == ctx.urls.end())
        {
            ctx.urls[common->modelGUID] = new std::string(modelCard["url"]);
            ctx.jsonRequestBodies[common->modelGUID] = new std::string(modelCard["request_body"].dump(2, ' ', false, json::error_handler_t::replace));
        }
        s_cloudCaps.url = ctx.urls[common->modelGUID]->c_str();
        s_cloudCaps.jsonRequestBody = ctx.jsonRequestBodies[common->modelGUID]->c_str();
    }
    catch (std::exception& e)
    {
        NVIGI_LOG_ERROR("%s", e.what());
        return kResultInvalidState;
    }

    info->supportedBackends = nvigi::InferenceBackendLocations::eCloud;

    //! Must be called after we set supported backends to correctly filter models
    ai::populateCommonCapsAndRequirements(ctx.capsData, *common, *info, ctx.modelInfo);

    return kResultOk;
}

void searchAndReplace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    // Loop until all occurrences are replaced
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Move past the last replaced part
    }
}

// Recursively search and replace placeholders in the JSON object
void replacePlaceholdersInJson(json& jsonObj, const std::string& userValue, const std::string& assistantValue, const std::string& systemValue) {
    // Iterate through the JSON object
    for (auto& element : jsonObj.items()) {
        // If the element is an object, recurse
        if (element.value().is_object()) {
            replacePlaceholdersInJson(element.value(), userValue, assistantValue, systemValue);
        }
        // If the element is an array, iterate through the array and recurse
        else if (element.value().is_array()) {
            for (auto& item : element.value()) {
                if (item.is_object() || item.is_array()) {
                    replacePlaceholdersInJson(item, userValue, assistantValue, systemValue);
                }
                else if (item.is_string()) {
                    // Replace the placeholder if it matches
                    std::string tmp = item;
                    searchAndReplace(tmp, "$user", userValue);
                    searchAndReplace(tmp, "$assistant", assistantValue);
                    searchAndReplace(tmp, "$system", systemValue);
                    item = tmp;
                    /*if (item == "$user") {
                        item = userValue;
                    }
                    else if (item == "$assistant") {
                        item = assistantValue;
                    }
                    else if (item == "$system") {
                        item = systemValue;
                    }*/
                }
            }
        }
        // If the element is a string, replace the placeholder
        else if (element.value().is_string()) 
        {
            std::string tmp = element.value();
            searchAndReplace(tmp, "$user", userValue);
            searchAndReplace(tmp, "$assistant", assistantValue);
            searchAndReplace(tmp, "$system", systemValue);
            element.value() = tmp;
            /*if (element.value() == "$user") {
                element.value() = userValue;
            }
            else if (element.value() == "$assistant") {
                element.value() = assistantValue;
            }
            else if (element.value() == "$system") {
                element.value() = systemValue;
            }*/
        }
    }
}

nvigi::Result restEvaluate(nvigi::InferenceExecutionContext* execCtx, bool async = false)
{
    auto& ctx = (*gpt::getContext());

    // Validate all inputs first
    
    if (!execCtx)
    {
        NVIGI_LOG_ERROR("Unable to find 'InferenceExecutionContext' structure in the inputs");
        return kResultInvalidParameter;
    }

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

    auto responseCallback = [&execCtx](nvigi::gpt::InferenceContext* instance, const char* response, nvigi::InferenceExecutionState state = nvigi::kInferenceExecutionStateDataPending)->nvigi::InferenceExecutionState
    {
        nvigi::InferenceExecutionState res{};

        auto runtime = findStruct<GPTRuntimeParameters>(execCtx->runtimeParameters);

        std::string responseJSONStr = "{}";
        std::string content = response;
        try
        {
            json responseJSON = json::parse(response);
            responseJSONStr = responseJSON.dump(2, ' ', false, json::error_handler_t::replace);
            auto& firstChoice = responseJSON["choices"][0];
            if (firstChoice.contains("delta"))
            {
                // streaming
                content = firstChoice["delta"]["content"];
            }
            else
            {
                // full message
                content = firstChoice["message"]["content"];
            }
            if (runtime && runtime->interactive)
            {
                instance->prompt += "\n" + content;
            }
        }
        catch (std::exception&)
        {
            // This is OK, server can return plain text not JSON
        }

        nvigi::InferenceDataText* outputResponse{};
        nvigi::InferenceDataText* outputJSON{};
        if (execCtx->outputs && 
            execCtx->outputs->findAndValidateSlot(kGPTDataSlotResponse, &outputResponse) && 
            execCtx->outputs->findAndValidateSlot(kGPTDataSlotJSON, &outputJSON))
        {
            if(!ai::updateInferenceDataText(outputResponse, content) ||
               !ai::updateInferenceDataText(outputJSON, responseJSONStr))
            {
                return nvigi::kInferenceExecutionStateInvalid;
            }
            if (execCtx->callback)
            {
                res = execCtx->callback(execCtx, state, execCtx->callbackUserData);
            }
            else
            {
                res = instance->pollCtx.triggerCallback(state);
            }
        }
        else
        {
            //! Temporary outputs for the callback since host did not provide any
            nvigi::ai::InferenceDataTextHelper responseSlot(content);
            nvigi::ai::InferenceDataTextHelper jsonSlot(responseJSONStr);
            std::vector<nvigi::InferenceDataSlot> slots = { {kGPTDataSlotResponse, responseSlot}, { kGPTDataSlotJSON, jsonSlot } };
            nvigi::InferenceDataSlotArray outputs = { slots.size(), slots.data() };
            execCtx->outputs = &outputs;
            if (execCtx->callback)
            {
                res = execCtx->callback(execCtx, state, execCtx->callbackUserData);
            }
            else
            {
                res = instance->pollCtx.triggerCallback(state);
            }
            //! Clear outputs since these are all local variables
            execCtx->outputs = {};
        }
        return res;
    };

    
    auto instance = (nvigi::gpt::InferenceContext*)(execCtx->instance->data);

    const nvigi::InferenceDataText* systemSlot{};
    const nvigi::InferenceDataText* userSlot{};
    const nvigi::InferenceDataText* assistantSlot{};
    const nvigi::InferenceDataText* jsonSlot{};
    execCtx->inputs->findAndValidateSlot(kGPTDataSlotJSON, &jsonSlot);
    execCtx->inputs->findAndValidateSlot(kGPTDataSlotSystem, &systemSlot);
    execCtx->inputs->findAndValidateSlot(kGPTDataSlotUser, &userSlot);
    execCtx->inputs->findAndValidateSlot(kGPTDataSlotAssistant, &assistantSlot);

    std::string jsonBody;
    json jsonData;

    auto runtime = findStruct<GPTRuntimeParameters>(execCtx->runtimeParameters);

    if (userSlot || systemSlot || assistantSlot)
    {
        std::string system = systemSlot ? systemSlot->getUTF8Text() : "";
        std::string user = userSlot ? userSlot->getUTF8Text() : "";
        std::string assistant = assistantSlot ? assistantSlot->getUTF8Text() : "";

        if (systemSlot)
        {
            instance->prompt = system;
        }

        if (runtime && runtime->interactive)
        {
            if (!userSlot)
            {
                // MUST always send a DONE callback before returning success, even if it is an empty string
                responseCallback(instance, "", nvigi::kInferenceExecutionStateDone);
                return kResultOk;
            }
        }

        try
        {
            jsonData = instance->modelInfo["request_body"];
            if (runtime)
            {
                //! OpenAI API
                if (jsonData.contains("temperature"))
                {
                    jsonData["temperature"] = runtime->temperature;
                }
                if (jsonData.contains("top_p"))
                {
                    jsonData["top_p"] = runtime->topP;
                }
                if (jsonData.contains("max_tokens") && runtime->tokensToPredict != -1)
                {
                    jsonData["max_tokens"] = runtime->tokensToPredict;
                }
            }
            // Override streaming with what was provided from the host in the RESTParameters (if any, false is default)
            jsonData["stream"] = instance->stream;
            replacePlaceholdersInJson(jsonData, user, assistant, instance->prompt);
            instance->prompt += "\n" + user;
            jsonBody = jsonData.dump(2, ' ', false, json::error_handler_t::replace);
        }
        catch (std::exception& e)
        {
            NVIGI_LOG_ERROR("Malformed 'nvigi.model.config.json' - exception %s", e.what());
            return kResultInvalidState;
        }
    }
    else
    {
        if (!jsonSlot)
        {
            NVIGI_LOG_ERROR("Expecting inference 'kGPTDataSlotJSON' input of type 'nvigi::InferenceDataText'");
            return kResultInvalidParameter;
        }
        instance->prompt.clear();
    
        if (runtime)
        {
            NVIGI_LOG_WARN("When using JSON input slot 'GPTRuntimeParameters' are completely ignored");
        }

        jsonBody = jsonSlot->getUTF8Text();
        // Validate incoming JSON
        try
        {
            jsonData = json::parse(jsonBody.c_str());
        }
        catch (std::exception& e)
        {
            NVIGI_LOG_ERROR("Malformed JSON input slot - exception %s", e.what());
            return kResultInvalidState;
        }
        // NOTE: No change to streaming here since JSON request body is coming directly from host and we use it as is
    }

    // Cloud inference

    ctx.net->nvcfSetToken(instance->token.c_str());
    ctx.net->setVerboseMode(instance->verboseMode);

    net::Parameters hparams;
    hparams.url = instance->url.c_str();
    hparams.headers = { "accept: application/json", "Content-Type: application/json" };

    std::string jsonObj = jsonData.dump(2, ' ', false, json::error_handler_t::replace);

    //NVIGI_LOG_HINT("%s", jsonObj.c_str());

    hparams.data.resize(jsonObj.size());
    memcpy(hparams.data.data(), jsonObj.c_str(), jsonObj.size());

    // Check if streaming is enabled, if not in JSON use default
    instance->stream = extra::getJSONValue(jsonData, "stream", instance->stream);
    
    nvigi::Result res = kResultOk;

    if (instance->stream)
    {
        // Handle streaming response
        std::string streamBuffer;
        
        // Structure to hold callback context for streaming
        struct StreamingContext {
            decltype(responseCallback)* originalCallback;
            nvigi::gpt::InferenceContext* instance;
            std::string* streamBuffer;
        } streamingCtx = { &responseCallback, instance, &streamBuffer };
        
        auto streamingCallback = [](const char* data, size_t size, void* userdata) -> size_t {
            auto* ctx = static_cast<StreamingContext*>(userdata);
            ctx->streamBuffer->append(data, size);
            
            // Process Server-Sent Events (SSE) format
            size_t pos = 0;
            while ((pos = ctx->streamBuffer->find('\n')) != std::string::npos)
            {
                std::string line = ctx->streamBuffer->substr(0, pos);
                ctx->streamBuffer->erase(0, pos + 1);
                
                // Remove \r if present
                if (!line.empty() && line.back() == '\r')
                {
                    line.pop_back();
                }
                
                // Skip empty lines and comments
                if (line.empty() || line[0] == ':')
                {
                    continue;
                }
                
                // Parse SSE data lines
                if (line.substr(0, 6) == "data: ")
                {
                    std::string eventData = line.substr(6);
                    
                    // Check for end of stream
                    if (eventData == "[DONE]")
                    {
                        (*ctx->originalCallback)(ctx->instance, "", nvigi::kInferenceExecutionStateDone);
                        return size;
                    }
                    
                    try
                    {
                        json chunkJson = json::parse(eventData);
                        if (chunkJson.contains("choices") && !chunkJson["choices"].empty())
                        {
                            auto& choice = chunkJson["choices"][0];
                            if (choice.contains("delta") && choice["delta"].contains("content"))
                            {
                                if (choice["delta"]["content"].is_string()) {
                                    std::string content = choice["delta"]["content"];
                                    (*ctx->originalCallback)(ctx->instance, chunkJson.dump().c_str(), nvigi::kInferenceExecutionStateDataPending);
                                }
                            }
                        }
                    }
                    catch (std::exception& e)
                    {
                        NVIGI_LOG_WARN("Failed to parse streaming JSON chunk: %s", e.what());
                    }
                }
            }
            
            return size;
        };
        
        res = ctx.net->nvcfPostStreaming(hparams, streamingCallback, &streamingCtx);
    }
    else
    {
        // Handle non-streaming response (original code)
        json responseAsJSON;
        types::string response;
        res = ctx.net->nvcfPost(hparams, response);
        if (res == kResultOk)
        {
            responseCallback(instance, response.c_str(), nvigi::kInferenceExecutionStateDone);
        }
    }

    if (res != kResultOk)
    {
        return nvigi::kResultInvalidState;
    }

    return kResultOk;
}

nvigi::Result flushAsyncJob(nvigi::gpt::InferenceContext* instance)
{
    if (instance->asyncJob.valid())
    {
        // Release any results that are pending
        if (instance->pollCtx.checkResultPending())
        {
            instance->pollCtx.releaseResults(nvigi::kInferenceExecutionStateDone);
        }
        // Wait for the job with timeout
        auto result = instance->asyncJob.wait_for(std::chrono::seconds(10));
        if (result == std::future_status::ready)
        {
            if (NVIGI_FAILED(result, instance->asyncJob.get()))
            {
                return result;
            }
        }
        else
        {
            NVIGI_LOG_WARN("Async job timed out.");
            return kResultTimedOut;
        }
    }
    return kResultOk;
}

nvigi::Result restEvaluateAsync(nvigi::InferenceExecutionContext* execCtx)
{
    // For REST/cloud inference, we can run the evaluation asynchronously
    // The actual HTTP call will be synchronous but we run it in a separate thread
    auto instance = (nvigi::gpt::InferenceContext*)(execCtx->instance->data);
    
    // If there's already an async job running, wait for it to complete
    if (NVIGI_FAILED(res, flushAsyncJob(instance)))
    {
        return res;
    }
    
    // Launch async evaluation
    instance->asyncJob = std::async(std::launch::async, [execCtx]() -> nvigi::Result {
        return restEvaluate(execCtx, true);
    });
    
    return kResultOk;
}

nvigi::Result restDestroyInstance(const nvigi::InferenceInstance* instance)
{
    if (instance)
    {
        auto gptInstance = static_cast<nvigi::gpt::InferenceContext*>(instance->data);
        // If there's already an async job running, wait for it to complete
        if (NVIGI_FAILED(res, flushAsyncJob(gptInstance)))
        {
            return res;
        }
        delete gptInstance;
        delete instance;
    }
    return nvigi::kResultOk;
}

nvigi::Result restGetResults(nvigi::InferenceExecutionContext* execCtx, bool wait, nvigi::InferenceExecutionState* state)
{
    if (!execCtx || !execCtx->instance)
        return kResultInvalidParameter;

    auto instance = static_cast<gpt::InferenceContext*>(execCtx->instance->data);
    return instance->pollCtx.getResults(wait, state);
}

nvigi::Result restReleaseResults(nvigi::InferenceExecutionContext* execCtx, nvigi::InferenceExecutionState state)
{
    if (!execCtx || !execCtx->instance)
        return kResultInvalidParameter;

    auto instance = static_cast<gpt::InferenceContext*>(execCtx->instance->data);
    return instance->pollCtx.releaseResults(state);
}

//! Exception handling wrappers
//! 
//! Note that we export these via our interface
//! 
namespace gpt
{
nvigi::Result createInstance(const nvigi::NVIGIParameter* params, nvigi::InferenceInstance** instance)
{
    NVIGI_CATCH_EXCEPTION(restCreateInstance(params, instance));
}
nvigi::Result destroyInstance(const nvigi::InferenceInstance* instance)
{
    NVIGI_CATCH_EXCEPTION(restDestroyInstance(instance));
}
nvigi::Result getCapsAndRequirements(nvigi::NVIGIParameter** modelInfo, const nvigi::NVIGIParameter* params)
{
    NVIGI_CATCH_EXCEPTION(restGetCapsAndRequirements(modelInfo, params));
}
nvigi::Result evaluate(nvigi::InferenceExecutionContext* execCtx)
{
    NVIGI_CATCH_EXCEPTION(restEvaluate(execCtx, false));
}
nvigi::Result evaluateAsync(nvigi::InferenceExecutionContext* execCtx)
{
    NVIGI_CATCH_EXCEPTION(restEvaluateAsync(execCtx));
}
nvigi::Result getResults(nvigi::InferenceExecutionContext* execCtx, bool wait, nvigi::InferenceExecutionState* state)
{
    NVIGI_CATCH_EXCEPTION(restGetResults(execCtx, wait, state));
}
nvigi::Result releaseResults(nvigi::InferenceExecutionContext* execCtx, nvigi::InferenceExecutionState state)
{
    NVIGI_CATCH_EXCEPTION(restReleaseResults(execCtx, state));
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
    info.description = "Cloud REST backend for GPT model";
    info.author = "NVIDIA";
    info.build = GIT_BRANCH_AND_LAST_COMMIT;
    info.interfaces = { plugin::getInterfaceInfo<IGeneralPurposeTransformer>()};

    //! Defaults indicate no restrictions - plugin can run on any system, even without any adapter
    info.requiredVendor = nvigi::VendorId::eNone;
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

    auto& ctx = (*gpt::getContext());

    ctx.feature = gpt::getFeatureId(nullptr);

    if (!framework::getInterface(framework, plugin::net::kId, &ctx.net))
    {
        NVIGI_LOG_ERROR("Missing interface from 'nvigi.plugin.net'");
        return kResultInvalidState;
    }

    ctx.api.createInstance = gpt::createInstance;
    ctx.api.destroyInstance = gpt::destroyInstance;
    ctx.api.getCapsAndRequirements = gpt::getCapsAndRequirements;
 
    framework->addInterface(ctx.feature, &ctx.api, 0);

    // Add polled interface
    ctx.polledApi.getResults = gpt::getResults;
    ctx.polledApi.releaseResults = gpt::releaseResults;
    framework->addInterface(ctx.feature, &ctx.polledApi, 0);

    return kResultOk;
}

//! Main exit point - shutting down our plugin
//! 
Result nvigiPluginDeregister()
{
    auto& ctx = (*gpt::getContext());

    ai::freeCommonCapsAndRequirements(ctx.capsData);

    framework::releaseInterface(plugin::getContext()->framework, plugin::net::kId, ctx.net);
    ctx.net = nullptr;

    // Cleanup cloud caps data
    for (auto& [key, value] : ctx.urls)
    {
        delete value;
    }
    for (auto& [key, value] : ctx.jsonRequestBodies)
    {
        delete value;
    }

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