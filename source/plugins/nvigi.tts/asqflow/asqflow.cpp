// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "source/plugins/nvigi.tts/asqflow/asqflow.h"
#include "../../../../nvigi_core/source/core/nvigi.system/system.h"
#include "_artifacts/gitVersion.h"
#include "source/core/nvigi.api/nvigi.h"
#include "source/core/nvigi.exception/exception.h"
#include "source/core/nvigi.extra/extra.h"
#include "source/core/nvigi.log/log.h"
#include "source/core/nvigi.plugin/plugin.h"
#include "source/core/nvigi.system/system.h"
#include "source/plugins/nvigi.tts/asqflow/asqflow_inference.h"
#include "source/plugins/nvigi.tts/asqflow/versions.h"
#include "source/plugins/nvigi.tts/nvigi_tts.h"
#include "source/utils/nvigi.ai/ai.h"
#include "source/utils/nvigi.hwi/cuda/runtime_context_scope.h"
#include "nvigi_stl_helpers.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <future>
#include <iosfwd>
#include <memory.h>
#include <mutex>
#include <nvigi_cuda.h>
#include <nvigi_result.h>
#include <nvigi_struct.h>
#include <string.h>
#include <string>
#include <vector>

#define TRT_USE_CIG // CiG on

#ifdef TRT_USE_CIG
#include "source/utils/nvigi.hwi/cuda/push_poppable_cuda_context.h"
#endif

namespace nvigi
{

namespace asqflow
{

// Blackwell arch id from https://github.com/NVIDIA/nvapi/blob/main/nvapi.h#L2729
constexpr uint32_t NV_GPU_ARCHITECTURE_GB200 = 0x000001B0;

constexpr int DEFAULT_GRAPHICS_ADAPTER_INDEX = 0;

static bool isBlackwell()
{
    using namespace nvigi::system;

    ISystem* const system = nvigi::system::getInterface();
    if (system)
    {
        const SystemCaps* const sysCaps = system->getSystemCaps();
        if (sysCaps && sysCaps->adapterCount > 0u)
        {
            Adapter* const adapter = sysCaps->adapters[DEFAULT_GRAPHICS_ADAPTER_INDEX];
            if (adapter)
            {
                NVIGI_LOG_INFO("Adapter architecture ID = 0x%x", adapter->architecture);
                if (adapter->architecture == NV_GPU_ARCHITECTURE_GB200)
                {
                    return true;
                }
            }
        }
    }
    return false;
}

struct InferenceContextSync
{
    std::promise<nvigi::Result> startedPromise;
    std::future<nvigi::Result> job;
    std::mutex mtx;
    std::atomic<bool> running = true;
    nvigi::InferenceExecutionContext *execCtx;
};

struct InferenceContext
{
    InferenceContext(const nvigi::NVIGIParameter *params) : cudaContext(params)
    {
    }

    json modelInfo;
    ASqFlow *asqflow{};
    PromptsBuffer promptsToProcess;

    InferenceContextSync sync{};
    InferenceParameters params;

    std::vector<cudaStream_t> cudaStreams{};

    // InferenceParamsASqFlow params;
// Even when using DML, embedding is using CUDA
#ifdef TRT_USE_CIG
    // Use PushPoppableCudaContext defined in push_poppable_cuda_context.h
#else
    // Use dummy PushPoppableCudaContext to avoid depending on CUDA
    struct PushPoppableCudaContext
    {
        bool constructorSucceeded = true;
        PushPoppableCudaContext(const nvigi::NVIGIParameter *params)
        {
        }
        void pushRuntimeContext()
        {
        }
        void popRuntimeContext()
        {
        }
    };
#endif
    PushPoppableCudaContext cudaContext;
};

PluginID getFeatureId(InferenceInstanceData *data)
{
#if USE_TRT
    return plugin::tts::asqflow::trt::kId;
#else
    return plugin::tts::asqflow::cpu::kId;
#endif
}

const nvigi::InferenceDataDescriptorArray *getInputSignature(InferenceInstanceData *data)
{
    static std::vector<InferenceDataDescriptor> slots = {
        {nvigi::kTTSDataSlotInputText, InferenceDataText::s_type, false},
        {nvigi::kTTSDataSlotInputTargetSpectrogramPath, InferenceDataText::s_type, false}};
    static InferenceDataDescriptorArray s_desc = {slots.size(), slots.data()};
    return &s_desc;
}

const nvigi::InferenceDataDescriptorArray *getOutputSignature(InferenceInstanceData *data)
{
    static std::vector<InferenceDataDescriptor> slots = {
        {nvigi::kTTSDataSlotOutputAudio, InferenceDataAudio::s_type, false},
        {nvigi::kTTSDataSlotOutputTextNormalized, InferenceDataText::s_type, true}};
    static InferenceDataDescriptorArray s_desc = {slots.size(), slots.data()};
    return &s_desc;
}

//! Our common context
//!
//! Here we can keep whatever global state we need
//!
struct ASqFlowContext
{
    NVIGI_PLUGIN_CONTEXT_CREATE_DESTROY(ASqFlowContext);

    // Called when plugin is loaded, do any custom constructor initialization here
    void onCreateContext() {
        //! For example
        //!
        //! onHeap = new std::map<...>
    };

    // Called when plugin is unloaded, destroy any objects on heap here
    void onDestroyContext() {
        //! For example
        //!
        //! delete onHeap;
    };

    nvigi::PluginID feature{};

    // For example, interface we will export
    // ITemplateInfer api{};
    ITextToSpeech api{};

    // NOTE: No instance promptData should be here
    std::atomic<bool> initialized = false;

    // Caps and requirements
    json modelInfo;
    ai::CommonCapsData capsData;

    nvigi::IHWICuda* icig{};
};

struct MyInstanceData
{
    // All per instance promptData goes here
};

nvigi::Result evaluate(nvigi::InferenceExecutionContext *execCtx);
nvigi::Result evaluateAsync(nvigi::InferenceExecutionContext *execCtx);

} // namespace asqflow

//! Define our plugin
//!
//! IMPORTANT: Make sure to place this macro right after the context declaration and always within the 'nvigi' namespace
//! ONLY.
NVIGI_PLUGIN_DEFINE("nvigi.plugin.tts.asqflow", Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH),
                    Version(API_MAJOR, API_MINOR, API_PATCH), asqflow, ASqFlowContext)

//! Example interface implementation
//!
nvigi::Result asqflowCreateInstance(const nvigi::NVIGIParameter *_params, nvigi::InferenceInstance **_instance)
{
    ////! An example showing how to obtain an optional chained parameters
    ////!
    ////! User can chain extra structure(s) using the params._next
    ////!
    // auto extraParams = findStruct<nvigi::TTSCreationParameters>(params);
    // if (extraParams)
    //{
    //     //! User provided extra parameters!
    // }
    auto common = findStruct<CommonCreationParameters>(_params);
    auto creationParams = findStruct<TTSCreationParameters>(_params);
    auto asqfCreationParams = findStruct<TTSASqFlowCreationParameters>(_params);
    if (!creationParams || !common)
    {
        NVIGI_LOG_ERROR("Missing required parameters");
        return nvigi::kResultInvalidParameter;
    }
    auto& params = *creationParams;

	// Use default values if TTSASqFlowCreationParameters is not provided
    if (!asqfCreationParams) {
		asqfCreationParams = new TTSASqFlowCreationParameters();
    }

    if (!_instance || !common->utf8PathToModels || !common->modelGUID)
    {
        NVIGI_LOG_ERROR("Missing instance pointer or pathToModels or modelGUID");
        return nvigi::kResultInvalidParameter;
    }

    if (!ai::isGuid(common->modelGUID))
    {
        NVIGI_LOG_ERROR("Provided model GUID '%s' is invalid", common->modelGUID);
        return kResultInvalidParameter;
    }

    *_instance = nullptr;

    nvigi::asqflow::InferenceContext* instanceData = nullptr;
    // CIG (CUDA In Graphics) is currently not supported with TensorRT on Blackwell architecture.
    // Ensure the CIG context is not used when running on Blackwell.
    if (nvigi::asqflow::isBlackwell())
    {
        nvigi::TTSCreationParameters cloneCreationParams{};
        cloneCreationParams.warmUpModels = creationParams->warmUpModels;

        nvigi::CommonCreationParameters cloneCommon{};
        cloneCommon.modelGUID = common->modelGUID;
        cloneCommon.numThreads = common->numThreads;
        cloneCommon.utf8PathToAdditionalModels = common->utf8PathToAdditionalModels;
        cloneCommon.utf8PathToModels = common->utf8PathToModels;
        cloneCommon.vramBudgetMB = common->vramBudgetMB;
        cloneCreationParams.chain(cloneCommon);

        nvigi::TTSASqFlowCreationParameters cloneAsqfCreationParams{};
        cloneAsqfCreationParams.extendedPhonemesDictPath = asqfCreationParams->extendedPhonemesDictPath;
        cloneCreationParams.chain(cloneAsqfCreationParams);

        auto cudaParams = findStruct<nvigi::CudaParameters>(_params);
        auto d3d12Parameters = findStruct<nvigi::D3D12Parameters>(_params);
        if (cudaParams)
        {
            nvigi::CudaParameters cloneCudaParams{};
            cloneCudaParams.device = cudaParams->device;
            cloneCudaParams.context = cudaParams->context;
            cloneCudaParams.stream = cudaParams->stream;
            cloneCudaParams.cudaMallocReportCallback = cudaParams->cudaMallocReportCallback;
            cloneCudaParams.cudaMallocReportUserContext = cudaParams->cudaMallocReportUserContext;
            cloneCudaParams.cudaFreeReportCallback = cudaParams->cudaFreeReportCallback;
            cloneCudaParams.cudaFreeReportUserContext = cudaParams->cudaFreeReportUserContext;
            cloneCudaParams.cudaMallocCallback = cudaParams->cudaMallocCallback;
            cloneCudaParams.cudaMallocUserContext = cudaParams->cudaMallocUserContext;
            cloneCudaParams.cudaFreeCallback = cudaParams->cudaFreeCallback;
            cloneCudaParams.cudaFreeUserContext = cudaParams->cudaFreeUserContext;
            cloneCreationParams.chain(cloneCudaParams);
        }
        if (d3d12Parameters) {
            NVIGI_LOG_WARN("Current plugin version is not compatible with CIG on Blackwell. D3d12Parameters not used.");
        }

        instanceData = new nvigi::asqflow::InferenceContext(cloneCreationParams);
    }
    else
    {
        instanceData = new nvigi::asqflow::InferenceContext(_params);
    }
    if (!instanceData)
    {
        return kResultInvalidState;
    }

    if (!instanceData->cudaContext.constructorSucceeded)
        return kResultInvalidState;

    auto &ctx = (*asqflow::getContext());

    using namespace nvigi::asqflow;

    int n_gpu_layers = -1;
#if USE_TRT
#ifndef NVIGI_PRODUCTION
    size_t currentUsageMB{};
    extra::ScopedTasks vram(
        [&currentUsageMB]() {
            system::VRAMUsage* usage;
            system::getInterface()->getVRAMStats(0, &usage);
            currentUsageMB = usage->currentUsageMB;
        },
        [&currentUsageMB]() {
            system::VRAMUsage* usage;
            system::getInterface()->getVRAMStats(0, &usage);
            currentUsageMB = usage->currentUsageMB - currentUsageMB;
            NVIGI_LOG_INFO("New instance using %lluMB budget %lluMB", currentUsageMB, usage->budgetMB);
        });
#endif
    n_gpu_layers = INT_MAX;
    if (common->numThreads > 1)
    {
        NVIGI_LOG_WARN("For optimal performance when using CUDA only one CPU thread is used");
    }
#else
#endif

    {
        // Phonemes dictionary for GraphemeToPhoneme is stored under config folder
        json modelInfoConfig;
        nvigi::CommonCreationParameters commonConfig = *common;

        // Find the three onnx models needed (base model, converter and g2p model)
        if (!ai::findModels(common, { "onnx", "txt", "engine" }, instanceData->modelInfo))
        {
            return kResultInvalidParameter;
        }

        std::string pathDpModel{}, pathGeneratorModel, pathVocoderModel{}, pathg2pModel{};
        std::string pathCMUDict{};

        try
        {
            // Trim down to our GUID for this instance
            instanceData->modelInfo = instanceData->modelInfo[common->modelGUID];
            if (n_gpu_layers > 0)
            {
                size_t neededVRAM = instanceData->modelInfo["vram"];
                if (common->vramBudgetMB < neededVRAM)
                {
                    NVIGI_LOG_WARN("Provided VRAM %uMB is insufficient, required VRAM is %uMB", common->vramBudgetMB,
                        neededVRAM);
                    return kResultInsufficientResources;
                }
            }

            // Find paths for : Base Model, Converter Model, G2P model
            std::vector<std::string> filesModels = instanceData->modelInfo["onnx"];
            std::vector<std::string> filesModelsTrt = instanceData->modelInfo["engine"];
            if (filesModels.empty())
            {
                NVIGI_LOG_ERROR("Failed to find model in the expected directory '%s'", common->utf8PathToModels);
                return kResultInvalidParameter;
            }

            std::string parentFolder = std::filesystem::path(filesModels[0]).parent_path().string();
            for (const std::string& file : filesModels)
            {
                if (file.find("G2P") != std::string::npos)
                {
                    if (pathg2pModel != "")
                    {
                        NVIGI_LOG_ERROR("Multiple G2P models have been found in the directory '%s'",
                            parentFolder.c_str());
                        return kResultInvalidParameter;
                    }
                    pathg2pModel = file;
                }
            }

            for (const std::string& file : filesModelsTrt)
            {
                if (file.find("DpModel") != std::string::npos)
                {
                    if (pathDpModel != "")
                    {
                        NVIGI_LOG_ERROR("Multiple Duration predictor models have been found in the directory '%s'",
                            parentFolder.c_str());
                        return kResultInvalidParameter;
                    }
                    pathDpModel = file;
                }
                else if (file.find("VocoderModel") != std::string::npos)
                {
                    if (pathVocoderModel != "")
                    {
                        NVIGI_LOG_ERROR("Multiple vocoders models have been found in the directory '%s'",
                            parentFolder.c_str());
                        return kResultInvalidParameter;
                    }
                    pathVocoderModel = file;
                }
                else if (file.find("GeneratorModel") != std::string::npos)
                {
                    if (pathGeneratorModel != "")
                    {
                        NVIGI_LOG_ERROR("Multiple generators models have been found in the directory '%s'",
                            parentFolder.c_str());
                        return kResultInvalidParameter;
                    }
                    pathGeneratorModel = file;
                }
            }

            if (pathDpModel == "")
            {
                NVIGI_LOG_ERROR("GP have not been found in the directory '%s'", parentFolder.c_str());
                return kResultInvalidParameter;
            }
            else if (pathGeneratorModel == "")
            {
                NVIGI_LOG_ERROR("Generator model have not been found in the directory '%s'", parentFolder.c_str());
                return kResultInvalidParameter;
            }
            else if (pathVocoderModel == "")
            {
                NVIGI_LOG_ERROR("Generator model have not been found in the directory '%s'", parentFolder.c_str());
                return kResultInvalidParameter;
            }
            else if (pathg2pModel == "")
            {
                NVIGI_LOG_ERROR("G2P model have not been found in the directory '%s'", parentFolder.c_str());
                return kResultInvalidParameter;
            }

            // Find the path for CMU dictionnary (in charge of converting grapheme to phonemes
            std::vector<std::string> filesDicts;
            const std::string nameDictPhonemes = "ipa_dict_phonemized.txt";


            filesDicts = instanceData->modelInfo["txt"];
            for (const std::string& file : filesDicts)
            {
                if (file.find(nameDictPhonemes) != std::string::npos)
                {
                    pathCMUDict = file;
                }
            }

            if (pathCMUDict == "")
            {
                NVIGI_LOG_ERROR("%s have not been found in the expected directory '%s'", nameDictPhonemes.c_str(),
                    parentFolder.c_str());
                return kResultInvalidParameter;
            }
        }
        catch (std::exception& e)
        {
            NVIGI_LOG_ERROR("%s", e.what());
            return kResultInvalidState;
        }

        NVIGI_LOG_INFO("Instantiating asqflow");
        NVIGI_LOG_INFO("Dp model : %s", pathDpModel.c_str());
        NVIGI_LOG_INFO("Generator model : %s", pathGeneratorModel.c_str());
        NVIGI_LOG_INFO("Vocoder model : %s", pathVocoderModel.c_str());
        NVIGI_LOG_INFO("G2P model : %s", pathg2pModel.c_str());
        {
            nvigi::RuntimeContextScope scope(*instanceData);
            CreationParameters paramsASqFlow;
            paramsASqFlow.warmUpModels = params.warmUpModels;
			paramsASqFlow.extendedPhonemesDictPath = asqfCreationParams->extendedPhonemesDictPath;
            try
            {
                instanceData->asqflow = new ASqFlow(pathDpModel, pathGeneratorModel, pathVocoderModel, pathg2pModel,
                                                  pathCMUDict, instanceData->modelInfo, paramsASqFlow);
            }
            catch (std::exception& e)
            {
                NVIGI_LOG_ERROR("Error while creating asqflow instance : %s", e.what());
                return kResultInvalidState;
            }
        }
    }

    {
        // We need RuntimeContextScope here to cause the streams to be created on the CIG context
        RuntimeContextScope cigScope(*instanceData);

        // We have to make one main stream, plus the max of the stream counts over all the models in AsqFlow
        int32_t auxStreamCount = instanceData->asqflow->getMaxAuxStreamCount();
        uint32_t streamCount = 1 + std::max(auxStreamCount, 0);

        for (uint32_t i = 0; i != streamCount; i++)
        {
            cudaStream_t stream;
            cudaError_t err = cudaStreamCreate(&stream);

            if (err == cudaSuccess)
            {
                instanceData->cudaStreams.push_back(stream);
            }
            else
            {
                NVIGI_LOG_ERROR("Failed to create CUDA stream: %s", cudaGetErrorString(err));
                for (auto createdStream : instanceData->cudaStreams)
                {
                    cudaStreamDestroy(createdStream);
                }
                return kResultInvalidState;
            }
        }

        // Apply the global priority to all streams
        if (ctx.icig->getVersion() >= 2)
        {
            nvigi::Result cuerr = ctx.icig->cudaApplyGlobalGpuInferenceSchedulingMode(instanceData->cudaStreams.data(), instanceData->cudaStreams.size());
            if (cuerr != kResultOk)
            {
                NVIGI_LOG_WARN_ONCE("Could not set relative priority of compute and graphics. Please use 575 driver or higher\n");
            }
        }

        // Tell TensorRT to use the streams we made
        instanceData->asqflow->setAuxStreams(instanceData->cudaStreams.data()+1, instanceData->cudaStreams.size()-1);
    }

    // Explicitly declaring that we are implementing v1
    auto instance = new InferenceInstance(kStructVersion1);
    instance->data = instanceData;
    instance->getFeatureId = asqflow::getFeatureId;
    instance->getInputSignature = asqflow::getInputSignature;
    instance->getOutputSignature = asqflow::getOutputSignature;
    instance->evaluate = asqflow::evaluate;
    instance->evaluateAsync = asqflow::evaluateAsync;

    *_instance = instance;

    return kResultOk;
}

nvigi::Result asqflowGetCapsAndRequirements(nvigi::NVIGIParameter **_info, const nvigi::NVIGIParameter *_params)
{
    // population of the CapabilitiesAndRequirements is doing a promptData oriented population of
    // ALL models supported by this plugin.  It is encumbent on the user to iterate over these vectors
    // find the model they are interested in and the corresponding caps and requirements for that model.

    auto common = findStruct<CommonCreationParameters>(_params);
    if (!common)
    {
        return nvigi::kResultInvalidParameter;
    }

    auto params = castTo<TTSCreationParameters>(_params);
    if (!_info)
        return nvigi::kResultInvalidParameter;

    static CommonCapabilitiesAndRequirements s_capsCommon{};
    static TTSCapabilitiesAndRequirements s_caps{};
    s_caps.common = &s_capsCommon;
    auto info = &s_caps;
    *_info = s_caps;

    auto &ctx = (*asqflow::getContext());

    if (ctx.modelInfo.empty())
    {
        if (!ai::findModels(common, {"onnx"}, ctx.modelInfo))
        {
            return kResultInvalidParameter;
        }
    }

#if USE_TRT
    info->common->supportedBackends = nvigi::InferenceBackendLocations::eGPU;
#else
    info->common->supportedBackends = nvigi::InferenceBackendLocations::eCPU;
#endif
    //! Must be called after we set supported backends to correctly filter models
    ai::populateCommonCapsAndRequirements(ctx.capsData, *common, *info->common, ctx.modelInfo);

    return kResultOk;
}

nvigi::Result asqflowEvaluate(nvigi::InferenceExecutionContext *execCtx, bool async = false)
{
    auto &ctx = (*asqflow::getContext());
    // Validate all inputs first

    if (!execCtx)
    {
        NVIGI_LOG_ERROR("Unable to find 'InferenceExecutionContext' structure in the inputs");
        return kResultInvalidParameter;
    }

    if (!execCtx->callback)
    {
        NVIGI_LOG_ERROR("ASqFlow inference callback not provided");
        return kResultInvalidParameter;
    }

    if (!execCtx->instance)
    {
        NVIGI_LOG_ERROR("ASqFlow inference instance not provided");
        return kResultInvalidParameter;
    }

    if (ctx.feature != execCtx->instance->getFeatureId(execCtx->instance->data))
    {
        NVIGI_LOG_ERROR("Invalid inference instance - expecting Embed %u got %u", ctx.feature,
                        execCtx->instance->getFeatureId(execCtx->instance->data));
        return kResultInvalidParameter;
    }

    using namespace nvigi::asqflow;
    auto instance = static_cast<nvigi::asqflow::InferenceContext *>(execCtx->instance->data);

    const nvigi::InferenceDataText *promptData{};
    if (!execCtx->inputs->findAndValidateSlot(kTTSDataSlotInputText, &promptData))
    {
        NVIGI_LOG_ERROR("Expecting single inference input of type 'sl::InferenceDataText'");
        return kResultInvalidParameter;
    }

    const nvigi::InferenceDataText *targetSpecData{};
    if (!execCtx->inputs->findAndValidateSlot(kTTSDataSlotInputTargetSpectrogramPath, &targetSpecData))
    {
        NVIGI_LOG_ERROR("Expecting single inference input of type 'sl::InferenceDataText'");
        return kResultInvalidParameter;
    }
    instance->params.targetSpectrogramPath = (const char *)targetSpecData->getUTF8Text();
    std::string basenameSpectogram = std::filesystem::path(instance->params.targetSpectrogramPath).stem().string();
    std::string parentFolder = std::filesystem::path(instance->params.targetSpectrogramPath).parent_path().string();
    std::string pathTranscripts = parentFolder + "/transcripts.json";

    auto runtime = findStruct<TTSASqFlowRuntimeParameters>(execCtx->runtimeParameters);
    if (runtime)
    {
        instance->params.speed = std::max(std::min(runtime->speed, 1.5f), 0.5f);
    }

    // Get Transcript data
    // It expected to find inside the same folder, the file transcripts.json which contains the transcript of the audio

    // Check if the transcripts file
    {
        std::ifstream transcriptFile(pathTranscripts);
        if (!transcriptFile.is_open())
        {
            NVIGI_LOG_ERROR("transcripts.json has not be find at '%s", pathTranscripts.c_str());
            return kResultInvalidParameter;
        }

        // Parse the JSON data
        json jsonData;
        try
        {
            transcriptFile >> jsonData;
        }
        catch (const json::parse_error &e)
        {
            NVIGI_LOG_ERROR("Failed parsing transcripts.json : '%s", e.what());
            return kResultInvalidParameter;
        }
        if (!jsonData.contains(basenameSpectogram))
        {
            NVIGI_LOG_ERROR("transcripts.json does not contains transcript for '%s' reference",
                            basenameSpectogram.c_str());
            return kResultInvalidParameter;
        }
        instance->params.transcriptTarget = jsonData.at(basenameSpectogram);
    }

    // We apply the global scheduling mode to our streams at every evaluate() 
    // to reflect any changes the user made to the global mode between calls

    if (ctx.icig->getVersion() >= 2)
    {
        nvigi::Result err = ctx.icig->cudaApplyGlobalGpuInferenceSchedulingMode(instance->cudaStreams.data(), instance->cudaStreams.size());
        if (err != kResultOk)
        {
            NVIGI_LOG_WARN_ONCE("Could not set relative priority of compute and graphics, insufficient driver\n");
        }
    }

    auto asqflowReturnOutputAudio =
        [execCtx, instance](const std::vector<int16_t> &audio, const std::string &textNormalized,
                            nvigi::InferenceExecutionState state =
                                nvigi::kInferenceExecutionStateDataPending) -> nvigi::InferenceExecutionState {
        // check for error
        if (state == nvigi::kInferenceExecutionStateInvalid)
        {
            return execCtx->callback(execCtx, state, execCtx->callbackUserData);
        }

#ifndef NVIGI_PRODUCTION
        if (state == nvigi::kInferenceExecutionStateDone)
        {
            for (const auto &pair : instance->asqflow->timers)
            {
                NVIGI_LOG_INFO("Step: %s, Time: %lld ms\n", pair.first.c_str(), pair.second);
            }
        }
#endif

        nvigi::InferenceExecutionState res{};

        const nvigi::InferenceDataByteArray *outputAudio{};
        std::vector<nvigi::InferenceDataSlot> slots;
        nvigi::InferenceDataByteArraySTLHelper dataAudio;
        // Output Audio
        if (execCtx->outputs && execCtx->outputs->findAndValidateSlot(kTTSDataSlotOutputAudio, &outputAudio))
        {
            CpuData *cpuBuffer = castTo<CpuData>(outputAudio->bytes);
            if (cpuBuffer->buffer == nullptr || cpuBuffer->sizeInBytes < audio.size() * sizeof(int16_t))
            {
                return nvigi::kInferenceExecutionStateInvalid;
            }
            memcpy_s((float *)(cpuBuffer->buffer), cpuBuffer->sizeInBytes, &(audio[0]), audio.size() * sizeof(float));
            // res = execCtx->callback(execCtx, state, execCtx->callbackUserData);
        }
        else
        {
            //! Temporary outputs for the callback since host did not provide any
            // The size is multiplied by 2 since int16 represents two bytes
            dataAudio = nvigi::InferenceDataByteArraySTLHelper((const uint8_t*)audio.data(), audio.size() * 2);
            slots.push_back(nvigi::InferenceDataSlot(kTTSDataSlotOutputAudio, dataAudio));
        }

        // Output Text Normalized. TODO : Send it only when requested
        const nvigi::InferenceDataText *outputTextNormalized{};
        nvigi::InferenceDataTextSTLHelper dataTextNormalized;
        if (execCtx->outputs &&
            execCtx->outputs->findAndValidateSlot(nvigi::kTTSDataSlotOutputTextNormalized, &outputTextNormalized))
        {
            auto cpuBuffer = castTo<CpuData>(outputTextNormalized->utf8Text);
            if (cpuBuffer->buffer && cpuBuffer->sizeInBytes >= textNormalized.size())
            {
                strcpy_s((char *)cpuBuffer->buffer, cpuBuffer->sizeInBytes, textNormalized.c_str());
            }
        }
        else
        {
            //! Temporary outputs for the callback since host did not provide any
            dataTextNormalized = nvigi::InferenceDataTextSTLHelper(textNormalized);
            slots.push_back(nvigi::InferenceDataSlot(kTTSDataSlotOutputTextNormalized, dataTextNormalized));
        }

        // Send output audio + text normalized
        nvigi::InferenceDataSlotArray outputs;
        if (slots.size() > 0)
        {
            outputs = {slots.size(), slots.data()};
            execCtx->outputs = &outputs;
        }
        res = execCtx->callback(execCtx, state, execCtx->callbackUserData);
        //! Clear outputs since these are all local variables
        execCtx->outputs = {};

        return res;
    };

    auto responseCallbackASqFlow = [execCtx, async, instance, asqflowReturnOutputAudio](const std::vector<int16_t> &audio,
                                                                                      const std::string &textNormalized,
                                                                                      InferenceState state) -> void {
        if (state == asqflow::InferenceState::RUNNING || (async && state == asqflow::InferenceState::FINISHED))
        {
            asqflowReturnOutputAudio(audio, textNormalized, nvigi::kInferenceExecutionStateDataPending);
        }
        else if (state == asqflow::InferenceState::ERROR_HAPPENED)
        {
            asqflowReturnOutputAudio(audio, textNormalized, nvigi::kInferenceExecutionStateInvalid);
        }
        else
        {
            asqflowReturnOutputAudio(audio, textNormalized, nvigi::kInferenceExecutionStateDone);
        }
    };

    instance->asqflow->initializeTimers();
    if (async)
    {
        instance->promptsToProcess.write(promptData->getUTF8Text());
        instance->sync.execCtx = execCtx;

        // If a previous job has finished and is ready, we terminate it with the .get() function
        if (instance->sync.job.valid() &&
            instance->sync.job.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            instance->sync.job.get();
            instance->sync.running.store(false);
        }

        // If the job hasnt been initialized, we create it
        if (!instance->sync.job.valid())
        {
            instance->sync.running.store(true);
            instance->sync.job =
                std::async(std::launch::async,
                           [execCtx, instance, responseCallbackASqFlow, asqflowReturnOutputAudio]() -> nvigi::Result {
                               while (instance->sync.running.load() && !instance->promptsToProcess.empty())
                               {
                                   nvigi::RuntimeContextScope scope(*instance);
                                   instance->promptsToProcess.read_and_pop(instance->params.textPrompt);

                                   if (instance->params.textPrompt == END_PROMPT_ASYNC)
                                   {
                                       asqflowReturnOutputAudio({}, "", nvigi::kInferenceExecutionStateDone);
                                       return nvigi::kResultOk;
                                   }
                                   else
                                   {
                                       auto res = instance->asqflow->evaluate(instance->params, responseCallbackASqFlow, instance->cudaStreams[0]);
                                       if (res != nvigi::kResultOk)
                                       {
                                           return res;
                                       }
                                   }
                               }
                               return nvigi::kResultOk;
                           });
        }
        return kResultOk;
    }
    else
    {
        instance->sync.running.store(false);
        if (instance->sync.job.valid())
        {
            NVIGI_LOG_WARN(
                "Asynchronous job already running ... Waiting for his termination before running a synchronous job.");
            instance->sync.job.get();
        }
        nvigi::RuntimeContextScope scope(*instance);
        instance->params.textPrompt = promptData->getUTF8Text();
        auto res = instance->asqflow->evaluate(instance->params, responseCallbackASqFlow, instance->cudaStreams[0]);
        return res;
    }
}

nvigi::Result asqflowDestroyInstance(const nvigi::InferenceInstance *instance)
{
    if (!instance)
        return kResultInvalidParameter;
    auto asqflowInstance = static_cast<asqflow::InferenceContext *>(instance->data);
    if (asqflowInstance->sync.job.valid())
    {
        {
            std::lock_guard lock(asqflowInstance->sync.mtx);
            asqflowInstance->sync.running = false;
        }
        asqflowInstance->sync.job.get();
    }
    {
        nvigi::RuntimeContextScope scope(*asqflowInstance);
        if (asqflowInstance->asqflow != nullptr)
        {
            delete asqflowInstance->asqflow;
            asqflowInstance->asqflow = nullptr;
        }
    }

    for (auto stream : asqflowInstance->cudaStreams)
    {
        cudaError_t err = cudaStreamDestroy(stream);
        if (err != cudaSuccess)
        {
            NVIGI_LOG_ERROR("Failed to destroy CUDA stream: %s", cudaGetErrorString(err));
        }
    }
    asqflowInstance->cudaStreams.clear();

    delete asqflowInstance;
    asqflowInstance = nullptr;
    delete instance;

    return kResultOk;
}

//! Making sure our implementation is covered with our exception handler
//!
namespace asqflow
{
nvigi::Result createInstance(const nvigi::NVIGIParameter *params, nvigi::InferenceInstance **instance)
{
    NVIGI_CATCH_EXCEPTION(asqflowCreateInstance(params, instance));
}

nvigi::Result destroyInstance(const nvigi::InferenceInstance *instance)
{
    NVIGI_CATCH_EXCEPTION(asqflowDestroyInstance(instance));
}

nvigi::Result getCapsAndRequirements(nvigi::NVIGIParameter **modelInfo, const nvigi::NVIGIParameter *params)
{
    NVIGI_CATCH_EXCEPTION(asqflowGetCapsAndRequirements(modelInfo, params));
}
nvigi::Result evaluate(nvigi::InferenceExecutionContext *execCtx)
{
    NVIGI_CATCH_EXCEPTION(asqflowEvaluate(execCtx, false));
}
nvigi::Result evaluateAsync(nvigi::InferenceExecutionContext *execCtx)
{
    NVIGI_CATCH_EXCEPTION(asqflowEvaluate(execCtx, true));
}

} // namespace asqflow

//! Main entry point - get information about our plugin
//!
Result nvigiPluginGetInfo(nvigi::framework::IFramework *framework, nvigi::plugin::PluginInfo **_info)
{
    ////! IMPORTANT:
    ////!
    ////! NO HEAVY LIFTING (MEMORY ALLOCATIONS, GPU RESOURCE CREATION ETC) IN THIS METHOD
    ////!
    ////! JUST PROVIDE INFORMATION REQUESTED AS QUICKLY AS POSSIBLE
    ////!

    if (!plugin::internalPluginSetup(framework))
        return kResultInvalidState;

    // Internal API, we know that incoming pointer is always valid
    auto &info = plugin::getContext()->info;
    *_info = &info;

    info.id = asqflow::getFeatureId(nullptr);
    info.description = "ASqFlow tts plugin implementation";
    info.author = "NVIDIA";
    info.build = GIT_BRANCH_AND_LAST_COMMIT;
    info.interfaces = {plugin::getInterfaceInfo<ITextToSpeech>()};

    //! Specify minimum spec for the OS, driver and GPU architecture
    //!
    //! Defaults indicate no restrictions - plugin can run on any system
    info.minDriver = {NVIGI_CUDA_MIN_DRIVER_MAJOR, NVIGI_CUDA_MIN_DRIVER_MINOR, NVIGI_CUDA_MIN_DRIVER_BUILD};
    info.minOS = {NVIGI_DEF_MIN_OS_MAJOR, NVIGI_DEF_MIN_OS_MINOR, NVIGI_DEF_MIN_OS_BUILD};
    info.minGPUArch = {NVIGI_CUDA_MIN_GPU_ARCH};
    info.requiredVendor = VendorId::eNVDA;

    return kResultOk;
}

//! Main entry point - starting our plugin
//!
Result nvigiPluginRegister(framework::IFramework *framework)
{
    if (!plugin::internalPluginSetup(framework))
        return kResultInvalidState;

    auto &ctx = (*asqflow::getContext());

    ctx.feature = asqflow::getFeatureId(nullptr);

    ctx.api.createInstance = asqflow::createInstance;
    ctx.api.destroyInstance = asqflow::destroyInstance;
    ctx.api.getCapsAndRequirements = asqflow::getCapsAndRequirements;

    framework->addInterface(ctx.feature, &ctx.api, 0);

    if (!framework::getInterface(plugin::getContext()->framework, plugin::hwi::cuda::kId, &ctx.icig))
    {
        NVIGI_LOG_ERROR("Missing interface from 'nvigi.plugin.hwi.cuda'");
        return kResultInvalidState;
    }

    return kResultOk;
    ////! Obtain interfaces from other plugins or core (if needed)
    ////!
    ////! For example, here we use our internal helper function to obtain a GPT interface (a rate example of a plugin
    /// using another plugin)
    ////!
    ////! NOTE: This will trigger loading of the 'nvigi.plugin.gpt.ggml.cuda' plugin if it wasn't loaded already
    ////!
    ////!
    ////! nvigi::IGeneralPurposeTransformer* igpt{};
    ////! if (!getInterface(framework, nvigi::plugin::gpt::ggml::cuda::kId, &igpt))
    ////! {
    ////!     NVIGI_LOG_ERROR("Failed to obtain interface");
    ////! }
    ////! else
    ////! {
    ////!     if (net->getVersion() >= kStructVersion2)
    ////!     {
    ////!         // OK to access v2 members
    ////!     }
    ////!     if (net->getVersion() >= kStructVersion3)
    ////!     {
    ////!         // OK to access v3 members
    ////!     }
    ////! }

    ////! Do any other startup tasks here
    ////!
    ////! IMPORTANT:
    ////!
    ////! NO HEAVY LIFTING WHEN REGISTERING PLUGINS
    ////!
    ////! Use your interface(s) to create instance(s) or context(s)
    ////! then do all initialization, memory allocations etc. within
    ////! the API which is exposed with your interface(s).

    return kResultOk;
}

//! Main exit point - shutting down our plugin
//!
Result nvigiPluginDeregister()
{

    auto &ctx = (*asqflow::getContext());

    framework::releaseInterface(plugin::getContext()->framework, plugin::hwi::cuda::kId, ctx.icig);
    ctx.icig = nullptr;

    ai::freeCommonCapsAndRequirements(ctx.capsData);
    return kResultOk;
}

//! The only exported function - gateway to all functionality
NVIGI_EXPORT void *nvigiPluginGetFunction(const char *functionName)
{
    //! Core API
    NVIGI_EXPORT_FUNCTION(nvigiPluginGetInfo);
    NVIGI_EXPORT_FUNCTION(nvigiPluginRegister);
    NVIGI_EXPORT_FUNCTION(nvigiPluginDeregister);

    return nullptr;
}

} // namespace nvigi
