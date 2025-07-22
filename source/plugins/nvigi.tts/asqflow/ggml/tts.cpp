// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <future>
#include <string>
#include <cstdint>
#include <ggml.h>
#include "source/core/nvigi.api/nvigi.h"
#include "source/core/nvigi.log/log.h"
#include "source/core/nvigi.exception/exception.h"
#include "source/core/nvigi.plugin/plugin.h"
#include "source/core/nvigi.file/file.h"
#include "source/core/nvigi.extra/extra.h"
#include "source/core/nvigi.thread/thread.h"
#include "source/plugins/nvigi.tts/asqflow/ggml/versions.h"
#include "source/plugins/nvigi.tts/nvigi_tts.h"
#include "source/utils/nvigi.ai/ai.h"
#include "_artifacts/gitVersion.h"
#include "external/json/source/nlohmann/json.hpp"
#include "external/asqflow.cpp/include/asqflow.h"
#include "source/plugins/nvigi.tts/asqflow/ggml/tts.h"
#include "source/utils/nvigi.hwi/cuda/runtime_context_scope.h"
#include "nvigi_stl_helpers.h"
#include <functional>
#include <chrono>

using json = nlohmann::json;

#if GGML_USE_CUDA
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
extern void ggml_d3d12_set_params(ID3D12Device *device, ID3D12CommandQueue *cmd_queue_direct,
                                  ID3D12CommandQueue *cmd_queue_compute, ID3D12CommandQueue *cmd_queue_copy,
                                  nvigi::PFun_createCommittedResource *createCommittedResource, nvigi::PFun_destroyResource *destroyResource,
                                  void *userContextCreate, void *userContextDestroy, bool allowReBAR);
#elif defined GGML_USE_VULKAN
#include "external/vulkanSDK/include/vulkan/vulkan.h"
#include "source/core/nvigi.api/nvigi_vulkan.h"
extern void ggml_set_vk_params(VkPhysicalDevice physicalDevice, VkDevice device, VkQueue cmd_queue_direct, VkQueue cmd_queue_compute, VkQueue cmd_queue_copy,
                               nvigi::PFun_allocateMemoryCallback *allocateMemory, nvigi::PFun_freeMemoryCallback *freeMemory,
                               void *userContextAlloc, void *userContextFree);
#endif


#if defined GGML_USE_VULKAN || defined GGML_USE_D3D12
extern void ggml_vk_backend_free();
#endif

namespace nvigi
{
    namespace asqflow
    {

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

        static void asqflowLogCallback(ggml_log_level level, const char* text, void* user_data)
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
                NVIGI_LOG("asqflow", LogType::eWarn, nvigi::log::YELLOW, "%s", msg.c_str());
            }
            else if (level == GGML_LOG_LEVEL_ERROR)
            {
                NVIGI_LOG("asqflow", LogType::eError, nvigi::log::RED, "%s", msg.c_str());
            }
            else
            {
                NVIGI_LOG("asqflow", LogType::eInfo, nvigi::log::WHITE, "%s", msg.c_str());
            }
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
#if GGML_USE_CUDA
            InferenceContext(const nvigi::NVIGIParameter *params) : cudaContext(params) {}
#else
            InferenceContext(const nvigi::NVIGIParameter *params) {}
#endif

            // Use pipeline instead of separate components
            asqflow_pipeline_state *pipeline{};

            std::string generatorModelPath;
            std::string vocoderModelPath;
            std::string g2pModelPath;
            std::string phonemesDictPath;
            std::string configPath;

            json modelInfo;

            std::mutex mtx;
            std::atomic<bool> running = true;

            // Async support structures
            InferenceContextSync sync{};
            PromptsBuffer promptsToProcess;

#if GGML_USE_CUDA
            // Used to set the relative priority of GPU inference and graphics
            std::vector<cudaStream_t> cuda_streams;
            // Use PushPoppableCudaContext defined in push_poppable_cuda_context.h
            PushPoppableCudaContext cudaContext;
#endif

#if GGML_USE_D3D12
            // Used to set the relative priority of GPU inference and graphics
            ID3D12Device *device{};
#endif
        };

        PluginID getFeatureId(InferenceInstanceData *data)
        {
#if GGML_USE_CUDA
            return plugin::tts::asqflow_ggml::cuda::kId;
#elif GGML_USE_VULKAN
            return plugin::tts::asqflow_ggml::vulkan::kId;
#elif GGML_USE_D3D12
            return plugin::tts::asqflow_ggml::d3d12::kId;
#else
            return plugin::tts::asqflow_ggml::cpu::kId;
#endif
        }

        const nvigi::InferenceDataDescriptorArray *getInputSignature(InferenceInstanceData *data)
        {
            static std::vector<InferenceDataDescriptor> slots = {{nvigi::kTTSDataSlotInputText, InferenceDataText::s_type, false}, {nvigi::kTTSDataSlotInputTargetSpectrogramPath, InferenceDataText::s_type, false}};
            static InferenceDataDescriptorArray s_desc = {slots.size(), slots.data()};
            return &s_desc;
        }

        const nvigi::InferenceDataDescriptorArray *getOutputSignature(InferenceInstanceData *data)
        {
            static std::vector<InferenceDataDescriptor> slots = {{nvigi::kTTSDataSlotOutputAudio, InferenceDataAudio::s_type, false}};
            static InferenceDataDescriptorArray s_desc = {slots.size(), slots.data()};
            return &s_desc;
        }

        struct TTSContext
        {
            NVIGI_PLUGIN_CONTEXT_CREATE_DESTROY(TTSContext);

            void onCreateContext() {};
            void onDestroyContext() {};

            ITextToSpeech api{};

            PluginID feature{};

            // Caps and requirements
            ai::CommonCapsData capsData;
            json modelInfo;

#ifdef GGML_USE_CUDA
            nvigi::IHWICuda *icig{};
#elif GGML_USE_D3D12
            nvigi::IHWID3D12 *iscg{};
            nvigi::system::ISystem *isystem{};
#endif
        };

        nvigi::Result evaluate(nvigi::InferenceExecutionContext *execCtx);
        nvigi::Result evaluateAsync(nvigi::InferenceExecutionContext *execCtx);

    }

    //! Define our plugin, make sure to update version numbers in versions.h
    NVIGI_PLUGIN_DEFINE("nvigi.plugin.tts.asqflow-ggml", Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH), Version(API_MAJOR, API_MINOR, API_PATCH), asqflow, TTSContext)

    nvigi::Result asqflowDestroyInstance(const nvigi::InferenceInstance *instance)
    {
        if (instance)
        {
            auto ttsInstance = (nvigi::asqflow::InferenceContext *)(instance->data);
            ttsInstance->running.store(false);

            // Stop any async job and wait for it to complete
            if (ttsInstance->sync.job.valid())
            {
                {
                    std::lock_guard lock(ttsInstance->sync.mtx);
                    ttsInstance->sync.running = false;
                }
                ttsInstance->sync.job.get();
            }

            {
#if GGML_USE_CUDA
                nvigi::RuntimeContextScope scope(*ttsInstance);
#endif
                if (ttsInstance->pipeline)
                {
                    asqflow_pipeline_free(ttsInstance->pipeline);
                }
            }

            delete ttsInstance;
            delete instance;
        }
        return nvigi::kResultOk;
    }

    nvigi::Result asqflowCreateInstance(const nvigi::NVIGIParameter *_params, nvigi::InferenceInstance **_instance)
    {
        auto common = findStruct<CommonCreationParameters>(_params);
        auto creationParams = findStruct<TTSCreationParameters>(_params);
        auto asqfCreationParams = findStruct<TTSASqFlowCreationParameters>(_params);
        if (!creationParams || !common)
        {
            NVIGI_LOG_ERROR("Missing required parameters");
            return nvigi::kResultInvalidParameter;
        }
        auto &params = *creationParams;

        // Use default values if TTSASqFlowCreationParameters is not provided
        if (!asqfCreationParams)
        {
            asqfCreationParams = new TTSASqFlowCreationParameters();
        }
        if (!_instance || !common->utf8PathToModels || !common->modelGUID)
            return nvigi::kResultInvalidParameter;

#ifdef GGML_USE_CUDA
        auto cudaParams = findStruct<CudaParameters>(_params);
        if (cudaParams && cudaParams->getVersion() >= kStructVersion2)
        {
            if (cudaParams->cudaMallocReportCallback)
                nvigi::asqflow::setCudaMallocReportCallback(cudaParams->cudaMallocReportCallback, cudaParams->cudaMallocReportUserContext);
            if (cudaParams->cudaFreeReportCallback)
                nvigi::asqflow::setCudaFreeReportCallback(cudaParams->cudaFreeReportCallback, cudaParams->cudaFreeReportUserContext);
            if (cudaParams->cudaMallocCallback)
                nvigi::asqflow::setCudaMallocCallback(cudaParams->cudaMallocCallback, cudaParams->cudaMallocUserContext);
            if (cudaParams->cudaFreeCallback)
                nvigi::asqflow::setCudaFreeCallback(cudaParams->cudaFreeCallback, cudaParams->cudaFreeUserContext);
        }
#endif

        using namespace nvigi::asqflow;
        auto &ctx = (*asqflow::getContext());

        *_instance = nullptr;

        auto instanceData = new nvigi::asqflow::InferenceContext(_params);
#if GGML_USE_CUDA
        if (!instanceData->cudaContext.constructorSucceeded)
            return kResultInvalidState;
#endif

#if defined(GGML_USE_CUDA) || defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
#ifndef NVIGI_PRODUCTION
        size_t currentUsageMB{};
        extra::ScopedTasks vram([&currentUsageMB]()
                                {
            system::VRAMUsage* usage;
            system::getInterface()->getVRAMStats(0, &usage);
            currentUsageMB = usage->currentUsageMB; },
                                [&currentUsageMB]()
                                {
                                    system::VRAMUsage *usage;
                                    system::getInterface()->getVRAMStats(0, &usage);
                                    currentUsageMB = usage->currentUsageMB - currentUsageMB;
                                    NVIGI_LOG_INFO("New instance using %lluMB budget %lluMB", currentUsageMB, usage->budgetMB);
                                });
#endif
#endif
        {
            if (instanceData->modelInfo.empty())
            {
                if (!ai::findModels(common, {"gguf", "txt"}, instanceData->modelInfo))
                {
                    NVIGI_LOG_ERROR("Failed to find models in the expected directory '%s'", common->utf8PathToModels);
                    return kResultInvalidParameter;
                }
            }

            std::string pathDpModel{}, pathGeneratorModel, pathVocoderModel{}, pathg2pModel{};
            std::string pathCMUDict{};

            try
            {
                // Trim down to our GUID for this instance
                instanceData->modelInfo = instanceData->modelInfo[common->modelGUID];

#if defined(GGML_USE_CUDA) || defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
                size_t neededVRAM = instanceData->modelInfo["vram"];
                if (common->vramBudgetMB < neededVRAM)
                {
                    NVIGI_LOG_WARN("Provided VRAM %uMB is insufficient, required VRAM is %uMB", common->vramBudgetMB, neededVRAM);
                    return kResultInsufficientResources;
                }
#endif

                // Find paths for models using filename patterns
                std::vector<std::string> filesModels = instanceData->modelInfo["gguf"];
                if (filesModels.empty())
                {
                    NVIGI_LOG_ERROR("Failed to find GGUF models in the expected directory '%s'", common->utf8PathToModels);
                    return kResultInvalidParameter;
                }

                std::string parentFolder = std::filesystem::path(filesModels[0]).parent_path().string();

                // Find generator, vocoder, and G2P models in GGUF files
                for (const std::string &file : filesModels)
                {
                    if (file.find("Generator") != std::string::npos)
                    {
                        if (pathGeneratorModel != "")
                        {
                            NVIGI_LOG_ERROR("Multiple generator models have been found in the directory '%s'",
                                            parentFolder.c_str());
                            return kResultInvalidParameter;
                        }
                        pathGeneratorModel = file;
                    }
                    else if (file.find("Vocoder") != std::string::npos)
                    {
                        if (pathVocoderModel != "")
                        {
                            NVIGI_LOG_ERROR("Multiple vocoder models have been found in the directory '%s'",
                                            parentFolder.c_str());
                            return kResultInvalidParameter;
                        }
                        pathVocoderModel = file;
                    }
                    else if (file.find("G2P") != std::string::npos)
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

                if (pathGeneratorModel == "")
                {
                    NVIGI_LOG_ERROR("Generator model have not been found in the directory '%s'", parentFolder.c_str());
                    return kResultInvalidParameter;
                }
                else if (pathVocoderModel == "")
                {
                    NVIGI_LOG_ERROR("Vocoder model have not been found in the directory '%s'", parentFolder.c_str());
                    return kResultInvalidParameter;
                }
                else if (pathg2pModel == "")
                {
                    NVIGI_LOG_ERROR("G2P model have not been found in the directory '%s'", parentFolder.c_str());
                    return kResultInvalidParameter;
                }

                // Find the path for CMU dictionary (in charge of converting grapheme to phonemes)
                std::vector<std::string> filesDicts;
                const std::string nameDictPhonemes = "ipa_dict_phonemized.txt";

                filesDicts = instanceData->modelInfo["txt"];
                for (const std::string &file : filesDicts)
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
            catch (const std::exception &e)
            {
                NVIGI_LOG_ERROR("Exception %s", e.what());
                return kResultJSONException;
            }

            instanceData->generatorModelPath = pathGeneratorModel;
            instanceData->vocoderModelPath = pathVocoderModel;
            instanceData->g2pModelPath = pathg2pModel;
            instanceData->phonemesDictPath = pathCMUDict;

            NVIGI_LOG_INFO("Generator model : %s", instanceData->generatorModelPath.c_str());
            NVIGI_LOG_INFO("Vocoder model : %s", instanceData->vocoderModelPath.c_str());
            NVIGI_LOG_INFO("G2P model : %s", instanceData->g2pModelPath.c_str());
            NVIGI_LOG_INFO("Phonemes dictionary : %s", instanceData->phonemesDictPath.c_str());

            {
#if GGML_USE_CUDA
                nvigi::RuntimeContextScope scope(*instanceData);
#endif
#ifdef GGML_USE_VULKAN
                auto vkParams = findStruct<VulkanParameters>(_params);
                if (vkParams)
                {
                    ggml_set_vk_params(vkParams->physicalDevice, vkParams->device, vkParams->queue, vkParams->queueCompute, vkParams->queueTransfer,
                                       vkParams->allocateMemoryCallback, vkParams->freeMemoryCallback,
                                       vkParams->allocateMemoryCallbackUserContext, vkParams->freeMemoryCallbackUserContext);
                }
#elif GGML_USE_D3D12
                auto d3d12Params = findStruct<D3D12Parameters>(_params);
                if (NVIGI_FAILED(res, d3d12::validateParameters(d3d12Params)))
                {
                    return res;
                }
                bool allowReBAR = d3d12Params->getVersion() < 3 || !(d3d12Params->flags & nvigi::D3D12ParametersFlags::eDisableReBAR);
                ggml_d3d12_set_params(d3d12Params->device, d3d12Params->queue, d3d12Params->queueCompute, d3d12Params->queueCopy,
                                      d3d12Params->createCommittedResourceCallback, d3d12Params->destroyResourceCallback, d3d12Params->createCommitResourceUserContext, d3d12Params->destroyResourceUserContext, allowReBAR);
#endif

                // Initialize pipeline with all models
                std::string config_data_storage = instanceData->modelInfo.dump();
                asqflow_pipeline_params pipelineParams = asqflow_pipeline_default_params();
                pipelineParams.generator_model = instanceData->generatorModelPath.c_str();
                pipelineParams.vocoder_model = instanceData->vocoderModelPath.c_str();
                pipelineParams.g2p_model = instanceData->g2pModelPath.c_str();
                pipelineParams.phonemes_dict = instanceData->phonemesDictPath.c_str();
                pipelineParams.config_data = config_data_storage.c_str();
#if GGML_USE_CUDA
                // Set GPU device
                auto cudaParams = findStruct<CudaParameters>(_params);
                pipelineParams.gpu_device = cudaParams ? cudaParams->device : 0;
#endif
                try
                {
                    instanceData->pipeline = asqflow_pipeline_init(&pipelineParams);
                }
                catch (std::exception &e)
                {
                    NVIGI_LOG_ERROR("Pipeline initialization failed: %s", e.what());
                    delete instanceData;
                    return kResultInvalidState;
                }

                if (!instanceData->pipeline)
                {
                    NVIGI_LOG_ERROR("Call to 'asqflow_pipeline_init' failed");
                    delete instanceData;
                    return kResultInvalidState;
                }

                NVIGI_LOG_INFO("Pipeline initialized successfully");
            }
#if GGML_USE_CUDA
            // We ask asqflow for all the cuda_streams it is going to use and
            // store them to enable us to change their priorities dynamically
            size_t stream_count = asqflow_get_cuda_stream_count(instanceData->pipeline);
            instanceData->cuda_streams.resize(stream_count);
            asqflow_get_cuda_streams(instanceData->pipeline, (void **)instanceData->cuda_streams.data(), stream_count);

            // Apply the global priority to all streams
            if (instanceData->cudaContext.usingCiG && ctx.icig->getVersion() >= 2)
            {
                nvigi::Result cuerr = ctx.icig->cudaApplyGlobalGpuInferenceSchedulingMode(instanceData->cuda_streams.data(), instanceData->cuda_streams.size());
                if (cuerr != kResultOk)
                {
                    NVIGI_LOG_WARN_ONCE("Could not set relative priority of compute and graphics. Please use 575 driver or higher\n");
                }
            }
#elif GGML_USE_D3D12
            VendorId vendor{};
            if (NVIGI_FAILED(res, d3d12::getDeviceVendor(d3d12Params, ctx.isystem, vendor)))
            {
                return res;
            }
            if (vendor == VendorId::eNVDA)
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
#endif

#if GGML_USE_CUDA
        auto platform = "ggml.cuda";
#elif GGML_USE_VULKAN
        auto platform = "ggml.vulkan";
#elif GGML_USE_D3D12
        auto platform = "ggml.d3d12";
#else
        auto platform = "ggml.cpu";
#endif
            NVIGI_LOG_VERBOSE("Created TTS instance for backend '%s'", platform);
        }

        auto instance = new InferenceInstance();
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
        auto common = findStruct<CommonCreationParameters>(_params);
        if (!common)
            return nvigi::kResultInvalidParameter;

        static CommonCapabilitiesAndRequirements s_capsCommon{};
        static TTSCapabilitiesAndRequirements s_caps{};
        s_caps.common = &s_capsCommon;
        auto info = &s_caps;
        *_info = s_caps;

        auto &ctx = (*asqflow::getContext());
        if (!ai::findModels(common, {"gguf"}, ctx.modelInfo))
        {
            return kResultInvalidParameter;
        }

        //// Supported languages (can be extended based on models)
        // static const char* s_languages[] = { "en" };
        // info->supportedLanguages = s_languages;

        // CUDA or CPU backend
#if defined(GGML_USE_CUDA) || defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
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
#if GGML_USE_CUDA || (GGML_USE_D3D12 && PROFILE_D3D)
        nvtxRangePushA("asqflowEvaluate");
#endif

        auto &ctx = (*asqflow::getContext());

        // Validate all inputs first
        if (!execCtx)
        {
            NVIGI_LOG_ERROR("Unable to find 'InferenceExecutionContext' structure in the inputs");
            return kResultInvalidParameter;
        }

        if (!execCtx->callback)
        {
            NVIGI_LOG_ERROR("TTS inference callback not provided");
            return kResultInvalidParameter;
        }

        if (!execCtx->instance)
        {
            NVIGI_LOG_ERROR("TTS inference instance not provided");
            return kResultInvalidParameter;
        }

        if (ctx.feature != execCtx->instance->getFeatureId(execCtx->instance->data))
        {
            NVIGI_LOG_ERROR("Invalid inference instance - expecting TTS %u got %u", ctx.feature, execCtx->instance->getFeatureId(execCtx->instance->data));
            return kResultInvalidParameter;
        }

        // Now we are good to go!
        using namespace nvigi::asqflow;

        const nvigi::InferenceDataText *textInput{};
        if (!execCtx->inputs->findAndValidateSlot(kTTSDataSlotInputText, &textInput))
        {
            NVIGI_LOG_ERROR("Expecting single inference input of type 'nvigi::InferenceDataText'");
            return kResultInvalidParameter;
        }

        const nvigi::InferenceDataText *targetSpecData{};
        if (!execCtx->inputs->findAndValidateSlot(kTTSDataSlotInputTargetSpectrogramPath, &targetSpecData))
        {
            NVIGI_LOG_ERROR("Expecting target spectrogram path input of type 'nvigi::InferenceDataText'");
            return kResultInvalidParameter;
        }

        auto instance = (nvigi::asqflow::InferenceContext *)(execCtx->instance->data);

#if GGML_USE_CUDA
        // We apply the global scheduling mode to our streams at every evaluate()
        // to reflect any changes the user made to the global mode between calls
        if (instance->cudaContext.usingCiG && ctx.icig->getVersion() >= 2)
        {
            nvigi::Result err = ctx.icig->cudaApplyGlobalGpuInferenceSchedulingMode(instance->cuda_streams.data(), instance->cuda_streams.size());
            if (err != kResultOk)
            {
                NVIGI_LOG_WARN_ONCE("Could not set relative priority of compute and graphics, insufficient driver\n");
            }
        }
#elif GGML_USE_D3D12
    if (ctx.iscg)
    {
#if PROFILE_D3D
        nvtxRangePushA("TTS: Set D3D priority");
#endif
        nvigi::Result d3derr = ctx.iscg->d3d12ApplyGlobalGpuInferenceSchedulingModeToThread(instance->device);
        if (d3derr != kResultOk)
        {
            NVIGI_LOG_WARN_ONCE("Could not set relative priority of D3D12 compute and graphics. Please use 575 driver or higher\n");
        }
#if PROFILE_D3D
        nvtxRangePop();
#endif
    }
#endif

        // Get input text
        std::string inputText = textInput->getUTF8Text();
        NVIGI_LOG_VERBOSE("Processing text: '%s'", inputText.c_str());

        // Get spectrogram path
        std::string spectrogramPath = targetSpecData->getUTF8Text();
        NVIGI_LOG_VERBOSE("Using spectrogram path: '%s'", spectrogramPath.c_str());

        auto runtime = findStruct<TTSASqFlowRuntimeParameters>(execCtx->runtimeParameters);
        if (!runtime)
        {
            runtime = new TTSASqFlowRuntimeParameters();
        }
        
        const float speechRate = std::max(std::min(runtime->speed, 1.5f), 0.5f);
        const int nTimesteps = std::max(std::min(runtime->n_timesteps, 32), 12);
        const int minChunkSize = std::max(runtime->minChunkSize, 50);   // Minimum 50 characters
        const int maxChunkSize = std::max(runtime->maxChunkSize, minChunkSize);  // At least minChunkSize
        const int seed = runtime->seed;
        const int sampler = std::max(std::min(runtime->sampler, 1), 0);  // Clamp to 0-1 range
        const int dpmpp_order = std::max(std::min(runtime->dpmpp_order, 3), 1);  // Clamp to 1-3 range
        const bool use_flash_attention = runtime->use_flash_attention;
        
        NVIGI_LOG_VERBOSE("Using runtime speed: %f", speechRate);
        NVIGI_LOG_VERBOSE("Using runtime n_timesteps: %d", nTimesteps);
        NVIGI_LOG_VERBOSE("Using runtime minChunkSize: %d, maxChunkSize: %d", minChunkSize, maxChunkSize);
        NVIGI_LOG_VERBOSE("Using runtime seed: %d, sampler: %d, dpmpp_order: %d, flash_attention: %s", 
                         seed, sampler, dpmpp_order, use_flash_attention ? "true" : "false");

        auto asqflowReturnOutputAudio =
            [execCtx, instance](const std::vector<int16_t> &audio, const std::string &textNormalized,
                                nvigi::InferenceExecutionState state =
                                    nvigi::kInferenceExecutionStateDataPending) -> nvigi::InferenceExecutionState
        {
            // check for error
            if (state == nvigi::kInferenceExecutionStateInvalid)
            {
                return execCtx->callback(execCtx, state, execCtx->callbackUserData);
            }

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
                memcpy_s((int16_t *)(cpuBuffer->buffer), cpuBuffer->sizeInBytes, &(audio[0]), audio.size() * sizeof(int16_t));
            }
            else
            {
                //! Temporary outputs for the callback since host did not provide any
                // The size is multiplied by 2 since int16 represents two bytes
                dataAudio = nvigi::InferenceDataByteArraySTLHelper((const uint8_t *)audio.data(), audio.size() * 2);
                slots.push_back(nvigi::InferenceDataSlot(kTTSDataSlotOutputAudio, dataAudio));
            }

            // Send output audio
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

        // Run TTS pipeline
        nvigi::Result result = kResultOk;
        
        if (async)
        {
            // Async mode: queue the input and manage background processing
            instance->promptsToProcess.write(inputText);
            instance->sync.execCtx = execCtx;

            // If a previous job has finished and is ready, we terminate it with the .get() function
            if (instance->sync.job.valid() &&
                instance->sync.job.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
            {
                instance->sync.job.get();
                instance->sync.running.store(false);
            }

            // If the job hasn't been initialized, we create it
            if (!instance->sync.job.valid())
            {
                instance->sync.running.store(true);
                instance->sync.job = std::async(std::launch::async,
                    [execCtx, instance, asqflowReturnOutputAudio, spectrogramPath, speechRate, 
                     nTimesteps, minChunkSize, maxChunkSize, seed, sampler, dpmpp_order, use_flash_attention]() -> nvigi::Result
                {
                    while (instance->sync.running.load() && !instance->promptsToProcess.empty())
                    {
#if GGML_USE_CUDA
                        nvigi::RuntimeContextScope scope(*instance);
#endif
                        std::string currentText;
                        instance->promptsToProcess.read_and_pop(currentText);

                        if (currentText == END_PROMPT_ASYNC)
                        {
                            asqflowReturnOutputAudio({}, "", nvigi::kInferenceExecutionStateDone);
                            return nvigi::kResultOk;
                        }
                        else
                        {
                            try
                            {
                                // Use the new TTS inference function with chunking support
                                std::vector<int16_t> audioData;
                                std::string textNormalized;

                                // Create a chunk callback function that uses the existing callback pattern
                                std::function<nvigi::InferenceExecutionState(const std::vector<int16_t> &, const std::string &, nvigi::InferenceExecutionState)> chunkCallback =
                                    [&asqflowReturnOutputAudio, &textNormalized](const std::vector<int16_t> &chunkAudio, const std::string &chunkText, nvigi::InferenceExecutionState state) -> nvigi::InferenceExecutionState
                                {
                                    // For each chunk, call the output callback
                                    NVIGI_LOG_VERBOSE("Returning chunk audio: %zu samples, text: '%s', state: %d", chunkAudio.size(), chunkText.c_str(), state);
                                    // In Asynch mode we return kInferenceExecutionStateDone only when receiving END_PROMPT_ASYNC
                                    if (state != nvigi::kInferenceExecutionStateInvalid)
                                    {
                                        return asqflowReturnOutputAudio(chunkAudio, textNormalized, nvigi::kInferenceExecutionStateDataPending);
                                    }
                                    else if (state == kInferenceExecutionStateInvalid)
                                    {
                                        return asqflowReturnOutputAudio(chunkAudio, textNormalized, nvigi::kInferenceExecutionStateInvalid);
                                    }

                                };

                                auto res = performTTSInference(
                                    instance->pipeline,
                                    currentText,
                                    spectrogramPath,
                                    speechRate,
                                    audioData,
                                    textNormalized,
                                    true, // enableChunking
                                    minChunkSize,
                                    maxChunkSize,
                                    &chunkCallback,
                                    nTimesteps,
                                    seed,
                                    sampler,
                                    dpmpp_order,
                                    use_flash_attention
                                );

                                if (res != nvigi::kResultOk)
                                {
                                    NVIGI_LOG_ERROR("Async TTS inference failed");
                                    asqflowReturnOutputAudio({}, "", nvigi::kInferenceExecutionStateInvalid);
                                    return res;
                                }
                            }
                            catch (std::exception &e)
                            {
                                NVIGI_LOG_ERROR("Async TTS inference failed: %s", e.what());
                                asqflowReturnOutputAudio({}, "", nvigi::kInferenceExecutionStateInvalid);
                                return kResultInvalidState;
                            }
                        }
                    }
                    return nvigi::kResultOk;
                });
            }
            result = kResultOk;
        }
        else
        {
            // Synchronous mode: stop any async job and run immediately
            instance->sync.running.store(false);
            if (instance->sync.job.valid())
            {
                NVIGI_LOG_WARN("Asynchronous job already running ... Waiting for its termination before running a synchronous job.");
                instance->sync.job.get();
            }

            try
            {
#if GGML_USE_CUDA
                nvigi::RuntimeContextScope scope(*instance);
#endif

                // Use the new TTS inference function with chunking support
                std::vector<int16_t> audioData;
                std::string textNormalized;

                // Create a chunk callback function that uses the existing callback pattern
                std::function<nvigi::InferenceExecutionState(const std::vector<int16_t> &, const std::string &, nvigi::InferenceExecutionState)> chunkCallback =
                    [&asqflowReturnOutputAudio, &textNormalized](const std::vector<int16_t> &chunkAudio, const std::string &chunkText, nvigi::InferenceExecutionState state) -> nvigi::InferenceExecutionState
                {
                    // For each chunk, call the output callback
                    NVIGI_LOG_VERBOSE("Returning chunk audio: %zu samples, text: '%s', state: %d", chunkAudio.size(), chunkText.c_str(), state);
                    return asqflowReturnOutputAudio(chunkAudio, chunkText, state);
                };

                // Enable chunking by default with reasonable chunk sizes
                result = performTTSInference(
                    instance->pipeline,
                    inputText,
                    spectrogramPath,
                    speechRate,
                    audioData,
                    textNormalized,
                    true, // enableChunking
                    minChunkSize,  // Use runtime parameter
                    maxChunkSize,  // Use runtime parameter
                    &chunkCallback,
                    nTimesteps, // Use runtime parameter
                    seed,      // Random seed
                    sampler,   // Sampler type (0=EULER, 1=DPM++)
                    dpmpp_order, // DPM++ order (1-3)
                    use_flash_attention // Flash attention flag
                );

                // If chunking failed or we want to return the complete audio at the end
                if (result == kResultOk && !audioData.empty())
                {
                    NVIGI_LOG_INFO("TTS inference completed successfully with %zu total audio samples", audioData.size());
                    // The chunked callback already handled the individual chunks
                    // We could optionally call the final callback here with the complete audio
                    // asqflowReturnOutputAudio(audioData, textNormalized, nvigi::kInferenceExecutionStateDone);
                }
                else if (result != kResultOk)
                {
                    NVIGI_LOG_ERROR("TTS inference failed");
                    asqflowReturnOutputAudio({}, "", nvigi::kInferenceExecutionStateInvalid);
                }
            }
            catch (std::exception &e)
            {
                NVIGI_LOG_ERROR("TTS inference failed: %s", e.what());
                result = kResultInvalidState;
            }
        }

#if GGML_USE_D3D12 && PROFILE_D3D
        nvtxRangePushA("TTS: Restore D3D priority");
        nvigi::Result d3derr = ctx.iscg->d3d12RestoreThreadsGpuInferenceSchedulingMode(instance->device);
        if (d3derr != kResultOk)
        {
            NVIGI_LOG_WARN_ONCE("Could not set relative priority of D3D12 compute and graphics. Please use 575 driver or higher\n");
        }
        nvtxRangePop();
#endif

#if GGML_USE_CUDA || (GGML_USE_D3D12 && PROFILE_D3D)
        nvtxRangePop();
#endif

        return result;
    }

    //! Exception handling wrappers
    //!
    //! Note that we export these via our interface
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
            NVIGI_CATCH_EXCEPTION(asqflowEvaluate(execCtx));
        }
        nvigi::Result evaluateAsync(nvigi::InferenceExecutionContext* execCtx)
        {
            NVIGI_CATCH_EXCEPTION(asqflowEvaluate(execCtx, true));
        }

    } // tts

    //! Main entry point - get information about our plugin
    //!
    Result nvigiPluginGetInfo(nvigi::framework::IFramework *framework, nvigi::plugin::PluginInfo **_info)
    {
        auto &ctx = (*asqflow::getContext());

        if (!plugin::internalPluginSetup(framework))
            return kResultInvalidState;

        // Internal API, we know that incoming pointer is always valid
        auto &info = plugin::getContext()->info;
        *_info = &info;

        info.id = asqflow::getFeatureId(nullptr);
        info.description = "ggml backend implementation for the 'tts' inference using ASQFlow";
        info.author = "NVIDIA";
        info.build = GIT_BRANCH_AND_LAST_COMMIT;
        info.interfaces = {plugin::getInterfaceInfo<ITextToSpeech>()};

        // Always the same OS requirements
        info.minOS = {NVIGI_DEF_MIN_OS_MAJOR, NVIGI_DEF_MIN_OS_MINOR, NVIGI_DEF_MIN_OS_BUILD};

        // Default to no GPU requirements for now
        info.minGPUArch = {};
        info.minDriver = {};

#if GGML_USE_CUDA
        info.minDriver = {NVIGI_CUDA_MIN_DRIVER_MAJOR, NVIGI_CUDA_MIN_DRIVER_MINOR, NVIGI_CUDA_MIN_DRIVER_BUILD};
        info.minGPUArch = {NVIGI_CUDA_MIN_GPU_ARCH};
        info.requiredVendor = VendorId::eNVDA;

        nvigi::system::ISystem* isystem{};
        if (!framework::getInterface(framework, nvigi::core::framework::kId, &isystem))
        {
            NVIGI_LOG_ERROR("Missing interface from 'nvigi.plugin.hwi.cuda'");
            return kResultInvalidState;
        }

        const nvigi::system::SystemCaps* caps = isystem->getSystemCaps();
        if (caps && caps->driverVersion.major < 580)
        {
            NVIGI_LOG_WARN_ONCE("CUDA backend recommends driver version 580 or higher, current version is %d.%d.%d - see documentation for details", caps->driverVersion.major, caps->driverVersion.minor, caps->driverVersion.build);
        }

        // Must release, as we may not have the chance to release it later if we are not registered
        framework::releaseInterface(plugin::getContext()->framework, nvigi::core::framework::kId, isystem);
#elif defined(GGML_USE_VULKAN)
        // Requires SOME GPU with Vulkan support, no specific vendor or driver version
        info.requiredVendor = VendorId::eAny;
#else
        // No requirements for CPU backend
        info.requiredVendor = nvigi::VendorId::eNone;
#endif

        return kResultOk;
    }

    //! Main entry point - starting our plugin
    //!
    //! IMPORTANT: Plugins are started based on their priority.
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
        
        asqflow_log_set(nvigi::asqflow::asqflowLogCallback, nullptr);


#if GGML_USE_CUDA
        if (!framework::getInterface(plugin::getContext()->framework, plugin::hwi::cuda::kId, &ctx.icig))
        {
            NVIGI_LOG_ERROR("Missing interface from 'nvigi.plugin.hwi.cuda'");
            return kResultMissingInterface;
        }
#elif GGML_USE_D3D12
    if (!framework::getInterface(framework, nvigi::core::framework::kId, &ctx.isystem))
    {
        NVIGI_LOG_ERROR("Missing core interface 'nvigi::system::ISystem'");
        return kResultMissingInterface;
    }
#endif

        return kResultOk;
    }

    //! Main exit point - shutting down our plugin
    //!
    Result nvigiPluginDeregister()
    {
        auto &ctx = (*asqflow::getContext());

        ai::freeCommonCapsAndRequirements(ctx.capsData);

#if GGML_USE_CUDA
        framework::releaseInterface(plugin::getContext()->framework, plugin::hwi::cuda::kId, ctx.icig);
        ctx.icig = nullptr;
#elif GGML_USE_D3D12
    framework::releaseInterface(plugin::getContext()->framework, plugin::hwi::d3d12::kId, ctx.iscg);
    ctx.iscg = nullptr;
    framework::releaseInterface(plugin::getContext()->framework, nvigi::core::framework::kId, ctx.isystem);
    ctx.isystem = nullptr;
#endif

    asqflow_log_set(nullptr, nullptr);

#if defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
        // ggml does not provide a way to free backends explicitly hence the hack
        ggml_vk_backend_free();
#endif

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

}