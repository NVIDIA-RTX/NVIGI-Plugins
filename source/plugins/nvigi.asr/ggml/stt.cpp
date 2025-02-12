// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <future>
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
#include "_artifacts/gitVersion.h"
#include "external/json/source/nlohmann/json.hpp"
#include "external/whisper.cpp/include/whisper.h"
#include "source/plugins/nvigi.asr/ggml/stt.h"
#include "source/utils/nvigi.ai/ai_data_helpers.h"
#include "source/utils/nvigi.hwi/cuda/runtime_context_scope.h"

using json = nlohmann::json;

#if GGML_USE_CUBLAS
#include "source/core/nvigi.api/nvigi_cuda.h"
#include "source/utils/nvigi.hwi/cuda/push_poppable_cuda_context.h"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define NUM_BUFFERS 2
#define BUFFER_SIZE 4096

namespace nvigi
{
namespace asr
{

constexpr StreamSignal kStreamTypeNone = (StreamSignal)-1;
constexpr float kMaxAudioBufferSizeMs = 5 * 60 * 1000;
constexpr int kAvailableAudioBufferSize = 0;

#define NUM_BUFFERS 2
#define BUFFER_SIZE 4096

static void whisperLogCallback(ggml_log_level level, const char* text, void* user_data) 
{
    if (level == GGML_LOG_LEVEL_WARN)
    {
        NVIGI_LOG("whisper", LogType::eWarn, nvigi::log::YELLOW, "%s", text);
    }
    else if (level == GGML_LOG_LEVEL_ERROR)
    {
        NVIGI_LOG("whisper", LogType::eError, nvigi::log::RED, "%s", text);
    }
    else
    {
        NVIGI_LOG("whisper", LogType::eInfo, nvigi::log::WHITE, "%s", text);
    }
}

struct InferenceContext
{
    InferenceContext(const nvigi::NVIGIParameter* params) : cudaContext(params) {}

#ifndef NVIGI_ASR_GFN_NVCF
    whisper_context* model{};
#endif
    whisper_params params{};
    CircularBuffer audio;
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32;
    std::vector<float> pcmf32_new;

    std::vector<whisper_token> prompt_tokens;

    std::mutex mtx;
    std::future<Result> job;
    std::atomic<bool> running = true;

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

    PluginID feature{};

    // Caps and requirements
    json modelInfo;
    ai::CommonCapsData capsData;
};

nvigi::Result evaluate(nvigi::InferenceExecutionContext* execCtx);
nvigi::Result evaluateAsync(nvigi::InferenceExecutionContext* execCtx);

}

//! Define our plugin, make sure to update version numbers in versions.h
NVIGI_PLUGIN_DEFINE("nvigi.plugin.asr", Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH), Version(API_MAJOR, API_MINOR, API_PATCH), asr, ASRContext)

nvigi::Result whisperDestroyInstance(const nvigi::InferenceInstance* instance)
{
    if (instance)
    {
        auto sttInstance = (nvigi::asr::InferenceContext*)(instance->data);
        if (sttInstance->job.valid())
        {
            sttInstance->running.store(false);
            sttInstance->job.get();
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

    using namespace nvigi::asr;
    auto& ctx = (*asr::getContext());

    *_instance = nullptr;

    auto instanceData = new nvigi::asr::InferenceContext(_params);
    if (!instanceData->cudaContext.constructorSucceeded) return kResultInvalidState;

#if GGML_USE_CUBLAS
    if (common->numThreads > 1)
    {
        NVIGI_LOG_WARN("For optimal performance when using CUDA only one CPU thread is used");
    }
    instanceData->params.n_threads = 1;
#else
    instanceData->params.n_threads = common->numThreads;
#endif

    instanceData->params.language = params.language ? params.language : "en";

#if defined(GGML_USE_CUBLAS)
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

#if defined(GGML_USE_CUBLAS)
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

        pathToModel = files[0];

        NVIGI_LOG_INFO("Loading model '%s'", pathToModel.c_str());

        {
#if GGML_USE_CUBLAS
            nvigi::RuntimeContextScope scope(*instanceData);
            cudaStream_t stream{};
            cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
#endif
            struct whisper_context_params cparams = whisper_context_default_params();
            if (params._base.version >= kStructVersion2)
            {
                cparams.flash_attn = params.flashAtt;
            }
#if GGML_USE_CUBLAS
            cparams.use_gpu = true;
            auto cudaParams = findStruct<CudaParameters>(_params);
            cparams.gpu_device = cudaParams ? cudaParams->device : 0;
#endif
            instanceData->model = whisper_init_from_file_with_params(pathToModel.c_str(),cparams);
        }
        if (!instanceData->model)
        {
            NVIGI_LOG_ERROR("Call to 'whisper_init_from_file_with_params' failed");
            delete instanceData;
            return kResultInvalidState;
        }
#if GGML_USE_CUBLAS
        auto platform = "ggml.cuda";
#else
        auto platform = "ggml.cpu";
#endif
        NVIGI_LOG_VERBOSE("Created instance for backend '%s' - threads %d", platform, instanceData->params.n_threads);

        // Single channel, 60 seconds of samples as floats
        instanceData->audio.init(kMaxAudioBufferSizeMs);
    }
    
    auto instance = new InferenceInstance();
    instance->data = instanceData;
    instance->getFeatureId = asr::getFeatureId;
    instance->getInputSignature = asr::getInputSignature;
    instance->getOutputSignature = asr::getOutputSignature;
    instance->evaluate = asr::evaluate;
    instance->evaluateAsync = asr::evaluateAsync;
    
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

    // Supported languages
    static const char* s_languages[] = { "auto" };
    info->supportedLanguages = s_languages;

    // CUDA or CPU backend
#if defined(GGML_USE_CUBLAS)
    info->common->supportedBackends = nvigi::InferenceBackendLocations::eGPU;
#else
    info->common->supportedBackends = nvigi::InferenceBackendLocations::eCPU;
#endif

    //! Must be called after we set supported backends to correctly filter models
    ai::populateCommonCapsAndRequirements(ctx.capsData, *common, *info->common, ctx.modelInfo);

    return kResultOk;
}

nvigi::Result whisperEvaluate(nvigi::InferenceExecutionContext* execCtx, bool async)
{
    auto& ctx = (*asr::getContext());

    // Validate all inputs first

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
                res = execCtx->callback(execCtx, state, execCtx->callbackUserData);
            }
        }
        else
        {
            //! Temporary outputs for the callback since host did not provide any
            auto text = nvigi::CpuData(content.length()+1, (const void*)content.c_str());
            auto data = nvigi::InferenceDataText(text);
            std::vector<nvigi::InferenceDataSlot> slots = { {kASRWhisperDataSlotTranscribedText, &data} };
            nvigi::InferenceDataSlotArray outputs = { slots.size(), slots.data() };
            execCtx->outputs = &outputs;
            res = execCtx->callback(execCtx, state, execCtx->callbackUserData);
            //! Clear outputs since these are all local variables
            execCtx->outputs = {};
        }
        return res;
    };

    auto instance = (nvigi::asr::InferenceContext*)(execCtx->instance->data);
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
    }
    StreamSignal streamType = streaming && async ? streaming->signal : kStreamTypeNone;

    if (streamType == kStreamTypeNone || streamType == StreamSignal::eStreamSignalStart)
    {
        static const char* s_strategy[] = { "ASRSamplingStrategy::eGreedy","ASRSamplingStrategy::eBeamSearch" };
        NVIGI_LOG_VERBOSE("Processing audio on %u threads - strategy '%s' - best of %d - beam size %d", params.n_threads, s_strategy[strategy], params.best_of, params.beam_size);
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

        auto streamType = (StreamSignal)instance->audio.getStreamType();
        bool noMoreData = false;
        bool finalSegment = false;
        if (streamType != kStreamTypeNone)
        {
            if (instance->audio.read(instance->pcmf32_new, params.step_ms, noMoreData))
            {
                instance->pcmf32.clear();
                instance->pcmf32.insert(instance->pcmf32.end(),instance->pcmf32_old.begin(), instance->pcmf32_old.end());
                instance->pcmf32.insert(instance->pcmf32.end(), instance->pcmf32_new.begin(), instance->pcmf32_new.end());
                if (instance->pcmf32.size() < 2*WHISPER_SAMPLE_RATE)
                {
                    // Minimum 1sec of samples needed
                    instance->pcmf32_old.insert(instance->pcmf32_old.end(), instance->pcmf32_new.begin(), instance->pcmf32_new.end());
                    return kResultOk;
                }
                
                // Final segment if we detect silence at the end
                finalSegment = vad_simple(instance->pcmf32, WHISPER_SAMPLE_RATE, params.step_ms / 2, params.vad_thold, params.freq_thold, false);
                if (finalSegment)
                {
                    instance->pcmf32_old.clear();
                }
                else
                {
                    instance->pcmf32_old.insert(instance->pcmf32_old.end(),instance->pcmf32_new.begin(), instance->pcmf32_new.end());
                }
            }
        }
        else
        {
            // Read everything
            instance->audio.read(instance->pcmf32, kAvailableAudioBufferSize, noMoreData);
            finalSegment = true;
        }

        if (!instance->pcmf32.empty())
        {
            whisper_full_params wparams = whisper_full_default_params(strategy);

            wparams.single_segment = async;

            wparams.no_context = true;
            wparams.prompt_tokens = wparams.no_context ? nullptr : instance->prompt_tokens.data();
            wparams.prompt_n_tokens = wparams.no_context ? 0 : (int)instance->prompt_tokens.size();

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
            for (int i = 0; i < n_segments; ++i)
            {
                const char* text = whisper_full_get_segment_text(instance->model, i);
                // NOTE: In non production we send stats later with state set to done
                nvigi::InferenceExecutionState state = finalSegment ? nvigi::kInferenceExecutionStateDataPending : nvigi::kInferenceExecutionStateDataPartial;
#ifdef NVIGI_PRODUCTION
                // If last segment and no streaming or last chunk in a stream report done
                if (i == n_segments - 1 && noMoreData)
                {
                    state = nvigi::kInferenceExecutionStateDone;
                    instance->audio.clear();
                    instance->pcmf32.clear();
                    instance->pcmf32_new.clear();
                    instance->pcmf32_old.clear();
                }
#endif
                triggerCallback(execCtx, text, state);
            }
        }

#ifndef NVIGI_PRODUCTION
        if (noMoreData)
        {
            instance->audio.clear();
            instance->pcmf32_old.clear();
            instance->pcmf32.clear();
            instance->pcmf32_new.clear();
            
            triggerCallback(execCtx, "", nvigi::kInferenceExecutionStateDone);

            auto timings = whisper_get_timings(instance->model);

            const int32_t n_sample = std::max(1, timings.n_sample);
            const int32_t n_encode = std::max(1, timings.n_encode);
            const int32_t n_decode = std::max(1, timings.n_decode);
            const int32_t n_prompt = std::max(1, timings.n_prompt);

            NVIGI_LOG_INFO("timings:mel    %.2f", timings.t_mel_us / 1000.0f);
            NVIGI_LOG_INFO("timings:sample %.2f", 1e-3f * timings.t_sample_us);
            NVIGI_LOG_INFO("timings:encode %.2f", 1e-3f * timings.t_encode_us);
            NVIGI_LOG_INFO("timings:decode %.2f", 1e-3f * timings.t_decode_us);
            NVIGI_LOG_INFO("timings:prompt %.2f", 1e-3f * timings.t_prompt_us);
            NVIGI_LOG_INFO("timings:total  %.2f", timings.t_total_ms);
        }
#endif
        return kResultOk;
    };

    nvigi::Result result = kResultOk;
    if (async)
    {
        // Non-blocking, schedule work
        if (!instance->job.valid())
        {
            instance->running.store(true);
            instance->job = std::async(std::launch::async, [instance, triggerCallback, runInference]()->Result
            {
                auto res = kResultOk;
                while (instance->running.load() && res == kResultOk)
                {
                    res = runInference();
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
                instance->job = std::async(std::launch::async, [instance, triggerCallback, runInference]()->Result
                {
                    auto res = kResultOk;
                    while (instance->running.load() && res == kResultOk)
                    {
                        res = runInference();
                    }
                    return res;
                });
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
    return result;
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

#ifdef NVIGI_ASR_GFN_NVCF
    //! Defaults indicate no restrictions - plugin can run on any system, even without any adapter
    info.requiredVendor = nvigi::VendorId::eNone;
    info.minDriver = {};
    info.minOS = { NVIGI_DEF_MIN_OS_MAJOR, NVIGI_DEF_MIN_OS_MINOR, NVIGI_DEF_MIN_OS_BUILD };
    info.minGPUArch = {};
#elif GGML_USE_CUBLAS
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

    auto& ctx = (*asr::getContext());

    ctx.feature = asr::getFeatureId(nullptr);

#ifdef NVIGI_ASR_GFN_NVCF
    if (!framework::getInterface(framework, plugin::net::kId, &ctx.net))
    {
        return kResultInvalidState;
    }
#endif

    ctx.api.createInstance = asr::createInstance;
    ctx.api.destroyInstance = asr::destroyInstance;
    ctx.api.getCapsAndRequirements = asr::getCapsAndRequirements;

    framework->addInterface(ctx.feature, &ctx.api, 0);

    whisper_log_set(nvigi::asr::whisperLogCallback, nullptr);
    
    return kResultOk;
}

//! Main exit point - shutting down our plugin
//! 
Result nvigiPluginDeregister()
{
    auto& ctx = (*asr::getContext());

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