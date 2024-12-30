
# Automatic Speech Recognition (ASR) - Whisper Programming Guide

The focus of this guide is on using In-Game Inferencing to integrate an ASR Whisper model into an application. More details can be found here [OpenAI Whisper](https://openai.com/research/whisper)

Please read the `docs/ProgrammingGuideAI.md` located in the NVIGI Core package to learn more about overall AI inference API in NVIGI SDK. :only:`binary_pack:[Which may be found here in combined binary packs](../../../nvigi_core/docs/ProgrammingGuideAI.md)` 

> **IMPORTANT**: This guide might contain pseudo code, for the up to date implementation and source code which can be copy pasted please see the  [basic sample](../source/samples/nvigi.basic/basic.cpp)

## Version 1.0.0 General Access

## 1.0 INITIALIZE AND SHUTDOWN

Please read the `docs/ProgrammingGuide.md` located in the NVIGI Core package to learn more about initializing and shutting down NVIGI SDK. :only:`binary_pack:[Which may be found here in combined binary packs](../../../nvigi_core/docs/ProgrammingGuide.md)` 

## 2.0 OBTAIN ASR INTERFACE

Next, we need to retrieve ASR's API interface based on what variant we need (CPU, CUDA etc.)   **NOTE** only the local inference plugins are provided/supported in this early release.  The cloud plugins will be added in a later release:

```cpp

nvigi::IAutoSpeechRecognition* iasrLocal{};
// Here we are requesting interface for the GGML_CUDA implementation
if(NVIGI_FAILED(result, nvigiGetInterface(nvigi::plugin::asr::ggml::cuda::kId &iasrLocal))
{
    // Handle error here
}
```

> **NOTE:**
One can only obtain interface for a feature which is available on user system. Interfaces are valid as long as the NVIGI ASR feature (plugin) is loaded and active.

## 3.0 CREATE ASR INSTANCE(S)

Now that we have our interface we can use it to create our ASR instance. To do this, we need to provide information about ASR model we want to use, CPU/GPU resources which are available and various other creation parameters.  

> **NOTE** Only the local inference plugins are provided/supported in this release.  The cloud plugin(s) might be added in a later release.

We can obtain this information by requesting capabilities and requirements for a model or models. 

### 3.1 OBTAIN CAPABILITIES AND REQUIREMENTS FOR MODEL(S)

There are few options here:

#### LOCAL

* provide specific model GUID and VRAM budget and check if that particular model can run within the budget
* provide null model GUID and VRAM budget to get a list of models that can run within the budget
* provide null model GUID and 'infinite' (SIZE_MAX) VRAM budget to get a list of ALL models
 
#### CLOUD (VRAM ignored)

* provide specific model GUID to obtain CloudCapabilities which include URL and JSON request body for the endpoint used by the model
* provide null model GUID to get a list of ALL models (CloudCapabilities in this case will NOT provide any info)

Here is an example:

```cpp
nvigi::CommonCreationParameters common{};
common.utf8PathToModels = myPathToNVIGIModelRepository; // Path to provided NVIGI model repository (using UTF-8 encoding)
common.vramBudgetMB = myVRAMBudget;  // VRAM budget (SIZE_MAX if we want ALL models) - IGNORED FOR CLOUD MODELS
common.modelGUID = myModelGUID; // Model GUID, set to `nullptr` if we want all models
nvigi::CommonCapabilitiesAndRequirements* caps{};
if (NVIGI_FAILED(result, getCapsAndRequirements(igptLocal, common, &caps)))
{
    LOG("'getCapsAndRequirements' failed");
}

for (size_t i = 0; i < caps->numSupportedModels; i++)
{
    if (caps->modelFlags[i] & kModelFlagRequiresDownload)
    {
        // Local model, requires download
        continue;
    }        
    LOG("MODEL: %s VRAM: %llu", caps->supportedModelNames[i], caps->modelMemoryBudgetMB[i]);
}
```

Once we know which model we want here is an example on how to create an instance for it:


Here is an example:

```cpp
nvigi::InferenceInstance* asrInstanceLocal;
{    
    nvigi::CommonCreationParameters common{};
    nvigi::ASRCreationParameters params{};    
    common.numThreads = myNumCPUThreads; // How many CPU threads is instance allowed to use, relevant only if using CPU feature
    common.vramBudgetMB = myVRAMBudget;  // How much VRAM is instance allowed to occupy
    common.utf8PathToModels = myPathToNVIGIModelRepository; // Path to provided NVIGI model repository (using UTF-8 encoding)
    common.modelGUID = "{5CAD3A03-1272-4D43-9F3D-655417526170}"; // Model GUID for Whisper, for details please see NVIGI models repository
    params.chain(common);

    //! Optional but highly recommended if using D3D context, if NOT provided performance might not be optimal
    nvigi::D3D12Parameters d3d12Params{};
    d3d12Params.device = myDevice;
    d3d12Params.queue = myQueue;
    params2.chain(d3d12Params);
    
    if(NVIGI_FAILED(res, iasrLocal->createInstance(params, &asrInstanceLocal)))
    {
        LOG("NVIGI call failed, code %d", res);
    }
}
```

> **IMPORTANT**: Providing D3D or Vulkan device and queue is highly recommended to ensure optimal performance

## 4.0 AUDIO INPUT

### 4.1 PROCESS COMPLETE AUDIO CLIP

The NVIGISample, a 3D rendered interactive NVIGI sample (provided with the SDK) provides helper functions which can be used to record a user's voice in an optimal format for the inference. These may be found in the files `nvigi/AudioRecordingHelper.*`
Here is an example of their use:

```cpp
AudioRecordingHelper::RecordingInfo* audioInfo = AudioRecordingHelper::StartRecordingAudio();

if(audioInfo == nullptr) 
{ 
    //! Check error 
} 

// ...

//! Incoming audio is stored in special inference audio variable
//! 
//! 16000 samples mono WAVE suitable for use with most ASR models
nvigi::CpuData audioData;
nvigi::InferenceDataAudio wavData(audioData);
if(!AudioRecordingHelper::StopRecordingAudio(audioInfo, &wavData)) 
{ 
    //! Check error 
} 
```

It is not mandatory to use the NVIGI helpers to record voice, your own recording method can be used but it **must record `WAVE` with a sampling rate of 16000 and 16bits per sample**. Here is an example setup for the audio input slot:

### 4.2 STREAM AUDIO 

NVIGI does not provide a way to record audio stream since that is out of the scope of this SDK. Any recording method can be used as long as produced audio samples are **single channel (mono) with 16000 sample rate**.

```cpp

// Mark each input based on if this is start, middle or end of a stream
nvigi::StreamingParameters audioChunkInfo = audioChunkInfo.type = nvigi::StreamSignalType::eStreamData;
if(firstChunk)
{
    audioChunkInfo.type = nvigi::StreamSignalType::eStreamStart;
}
else if(lastChunk)
{
    audioChunkInfo.type = nvigi::StreamSignalType::eStreamStop;
}
// We will chain this later to the runtime parameters in our execution context

```

> NOTE: See [section 7 below](#70-add-asr-inference-to-the-pipeline) for more details

### 4.3 PREPARE INPUT SLOT

```cpp
// Can be full audio clip or streamed audio chunk, in this case pcm16 as the most commonly used audio format
std::vector<int16> pcm16 = getMyAudio();
nvigi::CpuData _audio{pcm16.size() * sizeof(int16_t), pcm16.data()}; 
nvigi::InferenceDataAudio audio{_audio};
//! These are defaults and required to run inference with the OpenAI Whisper model
audio.bitsPerSample = 16;
audio.samplingRate = 16000;
audio.channels = 1;
```

## 5.0 SETUP CALLBACK TO RECEIVE INFERRED DATA

In order to receive transcribed text from the ASR model inference a special callback needs to be setup like this:

```cpp
auto asrCallback = [](const nvigi::InferenceExecutionContext* execCtx, nvigi::InferenceExecutionState state, void* userData)->nvigi::InferenceExecutionState 
{     
    //! Optional user context to control execution 
    auto userCtx = (HostProvidedASRCallbackCtx*)userData; 
    if (execCtx->outputs) 
    { 
       const nvigi::InferenceDataText* text{}; 
       execCtx->outputs->findAndValidateSlot(nvigi::kASRDataSlotTranscribedText, &text); 
       if(text)
       {
            std::string transcribedText = text->getUtf8Text();
            //! Do something with the received text 
            if (state == nvigi::kInferenceExecutionStateDataPending) 
            { 
                //! This is final data, more data is pending
            } 
            else if (state == nvigi::kInferenceExecutionStateDataPartial) 
            { 
                //! This is partial data and subject to change, more data is pending
            }
       }
    } 
    
    if (state == nvigi::kInferenceExecutionStateDone) 
    { 
        //! This is all the data we can expect to receive 
    }    
    
    if(userCtx->needToInterruptInference) 
    { 
        //! Inform NVIGI that inference should be cancelled 
        return nvigi::kInferenceExecutionStateCancel; 
    } 
    return state; 
}; 
```
> **IMPORTANT:**
> Input and output data slots provided within the execution context are **only valid during the callback execution**. Host application must be ready to handle callbacks until reaching `nvigi::InferenceExecutionStateDone` or `nvigi::InferenceExecutionStateCancel` state.

> **NOTE:**
> To cancel ASR inference make sure to return `nvigi::InferenceExecutionStateCancel` state in the callback.

## 6.0 PREPARE THE EXECUTION CONTEXT

Before ASR can be evaluated the `nvigi::InferenceExecutionContext` needs to be defined:

```cpp
//! Audio data slot is coming from our previous step
std::vector<nvigi::InferenceDataSlot> slots = { {nvigi::kASRDataSlotAudio, &audio} };
nvigi::InferenceDataSlotArray inputs = { slots.size(), slots.data() }; // Input slots

//! OPTIONAL Runtime parameters, we can for example switch between Greedy or BeamSearch sampling strategies
nvigi::ASRRuntimeParameters asrRuntime{};
runtime.sampling = nvigi::ASRSamplingStrategy::eBeamSearch;
   
nvigi::InferenceExecutionContext asrContext{};
asrContext.instance = asrInstanceLocal;         // The instance we created and we want to run inference on
asrContext.callback = asrCallback;              // Callback to receive transcribed text
asrContext.callbackUserData = &asrCallback;     // Optional context for the callback, can be null if not needed
asrContext.inputs = &inputs;
asrContext.runtimeParameters = &asrRuntime;      
```

> **IMPORTANT:**
> The execution context and all provided data must be valid at the time `instance->evaluate{Async}` is called (see below for more details).

## 7.0 ADD ASR INFERENCE TO THE PIPELINE

In your execution pipeline, call `instance->evaluate` to process single audio clip or `instance->evaluateAsync` to process audio stream at the appropriate location where audio needs to be transcribed. 

```cpp
// Make sure ASR is available and user selected this option in the UI
if(useASR) 
{
    if(audioStream)
    {
        // Non-blocking call - evaluate our instance asynchronously with all the provided parameters, transcribed text is received via callback on a different thread

        // Mark each input based on if this is start, middle or end of a stream
        nvigi::StreamingParameters audioChunkInfo{};
        audioChunkInfo.type = nvigi::StreamSignalType::eStreamData;
        if(firstChunk)
        {
            audioChunkInfo.type = nvigi::StreamSignalType::eStreamStart;
        }
        else if(lastChunk)
        {
            audioChunkInfo.type = nvigi::StreamSignalType::eStreamStop;
        }
        asrRuntime.chain(audioChunkInfo);

        // IMPORTANT: In this case execution context and all input data MUST BE VALID while audio streaming is active
        if(NVIGI_FAILED(res, asrContext.instance->evaluateAsync(asrContext)))
        {
            LOG("NVIGI call failed, code %d", res);
        }
    }   
    else
    {
        // Blocking call - evaluate our instance with all the provided parameters, transcribed text is received via callback on the same thread
        if(NVIGI_FAILED(res, asrContext.instance->evaluate(asrContext)))
        {
            LOG("NVIGI call failed, code %d", res);
        }
    }         
}
```

> **IMPORTANT:**
> When using `instance->evaluateAsync` the host app must ensure that the execution context and all inputs are valid until all streaming data is processed fully and callback receives `nvigi::kInferenceExecutionStateDone` state.

## 8.0 DESTROY INSTANCE(S)

Once ASR is no longer needed each instance should be destroyed like this:

```cpp
//! Finally, we destroy our instance(s)
if(NVIGI_FAILED(result, iasrLocal->destroyInstance(asrInstanceLocal))) 
{ 
    //! Check error
}
```

## 9.0 UNLOAD INTERFACE(S)

Once ASR is no longer needed each interface should be unloaded like this.  **NOTE** only the local inference plugins are provided/supported in this early release.  The cloud plugins will be added in a later release:

```cpp
//! Finally, we unload the interface since we no longer need ASR
if(NVIGI_FAILED(result, nvigiUnloadInterface(nvigi::plugin::asr::ggml::cuda::kId, &iasrLocal))) 
{ 
    //! Check error
} 
```
