# Automatic Speech Recognition (ASR) - Whisper Programming Guide

The focus of this guide is on using In-Game Inferencing to integrate an ASR Whisper model into an application. More details can be found here [OpenAI Whisper](https://openai.com/research/whisper)

Please read the [Programming Guide for AI](nvigi_core/docs/ProgrammingGuideAI.md) located in the NVIGI Core package to learn more about overall AI inference API in NVIGI SDK.

> **MIN RUNTIME SPEC:** Note that all Whisper ASR backends require a CPU supporting AVX2 instructions.  Support for this instruction extension is ubiquitous in modern gaming CPUs, but older hardware may not support it.

> **IMPORTANT**: This guide might contain pseudo code, for the up to date implementation and source code which can be copy pasted please see the SDK's Basic command line sample [Source Code](../source/samples/nvigi.basic/basic.cpp) and [Docs](Samples.md).  The Basic command-line sample includes the option to record audio and converting it to a text query for an LLM via the ASR plugins.

> **IMPORTANT NOTE: The D3D12 backend (nvigi.plugin.asr.ggml.d3d12.dll) is provided only precompiled as a part of the downloadable binary pack (`nvigi_pack`).  It is not possible for developers to compile the D3D12 backend plugin from source in this release.**

> **IMPORTANT NOTE: The D3D12 backend (nvigi.plugin.asr.ggml.d3d12.dll) requires an NVIDIA R580 driver or newer in order to be available at runtime.**

> **IMPORTANT NOTE: The CUDA backend (nvigi.plugin.asr.ggml.cuda.dll) strongly recommends an NVIDIA R580 driver or newer in order to avoid a potential memory leak if CiG (CUDA in Graphics) is used and the application deletes D3D12 command queues mid-application.**

## 1.0 INITIALIZE AND SHUTDOWN

Please read the [Programming Guide](nvigi_core/docs/ProgrammingGuide.md) located in the NVIGI Core package to learn more about initializing and shutting down NVIGI SDK. 

### 1.1 MODERN C++ WRAPPER (RECOMMENDED)

The NVIGI SDK provides modern C++ wrappers that simplify initialization and provide a cleaner API with RAII, `std::expected`, and builder patterns. The wrappers are located in `source/samples/nvigi.basic.cxx/` and can be used in your projects.

```cpp
#include "core.hpp"
#include "asr.hpp"

using namespace nvigi::asr;

// Initialize NVIGI core with builder pattern
nvigi::Core core({ 
    .sdkPath = "path/to/sdk",
    .logLevel = nvigi::LogLevel::eDefault,
    .showConsole = true 
});

// Access system information
const auto& sysInfo = core.getSystemInfo();
std::cout << "Detected " << sysInfo.getNumPlugins() << " plugins\n";
std::cout << "Detected " << sysInfo.getNumAdapters() << " adapters\n";
```

> **NOTE:** The C++ wrappers provide the same functionality as the low-level API but with modern C++ idioms. Both approaches are valid and can be mixed if needed.

## 2.0 OBTAIN ASR INTERFACE

Next, we need to retrieve ASR's API interface based on what variant we need (CPU, CUDA, D3D12, Vulkan etc.)   **NOTE** only the local inference plugins are provided/supported in this early release.  The cloud plugins will be added in a later release:

> **IMPORTANT NOTE: D3D12 and Vulkan backends are experimental and might not behave or perform as expected.**

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

### 2.1 MODERN C++ WRAPPER APPROACH

The C++ wrapper handles interface loading automatically during instance creation. You don't need to manually obtain interfaces:

```cpp
// No manual interface loading needed!
// Just create the instance with your desired backend
```

See section 3.2 for complete instance creation examples using the wrapper.

## 3.0 CREATE ASR INSTANCE(S)

Now that we have our interface we can use it to create our ASR instance. To do this, we need to provide information about ASR model we want to use, CPU/GPU resources which are available and various other creation parameters.  

> **NOTE** Only the local inference plugins are provided/supported in this release.  The cloud plugin(s) might be added in a later release.

We can obtain this information by requesting capabilities and requirements for a model or models. 

### 3.1 OBTAIN CAPABILITIES AND REQUIREMENTS FOR MODEL(S)

> **IMPORTANT NOTE**: This section covers a scenario where the host application can instantiate models that were not included when the application was packaged and shipped. If the models and their capabilities are predefined and there is no need for dynamically downloaded models, you can skip to the next section.

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

### 3.2 CREATE MODEL INSTANCE

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
    if(NVIGI_FAILED(params.chain(common)))
    {
        // handle error
    }
```
Next we need to provide information about D3D12 properties if our application is running with a D3D12 context or planning to leverage `CiG` (CUDA In Graphics)

> NOTE: For Vulkan requirements please see section in the Appendix

```cpp
    //! Required if using D3D12 context
    nvigi::D3D12Parameters d3d12Params{};
    d3d12Params.device = myDevice; // mandatory, must support sm_6_6 or higher if using D3D12
    d3d12Params.queue = myDirectQueue; // mandatory to use CIG, optional for D3D12, if provided AI miscellaneous D3D12 workloads will be executed on this queue
    d3d12Params.queueCompute = myComputeQueue; // optional, if provided AI compute D3D12 workloads will be executed on this queue
    d3d12Params.queueCopy = myCopyQueue; // optional, if provided AI copy D3D12 workloads will be executed on this queue
    if(NVIGI_FAILED(params.chain(d3d12Params)))
    {
        // handle error
    }
```

> IMPORTANT: Do NOT chain the same parameters to the multiple parameter chains, the recommended approach is to make a copy per chain. For example, creating an ASR and GPT instance with shared d3d12Params can result in re-chaining the input parameters the wrong way which then results in failed instance creation.

```cpp
    
    if(NVIGI_FAILED(res, iasrLocal->createInstance(params, &asrInstanceLocal)))
    {
        LOG("NVIGI call failed, code %d", res);
    }
}
```

> **IMPORTANT**: Providing D3D or Vulkan device and queue is highly recommended to ensure optimal performance

### 3.3 MODERN C++ WRAPPER APPROACH

The C++ wrapper simplifies instance creation with builder patterns and automatic resource management:

```cpp
#include "d3d12.hpp"  // or "vulkan.hpp"

using namespace nvigi::asr;

// Setup D3D12 (if using D3D12 or CUDA backend)
auto deviceAndQueue = nvigi::d3d12::D3D12Helper::create_best_compute_device();
nvigi::d3d12::D3D12Config d3d12_config = {
    .device = deviceAndQueue.device.Get(),
    .command_queue = deviceAndQueue.compute_queue.Get(),
    .create_committed_resource_callback = nvigi::d3d12::default_create_committed_resource,
    .destroy_resource_callback = nvigi::d3d12::default_destroy_resource
};

// Or setup Vulkan (if using Vulkan backend)
auto vk_objects = nvigi::vulkan::VulkanHelper::create_best_compute_device();
nvigi::vulkan::VulkanConfig vk_config = {
    .instance = vk_objects.instance,
    .physical_device = vk_objects.physical_device,
    .device = vk_objects.device,
    .compute_queue = vk_objects.compute_queue,
    .transfer_queue = vk_objects.transfer_queue,
    .allocate_memory_callback = nvigi::vulkan::default_allocate_memory,
    .free_memory_callback = nvigi::vulkan::default_free_memory
};

// Create ASR instance with builder pattern
auto instance = Instance::create(
    ModelConfig{
        .backend = "d3d12",  // or "cuda", "vulkan"
        .guid = "{5CAD3A03-1272-4D43-9F3D-655417526170}",  // Whisper Small
        .model_path = "path/to/nvigi.models",
        .num_threads = 8,
        .vram_budget_mb = 2048,
        .flash_attention = true,
        .language = "en",       // or "auto" for detection
        .translate = false,     // translate to English
        .detect_language = false
    },
    d3d12_config,      // Pass your config based on backend
    vk_config,         // Can pass both, unused ones are ignored
    core.loadInterface(),
    core.unloadInterface()
).value();  // Will throw if creation fails

// Instance is ready to use!
// RAII ensures proper cleanup when instance goes out of scope
```

The wrapper automatically:
- Loads the correct plugin based on backend
- Chains all creation parameters correctly
- Manages interface lifetimes
- Provides clear error messages via `std::expected`
- Cleans up resources when destroyed

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
nvigi::InferenceDataAudioSTLHelper wavData;
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
std::vector<int16> pcm16 = getMyMonoAudio();
nvigi::InferenceDataAudioSTLHelper audio{pmc16, 1}; // assuming single channel mono audio
```

## 5.0 RECEIVE INFERRED DATA
There are two ways to receive data from ASR inference when using evaluateAsync: using a callback or polling for results.

### 5.1 CALLBACK APPROACH
To receive transcribed text via callback, set up the callback handler like this:

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

### 5.1.1 MODERN C++ WRAPPER APPROACH

The C++ wrapper simplifies callbacks with modern C++ lambdas and cleaner state management:

```cpp
using namespace nvigi::asr;

// Record audio (using helper from wrapper)
auto* recording = AudioRecorder::StartRecording();
std::this_thread::sleep_for(std::chrono::seconds(5));
auto audioData = AudioRecorder::StopRecording(recording);

// Simple callback - just process the text
auto result = instance->transcribe(
    audioData.data(),
    audioData.size(),
    RuntimeConfig{}
        .set_sampling(SamplingStrategy::Greedy)
        .set_temperature(0.0f),
    [](std::string_view text, ExecutionState state) -> ExecutionState {
        // Callback receives text and state
        std::cout << text;  // Print transcribed text
        
        // Cancel if needed
        if (should_cancel) {
            return ExecutionState::Cancel;
        }
        
        return state;  // Continue with current state
    }
);

if (!result) {
    std::cerr << "Error: " << result.error().what() << "\n";
}
```

The wrapper provides:
- Cleaner lambda syntax with `std::string_view`
- Enum-based state management (`ExecutionState::Done`, `ExecutionState::Cancel`)
- `std::expected` for error handling
- No manual memory management needed

### 5.2 POLLING APPROACH

Alternatively, when using evaluateAsync, you can poll for results instead of using a callback. This is useful when you want more control over when to process results or need to integrate with a polling-based architecture:

```cpp
// Start async evaluation without a callback
asrContext.callback = nullptr;
if (NVIGI_FAILED(res, asrContext.instance->evaluateAsync(&asrContext))) {
    LOG("NVIGI async evaluation failed, code %d", res);
    return;
}

// Poll for results
while (true) {
    nvigi::InferenceExecutionState state;
    
    // Get current results - pass true to wait for new data, false to check immediately
    if (NVIGI_FAILED(res, asrContext.instance->getResults(&asrContext, true, &state))) {
        LOG("Failed to get results, code %d", res);
        break;
    }
    
    // Process the current results if available
    if (asrContext.outputs) {
        const nvigi::InferenceDataText* text{};
        asrContext.outputs->findAndValidateSlot(nvigi::kASRDataSlotTranscribedText, &text);
        if (text) {
            std::string transcribedText = text->getUtf8Text();
            // Process the transcribed text
        }
    }
    
    // Release the current results to free resources
    if (NVIGI_FAILED(res, asrContext.instance->releaseResults(&asrContext, state))) {
        LOG("Failed to release results, code %d", res);
        break;
    }
    
    // Check if inference is complete
    if (state == nvigi::kInferenceExecutionStateDone) {
        break;
    }
}
```

## 6.0 PREPARE THE EXECUTION CONTEXT

### 6.1 RUNTIME PARAMETERS

The `ASRWhisperRuntimeParameters` structure provides several options to control the behavior of the ASR inference:

```cpp
struct ASRWhisperRuntimeParameters {
    ASRWhisperSamplingStrategy sampling = ASRWhisperSamplingStrategy::eGreedy; // Sampling strategy
    int32_t bestOf = 1;            // For greedy sampling, number of candidates to consider (1 = disabled)
    int32_t beamSize = -1;         // For beam search, number of beams (-1 = disabled)
    
    // v2 parameters
    const char* prompt{};          // Optional prompt to guide the transcription
    bool noContext = true;         // If true, do not use previous context
    bool suppressBlank = true;     // If true, suppresses blank tokens in output
    bool suppressNonSpeechTokens = false;  // If true, suppresses non-speech tokens
    float temperature = 0.0f;      // Sampling temperature (0.0 = greedy, higher = more random)
    float entropyThold = 2.4f;     // Entropy-based suppression threshold (0.0 = disabled)
    float logprobThold = -1.0f;    // Log-probability suppression threshold (0.0 = disabled)
    float noSpeechThold = 0.6f;    // No-speech detection threshold (0.0 = disabled)
};
```

**Sampling Parameters:**
- `sampling`: Choose between greedy decoding or beam search
- `bestOf`: For greedy sampling, consider multiple candidates (useful with temperature > 0)
- `beamSize`: For beam search, specify number of beams to use

**Control Parameters (v2):**
- `prompt`: Provide context to guide transcription
- `noContext`: Enable/disable using previous context
- `suppressBlank`: Control blank token suppression
- `suppressNonSpeechTokens`: Filter out non-speech tokens
- `temperature`: Control randomness in sampling
- `entropyThold`: Threshold for entropy-based token filtering
- `logprobThold`: Threshold for probability-based filtering
- `noSpeechThold`: Sensitivity for no-speech detection

### 6.2 EXECUTION CONTEXT

Before ASR can be evaluated the `nvigi::InferenceExecutionContext` needs to be defined:

```cpp
//! Audio data slot is coming from our previous step
std::vector<nvigi::InferenceDataSlot> slots = { {nvigi::kASRDataSlotAudio, audio} };
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
        if(NVIGI_FAILED(asrRuntime.chain(audioChunkInfo)))
        {
            // handle error
        }

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

### 7.1 MODERN C++ WRAPPER APPROACH

The C++ wrapper provides a high-level `Stream` API for audio streaming:

**Complete Audio Processing (Blocking):**

```cpp
using namespace nvigi::asr;

// Record audio
auto* recording = AudioRecorder::StartRecording();
std::this_thread::sleep_for(std::chrono::seconds(5));
auto audioData = AudioRecorder::StopRecording(recording);

// Transcribe (blocking with callback)
std::string transcription;
auto result = instance->transcribe(
    audioData.data(),
    audioData.size(),
    RuntimeConfig{}
        .set_sampling(SamplingStrategy::Greedy)
        .set_temperature(0.0f)
        .set_best_of(2),
    [&transcription](std::string_view text, ExecutionState state) -> ExecutionState {
        transcription += text;
        std::cout << text << std::flush;
        return state;
    }
);

if (result) {
    std::cout << "\nFinal: " << transcription << "\n";
}
```

**Real-Time Streaming Mode:**

```cpp
// Create a stream
auto stream = instance->create_stream(
    RuntimeConfig{}
        .set_sampling(SamplingStrategy::Greedy)
        .set_temperature(0.0f)
);

// Start recording
auto* recording = AudioRecorder::StartRecording();

// Stream audio chunks
size_t chunk_size = 6400;  // 200ms at 16kHz
size_t bytes_processed = 0;
bool is_first = true;

while (true) {
    // Get chunk (thread-safe)
    std::vector<uint8_t> chunk_data;
    {
        std::lock_guard<std::mutex> lock(recording->mutex);
        size_t available = recording->bytesWritten - bytes_processed;
        
        if (available >= chunk_size) {
            chunk_data.resize(chunk_size);
            memcpy(chunk_data.data(), 
                   recording->audioBuffer.data() + bytes_processed,
                   chunk_size);
            bytes_processed += chunk_size;
        }
    }
    
    if (!chunk_data.empty()) {
        // Send chunk async (non-blocking!)
        auto op = stream.send_audio_async(
            chunk_data.data(),
            chunk_data.size(),
            is_first,
            false  // Not last chunk
        ).value();
        
        is_first = false;
        
        // Poll for results in game loop
        if (auto result = op.try_get_results()) {
            if (!result->text.empty() && result->text != "[BLANK_AUDIO]") {
                std::cout << result->text << std::flush;
            }
        }
    }
    
    // Check for stop condition...
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

auto audioData = AudioRecorder::StopRecording(recording);
```

**Key Features:**
- `transcribe()` - Blocking transcription with callbacks
- `create_stream()` - Real-time streaming with automatic chunk management
- `send_audio_async()` - Non-blocking audio processing
- Builder pattern for runtime configuration
- RAII resource management

### 7.2 CANCELLING ASYNC EVALUATION

When using `evaluateAsync`, you can cancel an ongoing inference operation early using the `cancelAsyncEvaluation` API. This is useful when you need to interrupt audio processing due to user actions (e.g., pressing ESC), timeouts, or changing contexts.

The cancellation mechanism is designed to interrupt the evaluation loop as early as possible, including during audio segment processing.

Here's how to cancel an async evaluation:

```cpp
// Start async evaluation for audio streaming
asrContext.callback = nullptr;
nvigi::StreamingParameters audioChunkInfo{};
audioChunkInfo.signal = nvigi::StreamSignal::eStreamSignalStart;
asrRuntime.chain(audioChunkInfo);

if (NVIGI_FAILED(res, asrContext.instance->evaluateAsync(&asrContext))) {
    LOG("NVIGI async evaluation failed, code %d", res);
    return;
}

// ... continue sending audio chunks ...

// User decides to cancel
if (NVIGI_FAILED(res, asrInstance->cancelAsyncEvaluation(&asrContext))) {
    if (res == kResultNoImplementation) {
        LOG("No async evaluation is currently running");
    } else {
        LOG("Failed to cancel evaluation, code %d", res);
    }
}

// The processing will stop as soon as possible
// Continue polling to clean up
nvigi::InferenceExecutionState state;
while (true) {
    if (NVIGI_FAILED(res, asrInstance->getResults(&asrContext, false, &state))) {
        break;
    }
    
    // Release any remaining results
    asrInstance->releaseResults(&asrContext, state);
    
    if (state == nvigi::kInferenceExecutionStateDone || 
        state == nvigi::kInferenceExecutionStateInvalid) {
        break;
    }
}
```

#### Important Notes:

- **`cancelAsyncEvaluation` returns `kResultNoImplementation`** if no async job is running (i.e., `evaluateAsync` was not called or the job has already completed)
- The cancellation is **thread-safe** and can be called from any thread
- After calling `cancelAsyncEvaluation`, continue polling with `getResults` to properly clean up any remaining resources
- The evaluation loop checks for cancellation at **strategic points**:
  - During the main async processing loop
  - Inside audio segment processing
- Cancellation is designed to be **as fast as possible**, typically interrupting within a few milliseconds

#### Example: User-Initiated Cancellation During Streaming

```cpp
// Track async state
std::atomic<bool> userRequestedCancel = false;
std::thread monitorThread;

// Start monitoring for user input
monitorThread = std::thread([&]() {
    while (!userRequestedCancel) {
        if (checkUserPressedEscape()) {
            userRequestedCancel = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
});

// Stream audio chunks
asrContext.callback = nullptr;
nvigi::StreamingParameters streamParams{};
asrRuntime.chain(streamParams);

std::vector<AudioChunk> chunks = getAudioChunks();
for (size_t i = 0; i < chunks.size(); i++) {
    // Check if user wants to cancel
    if (userRequestedCancel) {
        asrInstance->cancelAsyncEvaluation(&asrContext);
        LOG("User cancelled audio processing");
        break;
    }
    
    // Set stream signal
    streamParams.signal = (i == 0) ? nvigi::StreamSignal::eStreamSignalStart :
                          (i == chunks.size() - 1) ? nvigi::StreamSignal::eStreamSignalStop :
                          nvigi::StreamSignal::eStreamSignalData;
    
    // Send audio chunk
    audioData.buffer = chunks[i].data();
    audioData.sizeInBytes = chunks[i].size();
    
    if (NVIGI_FAILED(res, asrContext.instance->evaluateAsync(&asrContext))) {
        LOG("Failed to send audio chunk");
        break;
    }
}

// Poll for any remaining results
nvigi::InferenceExecutionState state;
while (true) {
    if (NVIGI_FAILED(res, asrInstance->getResults(&asrContext, true, &state))) {
        break;
    }
    
    // Process partial results if available
    if (asrContext.outputs && !userRequestedCancel) {
        const nvigi::InferenceDataText* text{};
        if (asrContext.outputs->findAndValidateSlot(kASRWhisperDataSlotTranscribedText, &text)) {
            std::string transcribed = text->getUTF8Text();
            displayPartialTranscription(transcribed);
        }
    }
    
    asrInstance->releaseResults(&asrContext, state);
    
    if (state == nvigi::kInferenceExecutionStateDone || 
        state == nvigi::kInferenceExecutionStateInvalid) {
        break;
    }
}

monitorThread.join();
```

> **NOTE**: Cancellation via `cancelAsyncEvaluation` is only available for async evaluation started with `evaluateAsync`. For synchronous evaluation, use the callback return value mechanism (return `kInferenceExecutionStateCancel` from the callback) as described in section 5.1.

## 8.0 DESTROY INSTANCE(S)

Once ASR is no longer needed each instance should be destroyed like this:

```cpp
//! Finally, we destroy our instance(s)
if(NVIGI_FAILED(result, iasrLocal->destroyInstance(asrInstanceLocal))) 
{ 
    //! Check error
}
```

### 8.1 MODERN C++ WRAPPER APPROACH

The C++ wrapper uses RAII for automatic resource management - no manual cleanup needed:

```cpp
{
    // Initialize core
    nvigi::Core core({ .sdkPath = "path/to/sdk" });
    
    // Create instance
    auto instance = Instance::create(
        ModelConfig{ /* ... */ },
        d3d12_config,
        vk_config,
        core.loadInterface(),
        core.unloadInterface()
    ).value();
    
    // Use instance...
    auto result = instance->transcribe(audio_data, audio_size);
    
    // Automatic cleanup when leaving scope!
    // 1. instance is destroyed -> calls destroyInstance()
    // 2. core is destroyed -> calls nvigiShutdown()
}
// All resources cleaned up automatically!
```

**Key Benefits:**
- No manual `destroyInstance()` calls needed
- No manual `nvigiUnloadInterface()` calls needed
- Exception-safe: cleanup happens even if exceptions are thrown
- Impossible to forget cleanup or get order wrong
- Reference counting ensures interfaces stay valid

## 9.0 UNLOAD INTERFACE(S)

Once ASR is no longer needed each interface should be unloaded like this.  **NOTE** only the local inference plugins are provided/supported in this early release.  The cloud plugins will be added in a later release:

```cpp
//! Finally, we unload the interface since we no longer need ASR
if(NVIGI_FAILED(result, nvigiUnloadInterface(nvigi::plugin::asr::ggml::cuda::kId, &iasrLocal))) 
{ 
    //! Check error
} 
```

### 9.1 MODERN C++ WRAPPER APPROACH

Interface unloading is handled automatically by the wrapper - see section 8.1 above.

## APPENDIX

### Supported Whisper Models

The SDK supports the following Whisper models in GGML format (quantized for optimal performance):

| Model | GUID | Description | Size |
|-------|------|-------------|------|
| Tiny | {4D180D1F-9267-44A8-A862-5575AB8E93EB} | Fastest, English-only model suitable for simple transcription tasks | 75 MB |
| Small | {5CAD3A03-1272-4D43-9F3D-655417526170} | Good balance of speed and accuracy for English content | 150 MB |
| Base | {807D0AEC-DD9A-4BD8-ADB1-EEADDA89DC00} | Improved accuracy over small, maintains good performance | 300 MB |
| Medium | {ABD526A2-4AFE-4550-88E2-D9F70F68D8C3} | High accuracy model with reasonable performance | 1.5 GB |
| Large | {9FE5583A-AF35-4823-9C9D-5FE990E8D868} | Most accurate model, requires more computational resources | 2.9 GB |

### Model Selection Guide

Choose a model based on your requirements:
- **Tiny**: Best for real-time applications with limited resources
- **Small/Base**: Good for general purpose transcription
- **Medium**: Recommended for most applications requiring high accuracy
- **Large**: Best for applications requiring maximum accuracy where performance is not critical

> NOTE: 8-bit quantization is probably the best choice when it comes to balancing between memory consumption and accuracy

### WHISPER.CPP

The `nvigi.plugin.asr.ggml.{$backend}` plugins use a specific snapshot of whisper.cpp therefore it is not guaranteed that NVIGI version will match the latest whisper.cpp capabilities. When comparing the two please note the following:

* NVIGI version is compiled with potentially different CPU flags (lowest common denominator to allow wide CPU support, not necessarily including the latest greatest CPU features)
* NVIGI input parameters should be modified to match whisper.cpp 1:1 (context size, batch size, number of threads etc.) when comparing performance
* NVIGI version is modified to **allow optimal execution inside of a process** (especially when it comes to CUDA in Graphics) hence it might NOT perform as fast as whisper.cpp on an idle GPU
* Performance and capabilities of whisper.cpp change on daily basis, NVIGI version will be updated at much slower cadence 

#### D3D12

When using D3D12 backend, the host application must created a device which supports shader model 6.6 or higher. To ensure proper support across various Windows OS versions the recommended approach is to include `Microsoft Agility SDK version 1.600.0` or newer with your executable by adding the following code:

```cpp
extern "C" __declspec(dllexport) UINT         D3D12SDKVersion = 610; // Change this as needed to reflect the version you want to use
extern "C" __declspec(dllexport) const char * D3D12SDKPath    = ".\\D3D12\\";
```
> NOTE: `D3D12` folder must be created next to the executable and it must contain `D3D12Core.dll` which is provided with the Agility SDK

The additional benefit of including the latest Agility SDK is the performance enhancement which comes with the introduction of the new heap type `D3D12_HEAP_TYPE_GPU_UPLOAD`. This new feature enables simultaneous CPU and GPU access to VRAM via the Resizable BAR (ReBAR) mechanism-was introduced to the DirectX 12 API through the Direct3D Agility SDK and corresponding Windows updates. This feature allows for more efficient data transfers, reducing the need for CPU-to-GPU copy operations and potentially improving performance in certain scenarios. For more details please visit https://devblogs.microsoft.com/directx/preview-agility-sdk-1-710-0/

| Feature                | First Supported Windows OS                       | First Supported Agility SDK Version   |
|------------------------|--------------------------------------------------|---------------------------------------|
| GPU UPLOAD HEAP (ReBAR)| Windows 11 Insider Preview Build 26080 or later  | 1.613.0                               |

> IMPORTANT: Please note that on some systems ReBAR must be explicitly enabled in the BIOS.

In addition to the above, it is also required to distribute `dxcompiler.dll` with your application.


#### VULKAN

> NOTE: This section is relevant only if the host application is providing `nvigi::VulkanParameters` to the NVIGI ASR plugin

Here are the Vulkan requirements:

* `VkInstance` must be created with the API 1.3.0 or higher
* `VkDevice` must be created with `VkPhysicalDeviceFeatures2`, `VkPhysicalDeviceVulkan11Features` and `VkPhysicalDeviceVulkan12Features` chained to the `VkDeviceCreateInfo`
* The following extensions must be enabled if physical device supports them:

```cpp
"VK_EXT_pipeline_robustness",
"VK_KHR_maintenance4",
"VK_EXT_subgroup_size_control",
"VK_KHR_16bit_storage",
"VK_KHR_shader_float16_int8",
"VK_KHR_cooperative_matrix",
"VK_NV_cooperative_matrix2"
```
> NOTE: If certain extensions are not available the appropriate fallbacks will be used if possible

#### MEMORY TRACKING 

##### VULKAN

NVIGI provides callback mechanism to track/allocated/free GPU resources as defined in the `nvigi_vulkan.h` header. Here is an example:

```cpp
VkResult allocateMemoryVK(VkDevice device, VkDeviceSize size, uint32_t memoryTypeIndex, VkDeviceMemory* outMemory) {
    // Define the memory allocation info
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = size;           // Size of memory to allocate
    allocInfo.memoryTypeIndex = memoryTypeIndex; // Memory type (e.g., from vkGetPhysicalDeviceMemoryProperties)

    // Allocate the memory
    VkDeviceMemory memory;
    VkResult result = vkAllocateMemory(device, &allocInfo, nullptr, &memory);

    if (result != VK_SUCCESS) {
        // Handle error (e.g., VK_ERROR_OUT_OF_DEVICE_MEMORY)
        return result;
    }

    // Output the allocated memory handle
    *outMemory = memory;
    gpuResourceCount++;
    gpuResourceAllocation = std::max(gpuResourceAllocation.load(), size);
    return VK_SUCCESS;
}

void freeMemoryVK(VkDevice device, VkDeviceMemory memory) {
    if (memory != VK_NULL_HANDLE) {
        gpuResourceCount--;
        vkFreeMemory(device, memory, nullptr);
    }
}

VulkanParameters params = {};
params.allocateMemoryCallback = allocateMemoryVK;
params.freeMemoryCallback = freeMemoryVK;
```

##### D3D12

NVIGI provides callback mechanism to track/allocated/free GPU resources as defined in the `nvigi_d3d12.h` header. Here is an example:

```cpp
// Example definition of the d3d12 callbacks when passed as D3D12Parameters:
//
// NOTE: "REFIID riidResource" is not passed as a parameter as the ID3D12Resource has a fixed UID derived with the IID_PPV_ARGS macro
// 
ID3D12Resource* createCommittedResource(
     ID3D12Device* device, const D3D12_HEAP_PROPERTIES* pHeapProperties,
     D3D12_HEAP_FLAGS HeapFlags, const D3D12_RESOURCE_DESC* pDesc,
     D3D12_RESOURCE_STATES InitialResourceState, const D3D12_CLEAR_VALUE* pOptimizedClearValue,
     void* userContext
 )
{
    ID3D12Resource* resource = nullptr;
    HRESULT hr = device->CreateCommittedResource(pHeapProperties, HeapFlags, pDesc, InitialResourceState, pOptimizedClearValue, IID_PPV_ARGS(&resource));
    if (FAILED(hr))
    {
        // Handle error
        return nullptr;
    }
    if(userContext)
    {
        // Do something with userContext
    }
    return resource;
}

void destroyResource(ID3D12Resource* pResource, void* userContext)
{
     pResource->Release();
}

D3D12Parameters params = {};
params.createCommittedResourceCallback = createCommittedResource;
params.destroyResourceCallback = destroyResource;
params.createCommitResourceUserContext = nullptr;
params.destroyResourceUserContext = nullptr;
```

##### CUDA

NVIGI provides callback mechanism to track/allocated/free GPU resources as defined in the `nvigi_cuda.h` header. Here is an example:

```cpp
// Callback implementations
void MallocReportCallback(void* ptr, size_t size, void* user_context) {
    auto* context = static_cast<int*>(user_context);
    std::cout << "Malloc Report: Allocated " << size << " bytes at " << ptr 
              << " (User context value: " << *context << ")\n";
}

void FreeReportCallback(void* ptr, size_t size, void* user_context) {
    auto* context = static_cast<int*>(user_context);
    std::cout << "Free Report: Freed memory at " << ptr 
              << " (User context value: " << *context << ")\n";
}

int32_t MallocCallback(void** ptr, size_t size, int device, bool managed, bool hip, void* user_context) {
    auto* context = static_cast<int*>(user_context);
    *ptr = malloc(size); // Simulate CUDA malloc
    if (*ptr) {
        std::cout << "Malloc Callback: Allocated " << size << " bytes on device " << device 
                  << " (Managed: " << managed << ", HIP: " << hip << ", Context: " << *context << ")\n";
        return 0; // Success
    }
    return -1; // Failure
}

int32_t FreeCallback(void* ptr, void* user_context) {
    auto* context = static_cast<int*>(user_context);
    if (ptr) {
        free(ptr); // Simulate CUDA free
        std::cout << "Free Callback: Freed memory at " << ptr 
                  << " (User context value: " << *context << ")\n";
        return 0; // Success
    }
    return -1; // Failure
}

// Example usage
CudaParameters params{};

// User context for tracking (e.g., an integer counter)
int userContextValue = 42;

// Set up callbacks
params.cudaMallocReportCallback = MallocReportCallback;
params.cudaMallocReportUserContext = &userContextValue;

params.cudaFreeReportCallback = FreeReportCallback;
params.cudaFreeReportUserContext = &userContextValue;

params.cudaMallocCallback = MallocCallback;
params.cudaMallocUserContext = &userContextValue;

params.cudaFreeCallback = FreeCallback;
params.cudaFreeUserContext = &userContextValue;
```
