
# Automatic Speech Recognition (ASR) - Whisper Programming Guide

The focus of this guide is on using In-Game Inferencing to integrate an ASR Whisper model into an application. More details can be found here [OpenAI Whisper](https://openai.com/research/whisper)

Please read the [Programming Guide for AI](nvigi_core/docs/ProgrammingGuideAI.md) located in the NVIGI Core package to learn more about overall AI inference API in NVIGI SDK.

> **IMPORTANT**: This guide might contain pseudo code, for the up to date implementation and source code which can be copy pasted please see the SDK's Basic command line sample [Source Code](../source/samples/nvigi.basic/basic.cpp) and [Docs](Samples.md).  The Basic command-line sample includes the option to record audio and converting it to a text query for an LLM via the ASR plugins.

> **IMPORTANT NOTE: The D3D12 backend (nvigi.plugin.asr.ggml.d3d12.dll) is provided only precompiled as a part of the downloadable binary pack (`nvigi_pack`).  It is not possible for developers to compile the D3D12 backend plugin from source in this release.**

> **IMPORTANT NOTE: The D3D12 backend (nvigi.plugin.asr.ggml.d3d12.dll) requires an NVIDIA R580 driver or newer in order to be available at runtime.**

> **IMPORTANT NOTE: The CUDA backend (nvigi.plugin.asr.ggml.cuda.dll) strongly recommends an NVIDIA R580 driver or newer in order to avoid a potential memory leak if CiG (CUDA in Graphics) is used and the application deletes D3D12 command queues mid-application.**

## 1.0 INITIALIZE AND SHUTDOWN

Please read the [Programming Guide](nvigi_core/docs/ProgrammingGuide.md) located in the NVIGI Core package to learn more about initializing and shutting down NVIGI SDK. 

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

## APPENDIX

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
