# Generative Pre-Trained Transformers (GPT) Programming Guide

The focus of this guide is on using In-Game Inferencing to integrate a GPT model into an application. One example would be [Meta Llama2](https://llama.meta.com)

Please read the [Programming Guide for AI](nvigi_core/docs/ProgrammingGuideAI.md) located in the NVIGI Core package to learn more about overall AI inference API in NVIGI SDK.

> **MIN RUNTIME SPEC:** Note that all GPT GGML-based backends require a CPU supporting AVX2 instructions.  Support for this instruction extension is ubiquitous in modern gaming CPUs, but older hardware may not support it.

> **IMPORTANT**: This guide might contain pseudo code, for the up to date implementation and source code which can be copy pasted please see the SDK's Basic command line sample [Source Code](../source/samples/nvigi.basic/basic.cpp) and [Docs](Samples.md).  The Basic command-line sample includes use of the GPT plugins to respond to text queries.

> **NOTE** The NVIGI code currently uses the now-outdated term "General Purpose Transformer" for Generative Pre-Trained Transformers in its headers/classes/types.  This will be rectified in a coming release.

> **IMPORTANT NOTE: D3D12 and Vulkan backends are experimental and might not behave or perform as expected.**

> **IMPORTANT NOTE: The D3D12 backend (nvigi.plugin.gpt.ggml.d3d12.dll) is provided only precompiled as a part of the downloadable binary pack (`nvigi_pack`).  It is not possible for developers to compile the D3D12 backend plugin from source in this release.**

> **IMPORTANT NOTE: The D3D12 backend (nvigi.plugin.gpt.ggml.d3d12.dll) requires an NVIDIA R580 driver or newer in order to be available at runtime.** For optimal performance MS Agility SDK is required, for details please read [GPU UPLOAD HEAP (ReBar)](#d3d12)

> **IMPORTANT NOTE: The CUDA backend (nvigi.plugin.gpt.ggml.cuda.dll) strongly recommends an NVIDIA R580 driver or newer in order to avoid a potential memory leak if CiG (CUDA in Graphics) is used and the application deletes D3D12 command queues mid-application.**

## 1.0 INITIALIZE AND SHUTDOWN

Please read the [Programming Guide](nvigi_core/docs/ProgrammingGuide.md) located in the NVIGI Core package to learn more about initializing and shutting down NVIGI SDK. 

### 1.1 MODERN C++ WRAPPER (RECOMMENDED)

The NVIGI SDK provides modern C++ wrappers that simplify initialization and provide a cleaner API with RAII, `std::expected`, and builder patterns. The wrappers are located in `source/samples/nvigi.basic.cxx/` and can be used in your projects.

```cpp
#include "core.hpp"
#include "gpt.hpp"

using namespace nvigi::gpt;

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

## 2.0 OBTAIN GPT INTERFACE(S)

Next, we need to retrieve GPT's API interface based on what variant we need (cloud, CUDA, D3D12, Vulkan etc.). 


```cpp

nvigi::IGeneralPurposeTransformer igptLocal{};
// Here we are requesting interface for the GGML_CUDA implementation
if(NVIGI_FAILED(res, nvigiGetInterface(nvigi::plugin::gpt::ggml::cuda::kId, igptLocal))
{
    LOG("NVIGI call failed, code %d", res);
}

nvigi::IGeneralPurposeTransformer igptCloud{};
// Here we are requesting interface for the GFN cloud implementation
if(NVIGI_FAILED(res, nvigiGetInterface(nvigi::plugin::gpt::cloud::rest::kId, igptCloud))
{
    LOG("NVIGI call failed, code %d", res);
}
```

> **NOTE:**
One can only obtain interface for a feature which is available on user system. Interfaces are valid as long as the underlying plugin is loaded and active.

### 2.1 MODERN C++ WRAPPER APPROACH

The C++ wrapper handles interface loading automatically during instance creation. You don't need to manually obtain interfaces:

```cpp
// No manual interface loading needed!
// Just create the instance with your desired backend
```

See section 3.1 for complete instance creation examples using the wrapper.

## 3.0 CREATE GPT INSTANCE(S)

Now that we have our interface we can use it to create our GPT instance. To do this, we need to provide information about GPT model we want to use, CPU/GPU resources which are available and various other creation parameters.
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
```

To check if an optional feature is supported by this backend one can check if it is chained to the returned `CommonCapabilitiesAndRequirements`. Here is an example:

```cpp
auto sampler = findStruct<GPTSamplerParameters>(*caps);
if (sampler)
{
    // This backend supports additional sampler parameters
    // 
    // Set them here as desired ...
    sampler->mirostat = 2; // for example, use mirostat v2    
}
```
In this example, the optional `GPTSamplerParameters` structure is found hence it can be populated with desired values and chained to the `GPTCreationParameter` or runtime parameters (see below for details)
```cpp
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

```cpp

//! Here we are creating two instances for different backends/APIs
//!
//! IMPORTANT: This is totally optional and only used to demonstrate runtime switching between different backends

nvigi::InferenceInstance* gptInstanceLocal;
{
    //! Creating local instance and providing our D3D12 or VK and CUDA information (all optional)
    //!
    //! This allows host to control how instance interacts with DirectX, Vulkan (if at all) or any existing CUDA contexts (if any)
    //!
    //! Note that providing DirectX/Vulkan information is mandatory if at runtime we expect instance to run on a command list.

    nvigi::CommonCreationParameters common{};
    nvigi::GPTCreationParameters params{};    
```
First we need to decide how many CPU threads should our instance use (assuming selected backend provides multi-threading):
```cpp
    common.numThreads = myNumCPUThreads; // How many CPU threads is instance allowed to use 
```
> NOTE: Spawning more than one thread when using GPU based backends might not be beneficial

Next section sets the VRAM budget, this is important since some backends (like for example `ggml.cuda`) allow splitting model between CPU/GPU if there isn't enough VRAM available. If selected backend does not provide such functionality providing insufficient VRAM budget will result in failure to create an instance.
```cpp
    common.vramBudgetMB = myVRAMBudget;  // How much VRAM is instance allowed to occupy
```
Now we continue with setting up the rest of the creation parameters, like model repository location and model GUID:
```cpp
    common.utf8PathToModels = myPathToNVIGIModelRepository; // Path to provided NVIGI model repository (using UTF-8 encoding)
    common.modelGUID = "{175C5C5D-E978-41AF-8F11-880D0517C524}"; // Model GUID, for details please see NVIGI models repository
    if(NVIGI_FAILED(params.chain(common)))
    {
        // Handle error
    }
```
As mention in the previous section, we detected that `GPTSamplerParameters` is supported so we can chain it with the rest of our creation paramters:
```cpp
   // Using helper operator hence *sampler
   if(NVIGI_FAILED(params.chain(*sampler)))
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
    if(NVIGI_FAILED(res, igptLocal.createInstance(params, &gptInstanceLocal)))
    {
        LOG("NVIGI call failed, code %d", res);
    }
}
```
> **IMPORTANT**: D3D12 device must support shader model 6_6 or higher. If queues are NOT provided plugin will generate its own compute and direct/copy queue.
```cpp
nvigi::InferenceInstance* gptInstanceCloud;
{
    nvigi::CommonCreationParameters common{};
    nvigi::GPTCreationParameters params{};    
    common.modelGUID = "{175C5C5D-E978-41AF-8F11-880D0517C524}"; // Model GUID, for details please see NVIGI models repository
    if(NVIGI_FAILED(params.chain(common)))
    {
        // handle error
    }

    //! Cloud parameters
    nvigi::RESTParameters nvcfParams{};
    std::string token;
    getEnvVar("MY_TOKEN", token);
    nvcfParams.url = myURL;
    nvcfParams.authenticationToken = token.c_str();
    if(NVIGI_FAILED(params.chain(nvcfParams)))
    {
        // handle error
    }
    
    if(NVIGI_FAILED(res, igptCloud.createInstance(params, &gptInstanceCloud, inputs, countof(inputs))))
    {
        LOG("NVIGI call failed, code %d", res);
    }
}

```

> **NOTE:**
> NVIGI model repository is provided with the pack under `data/nvigi.models`.

### 3.3 MODERN C++ WRAPPER APPROACH

The C++ wrapper simplifies instance creation with builder patterns and automatic resource management:

```cpp
#include "d3d12.hpp"  // or "vulkan.hpp"

using namespace nvigi::gpt;

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

// Or setup Cloud
CloudConfig cloud_config = {
    .url = "https://my-endpoint.com",  // optional, default from model JSON
    .token = "your_api_token",          // mandatory for cloud
    .verbose = false,
    .streaming = true
};

// Create instance with builder pattern
auto instance = Instance::create(
    ModelConfig{
        .backend = "d3d12",  // or "cuda", "vulkan", "cloud"
        .guid = "{175C5C5D-E978-41AF-8F11-880D0517C524}",
        .model_path = "path/to/nvigi.models",
        .context_size = 4096,
        .num_threads = 4,
        .vram_budget_mb = 8192,
        .flash_attention = true,
        .cache_type = "fp16"  // or "fp32", "q4_0", "q8_0"
    },
    d3d12_config,      // Pass your config based on backend
    vk_config,         // Can pass both, unused ones are ignored
    cloud_config,      // Only used if backend is "cloud"
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

## 4.0 SETUP CALLBACK TO RECEIVE INFERRED DATA

In order to receive a text response from the GPT model inference a special callback needs to be setup like this:

```cpp
auto gptCallback = [](const nvigi::InferenceExecutionContext* execCtx, nvigi::InferenceExecutionState state, void* userData)->nvigi::InferenceExecutionState 
{     
    //! Optional user context to control execution 
    auto userCtx = (HostProvidedGPTCallbackCtx*)userData; 
    if (execCtx->outputs) 
    { 
       const nvigi::InferenceDataText* text{};        
       execCtx->outputs->findAndValidateSlot(nvigi::kGPTDataSlotResponse, &text);        
       //! OPTIONAL - Cloud only, REST response from the server
       if(execCtx->instance == gptInstanceCloud)
       {
            // Full response from the server, normally as JSON but could be plain text in case of an error
            const nvigi::InferenceDataText* responseJSON{};
            execCtx->outputs->findAndValidateSlot(nvigi::kGPTDataSlotJSON, &responseJSON); 
            // Examine response from the server
            std::string receivedResponse = responseJSON->getUtf8Text();
       }       
       std::string receivedAnswer = text->getUtf8Text();
       //! Do something with the received answer
    } 
    if (state == nvigi::kInferenceExecutionStateDone) 
    { 
        //! This is all the data we can expect to receive 
    } 
    else if(userCtx->needToInterruptInference) 
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
> To cancel GPT inference make sure to return `nvigi::InferenceExecutionStateCancel` state in the callback.

### 4.1 MODERN C++ WRAPPER APPROACH

The C++ wrapper simplifies callbacks with modern C++ lambdas and cleaner state management:

```cpp
using namespace nvigi::gpt;

// Simple callback - just process the text
auto result = instance->generate(
    "You are a helpful assistant",  // system
    "Tell me about AI",             // user
    "",                              // assistant
    RuntimeConfig{},                 // config (optional)
    [](std::string_view response, ExecutionState state) -> ExecutionState {
        // Callback receives text and state
        std::cout << response;  // Print each token as it arrives
        
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

## 5.0 PREPARE THE EXECUTION CONTEXT

Before GPT can be evaluated the `nvigi::InferenceExecutionContext` needs to be defined. Among other things, this includes specifying input and output slots. GPT can operate in two modes:

* Instruct a model and receive an answer.
* Interact with a model while taking turns in a guided conversation.

We will be using the `InferenceDataTextSTLHelper` helper class to convert text to `nvigi::InferenceDataText`:

### 5.1 INSTRUCT MODE

In this mode GPT receives **any combination** of `system`, `user` and `assistant` input slots with instruction(s) and produces adequate response. Here is an example:

```cpp
//! We can provide ANY combination of input slots, it can be just $user for example.
std::string system("You are a world renown and extremely talented writer.");
std::string user("Write a poem about transformers, robots in disguise");
// Using our helper from the section above
InferenceDataTextSTLHelper systemSlot(system);
InferenceDataTextSTLHelper userSlot(user);
std::vector<nvigi::InferenceDataSlot> slots = { {nvigi::kGPTDataSlotSystem, systemSlot}, {nvigi::kGPTDataSlotUser, userSlot} };
nvigi::InferenceDataSlotArray inputs = { slots.size(), slots.data() }; // Input slots

nvigi::InferenceExecutionContext gptContext{};
gptContext.instance = gptInstanceLocal;         // The instance we created and we want to run inference on
gptContext.callback = gptCallback;              // Callback to receive transcribed text
gptContext.callbackUserData = &gptCallbackCtx;  // Optional context for the callback, can be null if not needed
gptContext.inputs = &inputs
```

> NOTE: Prompt templating is defined in `nvigi.model.config.json` for each model

### 5.2 INTERACTIVE (CHAT) MODE

In this mode, the first call to evaluate instance that includes `system` is used to setup the conversation. Consecutive calls which do NOT include `system` slot are considered a new "turn" in a conversation. Providing `system` input slot at any point in time resets conversation with new setup. Here is an example:

```cpp
std::string system("This is a transcript of a conversation between Rob and Bob. Rob enters the room.");
std::string user("Hey Bob, how are you?");
// Using our helper from the section above
InferenceDataTextSTLHelper systemSlot(system);
InferenceDataTextSTLHelper userSlot(user);
std::vector<nvigi::InferenceDataSlot> slots = { {nvigi::kGPTDataSlotSystem, systemSlot}, {nvigi::kGPTDataSlotUser, userSlot} };
nvigi::InferenceDataSlotArray inputs = { slots.size(), slots.data() }; // Input slots

nvigi::InferenceExecutionContext gptContext{};
gptContext.instance = gptInstanceLocal;         // The instance we created and we want to run inference on
gptContext.callback = gptCallback;              // Callback to receive transcribed text
gptContext.callbackUserData = &gptCallbackCtx;  // Optional context for the callback, can be null if not needed
gptContext.inputs = &inputs
```

Further down in this document we will explain how to setup interactive mode and reverse prompt when evaluating your GPT instance.

> NOTE: All input slots are considered optional, any combination of inputs is acceptable but **at least one has to be provided**.

### 5.3 CLOUD

Cloud inference can operate in two modes as described in next two sections.

#### 5.3.1 INSTRUCT OR CHAT MODE

In this mode cloud inference is **exactly the same as the local one** which is described in the previous section. The only difference would be to obtain interface from `nvigi::plugin::gpt::cloud::rest::kId` plugin and create instance with it as shown in [section 3](#30-create-gpt-instances)

#### 5.3.1 FULL CONTROL VIA JSON INPUT SLOT

To obtain full control over the prompt engineering the standard `system/user/assistant` input slots can be omitted and instead JSON input slot is used to define the entire REST request sent to the cloud end-point. Here is an example:

```cpp
//! When issuing REST request we need to tell the cloud endpoint what model we want and what parameters to use
//!
//! IMPORTANT: Make sure to replace $system, $user, $assistant as needed
json restJSONBody = R"({
    "model" : "meta/llama2-70b",
    "messages": [
        {
            "role":"system",
            "content":"$system"
        },
        {
            "role":"user",
            "content":"$user"
        },
        {
            "role":"assistant",
            "content":"$assistant"
        }
    ],
    "stream": false,
    "temperature": 0.5,
    "top_p" : 1,
    "max_tokens": 1024
})"_json;

// Creating extra input slot to provide JSON body
std::string restJSONBodyAsString = restJSONBody.dump(2, ' ', false, json::error_handler_t::replace);
InferenceDataTextSTLHelper jsonSlot(restJSONBodyAsString);
// Only providing one input slot, JSON
std::vector<nvigi::InferenceDataSlot> slots = { {nvigi::kGPTDataSlotJSON, jsonSlot} };
nvigi::InferenceDataSlotArray inputs = { slots.size(), slots.data() }; // Input slots
```

### 5.4 OUTPUTS

If the output slots are not provided, as in the above examples, NVIGI will allocate them. This is the simplest and recommended way, output slots will be provided via callback and will only be valid within the callback execution framework.

> **IMPORTANT:**
> The execution context and all provided data (input, output slots) must be valid at the time `instance->evaluate` is called (see below for more details).

## 6.0 ADD GPT INFERENCE TO THE PIPELINE

In your execution pipeline, call `instance->evaluate` at the appropriate location where a prompt needs to be processed to receive a response from the GPT.

### 6.1 INSTRUCT MODE

In this mode we evaluate given prompt and receive our response like this:

```cpp
// Make sure GPT is available and user selected this option in the UI
if(useGPT) 
{
    //! OPTIONAL Runtime properties, we can for example change seed or how many tokes to predict etc.
    nvigi::GPTRuntimeParameters gptRuntime{};
    gptRuntime.seed = myRNGSeed;
    gptRuntime.tokensToPredict = 100; // modify as needed
    gptRuntime.interactive = false;
    gptContext.runtimeParameters = &gptRuntime;

    //! OPTIONAL Runtime sampler properties, not necessarily supported by all backends
    nvigi::GPTSamplerParameters gptSampler{};
    gptSampler.penaltyRepeat = 0.1f;
    if(NVIGI_FAILED(gptRuntime.chain(gptSampler)))
    {
        // handle error
    }

    //! OPTIONAL - Switching backends at runtime (could depend on current latency, available resources etc.)
    if(useLocalInference)
    {
        gptContext.instance = gptInstanceLocal;
        gptRuntime.chain(d3d12Runtime); // makes sense only for local inference
    }
    else
    {
        gptContext.instance = gptInstanceCloud;
    }    
    
    // Evaluate our instance with all the additional parameters, transcribed text is received via callback    
    if(NVIGI_FAILED(res, gptContext.instance->evaluate(gptContext)))
    {
        LOG("NVIGI call failed, code %d", res);
    }    

    //! IMPORTANT: Wait for the callback to receive nvigi::InferenceExecutionStateDone

    //! Now we have received the response from the GPT via our gptCallback
}
```

### 6.2 INTERACT MODE

In this scenario, GPT must process the prompt first and then we need to feed user input based on the responses received. Here is an example:

```cpp
// Make sure GPT is available and user selected this option in the UI
if(useGPT) 
{
    //! OPTIONAL Runtime properties, we can for example change seed or how many tokes to predict etc.
    nvigi::GPTRuntimeParameters gptRuntime{};
    gptRuntime.seed = myRNGSeed;
    gptRuntime.tokensToPredict = 100; // modify as needed
    gptRuntime.interactive = true;     // Set to true if prompt engineering is used to start a guided conversation
    gptRuntime.reversePrompt = "Rob:"; // This has to match our prompt, it represents the user
    gptContext.runtimeParameters = &gptRuntime;
    
    //! OPTIONAL Runtime sampler properties, not necessarily supported by all backends
    nvigi::GPTSamplerParameters gptSampler{};
    gptSampler.penaltyRepeat = 0.1f;
    if(NVIGI_FAILED(gptRuntime.chain(gptSampler)))
    {
        // handle error
    }

    //! OPTIONAL - Switching backends at runtime (could depend on current latency, available resources etc.)
    if(useLocalInference)
    {
        gptContext.instance = gptInstanceLocal;
    }
    else
    {
        gptContext.instance = gptInstanceCloud;
    }    
    
    // Process system input slot first (assuming it was set already as shown in section 5.2)
    if(NVIGI_FAILED(res, gptContext.instance->evaluate(gptContext)))
    {
        LOG("NVIGI call failed, code %d", res);
    }    
    else
    {
        //! IMPORTANT: Wait for the callback to receive nvigi::InferenceExecutionStateDone before proceeding here

        // Display response from GPT on screen and or use TTS to play the response

        // Now we enter the conversation
        while(runConversation)
        {
            // Setting up new context and new input/output slots
            std::string input = getUserInputBasedOnResponseReceivedFromGPT();
            nvigi::InferenceDataTextSTLHelper userSlot(input);        
            // Note that here we are providing `user input` slot and no system
            std::vector<nvigi::InferenceDataSlot> inSlots = { {nvigi::kGPTDataSlotUser, userSlot} };        
            InferenceDataSlotArray inputs = { inSlots.size(), inSlots.data() };
            InferenceDataSlotArray outputs = { outSlots.size(), outSlots.data() };

            gptContext.inputs = &inputs;
            gptContext.outputs = &outputs;        
            
            if(NVIGI_FAILED(res, gptContext.instance->evaluate(gptContext)))
            {
                LOG("NVIGI call failed, code %d", res);
            }

            //! IMPORTANT: Wait for the callback to receive nvigi::InferenceExecutionStateDone before proceeding

            // Display response from GPT on screen and or use TTS to play the response
        }
    }    
}
```

To start a new conversation **simply evaluate your instance with the new SYSTEM input slot** and then repeat the same process.

### 6.3 TOKEN GENERATION RATE LIMIT

To control the rate of token generation during inference, you can use the `targetTokensPerSecond` and `frameTimeMS` fields in the `nvigi::GPTRuntimeParameters`. These parameters allow you to limit the number of tokens generated per second, ensuring that the inference process aligns with your application's performance requirements.

Here is an example of how to set up the token generation limiter:

```cpp
nvigi::GPTRuntimeParameters gptRuntime{};
gptRuntime.seed = myRNGSeed;
gptRuntime.targetTokensPerSecond = 10;  // Limit to 10 tokens per second
gptRuntime.frameTimeMS = 16;            // Frame time in milliseconds (e.g., 16ms for 60 FPS)

// Attach runtime parameters to the execution context
gptContext.runtimeParameters = &gptRuntime;

// Evaluate the instance
if (NVIGI_FAILED(res, gptContext.instance->evaluate(gptContext))) {
    LOG("NVIGI call failed, code %d", res);
}
```

#### Explanation:
- **`targetTokensPerSecond`**: Specifies the maximum number of tokens to generate per second. For example, setting this to `10` ensures that no more than 10 tokens are generated every second.
- **`frameTimeMS`**: Defines the frame time in milliseconds. This is useful for real-time applications where inference needs to align with a specific frame rate (e.g., 16ms for 60 FPS).

By combining these parameters, you can ensure that token generation is throttled to match your application's performance constraints, avoiding excessive resource usage or latency.

> **NOTE**: The token generation limiter is particularly useful in interactive or streaming scenarios where maintaining a consistent response rate is critical.

### 6.3.1 MODERN C++ WRAPPER APPROACH - CHAT API

The C++ wrapper provides a high-level `Chat` API that manages conversation state automatically:

```cpp
using namespace nvigi::gpt;

// Create a chat session with configuration
auto chat = instance->create_chat(
    RuntimeConfig{}
        .set_tokens(256)
        .set_temperature(0.3f)
        .set_top_p(0.8f)
        .set_interactive(true)
        .set_reverse_prompt("\nAssistant:")
);

// Send system message to setup conversation
chat.send_message(
    { .role = Instance::Chat::Message::Role::System,
      .content = "You are a helpful AI assistant."
    },
    [](std::string_view response, ExecutionState state) -> ExecutionState {
        std::cout << response;
        return state;
    }
);

// Interactive loop
std::string user_input;
while (true) {
    std::cout << "\nUser> ";
    std::getline(std::cin, user_input);
    
    if (user_input == "quit") break;
    
    // Send user message (blocking)
    chat.send_message(
        { .role = Instance::Chat::Message::Role::User,
          .content = user_input
        },
        [](std::string_view response, ExecutionState state) -> ExecutionState {
            std::cout << response;
            return state;
        }
    );
}

// Access conversation history
for (const auto& msg : chat.history()) {
    std::cout << "Role: " << (int)msg.role << " Content: " << msg.content << "\n";
}
```

**Polling-Based Async Chat (Perfect for Game Loops!):**

```cpp
// Start async operation (non-blocking!)
auto op = chat.send_message_polled(
    { .role = Instance::Chat::Message::Role::User,
      .content = "Tell me about AI"
    }
).value();

// Game loop integration
bool game_running = true;
while (game_running) {
    // Poll for tokens (non-blocking - returns immediately!)
    if (auto result = op.try_get_results()) {
        std::cout << result->tokens;  // Display new tokens
        
        if (result->state == ExecutionState::Done) {
            std::cout << "\n[Inference complete]\n";
        }
    }
    
    // Check if done
    if (op.is_complete()) {
        chat.finalize_async_response(op);
        break;
    }
    
    // Game continues running smoothly
    render_frame();      // Renders every frame at 60 FPS
    update_physics();    // Physics keeps running
    process_input();     // Player can still move
    
    // Optional: Cancel on user input
    if (user_pressed_cancel()) {
        op.cancel();
    }
}
```

**Token Generation Rate Limiting:**

```cpp
auto chat = instance->create_chat(
    RuntimeConfig{}
        .set_target_tokens_per_second(10)  // Limit to 10 tokens/sec
        .set_frame_time_ms(16.0f)           // Frame time for 60 FPS
);
```

**LoRA Support:**

```cpp
auto chat = instance->create_chat(
    RuntimeConfig{}
        .add_lora("mylora1", 1.0f)    // Full strength
        .add_lora("mylora2", 0.5f)    // Half strength
);
```

The Chat API automatically:
- Manages conversation history
- Handles message role formatting
- Provides blocking and non-blocking modes
- Supports token-by-token streaming
- Enables easy cancellation

### 6.4 ASYNC EVALUATION AND RESULT POLLING

When using `evaluateAsync`, you can poll for results instead of using a callback. This approach is useful in scenarios where you want more control over when to process the inference results, or when you need to integrate with an existing polling-based architecture.

Here's how to use the polling mechanism:

```cpp
// Start async evaluation without a callback
gptContext.callback = nullptr;
if (NVIGI_FAILED(res, gptContext.instance->evaluateAsync(gptContext))) {
    LOG("NVIGI async evaluation failed, code %d", res);
    return;
}

// Poll for results
while (true) {
    nvigi::InferenceExecutionState state;
    
    // Get current results - pass true to wait for new data, false to check immediately
    if (NVIGI_FAILED(res, gptInstance->getResults(&gptContext, true, &state))) {
        LOG("Failed to get results, code %d", res);
        break;
    }

    // Process the current state
    switch (state) {
        case nvigi::kInferenceExecutionStateDone:
            // All data has been processed
            gptInstance->releaseResults(&gptContext, state);
            return;

        case nvigi::kInferenceExecutionStateDataPending:
            // New data is available
            const nvigi::InferenceDataText* output;
            if (gptContext.outputs && 
                gptContext.outputs->findAndValidateSlot(kGPTDataSlotResponse, &output)) {
                // Process the output text
                std::string response = output->getUTF8Text();
                // ... handle the response ...
            }
            
            // Release current results to get next batch
            gptInstance->releaseResults(&gptContext, state);
            break;

        case nvigi::kInferenceExecutionStateInvalid:
            // Handle error state
            LOG("Inference failed");
            gptInstance->releaseResults(&gptContext, state);
            return;
    }
}
```

### 6.5 CANCELLING ASYNC EVALUATION

When using `evaluateAsync`, you can cancel an ongoing inference operation early using the `cancelAsyncEvaluation` API. This is useful when you need to interrupt generation due to user actions (e.g., pressing ESC), timeouts, or changing contexts.

The cancellation mechanism is designed to interrupt the generation loop as early as possible, including during GPU/CPU work, token processing, and other internal loops.

Here's how to cancel an async evaluation:

```cpp
// Start async evaluation
gptContext.callback = nullptr;
if (NVIGI_FAILED(res, gptContext.instance->evaluateAsync(gptContext))) {
    LOG("NVIGI async evaluation failed, code %d", res);
    return;
}

// ... some time later, user decides to cancel ...

// Cancel the ongoing async evaluation
if (NVIGI_FAILED(res, gptInstance->cancelAsyncEvaluation(&gptContext))) {
    if (res == kResultNoImplementation) {
        LOG("No async evaluation is currently running");
    } else {
        LOG("Failed to cancel evaluation, code %d", res);
    }
}

// The generation will stop as soon as possible
// Continue polling to clean up
nvigi::InferenceExecutionState state;
while (true) {
    if (NVIGI_FAILED(res, gptInstance->getResults(&gptContext, false, &state))) {
        break;
    }
    
    // Release any remaining results
    gptInstance->releaseResults(&gptContext, state);
    
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
- The generation loop checks for cancellation at **multiple strategic points**:
  - At the start of each generation iteration
  - After GPU/CPU evaluation work
  - Inside batch processing loops
  - During token display and callback execution
  - During session token matching
  - During image marker processing (for VLM models)
- Cancellation is designed to be **as fast as possible**, typically interrupting within a few milliseconds

#### Example: User-Initiated Cancellation

```cpp
// Track async state
std::atomic<bool> userRequestedCancel = false;
std::thread asyncThread;

// Start async generation
gptContext.callback = nullptr;
if (NVIGI_FAILED(res, gptContext.instance->evaluateAsync(gptContext))) {
    LOG("Failed to start async evaluation");
    return;
}

// Spawn a thread to handle user input
asyncThread = std::thread([&]() {
    while (true) {
        if (checkUserPressedEscape()) {
            userRequestedCancel = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
});

// Poll for results
nvigi::InferenceExecutionState state;
while (true) {
    // Check if user wants to cancel
    if (userRequestedCancel) {
        gptInstance->cancelAsyncEvaluation(&gptContext);
        LOG("User cancelled generation");
    }
    
    if (NVIGI_FAILED(res, gptInstance->getResults(&gptContext, true, &state))) {
        break;
    }
    
    // Process results...
    if (gptContext.outputs) {
        const nvigi::InferenceDataText* output;
        if (gptContext.outputs->findAndValidateSlot(kGPTDataSlotResponse, &output)) {
            std::string response = output->getUTF8Text();
            displayPartialResponse(response);
        }
    }
    
    gptInstance->releaseResults(&gptContext, state);
    
    if (state == nvigi::kInferenceExecutionStateDone || 
        state == nvigi::kInferenceExecutionStateInvalid) {
        break;
    }
}

asyncThread.join();
```

> **NOTE**: Cancellation via `cancelAsyncEvaluation` is only available for async evaluation started with `evaluateAsync`. For synchronous evaluation, use the callback return value mechanism (return `kInferenceExecutionStateCancel` from the callback) as described in section 4.0.

## 7.0 DESTROY INSTANCE(S)

Once GPT is no longer needed each instance should be destroyed like this:

```cpp
//! Finally, we destroy our instance(s)
if(NVIGI_FAILED(res, igptLocal.destroyInstance(gptInstanceLocal))) 
{ 
    LOG("NVIGI call failed, code %d", res);
}
if(NVIGI_FAILED(res, igptCloud.destroyInstance(gptInstanceCloud))) 
{ 
    LOG("NVIGI call failed, code %d", res);
} 
```

## 8.0 UNLOAD INTERFACE(S)

Once GPT is no longer needed each interface should be unloaded like this:

```cpp
//! Finally, we destroy our instance(s)
if(NVIGI_FAILED(result, nvigiUnloadInterface(nvigi::plugin::gpt::cloud::rest::kId , &igptCloud))) 
{ 
    //! Check error
}
if(NVIGI_FAILED(result, nvigiUnloadInterface(nvigi::plugin::gpt::ggml::cuda::kId, &igptLocal))) 
{ 
    //! Check error
} 
```

### 8.1 MODERN C++ WRAPPER APPROACH

Interface unloading is handled automatically by the wrapper:

```cpp
{
    // Initialize core
    nvigi::Core core({ .sdkPath = "path/to/sdk" });
    
    // Create instance
    auto instance = Instance::create(
        ModelConfig{ /* ... */ },
        d3d12_config,
        vk_config,
        cloud_config,
        core.loadInterface(),
        core.unloadInterface()
    ).value();
    
    // Use instance...
    auto chat = instance->create_chat();
    chat.send_message(/* ... */);
    
    // Automatic cleanup when leaving scope!
    // 1. instance destructor -> calls destroyInstance() and unloads interfaces
    // 2. core destructor -> calls nvigiShutdown()
}
// All resources cleaned up automatically - no manual calls needed!
```

**Key Benefits:**
- No manual `nvigiUnloadInterface()` calls needed
- Exception-safe: cleanup happens even if exceptions are thrown
- Reference counting ensures interfaces stay valid while in use
- Impossible to forget cleanup or get order wrong

## 9.0 VLM (Visual Lanuage Models)

The GPT plugin can also load some VILA based VLM models.  These models allows the prompts to discuss input images.
The setup and usage is the same as a standard LLM model.  For the input InferenceDataSlot, you may now also pass in an InferenceDataImage, 
currently limited to one image per prompt.  The input image should be a byte array in RGB format, 8 bits per channel.

```cpp
std::string prompt( "Describe this picture" );
std::string my_image("picture.jpg");
int w, h, c;
auto* rgb_data = stbi_load(my_image.c_str(), &w, &h, &c, 3);
...
// prompt 
nvigi::CpuData text(prompt.length() + 1, (void*)prompt.c_str());
nvigi::InferenceDataText prompt_data(text);

// image
nvigi::CpuData image(h * w * c, rgb_data);
nvigi::InferenceDataImage image_data(image, h, w, c);

std::vector<nvigi::InferenceDataSlot> inSlots = { {nvigi::kGPTDataSlotUser, &prompt_data}, {nvigi::kGPTDataSlotImage, &image_data} };
InferenceDataSlotArray inputs = { inSlots.size(), inSlots.data() }; // Input slots
...

if(NVIGI_FAILED(res, gptContext.instance->evaluate(gptContext)))
{
    LOG("NVIGI call failed, code %d", res);
}   

...
// handle output response text from VLM same as a standard LLM
...
stbi_image_free(rgb_data);
```

If no image is passed in, the VLM will still function as a typical LLM.

## 10.0 LORA

LORAs are small, lightweight adapters that can be applied to a base model to efficiently change the base model behavior.

### Training


Training is beyond the scope of this document, though you can look to documents like [unsloth's fine tuning llm guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) for examples of how to create a LORA.

### Setting up the model.config.json

In the root json, have a "loras" key that is a list of dictionaries, with "name", "filename", and "vram" keys, like so:

```json
{
  "name": "llama-3-8b-instruct",
  "vram": 5280,
  "n_layers": 33,  
  "prompt_template": [...
  ],
  "model":
  {
    ...
  },
  "loras":
  [
      {
          "name" : "mylora1",
          "filename" : "mylora1.lora",
          "vram" : 344
      },
      {
          "name" : "mylora2",
          "filename" : "smaller_lora.lora",
          "vram" : 163
      },
  ]
}
```

Loras are gguf files but we rename them to a .lora extension as to not collide with the model files they are based off of.  The .lora files should go in the same folder guid as the model they modify.  So if these loras were trained off of the llama-3-8B instruct located at GUID {D5E8DEB3-28C2-4B9E-9412-B9A012B23584}, the loras would go in

```
nvigi.models\nvigi.plugin.gpt.ggml\{D5E8DEB3-28C2-4B9E-9412-B9A012B23584}
    Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
    mylora1.lora
    smaller_lora.lora
```

Then in the code, you will reference the loras by the name key.

### Preparing a model for a lora in GPT Creation

There are three new parameters that allow you to set up your loras:

```cpp
    size_t numLoras{};          
    const char** loraNames{};   
    const float* loraScales{};
```

numLoras should be the size of the two following arrays, which should be the same length.
loraNames are the loras you wish to load.
loraScales are the strengths you want the loras to come in at, between 0 and 1.

scales default to 0 (deactivated), but you can either set this scale at creation time and forget it, or you can wait until you call evaluate and set them in your GPTRuntimeParameters.

In code, that looks like this:

```cpp
nvigi::GPTCreationParameters params1{};

const char* loraNames[] = { "mylora1", "mylora2" }
float loraScales[] = { 1.0f, 0.5f };

params1.numLoras = 2;
params1.loraNames = loraNames;
params1.loraScales = loraScales;
```

NOTE: references will be held to the loraNames and loraScales arrays, so it is essential that their lifetime exceeds the runtime of your plugin.  Plan accordingly.

### Changing LORA values at Runtime

The same values that exist in GPTCreationParameters also exist in GPTRuntimeParameters, but instead of preparing and loading the LORAs, the values here allow for modification of the strength of the LORA at runtime.  This is predominantly for turning on and off LORAs, though be warned, this will still require a weight reload, so it's not completely free.

```cpp
    size_t numLoras{};          
    const char** loraNames{};   
    const float* loraScales{};
```

When preparing for your evaluate calls, the code looks very similar to the creation

```cpp
nvigi::GPTRuntimeParameters runtime{};

const char* loraNames[] = { "mylora1", "mylora2" }
float loraScales[] = { 0.0f, 1.0f };

runtime.numLoras = 2;
runtime.loraNames = loraNames;
runtime.loraScales = loraScales;
```

In this case, mylora1 is loaded, but turned off, and mylora2 is loaded at full strength.

This following should turn off the mylora2 and set mylora1 to full strength (since mylora2 is missing, it is implicitly not applied)

```cpp
nvigi::GPTRuntimeParameters runtime{};

const char* loraNames[] = { "mylora1" }
float loraScales[] = { 1.0f };

runtime.numLoras = 1;
runtime.loraNames = loraNames;
runtime.loraScales = loraScales;
```

As during creation, the loraNames and loraScales lifetime must exceed the duration of the evaluate call before they can be properly disposed of.

Creation must always happen, as bookkeeping occurs to prepare for referencing loras, but the runtime modifications are only necessary if you intend to change the weighting of the loras dynamically per evaluate call.

## APPENDIX

### LLAMA.CPP

The `nvigi.plugin.gpt.ggml.{$backend}` plugins use a specific snapshot of llama.cpp therefore it is not guaranteed that NVIGI version will match the latest llama.cpp capabilities. When comparing the two please note the following:

* NVIGI version is compiled with potentially different CPU flags (lowest common denominator to allow wide CPU support, not necessarily including the latest greatest CPU features)
* NVIGI input parameters should be modified to match llama.cpp 1:1 (context size, batch size, number of threads etc.) when comparing performance
* NVIGI version is modified to **allow optimal execution inside of a process** (especially when it comes to CUDA in Graphics) hence it might NOT perform as fast as llama.cpp on an idle GPU
* Performance and capabilities of llama.cpp change on daily basis, NVIGI version will be updated at much slower cadence 

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

> NOTE: This section is relevant only if the host application is providing `nvigi::VulkanParameters` to the NVIGI GPT plugin

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

### CHAT MODE SETUP SUMMARY

Here is the summary of the steps needed to setup the interactive (chat) mode:

#### Using Callback Approach:

1) Provide `nvigi::GPTRuntimeParameters` in the execution context and set `interactive` flag to TRUE
2) Setup the conversation context by providing `nvigi::kGPTDataSlotSystem` input slot
3) Wait until callback returns `nvigi::kInferenceExecutionStateDone`
4) Get user's input and take turn in the conversation by providing `nvigi::kGPTDataSlotUser` input slot
5) Wait until callback returns `nvigi::kInferenceExecutionStateDone` and full response is provided by the plugin
6) Inform user about the response
7) To continue conversation go back to the step #4
8) To start a new conversation, go to the step #2

#### Using Polling Approach:

1) Provide `nvigi::GPTRuntimeParameters` in the execution context and set `interactive` flag to TRUE
2) Set callback to nullptr in execution context
3) Setup the conversation context by providing `nvigi::kGPTDataSlotSystem` input slot
4) Call `evaluateAsync` and poll results using `getResults` until state is `kInferenceExecutionStateDone`
5) Release results using `releaseResults`
6) Get user's input and take turn in the conversation by providing `nvigi::kGPTDataSlotUser` input slot
7) Call `evaluateAsync` and poll results using `getResults`:
   - Process response text when state is `kInferenceExecutionStateDataPending`
   - Continue polling until state is `kInferenceExecutionStateDone`
   - **Optional**: Call `cancelAsyncEvaluation` to interrupt generation early if needed
8) Release results using `releaseResults`
9) Inform user about the complete response
10) To continue conversation go back to step #6
11) To start a new conversation, go back to step #3

> NOTE: When changing models there is no need to change any of the setup above, only model GUID would be different when creating GPT instance