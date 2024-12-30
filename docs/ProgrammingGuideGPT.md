
# Generative Pre-Trained Transformers (GPT) Programming Guide

The focus of this guide is on using In-Game Inferencing to integrate a GPT model into an application. One example would be [Meta Llama2](https://llama.meta.com)

Please read the `docs\ProgrammingGuideAI.md` located in the NVIGI Core package to learn more about overall AI inference API in NVIGI SDK.

> **IMPORTANT**: This guide might contain pseudo code, for the up to date implementation and source code which can be copy pasted please see the [basic sample](../source/samples/nvigi.basic/basic.cpp)

> **NOTE** The NVIGI code currently uses the now-outdated term "General Purpose Transformer" for Generative Pre-Trained Transformers in its headers/classes/types.  This will be rectified in a coming release.

## Version 1.0.0 General Access

## 1.0 INITIALIZE AND SHUTDOWN

Please read the `docs/ProgrammingGuide.md` located in the NVIGI Core package to learn more about initializing and shutting down NVIGI SDK. :only:`binary_pack:[Which may be found here in combined binary packs](../../../nvigi_core/docs/ProgrammingGuide.md)` 

## 2.0 OBTAIN GPT INTERFACE(S)

Next, we need to retrieve GPT's API interface based on what variant we need (cloud, CUDA etc.).  **NOTE** only the local inference plugins are provided/supported in this early release.  The cloud plugins will be added in a later release:

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

## 3.0 CREATE GPT INSTANCE(S)

Now that we have our interface we can use it to create our GPT instance. To do this, we need to provide information about GPT model we want to use, CPU/GPU resources which are available and various other creation parameters.
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
    params.chain(common);
```
As mention in the previous section, we detected that `GPTSamplerParameters` is supported so we can chain it with the rest of our creation paramters:
```cpp
   // Using helper operator hence *sampler
   params.chain(*sampler);
```
Next we need to provide information about D3D properties if our application is running with a D3D context:
```cpp
    //! Optional but highly recommended if using D3D context, if NOT provided performance might not be optimal
    nvigi::D3D12Parameters d3d12Params{};
    d3d12Params.device = myDevice;
    d3d12Params.queue = myQueue;
    params.chain(d3d12Params);

    if(NVIGI_FAILED(res, igptLocal.createInstance(params, &gptInstanceLocal)))
    {
        LOG("NVIGI call failed, code %d", res);
    }
}
```
> **IMPORTANT**: Providing D3D or Vulkan device and queue is highly recommended to ensure optimal performance
```cpp
nvigi::InferenceInstance* gptInstanceCloud;
{
    nvigi::CommonCreationParameters common{};
    nvigi::GPTCreationParameters params{};    
    common.modelGUID = "{175C5C5D-E978-41AF-8F11-880D0517C524}"; // Model GUID, for details please see NVIGI models repository
    params.chain(common);

    //! Cloud parameters
    nvigi::RESTParameters nvcfParams{};
    std::string token;
    getEnvVar("MY_TOKEN", token);
    nvcfParams.url = myURL;
    nvcfParams.authenticationToken = token.c_str();
    params.chain(nvcfParams);
    
    if(NVIGI_FAILED(res, igptCloud.createInstance(params, &gptInstanceCloud, inputs, countof(inputs))))
    {
        LOG("NVIGI call failed, code %d", res);
    }
}

```

> **NOTE:**
> NVIGI model repository is provided with the pack under `data/nvigi.models`.

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

## 5.0 PREPARE THE EXECUTION CONTEXT

Before GPT can be evaluated the `nvigi::InferenceExecutionContext` needs to be defined. Among other things, this includes specifying input and output slots. GPT can operate in two modes:

* Instruct a model and receive an answer.
* Interact with a model while taking turns in a guided conversation.

We will be using the following helper class to convert text to `nvigi::InferenceDataText`:

```cpp
struct InferenceDataTextHelper
{
    InferenceDataTextHelper() {};
    InferenceDataTextHelper(const char* txt) : _text(txt){}
    InferenceDataTextHelper(const std::string& txt) : _text(txt){}
    operator InferenceDataText* ()
    {
        _data.buffer = _text.data();
        _data.sizeInBytes = _text.length();
        _slot.utf8Text = _data;
        return &_slot;
    };
    InferenceDataText _slot{};
    std::string _text{};
    CpuData _data{};
};
```

### 5.1 INSTRUCT MODE

In this mode GPT receives **any combination** of `system`, `user` and `assistant` input slots with instruction(s) and produces adequate response. Here is an example:

```cpp
//! We can provide ANY combination of input slots, it can be just $user for example.
std::string system("You are a world renown and extremely talented writer.");
std::string user("Write a poem about transformers, robots in disguise");
// Using our helper from the section above
InferenceDataTextHelper systemSlot(system);
InferenceDataTextHelper userSlot(user);
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
InferenceDataTextHelper systemSlot(system);
InferenceDataTextHelper userSlot(user);
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
InferenceDataTextHelper jsonSlot(restJSONBodyAsString);
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
    gptRuntime.chain(gptSampler);

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
    gptRuntime.chain(gptSampler);

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

        // Display response from GPT on screen

        // Now we enter the conversation
        while(runConversation)
        {
            // Setting up new context and new input/output slots
            std::string input = getUserInputBasedOnResponseReceivedFromGPT();
            nvigi::InferenceDataText userSlot(input);        
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

            // Display response from GPT on screen
        }
    }    
}
```

To start a new conversation **simply evaluate your instance with the new SYSTEM input slot** and then repeat the same process.

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

## APPENDIX

### CHAT MODE SETUP SUMMARY

Here is the summary of the steps needed to setup the interactive (chat) mode:

1) Provide `nvigi::GPTRuntimeParameters` in the execution context and set `interactive` flag to TRUE
2) Setup the conversation context by providing `nvigi::kGPTDataSlotSystem` input slot
3) Wait until callback returns `nvigi::kInferenceExecutionStateDone`
4) Get user's input and take turn in the conversation by providing `nvigi::kGPTDataSlotUser` input slot
5) Wait until callback returns `nvigi::kInferenceExecutionStateDone` and full response is provided by the plugin
6) Inform user about the response
7) To continue conversation go back to the step #4
8) To start a new conversation, go to the step #2 and setup new conversation context

> NOTE: When changing models there is no need to change any of the setup above, only model GUID would be different when creating GPT instance