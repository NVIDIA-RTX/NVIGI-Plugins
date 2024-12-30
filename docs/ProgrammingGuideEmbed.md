
# Embedding (EMBED) Programming Guide

The focus of this guide is on using In-Game Inferencing to integrate an embedding model into an application. One example would be [E5-Large-Unsupervised](https://huggingface.co/intfloat/e5-large-unsupervised)

> **IMPORTANT**: This guide might contain pseudo code, for the up to date implementation and source code which can be copy pasted please see the [rag sample](../source/samples/nvigi.rag/rag.cpp)

## Version 1.0.0 General Access

## 1.0 INITIALIZE AND SHUTDOWN

Please read the `docs/ProgrammingGuide.md` located in the NVIGI Core package to learn more about initializing and shutting down NVIGI SDK. :only:`binary_pack:[Which may be found here in combined binary packs](../../../nvigi_core/docs/ProgrammingGuide.md)` 

## 2.0 OBTAIN EMBED INTERFACE(S)

Next, we need to retrieve EMBED's API interface based on what variant we need (cloud, CUDA etc.).  **NOTE** only the local inference plugins are provided/supported in this early release.  The cloud plugins will be added in a later release:

```cpp

nvigi::IEmbed iembedLocal{};
// Here we are requesting interface for the GGML_CUDA implementation
if(NVIGI_FAILED(res, nvigiGetInterface(nvigi::plugin::embed::ggml::cuda, iembedLocal))
{
    LOG("NVIGI call failed, code %d", res);
}

nvigi::IEmbed iembedCloud{};
// Here we are requesting interface for the GFN cloud implementation (NOT IMPLEMENTED YET)
if(NVIGI_FAILED(res, nvigiGetInterface(nvigi::plugin::embed::gfn::nvcf::kId, iembedCloud))
{
    LOG("NVIGI call failed, code %d", res);
}
```

> **NOTE:**
One can only obtain interface for a feature which is available on user system. Interfaces are valid as long as the underlying plugin is loaded and active.

## 3.0 CREATE EMBED INSTANCE(S)

Now that we have our interface we can use it to create our EMBED instance. To do this, we need to provide information about Embedding model we want to use, CPU/GPU resources which are available and various other creation parameters.   **NOTE** only the local inference plugins are provided/supported in this early release.  The cloud plugins will be added in a later release.

Here is an example:

```cpp

//! Here we are creating two instances for different backends/APIs
//!
//! IMPORTANT: This is totally optional and only used to demonstrate runtime switching between different backends

nvigi::InferenceInstance* embedInstanceLocal;
{
    //! Creating local instance and providing our D3D12 or VK and CUDA information (all optional)
    //!
    //! This allows host to control how instance interacts with DirectX, Vulkan (if at all) or any existing CUDA contexts (if any)
    //!
    //! Note that providing DirectX/Vulkan information is mandatory if at runtime we expect instance to run on a command list.

    nvigi::EmbedCreationParameters params{};    

    nvigi::CommonCreationParameters common{};
    common.numThreads = myNumCPUThreads; // How many CPU threads is instance allowed to use
    common.vramBudgetMB = myVRAMBudget;  // How much VRAM is instance allowed to occupy
    common.utf8PathToModels = myPathToNVIGIModelRepository; // Path to provided NVIGI model repository (using UTF-8 encoding)
    common.modelGUID = "{5D458A64-C62E-4A9C-9086-2ADBF6B241C7}"; // Model GUID for e5-large-unsupervised, for details please see NVIGI models repository

    params.chain(common);

    //! OPTIONAL PSEUDO CODE - ONLY FOR ILLUSTRATION ON WHAT IS POSSIBLE

    // D3D12 creation parameters - tell our instance what device, adapter etc it should use or attach CUDA to
    nvigi::D3D12Parameters d3d12{};
    d3d12.device = myDevice;
    d3d12.adapter = myAdapter;
    d3d12.cmdQueue = myCmdQueue;
    d3d12.cmdList = myCmdList;

    // Vulkan creation parameters - tell our instance what device, adapter etc it should use or attach CUDA to
    nvigi::VulkanParameters vk{};
    vk.device = myDevice;
    vk.physicalDevice = myPhysicalDevice;
    vk.instance = myInstance;
    vk.cmdQueue = myQueue;
    vk.cmdBuffer = myCmdBuffer;

    // CUDA creation parameters - tell our instance what context, streams to use
    nvigi::CUDAParameters cuda{};
    cuda.context = myCUDAContext;
    cuda.stream = myStream;

    //! Example of how optional parameters can be chained as needed - most likely one would not be using d3d12, cuda and vk at the same time
    params.chain(d3d12) // or vk or cuda depending

    //! Query capabilities/models list and find the model we are interested in to use that index to find what embedding size our output data should be in.
    nvigi::EmbedCapabilitiesAndRequirements* info{};
    getCapsAndRequirements(iembedLocal, params1, &info);
    for (int i = 0; i < info.common->numSupportedModels; ++i)
    {
        if (strcmp(info.common->supportedModelGUIDs[i], params1.common->modelGUID) == 0)
        {
            embedding_size = info.embedding_numel[i];
            max_position_embeddings = info.max_position_embeddings[i];
            break;
        }
    }

    REQUIRE(embedding_size != 0);
    REQUIRE(max_position_embeddings != 0);

    if(NVIGI_FAILED(res, iembedLocal.createInstance(params, &embedInstanceLocal)))
    {
        LOG("NVIGI call failed, code %d", res);
    }
}

nvigi::InferenceInstance* embedInstanceCloud;
{
    // Note that CUDA and DirectX info here make no sense hence it is not provided
    nvigi::EmbedCreationParameters params{};

    nvigi::CommonCreationParameters common{};    
    common.modelGUID = "{5D458A64-C62E-4A9C-9086-2ADBF6B241C7}"; // Model GUID for e5-large-unsupervised, the exact same model is running in the GFN docker

    params.chain(common);

    //! Cloud parameters
    nvigi::RESTParameters nvcfParams{};
    std::string token;
    getEnvVar("MY_TOKEN", token);
    nvcfParams.url = myURL;
    nvcfParams.authenticationToken = token.c_str();
    params.chain(nvcfParams);
    
    if(NVIGI_FAILED(res, iembedCloud.createInstance(params, &embedInstanceCloud, inputs, countof(inputs))))
    {
        LOG("NVIGI call failed, code %d", res);
    }
}

```

> **NOTE:**
> NVIGI model repository is provided with the pack in nvigi.models.

## 5.0 SETUP CALLBACK TO RECEIVE INFERRED DATA

In order to receive transcribed text from the Embedding model inference a special callback needs to be setup like this:

```cpp
std::atomic<bool> embed_done = false;
nvigi::InferenceExecutionState embedOnComplete(const nvigi::InferenceExecutionContext* ctx, nvigi::InferenceExecutionState state, void* data)
{
    if (ctx)
    {
        auto slots = ctx->outputs;
        auto& answer = *(std::string*)data;
        const nvigi::InferenceDataByteArray* output_embedding{};
        slots->findAndValidateSlot(nvigi::kEmbedDataSlotOutEmbedding, &output_embedding);

        CpuData* cpuBuffer = castTo<CpuData>(output_embedding->bytes);
        float* fp32Buffer = (float*)(cpuBuffer->buffer);
    }

    embed_done.store(state == nvigi::InferenceExecutionStateDone);
    return state;
} 
```

> **IMPORTANT:**
> Input and output data slots provided within the execution context are **only valid during the callback execution**. Host application must be ready to handle callbacks until reaching `nvigi::InferenceExecutionStateDone` or `nvigi::InferenceExecutionStateCancel` state.

> **NOTE:**
> To cancel EMBED inference make sure to return `nvigi::InferenceExecutionStateCancel` state in the callback.

## 6.0 PREPARE THE EXECUTION CONTEXT AND EXECUTE INFERENCE

Before EMBED can be evaluated the `nvigi::InferenceExecutionContext` needs to be defined. Among other things, this includes specifying input and output slots.


```cpp

// Use nvigi::prompt_sep to separate prompts
std::string input = "Here one prompt." + nvigi::prompt_sep + "Here a second prompt."
int n_prompts = countLines(input, prompts_sep);
output_embeddings.clear();

output_embeddings.resize(n_prompts*embedding_size);

nvigi::CpuData text(input.length() + 1, (void*)input.c_str());
nvigi::InferenceDataText input_prompt(text);
std::vector<nvigi::InferenceDataSlot> inSlots = { {nvigi::kEmbedDataSlotInText, &input_prompt} };
InferenceDataSlotArray inputs = { inSlots.size(), inSlots.data() };

// If the output slots are not provided, NVIGI will allocate them. Alternatively, host can allocate and provide outputs as shown in the next lines
nvigi::CpuData cpu_data;
cpu_data.sizeInBytes = output_embeddings.size() * sizeof(float);
cpu_data.buffer = output_embeddings.data();
nvigi::InferenceDataByteArray output_param(cpu_data);
std::vector<nvigi::InferenceDataSlot> outSlots = { {nvigi::kEmbedDataSlotOutEmbedding, &output_param} };
InferenceDataSlotArray outputs = { outSlots.size(), outSlots.data() };

nvigi::InferenceExecutionContext embedContext{};
embedContext.instance = embedInstanceLocal; // The instance we created and we want to run inference on
embedContext.callback = embedOnComplete;
embedContext.inputs = &inputs;
embedContext.outputs = &outputs;

//! IMPORTANT: The execution context and all provided data (input, output slots) must be valid at the time `instance->evaluate` is called
// Evaluate our instance
nvigi::Result res = embedContext.instance->evaluate(&embedContext);

//! IMPORTANT: Wait for the callback to receive nvigi::InferenceExecutionStateDone
while (!embed_done && res == nvigi::ResultOk)
    continue;

//! Now we have received the response from the Embedding model via our embedOnComplete Callback

```

> **IMPORTANT:**
> The execution context and all provided data (input, output slots) must be valid at the time `instance->evaluate` is called

## 7.0 Do something with the embedding

In your execution pipeline, call `instance->evaluate` at the appropriate location where a prompt needs to be processed to receive a response from the EMBED.

### 7.1 Use it to match a player's answer to a set of predefined answers.
Imagine we are in a game where the player is asked a question. The game has a set of predefined answers, but we want to allow the player to respond freely. To match the player's response with the closest predefined answer, we can use embeddings to analyze and compare the answers.

Let's suppose the game asks this question : 

```
"You come across a locked door with a numeric keypad. There is a note saying \"The key is in the stars.\" What do you do?",
    "A. Look for a constellation map",
    "B. Try random numbers",
    "C. Look for another way around",
    "D. Give up"
```

The player answer : "I'd look up at the stars and try to find a clue."

```cpp

// Function to encode prompts
std::string encodePrompt(std::vector<std::string>  prompts){

    std::string full_prompts_str = "";

    for (int i_prompt = 0; i_prompt < prompts.size(); i_prompt += 1) {
        full_prompts_str += prompts[i_prompt];

        if (i_prompt < prompts.size() - 1)
            full_prompts_str += nvigi::prompts_sep;
    }

    return full_prompts_str;

}

// Let's prepare our prompts
std::vector<std::string> predefined_answers = {
    "A. Confront the butler",
    "B. Search the kitchen",
    "C. Investigate the study further",
    "D. Ask the maid if she saw anything"
};

std::string player_answer = "I'd look up at the stars and try to find a clue.";
std::string predefined_answer_prompts = encode_prompt(predefined_answers);
std::string full_prompts = predefined_answer_prompts + nvigi::prompts_sep + player_answer;

// Let's assume we encapsulated the evaluation in a function called embed (See Section 6)
embed(iembed, instance, full_prompts, output_embeddings);

// Let's get the data and compute cosinus simularity between embeddings to match the closest answer
int n_predefined_answers = predefined_answers.size();

int answer_start_pos = n_predefined_answers * embedding_size;
std::vector<float> embedding_answer = std::vector<float>(emb + answer_start_pos, emb + answer_start_pos + embedding_size);

cout << "\tPlayer given Answer : " << player_answer << endl;
float max_score = 0;
int max_score_index = 0;
// Compute cosin similarity with each predefined answer
for (int i_predefined_answer = 0; i_predefined_answer < predefined_answers.size(); i_predefined_answer += 1) {
    int predefined_answer_start_pos = i_predefined_answer * embedding_size;
    std::vector<float> embedding_predefined_answer = std::vector<float>(emb + predefined_answer_start_pos, emb + predefined_answer_start_pos + embedding_size);

    // You can find the code of this function here : source\samples\nvigi.rag\rag.cpp
    float cosinScore = cosSimScore(embedding_answer, embedding_predefined_answer);

    if (cosinScore > max_score) {
        max_score = cosinScore;
        max_score_index = i_predefined_answer;
    }

    std::cout << "\tCosine Similarity between the player answer and the prompt ( " << questions[i_predefined_answer] << " ) : " << cosinScore << std::endl << std::endl;
}

std:: cout << "The player has choosen the answer : " << predefined_answers[max_score_index] << endl;

```


> **IMPORTANT:**
> The host app CANNOT assume that the inference callback will be invoked on the thread that calls `instance->evaluate`. In addition, inference (and thus callback invocations) is NOT guaranteed to be done when `instance->evaluate` returns.

## 8.0 DESTROY INSTANCE(S)

Once EMBED is no longer needed each instance should be destroyed like this:

```cpp
//! Finally, we destroy our instance(s)
if(NVIGI_FAILED(res, iembedLocal.destroyInstance(embedInstanceLocal))) 
{ 
    LOG("NVIGI call failed, code %d", res);
}
if(NVIGI_FAILED(res, iembedCloud.destroyInstance(embedInstanceCloud))) 
{ 
    LOG("NVIGI call failed, code %d", res);
} 
```

## 9.0 UNLOAD INTERFACE(S)

Once EMBED is no longer needed each interface should be unloaded like this:

**NOTE** only the local inference plugins are provided/supported in this early release.  The cloud plugins will be added in a later release.

```cpp
//! Finally, we destroy our instance(s)
if(NVIGI_FAILED(result, nvigiUnloadInterface(nvigi::plugin::embed::gfn::nvcf::kId , &iembedCloud))) 
{ 
    //! Check error
}
if(NVIGI_FAILED(result, nvigiUnloadInterface(nvigi::plugin::embed::ggml::cuda::kId, &iembedLocal))) 
{ 
    //! Check error
} 
```

## 10.0 Exceptions

> **IMPORTANT:**
> During the evalutions some errors can happen. In this case the Result of ctx.instance->evaluate(&ctx) will not be nvigi::ResultOk

It can be : 
- nvigi::kResultOk : If everything goes well and embedding output has been generated.
- nvigi::kResultNonUtf8 : If an input prompt contains non-UTF8 characters, you can address it by either removing those characters or converting them to UTF8 using simple or advanced processing techniques.
- nvigi::kResultMaxTokensReached : If one prompt has reached the maximum number of tokens that the model can process. Reduce prompt length or switch to a model with a larger context size.
