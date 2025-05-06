# TTS (ASqFlow) Programming Guide

The focus of this guide is on using AI Inference Manager to integrate a TTS model into an application.

> **IMPORTANT:**
> **This feature is considered experimental in this release.  It is subject to significant change in later releases, possibly requring app modifications.**

> **IMPORTANT**: This guide might contain pseudo code, for the up to date implementation and source code which can be copy pasted please see the [basic sample]

A general overview of the components within the cpp ASqFlow implementation, including its capabilities, expected inputs, and outputs is shown in the diagram below:

![General overiew of ASqFlow implementation](media/ASqFlowSolution.png)

## 1.0 INITIALIZE AND SHUTDOWN

Please read the `docs\ProgrammingGuide.md` located in the NVIGI Core package to learn more about initializing and shutting down NVIGI SDK.

## 2.0 OBTAIN TTS INTERFACE(S)

Next, we need to retrieve TTS's API interface based on ASqFlow.

```cpp

nvigi::ITTS ittsLocal{};
// Here we are requesting interface for the GGML_CUDA implementation
if(NVIGI_FAILED(res, nvigiGetInterface(nvigi::plugin::tts::ASqFlow::trt, ittsLocal))
{
    LOG("NVIGI call failed, code %d", res);
}

```

## 3.0 CREATE TTS INSTANCE(S)

Now that we have our interface we can use it to create our TTS instance. To do this, we need to provide information about the TTS model we want to use, CPU/GPU resources which are available and various other creation parameters.

Here is an example:

```cpp

//! Here we are creating two instances for different backends/APIs
//!
//! IMPORTANT: This is totally optional and only used to demonstrate runtime switching between different backends

nvigi::InferenceInstance* ttsInstanceLocal;
{
    //! Creating local instance and providing our D3D12 or VK and CUDA information (all optional)
    //!
    //! This allows host to control how instance interacts with DirectX, Vulkan (if at all) or any existing CUDA contexts (if any)
    //!
    //! Note that providing DirectX/Vulkan information is mandatory if at runtime we expect instance to run on a command list.

    nvigi::TTSCreationParameters params{};
    nvigi::TTSASqFlowCreationParameters paramsAsqflow{};    

    nvigi::CommonCreationParameters common{};
    common.numThreads = myNumCPUThreads; // How many CPU threads is instance allowed to use
    common.vramBudgetMB = myVRAMBudget;  // How much VRAM is instance allowed to occupy
    common.utf8PathToModels = myPathToNVIGIModelRepository; // Path to provided NVIGI model repository (using UTF-8 encoding)
    common.modelGUID = "{81320D1D-DF3C-4CFC-B9FA-4D3FF95FC35F}"; // Model GUID for ASqFlow model
    params.warmUpModels = true; // faster inference, disable it if you want faster creation time. Default True.

    // Asqflow tts parameters
    paramsAsqflow.extendedPhonemesDictPath = "Path to a phoneme dictionary, which will extend the default dictionary present in the model's folder."

    // Note - this is pseudo code; the return value of chain() should always be checked
    params.chain(common);
    params.chain(paramsAsqflow);

    //! Optional but highly recommended if using D3D context, if NOT provided performance might not be optimal
    nvigi::D3D12Parameters d3d12Params{};
    d3d12Params.device = myDevice;
    d3d12Params.queue = myQueue;
    params.chain(d3d12Params);

    //! Query capabilities/models list and find the model we are interested in.
    nvigi::TTSCapabilitiesAndRequirements* info{};
    getCapsAndRequirements(ittsLocal, params1, &info);
    REQUIRE(info != nullptr);

    if(NVIGI_FAILED(res, ittsLocal.createInstance(params, &ttsInstanceLocal)))
    {
        LOG("NVIGI call failed, code %d", res);
    }
}

```

### TTSCreationParameters and TTSASqFlowCreationParameters

The `TTSCreationParameters` structure allows you to specify some parameters for creating a TTS instance. Here are the parameters:

- **warmUpModels**: 
  - **Type**: `bool`
  - **Description**: If set to `true`, the models will be warmed up during creation, leading to faster inference times. If set to `false`, the creation time will be faster, but the first inference might be slower. The default value is `true`.

The `TTSASqFlowCreationParameters` structure allows you to specify some parameters for creating a Asqflow TTS instance. Here are the parameters:

- **extendedPhonemesDictPath**: 
  - **Type**: `const char*`
  - **Description**: Path to a phoneme dictionary, which will extend the default dictionary. This allows you to provide additional phoneme mappings that are not present in the default dictionary.

> **IMPORTANT**: Providing D3D or Vulkan device and queue is highly recommended to ensure optimal performance

> **NOTE:**
> NVIGI model repository is provided with the pack in nvigi.models.

> **NOTE:**
One can only obtain interface for a feature which is available on user system. Interfaces are valid as long as the underlying plugin is loaded and active.



## 4.0 SETUP CALLBACK TO RECEIVE INFERRED DATA

In order to receive audio data from the TTS model inference a special callback needs to be setup like this:

```cpp
// Callback when tts Inference starts sending audio data
playAudioWhenReceivingData = true
// Callback when tts Inference starts sending audio data
auto ttsOnComplete = [](const nvigi::InferenceExecutionContext *ctx, nvigi::InferenceExecutionState state,
                        void *userData) -> nvigi::InferenceExecutionState {
    // In case an error happened
    if (state == nvigi::kInferenceExecutionStateInvalid)
    {
        tts_status.store(state);
        return state;
    }

    if (ctx)
    {
        auto outputData = (OutputData *)userData;
        auto slots = ctx->outputs;
        std::vector<int16_t> tempChunkAudio;
        const nvigi::InferenceDataByteArray *outputAudioData{};
        const nvigi::InferenceDataText *outputTextNormalized{};
        slots->findAndValidateSlot(nvigi::kTTSDataSlotOutputAudio, &outputAudioData);
        slots->findAndValidateSlot(nvigi::kTTSDataSlotOutputTextNormalized, &outputTextNormalized);

        CpuData *cpuBuffer = castTo<CpuData>(outputAudioData->bytes);

        for (int i = 0; i < cpuBuffer->sizeInBytes / 2; i++)
        {
            int16_t value = reinterpret_cast<const int16_t *>(cpuBuffer->buffer)[i];
            outputData->outputAudio.push_back(value);
            tempChunkAudio.push_back(value);
        }

        outputData->outputTextNormalized += outputTextNormalized->getUTF8Text();

        // Create threads to start playing audio
        if (playAudioWhenReceivingData)
        {
            std::lock_guard<std::mutex> lock(mtxAddThreads);
            playAudioThreads.push(std::make_unique<std::thread>(
                std::thread(savePlayAudioData<int16_t>, tempChunkAudio, "", 22050, true, false)));
        }
    }

    tts_status.store(state);
    return state;
};

```

> **IMPORTANT:**
> Input and output data slots provided within the execution context are **only valid during the callback execution**. Host application must be ready to handle callbacks until reaching `nvigi::InferenceExecutionStateDone` or `nvigi::InferenceExecutionStateCancel` state.

> **NOTE:**
> To cancel TTS inference make sure to return `nvigi::InferenceExecutionStateCancel` state in the callback.


## 5.0 PREPARE THE EXECUTION CONTEXT AND EXECUTE INFERENCE

Before TTS can be evaluated the `nvigi::InferenceExecutionContext` needs to be defined. Among other things, this includes specifying input slots.


```cpp

// Define inputs slots
std::string inputPrompt = "Here an example of imput prompt";
nvigi::InferenceDataTextSTLHelper inputPromptData(inputPrompt);

std::string targetPathSpectrogram = "../../../data/nvigi.test/nvigi.tts/ASqFlow/mel_spectrograms_targets/sample_3_neutral_se.bin";
nvigi::InferenceDataTextSTLHelper inputPathTargetSpectrogram(targetPathSpectrogram);

std::vector<nvigi::InferenceDataSlot> inSlots = { {nvigi::kTTSDataSlotInputText, inputPromptData},
                                    {nvigi::kTTSDataSlotInputTargetSpectrogramPath, inputPathTargetSpectrogram } };
InferenceDataSlotArray inputs = { inSlots.size(), inSlots.data() };


// Define Runtime parameters
nvigi::ASqFlowTTSRuntimeParameters runtime{};
runtime.speed = 1.0; // You can adjust the desired speed of the output audio if you like. It is recommended to not go lower than 0.7 and higher than 1.3. The value will be clipped between 0.5 and 1.5.

// Run inference
nvigi::InferenceExecutionContext ctx{};
ctx.instance = ttsInstanceLocal;
ctx.callback = ttsOnComplete;
ctx.callbackUserData = &outputAudio;
ctx.inputs = &inputs;
ctx.runtimeParameters = runtime;
ctx.outputs = nullptr;

//Evaluate
nvigi::Result res;
res = ctx.instance->evaluate(&ctx);

// Wait until the inference is done
while (!(tts_status == nvigi::kInferenceExecutionStateDone || tts_status == nvigi::kInferenceExecutionStateInvalid)
        && res == nvigi::kResultOk)
    continue;


// If an audio is playing, wait for it to finish and destroy the corresponding threads
while (true){
    std::lock_guard<std::mutex> lock(mtxAddThreads);
    std::unique_ptr<std::thread> thread;
    {
    if (playAudioThreads.empty())
        break;
    thread = std::move(playAudioThreads.front());
    playAudioThreads.pop();
    }

    if (thread->joinable()) {
        thread->join();
    }
}
tts_status.store(nvigi::kInferenceExecutionStateDataPending);

```

> **IMPORTANT:**
> The execution context and all provided data (input, output slots) must be valid at the time `instance->evaluate` is called


> **IMPORTANT:**
> The host app CANNOT assume that the inference callback will be invoked on the thread that calls `instance->evaluate`. In addition, inference (and thus callback invocations) is NOT guaranteed to be done when `instance->evaluate` returns.

## 6.0 DESTROY INSTANCE(S)

Once TTS is no longer needed each instance should be destroyed like this:

```cpp
//! Finally, we destroy our instance(s)
if(NVIGI_FAILED(res, ittsLocal.destroyInstance(ttsInstanceLocal))) 
{ 
    LOG("NVIGI call failed, code %d", res);
}
```


> **IMPORTANT:**
> Please review the transcription carefully to ensure it matches the audio, as any discrepancies could significantly affect the output quality.
## 7.0 AVAILABLE FEATURES IN ASQFLOW TTS

### Voice cloning
The A^2 Flow TTS model also allows for the creation of new voices via 'voice cloning'; a script for creating your own voices is available; please contact your NVIDIA Developer Relations representative for additional details.

### Async mode
The asynchronous mode (`evaluateAsync`) allows you to provide input prompts while processing continues in the background. This is particularly useful if you're expecting long outputs from a GPT model and need TTS to begin processing before the GPT model has finished responding.

## 8.0 KNOWN LIMITATIONS

### Currencies
The current text normalization may have certain limitations when handling currencies.

### Words that are not present inside the dictionary
When a word is not found in the dictionary, the system uses a graph-to-phoneme (g2p) model to predict its pronunciation. This model is based on a small neural network, sourced from https://github.com/Kyubyong/g2p. However, the predicted pronunciation may not always match your expectations.
You can create a custom phoneme dictionary to extend the default one. Provide the path to this custom dictionary during instance creation using the `extendedPhonemesDictPath` parameter.

### Words may have different pronounciations
Some words in the dictionary have multiple pronunciations. The system will always choose the first one, which may not be the desired pronunciation.

