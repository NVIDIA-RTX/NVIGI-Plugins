// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "nvigi_ai.h"

namespace nvigi
{
namespace plugin
{
namespace gpt
{
namespace ggml::cuda{
constexpr PluginID kId  = { {0x54bbefba, 0x535f, 0x4d77,{0x9c, 0x3f, 0x46, 0x38, 0x39, 0x2d, 0x23, 0xac}}, 0x4b9ee9 };  // {54BBEFBA-535F-4D77-9C3F-4638392D23AC} [nvigi.plugin.gpt.ggml.cuda]
}
namespace ggml::cpu{
constexpr PluginID kId   = { {0x1119fd8b, 0xfc4b, 0x425d,{0xa3, 0x72, 0xcc, 0xe7, 0xd5, 0x27, 0x34, 0x10}}, 0xaae2ed };  // {1119FD8B-FC4B-425D-A372-CCE7D5273410} [nvigi.plugin.gpt.ggml.cpu]
}
namespace cloud::rest
{
constexpr PluginID kId = { {0x3553c9f3, 0x686c, 0x4f08,{0x83, 0x8e, 0xf2, 0xe3, 0xb4, 0x01, 0x9a, 0x72}}, 0xa589b7 }; //{3553C9F3-686C-4F08-838E-F2E3B4019A72} [nvigi.plugin.gpt.cloud.rest]
}
namespace ggml::vulkan
{
constexpr PluginID kId = { {0xc36d66a8, 0x067c, 0x48e3,{0x98, 0x99, 0x6d, 0xd2, 0x36, 0x10, 0xca, 0x08}}, 0xdb9f72 }; //{C36D66A8-067C-48E3-9899-6DD23610CA08} [nvigi.plugin.gpt.ggml.vulkan]
}
namespace ggml::d3d12
{
constexpr PluginID kId = { {0xf007256a, 0xe71e, 0x4ffe,{0xb3, 0xa8, 0x30, 0xd7, 0x7e, 0x42, 0x58, 0x14}}, 0xb530c8 }; //{F007256A-E71E-4FFE-B3A8-30D77E425814} [nvigi.plugin.gpt.ggml.d3d12]
}


}
}

constexpr const char* kGPTDataSlotSystem = "system";
constexpr const char* kGPTDataSlotUser = "text"; // matching ASR output when used in a pipeline
constexpr const char* kGPTDataSlotAssistant = "assistant";
constexpr const char* kGPTDataSlotImage = "image"; // Begin VLM Addition -- Added for VLM -- End VLM Addition
constexpr const char* kGPTDataSlotResponse = "text";
constexpr const char* kGPTDataSlotJSON = "json"; // JSON input/output for the cloud.rest implementation

//! NOTE: Not all quantized GGML types are necessarily allowed with the KV cache.
//! 
constexpr int32_t kQuantizedTypeFP32 = 0;
constexpr int32_t kQuantizedTypeFP16 = 1;
constexpr int32_t kQuantizedTypeQ4_0 = 2;
constexpr int32_t kQuantizedTypeQ8_0 = 8;

//! GPT Creation Parameters
//! 
//! NOTE: Chain GPTRuntimeParameters to this structure to provide runtime parameters
//! 
//! {506C5935-67C6-4136-9550-36BBA83C93BC}
struct alignas(8) GPTCreationParameters {
    GPTCreationParameters() {}; 
    NVIGI_UID(UID({ 0x506c5935, 0x67c6, 0x4136,{ 0x95, 0x50, 0x36, 0xbb, 0xa8, 0x3c, 0x93, 0xbc } }), kStructVersion3);
    int32_t maxNumTokensToPredict = 200;
    int32_t contextSize = 512;
    int32_t seed = -1;
    // v2
    bool flashAttention = false;                // if true, the model will use flash attention
    int32_t batchSize = 2048;                   // batch size for prompt processing (must be >=32 to use BLAS and can NOT be increased at runtime)
    int32_t physicalBatchSize = 512;            // physical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t cacheTypeK = kQuantizedTypeFP16;    // optional, only supported by the GGML backend and maps directly to ggml_type, default fp16
    int32_t cacheTypeV = kQuantizedTypeFP16;    // optional, only supported by the GGML backend and maps directly to ggml_type, default fp16

    //! v3+ members go here, remember to update the kStructVersionN in the above NVIGI_UID macro!
    size_t numLoras{};          // num loras specified.
    const char** loraNames{};   // names of the loras that should be loaded
    const float* loraScales{};  // optional: the scales that should be applied to the lora, between 0 and 1.  If not present, must set in runtime parameters to activate.
};

NVIGI_VALIDATE_STRUCT(GPTCreationParameters)

// {FEB5F4A9-8A02-4864-8757-081F42381160}
struct alignas(8) GPTRuntimeParameters {
    GPTRuntimeParameters() {}; 
    NVIGI_UID(UID({ 0xfeb5f4a9, 0x8a02, 0x4864,{ 0x87, 0x57, 0x8, 0x1f, 0x42, 0x38, 0x11, 0x60 } }), kStructVersion4);
    uint32_t seed = 0xFFFFFFFF;     // RNG seed
    int32_t tokensToPredict = -1;   // new tokens to predict
    int32_t batchSize = 512;        // batch size for prompt processing (must be >=32 to use BLAS)
    int32_t tokensToKeep = 0;       // number of tokens to keep from initial prompt
    int32_t tokensToDraft = 16;     // number of tokens to draft during speculative decoding
    int32_t numChunks = -1;         // max number of chunks to process (-1 = unlimited)
    int32_t numParallel = 1;        // number of parallel sequences to decode
    int32_t numSequences = 1;       // number of sequences to decode
    float temperature = 0.2f;       // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float topP = 0.7f;              // 1.0 = disabled
    bool interactive = true;        // chat mode by default
    const char* reversePrompt{};    // reverse prompt for the interactive mode
    const char* prefix{};           // prefix for the user input
    const char* suffix{};           // suffix for the user input
    // v2
    float frameTimeMs = 0.0f;           // optional, used to limit the token generation rate, disabled by default
    int32_t targetTokensPerSecond = -1; // optional, used to limit the token generation rate, disabled by default

    //! v3+ members go here, remember to update the kStructVersionN in the above NVIGI_UID macro!
    bool promptPretemplatized = false;          // if true, will not attempt to apply any sort of prompt templatization from the model.  User is responsible for getting it right.

    //! v4+ members go here, remember to update the kStructVersionN in the above NVIGI_UID macro!
    size_t numLoras{};              // num loras being updated
    const char** loraNames{};       // names of the loras being updated.  Must match one of the names set during GPTCreationParameters
    const float* loraScales{};      // scale of the loras being updated.  Can use this to dynamically turn on/off loras per evaluate call.
};

NVIGI_VALIDATE_STRUCT(GPTRuntimeParameters)


//! Interface 'GPTSamplerParameters'
//!
//! OPTIONAL - not necessarily supported by all backends!
//! 
//! Use 'getCapsAndRequirements' API to obtain 'CommonCapabilitiesAndRequirements' then check if this structure is chained to it
//! 
//! {FD183AA9-6E50-4021-9B0E-A7AEAB6EEF49}
struct alignas(8) GPTSamplerParameters
{
    GPTSamplerParameters() { };
    NVIGI_UID(UID({ 0xfd183aa9, 0x6e50, 0x4021,{0x9b, 0x0e, 0xa7, 0xae, 0xab, 0x6e, 0xef, 0x49} }), kStructVersion2)

    int32_t numPrev = 64;               // number of previous tokens to remember
    int32_t numProbs = 0;               // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t minKeep = 0;                // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t topK = 40;                  // <= 0 to use vocab size
    float   minP = 0.05f;               // 0.0 = disabled
    float   xtcProbability = 0.00f;     // 0.0 = disabled
    float   xtcThreshold = 0.10f;       // > 0.5 disables XTC
    float   tfsZ = 1.00f;               // DEPRECATED
    float   typP = 1.00f;               // typical_p, 1.0 = disabled
    float   dynatempRange = 0.00f;      // 0.0 = disabled
    float   dynatempExponent = 1.00f;   // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t penaltyLastN = 64;          // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   penaltyRepeat = 1.00f;      // 1.0 = disabled
    float   penaltyFreq = 0.00f;        // 0.0 = disabled
    float   penaltyPresent = 0.00f;     // 0.0 = disabled
    int32_t mirostat = 0;               // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostatTAU = 5.00f;        // target entropy
    float   mirostatETA = 0.10f;        // learning rate
    bool    penalizeNewLine = false;    // DEPRECATED
    bool    ignoreEOS = false;

    // v2
    bool persistentKVCache = false;         // if true, the KV cache will NOT be cleared between calls in instruct mode or when new system prompt is provided in chat (interactive) mode
    const char* grammar{};                  // optional BNF-like grammar to constrain sampling
    const char* utf8PathToSessionCache{};   // optional path to a session file to load/save the session state

    //! v3+ members go here, remember to update the kStructVersionN in the above NVIGI_UID macro!
};

NVIGI_VALIDATE_STRUCT(GPTSamplerParameters)


//! General Purpose Transformer (GPT) interface
//! 
using IGeneralPurposeTransformer = InferenceInterface;

}