// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

}
}

constexpr const char* kGPTDataSlotSystem = "system";
constexpr const char* kGPTDataSlotUser = "text"; // matching ASR output when used in a pipeline
constexpr const char* kGPTDataSlotAssistant = "assistant";
constexpr const char* kGPTDataSlotImage = "image";
constexpr const char* kGPTDataSlotResponse = "text";
constexpr const char* kGPTDataSlotJSON = "json"; // JSON input/output for the cloud.rest implementation

// {506C5935-67C6-4136-9550-36BBA83C93BC}
struct alignas(8) GPTCreationParameters {
    GPTCreationParameters() {}; 
    NVIGI_UID(UID({ 0x506c5935, 0x67c6, 0x4136,{ 0x95, 0x50, 0x36, 0xbb, 0xa8, 0x3c, 0x93, 0xbc } }), kStructVersion1);
    int32_t maxNumTokensToPredict = 200;
    int32_t contextSize = 512;
    int32_t seed = -1;
};

NVIGI_VALIDATE_STRUCT(GPTCreationParameters)

// {FEB5F4A9-8A02-4864-8757-081F42381160}
struct alignas(8) GPTRuntimeParameters {
    GPTRuntimeParameters() {}; 
    NVIGI_UID(UID({ 0xfeb5f4a9, 0x8a02, 0x4864,{ 0x87, 0x57, 0x8, 0x1f, 0x42, 0x38, 0x11, 0x60 } }), kStructVersion1);
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
    NVIGI_UID(UID({ 0xfd183aa9, 0x6e50, 0x4021,{0x9b, 0x0e, 0xa7, 0xae, 0xab, 0x6e, 0xef, 0x49} }), kStructVersion1)

    int32_t numPrev = 64;               // number of previous tokens to remember
    int32_t numProbs = 0;               // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t minKeep = 0;                // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t topK = 40;                  // <= 0 to use vocab size
    float   minP = 0.05f;               // 0.0 = disabled
    float   xtcProbability = 0.00f;     // 0.0 = disabled
    float   xtcThreshold = 0.10f;       // > 0.5 disables XTC
    float   tfsZ = 1.00f;               // 1.0 = disabled
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
    bool    penalizeNewLine = false;    // consider newlines as a repeatable token
    bool    ignoreEOS = false;

    //! v2+ members go here, remember to update the kStructVersionN in the above NVIGI_UID macro!
};

NVIGI_VALIDATE_STRUCT(GPTSamplerParameters)


//! General Purpose Transformer (GPT) interface
//! 
using IGeneralPurposeTransformer = InferenceInterface;

}