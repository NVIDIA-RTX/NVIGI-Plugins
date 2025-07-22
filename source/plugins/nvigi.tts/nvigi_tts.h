// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "nvigi_ai.h"
#include <memory>
#include <unordered_map>

namespace nvigi
{
namespace plugin
{
namespace tts
{
namespace asqflow_trt
{
constexpr PluginID kId = {{0x97b630c1, 0x1739, 0x4a6a, {0x9f, 0xce, 0x60, 0x73, 0xf6, 0xe8, 0x9a, 0x69}},
                          0xbbf95e}; //{97B630C1-1739-4A6A-9FCE-6073F6E89A69} [nvigi.plugin.tts.asqflow-trt]

} // namespace asqflow_trt

namespace asqflow_ggml
{
namespace cuda
{
constexpr PluginID kId = {{0x99951b3e, 0x3a71, 0x4436,{0xbb, 0x50, 0xd6, 0xb4, 0x33, 0xb8, 0xa3, 0x07}}, 0x86b18b}; //{99951B3E-3A71-4436-BB50-D6B433B8A307} [nvigi.plugin.tts.asqflow-ggml.cuda]
}
namespace vulkan
{
constexpr PluginID kId = {{0x957e46ab, 0x7f17, 0x4e85,{0x98, 0xb4, 0x7e, 0xd9, 0x86, 0x57, 0x49, 0x67}}, 0x600f8d}; //{957E46AB-7F17-4E85-98B4-7ED986574967} [nvigi.plugin.tts.asqflow-ggml.vulkan]
}
namespace d3d12
{
constexpr PluginID kId = {{0x80c16e23, 0xdf53, 0x41a8,{0x86, 0x32, 0x80, 0xea, 0x37, 0xcf, 0xa7, 0x4c}}, 0xe0bf19}; //{80C16E23-DF53-41A8-8632-80EA37CFA74C} [nvigi.plugin.tts.asqflow-ggml.d3d12]
}
} // namespace asqflow_ggml

} // namespace tts
} // namespace plugin

constexpr const char *kTTSDataSlotInputText = "text";
constexpr const char *kTTSDataSlotInputTargetSpectrogramPath = "target_spec";
constexpr const char *kTTSDataSlotOutputAudio = "audio";
constexpr const char *kTTSDataSlotOutputTextNormalized = "text_normalized";

// Custom Exceptions
constexpr uint32_t kResultErrorG2pModelASqFlow = 1 << 24 | plugin::tts::asqflow_trt::kId.crc24;
constexpr uint32_t kResultErrorDpModel = 2 << 24 | plugin::tts::asqflow_trt::kId.crc24;
constexpr uint32_t kResultErrorGeneratorModel = 3 << 24 | plugin::tts::asqflow_trt::kId.crc24;
constexpr uint32_t kResultErrorVocoderModel = 4 << 24 | plugin::tts::asqflow_trt::kId.crc24;

constexpr const char *END_PROMPT_ASYNC =
    "END_PROMPT_ASYNC"; // string to signify that we finished sending asynchronous evaluation

// {435123D8-C97A-4198-86A3-BC1ECFD077E1}
struct alignas(8) TTSCreationParameters
{
    TTSCreationParameters() {};
    NVIGI_UID(UID({0x435123d8, 0xc97a, 0x4198, {0x86, 0xa3, 0xbc, 0x1e, 0xcf, 0xd0, 0x77, 0xe1}}), kStructVersion1);
    bool warmUpModels = true;
};

NVIGI_VALIDATE_STRUCT(TTSCreationParameters);

//! {02025A22-215B-4016-A08C-5607B033BC5B}
struct alignas(8) TTSASqFlowCreationParameters
{
    TTSASqFlowCreationParameters() {};
    NVIGI_UID(UID({0x02025a22, 0x215b, 0x4016, {0xa0, 0x8c, 0x56, 0x07, 0xb0, 0x33, 0xbc, 0x5b}}), kStructVersion1)

    //! v1 members go here, please do NOT break the C ABI compatibility:
    const char *extendedPhonemesDictPath = ""; // Path to the optional extended phoneme dictionary for updating phonemes
                                               // of existing words or adding new word pronunciations

    //! * do not use virtual functions, volatile, STL (e.g. std::vector) or any other C++ high level functionality
    //! * do not use nested structures, always use pointer members
    //! * do not use internal types in _public_ interfaces (like for example 'nvigi::types::vector' etc.)
    //! * do not change or move any existing members once interface has shipped

    //! v2+ members go here, remember to update the kStructVersionN in the above NVIGI_UID macro!
};

NVIGI_VALIDATE_STRUCT(TTSASqFlowCreationParameters);

// {F7359A0A-5387-47D6-BCA4-BB6CEFD4172B}
struct alignas(8) TTSCapabilitiesAndRequirements
{
    TTSCapabilitiesAndRequirements() {};
    NVIGI_UID(UID({0xf7359a0a, 0x5387, 0x47d6, {0xbc, 0xa4, 0xbb, 0x6c, 0xef, 0xd4, 0x17, 0x2b}}), kStructVersion1);
    CommonCapabilitiesAndRequirements *common;
    // Unused
    const char** speakers{};
};

NVIGI_VALIDATE_STRUCT(TTSCapabilitiesAndRequirements);

//! Interface 'TTSASqFlowRuntimeParameters'
//!
//! {41ED9DCB-0AC3-41C0-ADE3-ABBEEB0568C7}
struct alignas(8) TTSASqFlowRuntimeParameters
{
    TTSASqFlowRuntimeParameters() {};
    NVIGI_UID(UID({ 0x41ed9dcb, 0x0ac3, 0x41c0, {0xad, 0xe3, 0xab, 0xbe, 0xeb, 0x05, 0x68, 0xc7} }), kStructVersion3)
    //! v1 members go here, please do NOT break the C ABI compatibility:

    //! * do not use virtual functions, volatile, STL (e.g. std::vector) or any other C++ high level functionality
    //! * do not use nested structures, always use pointer members
    //! * do not use internal types in _public_ interfaces (like for example 'nvigi::types::vector' etc.)
    //! * do not change or move any existing members once interface has shipped

    //! v2+ members go here, remember to update the kStructVersionN in the above NVIGI_UID macro!
    float speed = 1.0;

    //! v3+ members go here

    // These parameters control how the input text is split into chunks for processing
    // The algorithm will try to split the text into chunks of size between minChunkSize and maxChunkSize
    // and avoid splitting sentences in the middle.
    // Having a low minChunkSize can lead to a better time to first audio.
    int minChunkSize = 100;  // Minimum chunk size in characters
    int maxChunkSize = 200;  // Maximum chunk size in characters

    // Random seed for generation
    int seed = -1;

    //! The following parameters apply exclusively to the GGML backend.

    // Number of timesteps for TTS inference
    // minimum suggested value is 16, maximum suggested value is 32
    // Choose lower value for faster inference, higher value for better quality
    int n_timesteps = 16; //Only for GGML backend


    // Sampler type: 0 = EULER, 1 = DPM_SOLVER_PLUS_PLUS
    int sampler = 1;  // Only for GGML backend. Default to DPM_SOLVER_PLUS_PLUS

    // DPM++ solver order: 1, 2, 3
    // Higher order can provide better quality but may be slower
    int dpmpp_order = 2;//Only for GGML backend

    // Enable flash attention for better performance
    bool use_flash_attention = true; //Only for GGML backend
};

NVIGI_VALIDATE_STRUCT(TTSASqFlowRuntimeParameters)

//! Text To Speech (TTS) interface
using ITextToSpeech = InferenceInterface;

} // namespace nvigi
