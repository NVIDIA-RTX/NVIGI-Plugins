// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#include "nvigi_ai.h"

namespace nvigi
{
namespace plugin::asr::ggml::cuda
{
constexpr PluginID kId = {{0x731fdb34, 0x8c5b, 0x4bd5,{0xb6, 0xd0, 0x09, 0xad, 0xdf, 0x89, 0x8b, 0x2b}}, 0x4429e2}; // {731FDB34-8C5B-4BD5-B6D0-09ADDF898B2B} [nvigi.plugin.asr.ggml.cuda]
}
namespace plugin::asr::ggml::cpu
{
constexpr PluginID kId = {{0x2654567f, 0x2cf4, 0x4e4e,{0x95, 0x45, 0x5d, 0xa8, 0x39, 0x69, 0x5c, 0x43}}, 0x87c5d4};  // {2654567F-2CF4-4E4E-9545-5DA839695C43} [nvigi.plugin.asr.ggml.cpu]
}
namespace plugin::asr::ggml::vulkan
{
constexpr PluginID kId = { {0xe604e2bb, 0xe08d, 0x4573,{0x9d, 0xdd, 0x81, 0xe1, 0xb0, 0x10, 0x66, 0x61}}, 0x70731d }; //{E604E2BB-E08D-4573-9DDD-81E1B0106661} [nvigi.plugin.asr.ggml.vulkan]
}
namespace plugin::asr::ggml::d3d12
{
constexpr PluginID kId = { {0x65dbd120, 0x3a70, 0x45c9,{0x80, 0x8b, 0x80, 0x99, 0xa2, 0x46, 0xba, 0x0d}}, 0xe4faf }; //{65DBD120-3A70-45C9-808B-8099A246BA0D} [nvigi.plugin.asr.ggml.d3d12]
}



//! Data slot keys
constexpr const char* kASRWhisperDataSlotAudio = "audio";
constexpr const char* kASRWhisperDataSlotTranscribedText = "text";

//! Available sampling strategies
enum class ASRWhisperSamplingStrategy : uint32_t
{
    eGreedy,      // similar to OpenAI's GreedyDecoder
    eBeamSearch, // similar to OpenAI's BeamSearchDecoder
};

// {08DB14D4-A87F-4BBB-B3FF-5C848259EDFD}
struct alignas(8) ASRWhisperCreationParameters {
    ASRWhisperCreationParameters() {}; 
    NVIGI_UID(UID({ 0x8db14d4, 0xa87f, 0x4bbb,{ 0xb3, 0xff, 0x5c, 0x84, 0x82, 0x59, 0xed, 0xfd } }), kStructVersion2);
    //! Defaults to "en" if not specified
    const char* language{};

    //! v2
    
    //! Use flash attenuation
    bool flashAtt{};
};

NVIGI_VALIDATE_STRUCT(ASRWhisperCreationParameters)

// {53068401-DD81-41B8-9896-FE9DD613F852}
struct alignas(8) ASRWhisperRuntimeParameters {
    ASRWhisperRuntimeParameters() {}; 
    NVIGI_UID(UID({ 0x53068401, 0xdd81, 0x41b8,{ 0x98, 0x96, 0xfe, 0x9d, 0xd6, 0x13, 0xf8, 0x52 } }), kStructVersion1);
    ASRWhisperSamplingStrategy sampling = ASRWhisperSamplingStrategy::eGreedy;
    int32_t bestOf = 1;
    int32_t beamSize = -1;
};

NVIGI_VALIDATE_STRUCT(ASRWhisperRuntimeParameters)

// {C8A416E6-F387-4E88-A333-A755054A2F3B}
struct alignas(8) ASRWhisperCapabilitiesAndRequirements {
    ASRWhisperCapabilitiesAndRequirements() {}; 
    NVIGI_UID(UID({ 0xc8a416e6, 0xf387, 0x4e88,{ 0xa3, 0x33, 0xa7, 0x55, 0x5, 0x4a, 0x2f, 0x3b } }), kStructVersion1);
    CommonCapabilitiesAndRequirements* common;
    //! "auto" indicates multi-language support with optional auto detection
    const char** supportedLanguages{};
};

NVIGI_VALIDATE_STRUCT(ASRWhisperCapabilitiesAndRequirements)

using IAutoSpeechRecognition = InferenceInterface;

}