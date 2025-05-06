// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
namespace asqflow
{
namespace trt
{
constexpr PluginID kId = {{0x97b630c1, 0x1739, 0x4a6a, {0x9f, 0xce, 0x60, 0x73, 0xf6, 0xe8, 0x9a, 0x69}},
                          0xbbf95e}; //{97B630C1-1739-4A6A-9FCE-6073F6E89A69} [nvigi.plugin.tts.asqflow.trt]

}

} // namespace asqflow

} // namespace tts
} // namespace plugin

constexpr const char *kTTSDataSlotInputText = "text";
constexpr const char *kTTSDataSlotInputTargetSpectrogramPath = "target_spec";
constexpr const char *kTTSDataSlotOutputAudio = "audio";
constexpr const char *kTTSDataSlotOutputTextNormalized = "text_normalized";

// Custom Exceptions
constexpr uint32_t kResultErrorG2pModelASqFlow = 1 << 24 | plugin::tts::asqflow::trt::kId.crc24;
constexpr uint32_t kResultErrorDpModel = 2 << 24 | plugin::tts::asqflow::trt::kId.crc24;
constexpr uint32_t kResultErrorGeneratorModel = 3 << 24 | plugin::tts::asqflow::trt::kId.crc24;
constexpr uint32_t kResultErrorVocoderModel = 4 << 24 | plugin::tts::asqflow::trt::kId.crc24;

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
};

NVIGI_VALIDATE_STRUCT(TTSCapabilitiesAndRequirements);

//! Interface 'TTSASqFlowRuntimeParameters'
//!
//! {41ED9DCB-0AC3-41C0-ADE3-ABBEEB0568C7}
struct alignas(8) TTSASqFlowRuntimeParameters
{
    TTSASqFlowRuntimeParameters() {};
    NVIGI_UID(UID({0x41ed9dcb, 0x0ac3, 0x41c0, {0xad, 0xe3, 0xab, 0xbe, 0xeb, 0x05, 0x68, 0xc7}}), kStructVersion2)
    //! v1 members go here, please do NOT break the C ABI compatibility:

    //! * do not use virtual functions, volatile, STL (e.g. std::vector) or any other C++ high level functionality
    //! * do not use nested structures, always use pointer members
    //! * do not use internal types in _public_ interfaces (like for example 'nvigi::types::vector' etc.)
    //! * do not change or move any existing members once interface has shipped

    //! v2+ members go here, remember to update the kStructVersionN in the above NVIGI_UID macro!
    float speed = 1.0;
};

NVIGI_VALIDATE_STRUCT(TTSASqFlowRuntimeParameters)

//! Text To Speech (TTS) interface
using ITextToSpeech = InferenceInterface;

} // namespace nvigi
