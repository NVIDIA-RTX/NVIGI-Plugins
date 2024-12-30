// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "nvigi_ai.h"

namespace nvigi
{
    namespace plugin
    {
        namespace embed
        {
            namespace ggml::cpu {
                constexpr PluginID kId = { {0xac660fc4, 0x2459, 0x453b,{0x81, 0x0a, 0x3f, 0x9f, 0x37, 0xf9, 0x6e, 0x20}}, 0xc891b0 }; //{AC660FC4-2459-453B-810A-3F9F37F96E20} [nvigi.plugin.embed.ggml.cpu]}
            }
            namespace ggml::cuda {
                constexpr PluginID kId = { {0xf4c7454b, 0xa08e, 0x4f6c,{0xb2, 0x51, 0xfa, 0x56, 0x15, 0x0e, 0x1b, 0x20}}, 0xd3adc0 }; //{F4C7454B-A08E-4F6C-B251-FA56150E1B20} [nvigi.plugin.embed.ggml.cuda]}
            }
            namespace cloud::nvcf {
                constexpr PluginID kId = { {0x496bf5f7, 0x7813, 0x425b,{0xa0, 0x89, 0x38, 0x95, 0x99, 0xc6, 0x3d, 0xd8}}, 0x7d9686 }; //{496BF5F7-7813-425B-A089-389599C63DD8} [nvigi.plugin.embed.cloud.nvcf]    }
            }
        }
    }
constexpr const char* kEmbedDataSlotInText = "in_text";
constexpr const char* kEmbedDataSlotOutEmbedding = "out_embed";
constexpr const int32_t default_max_position_embeddings = 2048;

constexpr const char* prompts_sep = "PROMPT_SEP"; // string separator to use to different prompots inside a same string

// Custom Exceptions
constexpr uint32_t kResultNonUtf8 = 1 << 24 | plugin::embed::ggml::cpu::kId.crc24;
constexpr uint32_t kResultMaxTokensReached = 2 << 24 | plugin::embed::ggml::cpu::kId.crc24;

// {9b62a7cf-7c07-44b5-9a0a-2d97da5ba34c}
struct alignas(8) EmbedCreationParameters {
    EmbedCreationParameters() {}; 
    NVIGI_UID(UID({ 0x9b62a7cf, 0x7c07, 0x44b5, { 0x9a, 0x0a, 0x2d, 0x97, 0xda, 0x5b, 0xa3, 0x4c } }), kStructVersion1);
    int32_t main_gpu = -1;
};
NVIGI_VALIDATE_STRUCT(EmbedCreationParameters)

// {265930bc-5bf5-48e6-a5fa-183a4b1c83ed}
struct alignas(8) EmbedCapabilitiesAndRequirements {
    EmbedCapabilitiesAndRequirements() {}; 
    NVIGI_UID(UID({ 0x265930bc, 0x5bf5, 0x48e6, { 0xa5, 0xfa, 0x18, 0x3a, 0x4b, 0x1c, 0x83, 0xed } }), kStructVersion1);
    CommonCapabilitiesAndRequirements* common;
    size_t* embedding_numel{}; // dimensions of the embedding vector per modelGUID we support.
    int* max_position_embeddings{}; // maximum position embedding per modelGUID we support.
};
NVIGI_VALIDATE_STRUCT(EmbedCapabilitiesAndRequirements)

//! Embed interface
//! 
using IEmbed = InferenceInterface;

}