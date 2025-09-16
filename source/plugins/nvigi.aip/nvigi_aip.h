// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "nvigi_ai.h"

namespace nvigi
{

namespace plugin
{
namespace ai::pipeline
{
constexpr PluginID kId = { {0xe3787947, 0x3d3e, 0x4c5c,{0xa4, 0xff, 0xbf, 0xae, 0x98, 0x87, 0x48, 0x3a}}, 0x30eb3a }; //{E3787947-3D3E-4C5C-A4FF-BFAE9887483A} [nvigi.plugin.ai.pipeline]
}
}

namespace aip
{


//! Interface Creation Parameters
//!
//! {B35BE632-83EA-44FF-B6B5-6073EBBA1148}
struct alignas(8) AiPipelineCreationParameters {
    AiPipelineCreationParameters() {}; 
    NVIGI_UID(UID({ 0xb35be632, 0x83ea, 0x44ff,{ 0xb6, 0xb5, 0x60, 0x73, 0xeb, 0xba, 0x11, 0x48 } }), kStructVersion1);

    size_t numStages{};
    //! Plugins (backends, api) to use
    const PluginID* stages{};
    //! Creation parameters per plugin, additional parameters per plugin can be chained as needed
    const NVIGIParameter** stageParams{};
    
    //! NEW MEMBERS GO HERE, REMEMBER TO BUMP THE VERSION!
};

NVIGI_VALIDATE_STRUCT(AiPipelineCreationParameters)

//! Interface AI Pipeline
//!
using IAiPipeline = InferenceInterface;

}
}