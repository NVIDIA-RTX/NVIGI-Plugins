// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "nvigi_gpt.h"

namespace nvigi
{
namespace plugin
{
namespace gpt
{
namespace onnxgenai
{
namespace dml 
{
    constexpr PluginID kId = { {0xf966dbf7, 0xd9d8, 0x448d,{0x8c, 0x52, 0xa4, 0x68, 0x8f, 0xbe, 0x79, 0xf9}}, 0xa815f8 }; //{F966DBF7-D9D8-448D-8C52-A4688FBE79F9} [nvigi.plugin.gpt.onnxgenai.dml]
}
}
}
}


//! Interface 'GPTOnnxgenaiCreationParameters'
//!
//! {3AD189AE-C91E-4F84-B64A-34729EBB110C}
struct alignas(8) GPTOnnxgenaiCreationParameters
{
GPTOnnxgenaiCreationParameters() { };
NVIGI_UID(UID({ 0x3ad189ae, 0xc91e, 0x4f84,{0xb6, 0x4a, 0x34, 0x72, 0x9e, 0xbb, 0x11, 0x0c} }), kStructVersion1);

//! V1 MEMBERS GO HERE
bool backgroundMode = false; // applicable only with gpu mode
bool allowAsync = false; // ignored; always false
//! NEW MEMBERS GO HERE, REMEMBER TO BUMP THE VERSION!
};

NVIGI_VALIDATE_STRUCT(GPTOnnxgenaiCreationParameters)
}