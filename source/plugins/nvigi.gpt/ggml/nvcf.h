// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "source/core/nvigi.api/nvigi.h"
#include "source/plugins/nvigi.gpt/nvigi_gpt.h"
#include "source/core/nvigi.extra/extra.h"

namespace nvigi
{

namespace gpt
{

namespace nvcf
{

struct Context
{
    nvigi::IGeneralPurposeTransformer* illama2{};
    std::map<std::string,nvigi::InferenceInstance*> instances{};
};
Context ctx;

}
}
}