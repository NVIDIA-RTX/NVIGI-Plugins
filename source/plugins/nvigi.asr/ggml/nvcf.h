// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "source/core/nvigi.api/nvigi.h"
#include "source/core/nvigi.log/log.h"
#include "source/plugins/nvigi.asr/nvigi_asr.h"
#include "source/core/nvigi.extra/extra.h"
#include "source/core/nvigi.file/file.h"

namespace nvigi
{

namespace asr
{

namespace nvcf
{

struct Context
{
    nvigi::IAutoSpeechRecognition* iwhisper{};
    std::map<std::string, nvigi::InferenceInstance*> instances{};
};
Context ctx;

}
}
}