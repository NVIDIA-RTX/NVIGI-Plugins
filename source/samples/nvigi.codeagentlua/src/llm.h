// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <string>

#if NVIGI_WINDOWS
#include <windows.h>
#endif

#include <nvigi.h>
#include "nvigi_gpt.h"

struct NVIGIAppCtx
{
    HMODULE coreLib{};

    nvigi::IGeneralPurposeTransformer* igpt{};
    nvigi::InferenceInstance* gptInst{};
};

// Initialize the LLM
int llmInit(const std::string& modelDir, const std::string& systemPromptPath);

// called to get a new update or one_shot function for the AI
std::string llmCreateAIFunc(const std::string& prompt);

// Shutsdown the LLM
int llmShutdown();