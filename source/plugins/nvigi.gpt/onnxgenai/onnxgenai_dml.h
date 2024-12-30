// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "source/core/nvigi.api/nvigi.h"
#include "source/core/nvigi.api/internal.h"
#include "source/core/nvigi.log/log.h"
#include "source/core/nvigi.exception/exception.h"
#include "source/core/nvigi.plugin/plugin.h"
#include "source/core/nvigi.file/file.h"
#include "source/core/nvigi.extra/extra.h"
#include "source/core/nvigi.thread/thread.h"
#include "versions.h"
#include "source/utils/nvigi.ai/ai.h"
#include "_artifacts/gitVersion.h"
#include "external/json/source/nlohmann/json.hpp"
#include <functional>
#include "ort_genai_c.h"
#include "nvigi_gpt_onnxgenai.h"
#include "iostream"

#include "chat_history.h"

using json = nlohmann::json;

namespace nvigi
{
namespace onnxgenai
{
struct InferenceContext
{
    GPTCreationParameters GPT_params{};

    OgaTokenizer* tokenizer;
    OgaModel* model;
    OgaGeneratorParams* params;
    
    std::string prompt;

    uint32_t seed;
    int32_t n_predict;
    int32_t n_keep;
    int32_t n_draft;
    int32_t n_batch;
    int32_t n_chunks;
    int32_t n_parallel;
    int32_t n_sequences;

    // chat history
    ChatHistory history;
    // prompt template
    json modelInfo;
};

struct onnxgenaiContext
{
    NVIGI_PLUGIN_CONTEXT_CREATE_DESTROY(onnxgenaiContext);

public:

    void onCreateContext() { }
    void onDestroyContext() { }

    IGeneralPurposeTransformer api{};

    nvigi::PluginID feature{};

    // Caps and requirements
    ai::CommonCapsData capsData;
};
}
}