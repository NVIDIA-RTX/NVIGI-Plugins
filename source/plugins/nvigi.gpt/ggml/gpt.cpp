// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// This implementation uses the llama_server.h API for inference instead of the
// llama-cli style token generation approach. The server runs in-process and
// provides an OpenAI-compatible chat completions API.
//
// NOTES:
// - VLM (Vision Language Model) is supported when the model config includes "mmproj_weights".
//   Images are encoded as BMP and sent as base64 data URLs in OpenAI-compatible image_url content blocks.
// - The server handles conversation context internally, simplifying chat mode management.
// - LoRA adapters should be configured via server initialization parameters if supported.

#include <future>
#include <atomic>
#include <mutex>

#include "source/core/nvigi.api/nvigi.h"
#include "source/core/nvigi.api/nvigi_io.h"
#include "source/core/nvigi.api/internal.h"
#include "source/core/nvigi.log/log.h"
#include "source/core/nvigi.exception/exception.h"
#include "source/core/nvigi.plugin/plugin.h"
#include "source/core/nvigi.file/file.h"
#include "source/core/nvigi.extra/extra.h"
#include "source/plugins/nvigi.gpt/ggml/versions.h"
#include "source/utils/nvigi.ai/ai.h"
#include "source/utils/nvigi.ai/nvigi_stl_helpers.h"
#include "source/utils/nvigi.poll/poll.h"
#include "source/plugins/nvigi.gpt/nvigi_gpt.h"
#include "_artifacts/gitVersion.h"
#include "nvtx3/nvtx3.hpp"

#include "external/json/source/nlohmann/json.hpp"
using json = nlohmann::json;

#include <log.h>

#if GGML_USE_CUBLAS
#include "source/core/nvigi.api/nvigi_cuda.h"
#include "source/utils/nvigi.hwi/cuda/push_poppable_cuda_context.h"
#include "source/utils/nvigi.hwi/cuda/runtime_context_scope.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvtx3/nvToolsExt.h"
#include "ggml-cuda.h"
#elif defined(GGML_USE_D3D12)
#include "external/agility-sdk/build/native/include/d3d12.h"
#include "source/core/nvigi.api/nvigi_d3d12.h"
#include "source/plugins/nvigi.hwi/d3d12/nvigi_hwi_d3d12.h"
#include "source/utils/nvigi.d3d12/d3d12_helpers.h"
namespace nvigi {
// this should match the PFun_commandListAction defined in LlamaCPP - redefining as it's at a depth/scope in LlamaCPP not exposed to IGI
using PFun_commandListAction = void(ID3D12CommandList* pCommandList, int action);
}
#elif defined GGML_USE_VULKAN
#include "external/vulkanSDK/include/vulkan/vulkan.h"
#include "source/core/nvigi.api/nvigi_vulkan.h"
#endif

// Server API - available for all backends
#include <llama.h>
#include <llama_server.h>

namespace nvigi
{
namespace gpt
{

static void llamaLogCallback(ggml_log_level level, const char* text, void* user_data)
{
    //! Special case, llama prints progress bars with a stream of consecutive "." so we want to ignore that
    if (strcmp(text, ".") == 0) return;

    std::string msg(text);
    if (level == GGML_LOG_LEVEL_WARN)
    {
        NVIGI_LOG("llama", LogType::eWarn, nvigi::log::YELLOW, "%s", msg.c_str());
    }
    else if (level == GGML_LOG_LEVEL_ERROR)
    {
        NVIGI_LOG("llama", LogType::eError, nvigi::log::RED, "%s", msg.c_str());
    }
    else
    {
        NVIGI_LOG("llama", LogType::eInfo, nvigi::log::WHITE, "%s", msg.c_str());
    }
}

//=============================================================================
// Error Mapping
//=============================================================================

nvigi::Result mapServerError(const json& response)
{
    if (response.contains("error"))
    {
        auto& error = response["error"];
        std::string type = error.value("type", "");
        std::string message = error.value("message", "Unknown error");
        
        NVIGI_LOG_ERROR("Server error: %s - %s", type.c_str(), message.c_str());
        
        if (type == "invalid_request_error") return kResultInvalidParameter;
        if (type == "model_not_loaded") return kResultInvalidState;
        if (type == "context_length_exceeded") return kResultInvalidParameter;
        if (type == "exceed_context_size_error") return kResultInvalidParameter;
        if (type == "server_error") return kResultInvalidState;
        
        return kResultInvalidState;
    }
    return kResultOk;
}

//=============================================================================
// VLM Image Encoding Helpers
//=============================================================================

// Base64 encoding for image data URLs
static const char kBase64Chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64Encode(const uint8_t* data, size_t length)
{
    std::string result;
    result.reserve(((length + 2) / 3) * 4);
    
    for (size_t i = 0; i < length; i += 3)
    {
        uint32_t octet_a = i < length ? data[i] : 0;
        uint32_t octet_b = (i + 1) < length ? data[i + 1] : 0;
        uint32_t octet_c = (i + 2) < length ? data[i + 2] : 0;
        uint32_t triple = (octet_a << 16) + (octet_b << 8) + octet_c;
        
        result += kBase64Chars[(triple >> 18) & 0x3F];
        result += kBase64Chars[(triple >> 12) & 0x3F];
        result += (i + 1) < length ? kBase64Chars[(triple >> 6) & 0x3F] : '=';
        result += (i + 2) < length ? kBase64Chars[triple & 0x3F] : '=';
    }
    return result;
}

// Encode raw RGB(A) image data to BMP format in memory.
// Returns the BMP file as a byte vector.
// stb_image (used by the server's mtmd) can decode BMP natively.
static std::vector<uint8_t> encodeImageToBMP(const uint8_t* pixels, int width, int height, int channels)
{
    // BMP rows must be padded to 4-byte boundary
    int bytesPerPixel = 3; // BMP always stores BGR, 24-bit
    int rowSize = width * bytesPerPixel;
    int padding = (4 - (rowSize % 4)) % 4;
    int paddedRowSize = rowSize + padding;
    
    int pixelDataSize = paddedRowSize * height;
    int fileSize = 14 + 40 + pixelDataSize; // file header + info header + pixel data
    
    std::vector<uint8_t> bmp(fileSize, 0);
    
    // BMP File Header (14 bytes)
    bmp[0] = 'B'; bmp[1] = 'M';
    bmp[2] = fileSize & 0xFF;
    bmp[3] = (fileSize >> 8) & 0xFF;
    bmp[4] = (fileSize >> 16) & 0xFF;
    bmp[5] = (fileSize >> 24) & 0xFF;
    // bytes 6-9: reserved (0)
    bmp[10] = 54; // pixel data offset = 14 + 40
    
    // BMP Info Header (BITMAPINFOHEADER, 40 bytes)
    bmp[14] = 40; // header size
    bmp[18] = width & 0xFF;
    bmp[19] = (width >> 8) & 0xFF;
    bmp[20] = (width >> 16) & 0xFF;
    bmp[21] = (width >> 24) & 0xFF;
    bmp[22] = height & 0xFF;
    bmp[23] = (height >> 8) & 0xFF;
    bmp[24] = (height >> 16) & 0xFF;
    bmp[25] = (height >> 24) & 0xFF;
    bmp[26] = 1; // color planes
    bmp[28] = 24; // bits per pixel
    // bytes 30-33: compression (0 = BI_RGB)
    bmp[34] = pixelDataSize & 0xFF;
    bmp[35] = (pixelDataSize >> 8) & 0xFF;
    bmp[36] = (pixelDataSize >> 16) & 0xFF;
    bmp[37] = (pixelDataSize >> 24) & 0xFF;
    
    // Pixel data - BMP stores bottom-to-top, BGR order
    for (int y = 0; y < height; y++)
    {
        int srcRow = height - 1 - y; // BMP is bottom-up
        int dstOffset = 54 + y * paddedRowSize;
        for (int x = 0; x < width; x++)
        {
            int srcIdx = (srcRow * width + x) * channels;
            int dstIdx = dstOffset + x * 3;
            bmp[dstIdx + 0] = pixels[srcIdx + 2]; // B
            bmp[dstIdx + 1] = pixels[srcIdx + 1]; // G
            bmp[dstIdx + 2] = pixels[srcIdx + 0]; // R
        }
        // Padding bytes are already 0
    }
    
    return bmp;
}

// Build a base64 data URL from raw RGB(A) image data.
// Returns "data:image/bmp;base64,<encoded>" suitable for OpenAI image_url format.
static std::string buildImageDataUrl(const uint8_t* pixels, int width, int height, int channels)
{
    auto bmp = encodeImageToBMP(pixels, width, height, channels);
    return "data:image/bmp;base64," + base64Encode(bmp.data(), bmp.size());
}

//=============================================================================
// Inference Context - Server-based approach
//=============================================================================

struct InferenceContext
{
    poll::PollContext<nvigi::InferenceExecutionState> pollCtx;
    
    // Server handle for inference
    llama_server_handle serverHandle{};
    
    // IO callbacks for custom model loading (must persist for lifetime of the instance)
    llama_model_io_callbacks ioCbs{};

    // Model configuration from JSON
    json modelInfo;
    
    // Instance state
    std::atomic<nvigi::Result> state{kResultOk};
    std::future<nvigi::Result> asyncJob;
    std::atomic<bool> cancelled{false};
    std::mutex mtx;
    
    // Cached creation parameters
    int contextSize{512};
    int maxTokens{200};
    int batchSize{2048};
    bool flashAttention{false};
    float maxTimeoutSeconds{60.0f};
    
    // Convert the user-configurable timeout to a polling iteration count (100ms per iteration)
    int getTimeoutIterations() const { return std::max(10, static_cast<int>(maxTimeoutSeconds * 10.0f)); }
    
    // Session ID for stateful conversation (using new session API)
    int sessionId{-1};
    bool inChatMode{false};
    
    // Track if we've added any prompt to the session (for BOS token handling)
    bool sessionHasContent{false};
    
    // Track the number of tokens for the system prompt (for context truncation)
    int systemPromptTokens{0};
    
    // VLM (Vision Language Model) support
    bool hasMultimodal{false};
    
#if GGML_USE_CUBLAS
    // Used to set the relative priority of GPU inference and graphics
    std::vector<cudaStream_t> cuda_streams;
#else
    // Use dummy PushPoppableCudaContext to avoid depending on CUDA
    struct PushPoppableCudaContext
    {
        bool constructorSucceeded = true;
        PushPoppableCudaContext(const nvigi::NVIGIParameter* params) {}
        void pushRuntimeContext() {}
        void popRuntimeContext() {}
    };
#endif
    PushPoppableCudaContext cudaContext;
    
    // Constructor that initializes cudaContext with params (required for CUDA builds)
    InferenceContext(const nvigi::NVIGIParameter* params) : cudaContext(params) {}
};

//=============================================================================
// CUDA Scope Callbacks (for server thread context management)
//=============================================================================

#if GGML_USE_CUBLAS
// Called by the server before any CUDA/llama operations on a thread
// user_data is the InferenceContext pointer passed during server initialization
static void cudaScopeEnter(void* user_data)
{
    auto* instance = static_cast<InferenceContext*>(user_data);
    if (instance && instance->cudaContext.constructorSucceeded)
    {
        instance->cudaContext.pushRuntimeContext();
    }
}

// Called by the server after CUDA/llama operations complete on a thread
// user_data is the InferenceContext pointer passed during server initialization
static void cudaScopeExit(void* user_data)
{
    auto* instance = static_cast<InferenceContext*>(user_data);
    if (instance && instance->cudaContext.constructorSucceeded)
    {
        instance->cudaContext.popRuntimeContext();
    }
}
#endif

//=============================================================================
// Plugin Context and Registration
//=============================================================================

PluginID getFeatureId(InferenceInstanceData* data)
{
#ifdef GGML_USE_CUBLAS
    return plugin::gpt::ggml::cuda::kId;
#elif defined GGML_USE_VULKAN
    return plugin::gpt::ggml::vulkan::kId;
#elif defined(GGML_USE_D3D12)
    return plugin::gpt::ggml::d3d12::kId;
#else
    return plugin::gpt::ggml::cpu::kId;
#endif
}

const nvigi::InferenceDataDescriptorArray* getInputSignature(InferenceInstanceData* data)
{
    static std::vector<InferenceDataDescriptor> slots = 
    { 
        {nvigi::kGPTDataSlotSystem, InferenceDataText::s_type, true },      // optional
        {nvigi::kGPTDataSlotUser, InferenceDataText::s_type, false },
        {nvigi::kGPTDataSlotAssistant, InferenceDataText::s_type, true},    // optional
        {nvigi::kGPTDataSlotJSON, InferenceDataText::s_type, true},         // optional
    };
    static InferenceDataDescriptorArray s_desc = { slots.size(), slots.data() };
    return &s_desc;
}

const nvigi::InferenceDataDescriptorArray* getOutputSignature(InferenceInstanceData* data)
{
    static std::vector<InferenceDataDescriptor> slots = { 
        {nvigi::kGPTDataSlotResponse, InferenceDataText::s_type, false },
        {nvigi::kGPTDataSlotJSON, InferenceDataText::s_type, true}, // optional
    };
    static InferenceDataDescriptorArray s_desc = { slots.size(), slots.data() };
    return &s_desc;
}

struct GPTContext
{
    NVIGI_PLUGIN_CONTEXT_CREATE_DESTROY(GPTContext);

    void onCreateContext() {};
    void onDestroyContext() {};

    nvigi::PluginID feature{};

    IGeneralPurposeTransformer api{};
    IPolledInferenceInterface polledApi{};
    IGPTSessionState sessionStateApi{};

    // Caps and requirements
    ai::CommonCapsData capsData;

#ifdef GGML_USE_CUBLAS
    nvigi::IHWICuda* icig{};
#elif defined(GGML_USE_D3D12)
    nvigi::IHWID3D12* iscg{};
#endif
    nvigi::system::ISystem* isystem{};
};

nvigi::Result evaluate(nvigi::InferenceExecutionContext* execCtx);
nvigi::Result evaluateAsync(nvigi::InferenceExecutionContext* execCtx);
nvigi::Result cancelAsyncEvaluation(nvigi::InferenceExecutionContext* execCtx);
}

// Define our plugin, make sure to update version numbers in versions.h
NVIGI_PLUGIN_DEFINE("nvigi.plugin.gpt", Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH), Version(API_MAJOR, API_MINOR, API_PATCH), gpt, GPTContext)

//=============================================================================
// Session Management Helpers
//=============================================================================

// Check if context needs truncation and handle it
// userTokensToKeep: if > 0, overrides automatic calculation of how many tokens to preserve during truncation
// Returns true if context is OK to continue, false if truncation failed
bool ensureContextSpace(nvigi::gpt::InferenceContext* instance, int reservedTokens, int userTokensToKeep = 0)
{
    int contextLen = llama_server_session_get_context_length(instance->serverHandle, instance->sessionId);
    int contextSize = llama_server_session_get_context_size(instance->serverHandle, instance->sessionId);
    
    if (contextLen < 0 || contextSize < 0)
    {
        NVIGI_LOG_ERROR("Failed to get context info from session");
        return false;
    }
    
    // Check if we have enough space
    int availableTokens = contextSize - contextLen;
    if (availableTokens >= reservedTokens)
    {
        NVIGI_LOG_VERBOSE("Context: %d/%d tokens, %d available (need %d)", 
                          contextLen, contextSize, availableTokens, reservedTokens);
        return true;
    }
    
    // Determine how many tokens to keep during truncation
    // Priority: 1) user-specified tokensToKeep, 2) tracked system prompt tokens, 3) 10% estimate
    int tokensToKeep = userTokensToKeep;
    if (tokensToKeep <= 0)
    {
        tokensToKeep = instance->systemPromptTokens;
        if (tokensToKeep <= 0)
        {
            tokensToKeep = std::max(64, contextSize / 10);
        }
        else
        {
            tokensToKeep += 32;
        }
    }
    
    NVIGI_LOG_WARN("Context nearly full (%d/%d), truncating to keep %d tokens (system prompt: %d)", 
                   contextLen, contextSize, tokensToKeep, instance->systemPromptTokens);
    
    if (!llama_server_session_truncate(instance->serverHandle, instance->sessionId, tokensToKeep))
    {
        NVIGI_LOG_ERROR("Failed to truncate session context");
        return false;
    }
    
    // Verify truncation worked
    int newContextLen = llama_server_session_get_context_length(instance->serverHandle, instance->sessionId);
    NVIGI_LOG_VERBOSE("Context after truncation: %d/%d tokens", newContextLen, contextSize);
    
    return true;
}

// Streaming callback context
struct StreamContext
{
    nvigi::InferenceExecutionContext* execCtx;
    nvigi::gpt::InferenceContext* instance;
    std::string accumulatedContent;
    bool userCancelled;      // True only if user explicitly cancelled
    bool completedNormally;  // True when stream ended due to finish_reason or [DONE]
    
    // Token rate limiting (optional, driven by GPTRuntimeParameters)
    float frameTimeMs{0.0f};
    int32_t targetTokensPerSecond{-1};
    int32_t totalTokensAtFrameStart{0};
    std::chrono::steady_clock::time_point frameStartTime;
    bool rateLimitActive{false};
};

//=============================================================================
// Session-based Request Functions
//=============================================================================

// Extract text content from server JSON response
// The server may return JSON like {"content": "text"} or just plain text
std::string extractContentFromResponse(const std::string& response)
{
    // Try to parse as JSON
    try
    {
        json responseJson = json::parse(response);
        
        // Check for content field (session API format)
        if (responseJson.contains("content") && responseJson["content"].is_string())
        {
            return responseJson["content"].get<std::string>();
        }
        
        // Check for OpenAI-style choices array
        if (responseJson.contains("choices") && responseJson["choices"].is_array() && !responseJson["choices"].empty())
        {
            auto& choice = responseJson["choices"][0];
            if (choice.contains("message") && choice["message"].contains("content") &&
                choice["message"]["content"].is_string())
            {
                return choice["message"]["content"].get<std::string>();
            }
            if (choice.contains("text") && choice["text"].is_string())
            {
                return choice["text"].get<std::string>();
            }
        }
        
        // If it's a string at the root, return it
        if (responseJson.is_string())
        {
            return responseJson.get<std::string>();
        }
    }
    catch (const std::exception&)
    {
        // Not JSON, return as-is (plain text)
    }
    
    return response;
}

// Deliver generated response to the caller
// content: the text content to deliver to kGPTDataSlotResponse
// jsonResponse: optional full JSON to deliver to kGPTDataSlotJSON (can be empty)
nvigi::Result deliverResponse(
    const std::string& content,
    const std::string& jsonResponse,
    nvigi::InferenceExecutionContext* execCtx,
    nvigi::gpt::InferenceContext* instance,
    nvigi::InferenceExecutionState state)
{
    nvigi::InferenceExecutionState res{};
    
    // Check if host provided output slots
    const nvigi::InferenceDataText* responseOutput{};
    const nvigi::InferenceDataText* jsonOutput{};
    
    bool hasResponseSlot = execCtx->outputs && execCtx->outputs->findAndValidateSlot(kGPTDataSlotResponse, &responseOutput);
    bool hasJsonSlot = execCtx->outputs && execCtx->outputs->findAndValidateSlot(kGPTDataSlotJSON, &jsonOutput);
    
    if (hasResponseSlot || hasJsonSlot)
    {
        // Fill kGPTDataSlotResponse with text content
        if (hasResponseSlot && responseOutput)
        {
            auto cpuBuffer = castTo<CpuData>(responseOutput->utf8Text);
            if (cpuBuffer && cpuBuffer->buffer && cpuBuffer->sizeInBytes > content.size())
            {
                strcpy_s((char*)cpuBuffer->buffer, cpuBuffer->sizeInBytes, content.c_str());
            }
        }
        
        // Fill kGPTDataSlotJSON with full JSON response if requested
        if (hasJsonSlot && jsonOutput && !jsonResponse.empty())
        {
            auto cpuBuffer = castTo<CpuData>(jsonOutput->utf8Text);
            if (cpuBuffer && cpuBuffer->buffer && cpuBuffer->sizeInBytes > jsonResponse.size())
            {
                strcpy_s((char*)cpuBuffer->buffer, cpuBuffer->sizeInBytes, jsonResponse.c_str());
            }
        }
        
        if (execCtx->callback)
        {
            res = execCtx->callback(execCtx, state, execCtx->callbackUserData);
        }
        else
        {
            res = instance->pollCtx.triggerCallback(state);
        }
    }
    else
    {
        // Temporary outputs for the callback since host did not provide any
        nvigi::InferenceDataTextSTLHelper contentSlot(content);
        nvigi::InferenceDataTextSTLHelper jsonSlot(jsonResponse);
        std::vector<nvigi::InferenceDataSlot> slots = { {kGPTDataSlotResponse, contentSlot}, {kGPTDataSlotJSON, jsonSlot} };
        nvigi::InferenceDataSlotArray outputs = { slots.size(), slots.data() };
        execCtx->outputs = &outputs;
        
        if (execCtx->callback)
        {
            res = execCtx->callback(execCtx, state, execCtx->callbackUserData);
        }
        else
        {
            res = instance->pollCtx.triggerCallback(state);
        }
        
        // Clear outputs since these are all local variables
        execCtx->outputs = {};
    }
    
    if (res == nvigi::kInferenceExecutionStateCancel || res == nvigi::kInferenceExecutionStateInvalid)
    {
        return kResultOk; // User cancelled, not an error
    }
    
    return kResultOk;
}

// Streaming callback for session_generate
bool sessionStreamCallback(const char* token, void* userdata)
{
    auto* ctx = static_cast<StreamContext*>(userdata);
    
    // Check if user explicitly requested cancellation
    if (ctx->instance->cancelled.load())
    {
        ctx->userCancelled = true;
        return false;
    }
    
    if (!token) return true;
    
    std::string tokenStr(token);
    
    // Check for end of generation marker
    if (tokenStr == "[DONE]" || tokenStr.empty())
    {
        // Signal completion - content was already delivered incrementally via DataPending callbacks
        deliverResponse("", "", ctx->execCtx, ctx->instance, 
                       nvigi::kInferenceExecutionStateDone);
        ctx->completedNormally = true;
        return false; // Stop streaming
    }
    
    // The server sends JSON chunks like:
    //   {"content":"Hello","id_slot":0,"stop":false,"tokens_predicted":5}
    // Use lightweight string matching instead of full JSON parse for per-token performance.
    std::string contentDelta;
    bool isStop = false;
    int tokensPredicted = 0;
    
    std::string_view sv(tokenStr);
    
    // Check "stop":true
    isStop = (sv.find("\"stop\":true") != std::string_view::npos);
    
    // Extract "content":"..." value (handles JSON-escaped strings)
    auto contentKey = sv.find("\"content\":\"");
    if (contentKey != std::string_view::npos)
    {
        size_t valStart = contentKey + 11; // length of "content":"
        size_t valEnd = valStart;
        while (valEnd < sv.size() && !(sv[valEnd] == '"' && sv[valEnd - 1] != '\\'))
        {
            valEnd++;
        }
        if (valEnd < sv.size())
        {
            contentDelta = std::string(sv.substr(valStart, valEnd - valStart));
            // Unescape basic JSON escape sequences
            for (size_t p = 0; p < contentDelta.size(); p++)
            {
                if (contentDelta[p] == '\\' && p + 1 < contentDelta.size())
                {
                    char c = contentDelta[p + 1];
                    if (c == 'n') { contentDelta.replace(p, 2, "\n"); }
                    else if (c == 't') { contentDelta.replace(p, 2, "\t"); }
                    else if (c == '"') { contentDelta.replace(p, 2, "\""); }
                    else if (c == '\\') { contentDelta.replace(p, 2, "\\"); }
                    else if (c == '/') { contentDelta.replace(p, 2, "/"); }
                }
            }
        }
    }
    
    // Extract "tokens_predicted":N
    auto tpKey = sv.find("\"tokens_predicted\":");
    if (tpKey != std::string_view::npos)
    {
        size_t numStart = tpKey + 19; // length of "tokens_predicted":
        tokensPredicted = 0;
        for (size_t p = numStart; p < sv.size() && sv[p] >= '0' && sv[p] <= '9'; p++)
        {
            tokensPredicted = tokensPredicted * 10 + (sv[p] - '0');
        }
    }
    
    if (!contentDelta.empty())
    {
        ctx->accumulatedContent += contentDelta;
    }
    
    if (isStop)
    {
        // Signal completion - content was already delivered incrementally via DataPending callbacks
        deliverResponse("", "", ctx->execCtx, ctx->instance, 
                       nvigi::kInferenceExecutionStateDone);
        ctx->completedNormally = true;
        return false;
    }
    
    if (!contentDelta.empty())
    {
        deliverResponse(contentDelta, "", ctx->execCtx, ctx->instance, 
                       nvigi::kInferenceExecutionStateDataPending);
    }
    
    // Token rate limiting: use the server-reported tokens_predicted count to pace
    // token delivery according to the host's frame rate and desired throughput.
    if (ctx->rateLimitActive && tokensPredicted > 0)
    {
        int tokensSinceFrameStart = tokensPredicted - ctx->totalTokensAtFrameStart;
        float tokensPerFrame = ctx->targetTokensPerSecond * (ctx->frameTimeMs / 1000.0f);
        int budget = std::max(1, static_cast<int>(tokensPerFrame));
        
        if (tokensSinceFrameStart >= budget)
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<float, std::milli>(now - ctx->frameStartTime);
            float remainingMs = ctx->frameTimeMs - elapsed.count();
            if (remainingMs > 0.0f)
            {
                std::this_thread::sleep_for(std::chrono::microseconds(
                    static_cast<int64_t>(remainingMs * 1000.0f)));
            }
            ctx->totalTokensAtFrameStart = tokensPredicted;
            ctx->frameStartTime = std::chrono::steady_clock::now();
        }
    }
    
    return true;
}

// Build a complete JSON object with ALL runtime and sampler parameters.
// This matches the old CLI implementation which applied every parameter on every call.
// Used by both sendSessionRequest (generation) and session_add_messages (prompt processing).
json buildFullParamsJson(
    const GPTRuntimeParameters* runtime,
    const GPTSamplerParameters* sampler,
    int defaultMaxTokens,
    bool defaultChatMode)
{
    json p;
    
    // Runtime parameters
    if (runtime)
    {
        p["n_predict"] = runtime->tokensToPredict > 0 ? runtime->tokensToPredict : defaultMaxTokens;
        p["temperature"] = runtime->temperature;
        p["top_p"] = runtime->topP;
        p["n_batch"] = runtime->batchSize;
        p["n_keep"] = runtime->tokensToKeep;
        
        if (runtime->seed != 0xFFFFFFFF)
            p["seed"] = runtime->seed;
        
        if (runtime->reversePrompt && strlen(runtime->reversePrompt) > 0)
            p["stop"] = json::array({runtime->reversePrompt});
    }
    else
    {
        p["n_predict"] = defaultMaxTokens;
        p["temperature"] = defaultChatMode ? 0.7f : 0.2f;
        p["top_p"] = defaultChatMode ? 0.9f : 0.95f;
    }
    
    // Sampler parameters - ALL of them, every call, matching old implementation
    if (sampler)
    {
        p["top_k"] = sampler->topK;
        p["min_p"] = sampler->minP;
        p["typical_p"] = sampler->typP;
        p["repeat_penalty"] = sampler->penaltyRepeat;
        p["repeat_last_n"] = sampler->penaltyLastN;
        p["presence_penalty"] = sampler->penaltyPresent;
        p["frequency_penalty"] = sampler->penaltyFreq;
        p["ignore_eos"] = sampler->ignoreEOS;
        p["n_probs"] = sampler->numProbs;
        p["min_keep"] = sampler->minKeep;
        p["n_prev"] = sampler->numPrev;
        p["xtc_probability"] = sampler->xtcProbability;
        p["xtc_threshold"] = sampler->xtcThreshold;
        p["dynatemp_range"] = sampler->dynatempRange;
        p["dynatemp_exponent"] = sampler->dynatempExponent;
        p["mirostat"] = sampler->mirostat;
        p["mirostat_tau"] = sampler->mirostatTAU;
        p["mirostat_eta"] = sampler->mirostatETA;
        
        if (sampler->getVersion() >= 2 && sampler->grammar && strlen(sampler->grammar) > 0)
        {
            p["grammar"] = sampler->grammar;
        }
    }
    
    return p;
}

nvigi::Result sendSessionRequest(
    nvigi::gpt::InferenceContext* instance,
    nvigi::InferenceExecutionContext* execCtx,
    bool async,
    const std::string& customJsonStr = "")
{
    nvtx3::scoped_range r{"ggmlEvaluate sendSessionRequest"};
    auto runtime = findStruct<GPTRuntimeParameters>(execCtx->runtimeParameters);
    auto sampler = findStruct<GPTSamplerParameters>(execCtx->runtimeParameters);
    
    json genParams = buildFullParamsJson(runtime, sampler, instance->maxTokens, instance->inChatMode);
    
    // Merge custom JSON parameters if provided via kGPTDataSlotJSON input slot.
    // User-provided values override auto-generated ones, allowing direct control
    // over any server parameter not exposed through GPTRuntimeParameters/GPTSamplerParameters.
    if (!customJsonStr.empty())
    {
        try
        {
            json customParams = json::parse(customJsonStr);
            if (customParams.is_object())
            {
                genParams.merge_patch(customParams);
                NVIGI_LOG_VERBOSE("Merged %zu custom JSON parameters into generation request", customParams.size());
            }
            else
            {
                NVIGI_LOG_WARN("kGPTDataSlotJSON must be a JSON object, ignoring");
            }
        }
        catch (const std::exception& e)
        {
            NVIGI_LOG_WARN("Failed to parse kGPTDataSlotJSON: %s", e.what());
        }
    }
    
    std::string paramsStr = genParams.dump();
    
    // Token rate limiting: when both frameTimeMs and targetTokensPerSecond are set,
    // we limit how many tokens are produced per frame period. This is useful for real-time
    // applications (e.g. games) that need to spread token generation across frames.
    float frameTimeMs = 0.0f;
    int32_t targetTPS = -1;
    if (runtime && runtime->getVersion() >= 2)
    {
        frameTimeMs = runtime->frameTimeMs;
        targetTPS = runtime->targetTokensPerSecond;
    }
    bool rateLimitActive = (frameTimeMs > 0.0f && targetTPS > 0);
    
    if (rateLimitActive && !async)
    {
        // Blocking mode: cap n_predict to the per-frame token budget so the host
        // can call evaluate() once per frame and get at most this many tokens back.
        int tokensPerFrame = std::max(1, static_cast<int>(targetTPS * (frameTimeMs / 1000.0f)));
        int currentPredict = genParams.contains("n_predict") ? genParams["n_predict"].get<int>() : instance->maxTokens;
        genParams["n_predict"] = std::min(currentPredict, tokensPerFrame);
        paramsStr = genParams.dump();
        NVIGI_LOG_VERBOSE("Rate limiting (blocking): %d tokens/frame (%.1f ms frame, %d TPS)", 
            tokensPerFrame, frameTimeMs, targetTPS);
    }
    
    if (async)
    {
        // Streaming generation
        StreamContext ctx = {execCtx, instance, "", false, false,
                            frameTimeMs, targetTPS, 0,
                            std::chrono::steady_clock::now(), rateLimitActive};
        
        char* response = llama_server_session_generate(
            instance->serverHandle,
            instance->sessionId,
            paramsStr.c_str(),
            sessionStreamCallback,
            &ctx
        );
        
        if (response)
        {
            // Non-streaming response (callback wasn't used or returned final result)
            std::string jsonResponse(response);
            llama_server_free_response(response);
            
            if (!ctx.completedNormally && !ctx.userCancelled)
            {
                // Extract text content from JSON response
                std::string content = extractContentFromResponse(jsonResponse);
                deliverResponse(content, jsonResponse, execCtx, instance, nvigi::kInferenceExecutionStateDone);
            }
        }
        else if (!ctx.completedNormally && !ctx.userCancelled)
        {
            NVIGI_LOG_ERROR("Session generate failed");
            return kResultInvalidState;
        }
    }
    else
    {
        // Blocking generation (no streaming callback)
        char* response = llama_server_session_generate(
            instance->serverHandle,
            instance->sessionId,
            paramsStr.c_str(),
            nullptr,
            nullptr
        );
        
        if (!response)
        {
            NVIGI_LOG_ERROR("Session generate failed - null response");
            return kResultInvalidState;
        }
        
        std::string jsonResponse(response);
        llama_server_free_response(response);
        
        // Extract text content from JSON response
        std::string content = extractContentFromResponse(jsonResponse);
        return deliverResponse(content, jsonResponse, execCtx, instance, nvigi::kInferenceExecutionStateDone);
    }
    
    return kResultOk;
}

//=============================================================================
// Instance Management
//=============================================================================

nvigi::Result ggmlDestroyInstance(const nvigi::InferenceInstance* instance)
{
    nvtx3::scoped_range r{ "ggmlDestroyInstance" };
    if (instance)
    {
        auto gptInstance = static_cast<nvigi::gpt::InferenceContext*>(instance->data);
        
        // Cancel any running async job first
        gptInstance->cancelled.store(true);
        
        if (gptInstance->asyncJob.valid())
        {
            NVIGI_LOG_VERBOSE("Waiting for async job to complete...");
            
            // Wait for the async job to complete, repeatedly releasing any pending results.
            // This handles the race condition where the async task may not have started yet
            // when we first check - we keep trying to unblock it while waiting.
            // Note: recv_with_timeout in the server has a 1-second polling interval,
            // so streaming should notice cancellation within ~1 second.
            const int maxIterations = gptInstance->getTimeoutIterations();
            int iterations = 0;
            
            while (iterations < maxIterations)
            {
                // Try to release any pending results to unblock the async task's triggerCallback
                // This is safe to call even if nothing is pending (just notifies an empty CV)
                gptInstance->pollCtx.releaseResults(nvigi::kInferenceExecutionStateDone);
                
                auto status = gptInstance->asyncJob.wait_for(std::chrono::milliseconds(100));
                if (status == std::future_status::ready)
                {
                    gptInstance->asyncJob.get();
                    NVIGI_LOG_VERBOSE("Async job completed after %d iterations", iterations);
                    break;
                }
                iterations++;
            }
            
            if (iterations >= maxIterations)
            {
                // If we get here, something is seriously wrong. The server's 
                // recv_with_timeout will call std::terminate() if we try to stop
                // the server while a streaming request is in progress.
                NVIGI_LOG_ERROR("Async job failed to complete within timeout - server may be stuck");
                // We still need to try to clean up, but this may crash
            }
        }
        
        // Destroy the session first
        if (gptInstance->sessionId >= 0 && gptInstance->serverHandle)
        {
            NVIGI_LOG_VERBOSE("Destroying session %d...", gptInstance->sessionId);
            llama_server_session_destroy(gptInstance->serverHandle, gptInstance->sessionId);
            gptInstance->sessionId = -1;
            NVIGI_LOG_VERBOSE("Session destroyed");
        }
        
        // Free the server (llama_server_free internally calls llama_server_stop,
        // which is synchronous - it terminates the queues and joins the loop_thread)
        if (gptInstance->serverHandle)
        {
            NVIGI_LOG_VERBOSE("Freeing server...");
            llama_server_free(gptInstance->serverHandle);
            gptInstance->serverHandle = nullptr;
            NVIGI_LOG_VERBOSE("Server freed");
        }
        
        delete gptInstance;
        delete instance;
    }
    return nvigi::kResultOk;
}

nvigi::Result ggmlCreateInstance(const nvigi::NVIGIParameter* _params, nvigi::InferenceInstance** _instance)
{
    nvtx3::scoped_range r{"ggmlCreateInstance"};

    auto common = findStruct<CommonCreationParameters>(_params);
    auto creationParams = findStruct<GPTCreationParameters>(_params);
    auto ioCallbacks = findStruct<FileIOCallbacks>(_params);

    if (!common || !creationParams) return nvigi::kResultInvalidParameter;
    
    if (!_instance || ((!common->utf8PathToModels || !common->modelGUID) && !ioCallbacks)) return nvigi::kResultInvalidParameter;

    if (!ioCallbacks && !ai::isGuid(common->modelGUID))
    {
        NVIGI_LOG_ERROR("Provided model GUID '%s' is invalid", common->modelGUID);
        return kResultInvalidParameter;
    }

    *_instance = nullptr;

    auto instanceData = new nvigi::gpt::InferenceContext(_params);
    if (!instanceData->cudaContext.constructorSucceeded) return kResultInvalidState;
    auto& ctx = (*gpt::getContext());

    using namespace nvigi::gpt;

    // Find model information
    instanceData->modelInfo.clear();
    if (!ai::findModels(common, { "gguf" }, instanceData->modelInfo, ioCallbacks))
    {
        delete instanceData;
        return kResultInvalidParameter;
    }
    
    std::string modelPath;
    std::string mmprojPath;
    int n_layers = 0;
    int n_gpu_layers = 0;
    
    try
    {
        // Trim down to our GUID for this instance
        instanceData->modelInfo = instanceData->modelInfo[ioCallbacks ? ai::kSyntheticKey : common->modelGUID];

        n_layers = instanceData->modelInfo["n_layers"];
        
#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
        // Allow offloading to CPU as needed
        size_t neededVRAM = instanceData->modelInfo["vram"];
        auto gpuRatio = std::min(1.0, static_cast<double>(common->vramBudgetMB) / static_cast<double>(neededVRAM));
        n_gpu_layers = static_cast<int>(std::floor(gpuRatio * n_layers));
#endif
        
        std::vector<std::string> files = instanceData->modelInfo["gguf"];
        if (files.empty())
        {
            NVIGI_LOG_ERROR("Failed to find model in the expected directory '%s'", common->utf8PathToModels);
            delete instanceData;
            return kResultInvalidParameter;
        }
        modelPath = files[0];
        
        // VLM: Check if model config specifies a multimodal projector (mmproj)
        // When both "weights" and "mmproj_weights" are present, we match the gguf files
        // to their respective suffixes to correctly identify the main model and mmproj files
        if (instanceData->modelInfo.contains("weights") ||
            instanceData->modelInfo.contains("mmproj_weights"))
        {
            std::string weightsSuffix = instanceData->modelInfo.contains("weights") ? instanceData->modelInfo["weights"] : "";
            std::string mmprojSuffix = instanceData->modelInfo.contains("mmproj_weights") ? instanceData->modelInfo["mmproj_weights"] : "";
            
            auto endsWith = [](const std::string& str, const std::string& suffix) {
                return suffix.size() <= str.size() &&
                    str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
            };
            
            for (const auto& file : files)
            {
                if (endsWith(file, weightsSuffix))
                    modelPath = file;
                if (endsWith(file, mmprojSuffix))
                    mmprojPath = file;
            }
            
            if (!mmprojPath.empty())
            {
                instanceData->hasMultimodal = true;
                NVIGI_LOG_INFO("VLM model detected - multimodal projector: %s", mmprojPath.c_str());
            }
        }
    }
    catch (std::exception& e)
    {
        NVIGI_LOG_ERROR("%s", e.what());
        delete instanceData;
        return kResultInvalidState;
    }

    // Cache creation parameters
    instanceData->contextSize = creationParams->contextSize;
    instanceData->maxTokens = creationParams->maxNumTokensToPredict;
    instanceData->batchSize = creationParams->batchSize;
    if (creationParams->getVersion() >= 2)
    {
        instanceData->flashAttention = creationParams->flashAttention;
    }

    // Thread count: GPU backends benefit from fewer CPU threads to avoid contention
    int n_threads = common->numThreads;
#if defined(GGML_USE_CUBLAS)
    n_threads = 1;
#endif
    
    // Build server initialization parameters
    json serverParams = {
        {"model", modelPath},
        {"n_ctx", creationParams->contextSize},
        {"n_gpu_layers", n_gpu_layers},
        {"flash_attn", instanceData->flashAttention},
        {"n_threads", n_threads},
        {"n_threads_batch", n_threads},
        {"seed", creationParams->seed == -1 ? (uint32_t)time(nullptr) : (uint32_t)creationParams->seed},
        {"n_predict", creationParams->maxNumTokensToPredict},
        {"log_callback", reinterpret_cast<uint64_t>(llamaLogCallback)},
    };
    
    // VLM: Pass multimodal projector path to server if available
    if (!mmprojPath.empty())
    {
        serverParams["mmproj"] = mmprojPath;
    }
    
    if (creationParams->getVersion() >= 2)
    {
        serverParams["n_batch"] = creationParams->batchSize;
        serverParams["n_ubatch"] = creationParams->physicalBatchSize;
        // Map NVIGI cache types to server param names
        const char* cacheTypeNames[] = {"f32", "f16", "q4_0", "", "", "", "", "", "q8_0"};
        if (creationParams->cacheTypeK >= 0 && creationParams->cacheTypeK < 9)
        {
            serverParams["cache_type_k"] = cacheTypeNames[creationParams->cacheTypeK];
        }
        if (creationParams->cacheTypeV >= 0 && creationParams->cacheTypeV < 9)
        {
            serverParams["cache_type_v"] = cacheTypeNames[creationParams->cacheTypeV];
        }
    }

    // Match old implementation: pass ALL sampler and runtime params at create time.
    // Some of these affect common_init_from_params behavior (e.g., ignore_eos sets logit biases,
    // penalty_last_n=-1 resolves to ctx_size). The old code applied these BEFORE context creation.
    {
        auto initSampler = findStruct<GPTSamplerParameters>(_params);
        if (initSampler)
        {
            json sp;
            sp["n_prev"] = initSampler->numPrev;
            sp["n_probs"] = initSampler->numProbs;
            sp["min_keep"] = initSampler->minKeep;
            sp["top_k"] = initSampler->topK;
            sp["min_p"] = initSampler->minP;
            sp["xtc_probability"] = initSampler->xtcProbability;
            sp["xtc_threshold"] = initSampler->xtcThreshold;
            sp["typical_p"] = initSampler->typP;
            sp["dynatemp_range"] = initSampler->dynatempRange;
            sp["dynatemp_exponent"] = initSampler->dynatempExponent;
            sp["penalty_last_n"] = initSampler->penaltyLastN;
            sp["penalty_repeat"] = initSampler->penaltyRepeat;
            sp["penalty_freq"] = initSampler->penaltyFreq;
            sp["penalty_present"] = initSampler->penaltyPresent;
            sp["mirostat"] = initSampler->mirostat;
            sp["mirostat_tau"] = initSampler->mirostatTAU;
            sp["mirostat_eta"] = initSampler->mirostatETA;
            sp["ignore_eos"] = initSampler->ignoreEOS;
            serverParams["sampling"] = sp;
        }
        
        auto initRuntime = findStruct<GPTRuntimeParameters>(_params);
        if (initRuntime)
        {
            if (initRuntime->seed != 0xFFFFFFFF)
                serverParams["seed"] = initRuntime->seed;
            serverParams["n_predict"] = initRuntime->tokensToPredict;
            serverParams["n_keep"] = initRuntime->tokensToKeep;
            serverParams["n_parallel"] = initRuntime->numParallel;
            serverParams["n_sequences"] = initRuntime->numSequences;
            // n_batch from runtime at create time (creation params will override later via the existing serverParams["n_batch"])
            // temperature and top_p go into sampling
            if (!serverParams.contains("sampling")) serverParams["sampling"] = json::object();
            serverParams["sampling"]["temp"] = initRuntime->temperature;
            serverParams["sampling"]["top_p"] = initRuntime->topP;
        }
    }
    
    // Add LoRA adapters if specified (v3+)
    if (creationParams->getVersion() >= 3 && creationParams->numLoras > 0 && creationParams->loraNames)
    {
        json loraAdapters = json::array();
        for (size_t i = 0; i < creationParams->numLoras; i++)
        {
            if (creationParams->loraNames[i])
            {
                json loraEntry;
                loraEntry["path"] = creationParams->loraNames[i];
                if (creationParams->loraScales)
                {
                    loraEntry["scale"] = creationParams->loraScales[i];
                }
                else
                {
                    loraEntry["scale"] = 1.0f;
                }
                loraAdapters.push_back(loraEntry);
            }
        }
        if (!loraAdapters.empty())
        {
            serverParams["lora"] = loraAdapters;
        }
    }

    // Pass FileIOCallbacks to the server via a llama_model_io_callbacks struct
    if (ioCallbacks)
    {
        instanceData->ioCbs = {
            ioCallbacks->userData,
            ioCallbacks->open,
            ioCallbacks->close,
            ioCallbacks->size,
            ioCallbacks->tell,
            ioCallbacks->seek,
            ioCallbacks->read,
            nullptr
        };
        serverParams["io_callbacks"] = reinterpret_cast<uint64_t>(&instanceData->ioCbs);
    }

    // Check for runtime parameters at init time for template settings
    auto runtimeParams = findStruct<GPTRuntimeParameters>(_params);
    
    // Determine chat template - priority: 1) user-provided, 2) model config, 3) model's built-in
    bool useJinja = true;  // Default to Jinja
    std::string chatTemplate;
    
    // Check if model config has a chat template (it's stored as an array of strings)
    // Skip templates that contain <think> tags as they trigger thinking mode which outputs internal reasoning
    if (instanceData->modelInfo.contains("chat_template") && instanceData->modelInfo["chat_template"].is_array())
    {
        // Join the array of strings into a single template
        std::string configTemplate;
        for (const auto& line : instanceData->modelInfo["chat_template"])
        {
            if (line.is_string())
            {
                configTemplate += line.get<std::string>();
            }
        }
        
        chatTemplate = configTemplate;
        NVIGI_LOG_VERBOSE("Using chat template from model config (%zu chars)", chatTemplate.size());
    }
    
    if (runtimeParams)
    {
        // Chat template settings (v5+)
        if (runtimeParams->getVersion() >= 5)
        {
            useJinja = runtimeParams->useJinja;
            
            // User-provided template overrides model config
            if (runtimeParams->chatTemplate && strlen(runtimeParams->chatTemplate) > 0)
            {
                chatTemplate = runtimeParams->chatTemplate;
                NVIGI_LOG_VERBOSE("Using user-provided chat template (%zu chars)", chatTemplate.size());
            }
        }
        
        // Parallel sequences
        if (runtimeParams->numParallel > 1)
        {
            serverParams["n_parallel"] = runtimeParams->numParallel;
        }
    }
    
    // Fall back to ChatML if no template was found from model config or user
    // Older model JSON configs may not include chat_template and the GGUF may not have one embedded
    if (chatTemplate.empty())
    {
        chatTemplate =
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{'<|im_start|>assistant\n'}}"
            "{% endif %}";
        NVIGI_LOG_VERBOSE("No chat template in model config or user params, using ChatML default");
    }
    
    // Apply template settings to server params
    serverParams["use_jinja"] = useJinja;
    serverParams["chat_template"] = chatTemplate;

#if defined(GGML_USE_CUBLAS)

    // Set to match what embed does and that has this comment, so including here as well for posterity
    // ggml_backend_cuda_reg enumerates all devices which must be done before 
    // RuntimeContextScope is constructed
    ggml_backend_reg_t cuda_reg = ggml_backend_cuda_reg();
    nvigi::RuntimeContextScope scope(*instanceData);

    auto cudaParams = findStruct<CudaParameters>(_params);
    {
        // Query the CIG-active CUDA device (if CIG is enabled) or use the device from CudaParameters
        int targetDevice = 0;

        auto d3dParams = findStruct<D3D12Parameters>(_params);

        CudaParameters _dummy{};

        // If using CIG/D3D12, the CIG context is already current (pushed by RuntimeContextScope)
        // Query which device it's on so embed uses the correct device
        if (d3dParams && d3dParams->queue && instanceData->cudaContext.cudaCtx)
        {
            // CIG context is already current, just query which device
            CUdevice cuDevice;
            CUresult cuerr = cuCtxGetDevice(&cuDevice);
            if (cuerr == CUDA_SUCCESS)
            {
                targetDevice = (int)cuDevice;
                NVIGI_LOG_INFO("CIG context is already active on device %d, gpt.ggml.cuda plugin will use this device",
                    targetDevice);
            }
            else
            {
                targetDevice = 0;
                NVIGI_LOG_WARN("Failed to query current device, defaulting to device 0");
            }
            cudaParams = &_dummy;
        }
        else if (cudaParams)
        {
            targetDevice = cudaParams->device;
            NVIGI_LOG_INFO("User specified CUDA device %d in CudaParameters struct", targetDevice);
        }
        else
        {
            cudaParams = &_dummy;
            targetDevice = 0;
            NVIGI_LOG_INFO("No device specified, using default CUDA device 0");
        }

        // Pre-specify the device to prevent GGML from enumerating all devices (which changes context)
        // This must be done before llama_backend_init() and common_init_from_params()
        // IMPORTANT: The devices vector must be NULL-terminated (llama.cpp expects a NULL-terminated array)
        ggml_backend_dev_t cuda_dev = ggml_backend_reg_dev_get(cuda_reg, targetDevice);

        // Build CUDA JSON object for server initialization
        // Cast all pointers to uint64_t for JSON serialization
        // See llama_server.h for details on the JSON structure
        json cudaJson = {
            {"malloc_report_callback", reinterpret_cast<uint64_t>(cudaParams->cudaMallocReportCallback)},
            {"malloc_report_user_context", reinterpret_cast<uint64_t>(cudaParams->cudaMallocReportUserContext)},
            {"free_report_callback", reinterpret_cast<uint64_t>(cudaParams->cudaFreeReportCallback)},
            {"free_report_user_context", reinterpret_cast<uint64_t>(cudaParams->cudaFreeReportUserContext)},
            {"malloc_callback", reinterpret_cast<uint64_t>(cudaParams->cudaMallocCallback)},
            {"malloc_user_context", reinterpret_cast<uint64_t>(cudaParams->cudaMallocUserContext)},
            {"free_callback", reinterpret_cast<uint64_t>(cudaParams->cudaFreeCallback)},
            {"free_user_context", reinterpret_cast<uint64_t>(cudaParams->cudaFreeUserContext)},
            {"cuda_device", reinterpret_cast<uint64_t>(cuda_dev)}, // only item in the array
            {"main_gpu", 0 }, // index in the array
            // CUDA scope callbacks for thread/context management
            // Server calls these before/after any CUDA operations, passing scope_user_data
            {"scope_enter", reinterpret_cast<uint64_t>(&cudaScopeEnter)},
            {"scope_exit", reinterpret_cast<uint64_t>(&cudaScopeExit)},
            {"scope_user_data", reinterpret_cast<uint64_t>(instanceData)}
        };
        serverParams["cuda"] = cudaJson;
    }
#elif defined GGML_USE_VULKAN
    // Get Vulkan parameters from creation params
    auto vkParams = findStruct<VulkanParameters>(_params);
    if (vkParams)
    {
        // Build Vulkan JSON object for server initialization
        // Cast all pointers to uint64_t for JSON serialization
        json vulkanJson = {
            {"physical_device", reinterpret_cast<uint64_t>(vkParams->physicalDevice)},
            {"device", reinterpret_cast<uint64_t>(vkParams->device)},
            {"cmd_queue_direct", reinterpret_cast<uint64_t>(vkParams->queue)},
            {"cmd_queue_compute", reinterpret_cast<uint64_t>(vkParams->queueCompute)},
            {"cmd_queue_copy", reinterpret_cast<uint64_t>(vkParams->queueTransfer)},
            {"allocate_memory", reinterpret_cast<uint64_t>(vkParams->allocateMemoryCallback)},
            {"free_memory", reinterpret_cast<uint64_t>(vkParams->freeMemoryCallback)},
            {"user_context_alloc", reinterpret_cast<uint64_t>(vkParams->allocateMemoryCallbackUserContext)},
            {"user_context_free", reinterpret_cast<uint64_t>(vkParams->freeMemoryCallbackUserContext)}
        };
        serverParams["vulkan"] = vulkanJson;

        NVIGI_LOG_VERBOSE("Vulkan params: physical_device=%p, device=%p, direct_queue=%p, compute_queue=%p, copy_queue=%p",
            vkParams->physicalDevice, vkParams->device, vkParams->queue, vkParams->queueCompute, vkParams->queueTransfer);
    }
    else
    {
        NVIGI_LOG_WARN("Vulkan backend selected but no VulkanParameters provided");
    }
#elif defined(GGML_USE_D3D12)
    // Get D3D12 parameters from creation params
    auto d3d12Params = findStruct<D3D12Parameters>(_params);
    if (NVIGI_FAILED(res, d3d12::validateParameters(d3d12Params)))
    {
        NVIGI_LOG_ERROR("Invalid D3D12 parameters");
        delete instanceData;
        return res;
    }

    // Get adapter information for coreCount and architecture
    system::Adapter* adapter{};
    if (NVIGI_FAILED(res, d3d12::getDeviceAdapter(d3d12Params, ctx.isystem, &adapter)))
    {
        NVIGI_LOG_ERROR("Failed to get D3D12 adapter");
        delete instanceData;
        return res;
    }

    // Optionally get hwi.d3d12 interface for scheduling mode callback (NVIDIA only)
    PFun_commandListAction* CLresetCallback = nullptr;
    if (adapter->vendor == VendorId::eNVDA)
    {
        if (!ctx.iscg && !framework::getInterface(plugin::getContext()->framework, plugin::hwi::d3d12::kId, &ctx.iscg))
        {
            NVIGI_LOG_WARN("Missing interface from 'nvigi.plugin.hwi.d3d12', scheduling mode will not be applied");
        }

        // NOTE:  Current breaks if code is active.  Awaiting fix...
        if (ctx.iscg)
        {
            if (NVIGI_FAILED(res, d3d12::applyNVDASpecificSettings(d3d12Params, ctx.iscg)))
            {
                NVIGI_LOG_WARN("Failed to apply NVDA specific settings: %d", res);
            }

            if (ctx.iscg->getVersion() >= 4)
            {
                // Lambda to apply scheduling mode to command lists
                // GGML: kCommandListActionReset=0, Create=1, DispatchBegin=2, DispatchEnd=3 (ggml-vulkan-on-dx.hpp).
                static auto applySchedulingModeCallback = [](ID3D12CommandList* pCommandList, int action)
                {
                    constexpr int kGgmlCommandListActionDispatchBegin = 2;
                    if (action != kGgmlCommandListActionDispatchBegin)
                        return;

                    IHWID3D12* iscg = gpt::getContext()->iscg;
                    if (iscg)
                    {
                        ID3D12GraphicsCommandList* pGraphicsCommandList = nullptr;
                        if (SUCCEEDED(pCommandList->QueryInterface<ID3D12GraphicsCommandList>(&pGraphicsCommandList)))
                        {
                            iscg->d3d12ApplyGlobalGpuInferenceSchedulingModeToCommandList(pGraphicsCommandList);
                            pGraphicsCommandList->Release();
                        }
                    }
                };
                CLresetCallback = applySchedulingModeCallback;
            }
            else
            {
                NVIGI_LOG_WARN_ONCE("Need version 4 of hwi.d3d12 for D3D12 compute scheduling mode, available is %d. Performance may be suboptimal.", ctx.iscg->getVersion());
            }
        }
    }

    // Determine allowReBAR setting
    bool allowReBAR = d3d12Params->getVersion() < 3 || !(d3d12Params->flags & nvigi::D3D12ParametersFlags::eDisableReBAR);
    
    // Disabled by default for now
    bool allowAutoTuning = d3d12Params->getVersion() >= 3 && (d3d12Params->flags & nvigi::D3D12ParametersFlags::eEnableComputeAutoTuning);

    // Build D3D12 JSON object for server initialization
    // Cast all pointers to uint64_t for JSON serialization
    json d3d12Json = {
        {"device", reinterpret_cast<uint64_t>(d3d12Params->device)},
        {"cmd_queue_direct", reinterpret_cast<uint64_t>(d3d12Params->queue)},
        {"cmd_queue_compute", reinterpret_cast<uint64_t>(d3d12Params->queueCompute)},
        {"cmd_queue_copy", reinterpret_cast<uint64_t>(d3d12Params->queueCopy)},
        {"createCommittedResource", reinterpret_cast<uint64_t>(d3d12Params->createCommittedResourceCallback)},
        {"destroyResource", reinterpret_cast<uint64_t>(d3d12Params->destroyResourceCallback)},
        {"commandListAction", reinterpret_cast<uint64_t>(CLresetCallback)},
        {"userContextCreate", reinterpret_cast<uint64_t>(d3d12Params->createCommitResourceUserContext)},
        {"userContextDestroy", reinterpret_cast<uint64_t>(d3d12Params->destroyResourceUserContext)},
        {"allowReBAR", allowReBAR},
        {"bufferPadBytes", 0},
        {"maxDescriptorSets", 32 * 1024},
        {"constantBufferSize", 256 * 1024},
        {"coreCount", adapter->coreCount},
        {"architecture", adapter->architecture},
        {"allowAutoTuning", allowAutoTuning} // Allow server to choose optimal settings for this GPU
    };
    serverParams["d3d12"] = d3d12Json;

    NVIGI_LOG_VERBOSE("D3D12 params: device=%p, direct_queue=%p, compute_queue=%p, copy_queue=%p, allowReBAR=%d, coreCount=%u, arch=%u",
        d3d12Params->device, d3d12Params->queue, d3d12Params->queueCompute, d3d12Params->queueCopy,
        allowReBAR, adapter->coreCount, adapter->architecture);
#endif

    NVIGI_LOG_INFO("Loading model '%s'", modelPath.c_str());
#if GGML_USE_CUBLAS
    auto platform = "ggml.cuda";
#elif defined GGML_USE_VULKAN
    auto platform = "ggml.vulkan";
#elif defined(GGML_USE_D3D12)
    auto platform = "ggml.d3d12";
#else
    auto platform = "ggml.cpu";
#endif
    
    NVIGI_LOG_VERBOSE("# backend '%s'", platform);
    NVIGI_LOG_VERBOSE("# GPU layers %d[%d]", n_gpu_layers, n_layers);
    NVIGI_LOG_VERBOSE("# context size %d", creationParams->contextSize);
    NVIGI_LOG_VERBOSE("# max tokens %d", creationParams->maxNumTokensToPredict);
    NVIGI_LOG_VERBOSE("# batch %d", creationParams->batchSize);

    // Initialize the server
    std::string paramsStr = serverParams.dump();
    NVIGI_LOG_VERBOSE("Server params: %s", paramsStr.c_str());
    
    instanceData->serverHandle = llama_server_init(paramsStr.c_str());
    if (!instanceData->serverHandle)
    {
        NVIGI_LOG_ERROR("Failed to initialize llama server");
        delete instanceData;
        return kResultInvalidState;
    }

    // Load the model
    if (!llama_server_load_model(instanceData->serverHandle))
    {
        NVIGI_LOG_ERROR("Failed to load model into server");
        llama_server_free(instanceData->serverHandle);
        delete instanceData;
        return kResultInvalidState;
    }

    // Start the server inference loop
    llama_server_start(instanceData->serverHandle);
    
    NVIGI_LOG_INFO("Server started successfully for model '%s'", modelPath.c_str());
    
    // Create a stateful session for this instance
    instanceData->sessionId = llama_server_session_create(instanceData->serverHandle);
    if (instanceData->sessionId < 0)
    {
        NVIGI_LOG_ERROR("Failed to create session");
        llama_server_free(instanceData->serverHandle);
        delete instanceData;
        return kResultInvalidState;
    }
    
    NVIGI_LOG_VERBOSE("Created session %d", instanceData->sessionId);

#if GGML_USE_CUBLAS
    // We ask llama for all the cuda_streams it is going to use and 
    // store them to enable us to change their priorities dynamically
    
    void* streams[16]; // max 16 streams for now, should be enough for any model
    auto stream_count = llama_server_get_cuda_streams(instanceData->serverHandle, streams, 16);
    instanceData->cuda_streams.resize(stream_count);
    memcpy(instanceData->cuda_streams.data(), streams, stream_count * sizeof(void*));
    if (instanceData->cudaContext.usingCiG && ctx.icig->getVersion() >= 2)
    {
        // Apply the global priority to all streams
        nvigi::Result cuerr = ctx.icig->cudaApplyGlobalGpuInferenceSchedulingMode(instanceData->cuda_streams.data(), instanceData->cuda_streams.size());
        if (cuerr != kResultOk)
        {
            if (cuerr == kResultDriverOutOfDate)
            {
                NVIGI_LOG_WARN_ONCE("Could not set relative priority of CUDA compute and graphics because the driver is out of date\n");
            }
            else
            {
                NVIGI_LOG_ERROR("Could not set relative priority of CUDA compute and graphics because a CUDA error occurred.\n");
            }
        }
    }
#endif

    // At this point we are OK
    instanceData->state.store(nvigi::kResultOk);

    auto instance = new InferenceInstance();
    instance->data = instanceData;
    instance->getFeatureId = gpt::getFeatureId;
    instance->getInputSignature = gpt::getInputSignature;
    instance->getOutputSignature = gpt::getOutputSignature;
    instance->evaluate = gpt::evaluate;
    instance->evaluateAsync = gpt::evaluateAsync;
    instance->cancelAsyncEvaluation = gpt::cancelAsyncEvaluation;

    *_instance = instance;

    return kResultOk;
}

nvigi::Result ggmlGetCapsAndRequirements(nvigi::NVIGIParameter** _info, const nvigi::NVIGIParameter* _params)
{
    auto common = findStruct<CommonCreationParameters>(_params);
    if (!common) return nvigi::kResultInvalidParameter;

    static CommonCapabilitiesAndRequirements s_caps{};
    static GPTSamplerParameters s_sampler{}; // sets defaults
    auto info = &s_caps;
    *_info = s_caps;

    if (!findStruct<GPTSamplerParameters>(s_caps))
    {
        // Chaining this struct indicates that we support sampler parameters
        s_caps.chain(s_sampler);
    }

    auto& ctx = (*gpt::getContext());
    json modelInfo;
    // Nothing to do if path to models is not provided, user planning to use custom model loading
    if (common->utf8PathToModels)
    {
        if (!ai::findModels(common, { "gguf" }, modelInfo))
        {
            return kResultInvalidParameter;
        }
    }
    else
    {
        NVIGI_LOG_VERBOSE("Path to models not provided, assuming custom model loading will be used");
    }

#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
    info->supportedBackends = nvigi::InferenceBackendLocations::eGPU;
#else
    info->supportedBackends = nvigi::InferenceBackendLocations::eCPU;
#endif

    // Must be called after we set supported backends to correctly filter models
    ai::populateCommonCapsAndRequirements(ctx.capsData, *common, *info, modelInfo);

    return kResultOk;
}

//=============================================================================
// Evaluation
//=============================================================================

nvigi::Result ggmlEvaluate(nvigi::InferenceExecutionContext* execCtx, bool async)
{
    auto& ctx = (*gpt::getContext());

    // Validate all inputs first
    if (!execCtx)
    {
        NVIGI_LOG_ERROR("Unable to find 'InferenceExecutionContext' structure in the inputs");
        return kResultInvalidParameter;
    }

    // OK not to have callback in async mode with polled results
    if (!execCtx->callback && !async)
    {
        NVIGI_LOG_ERROR("GPT inference callback not provided");
        return kResultInvalidParameter;
    }

    if (!execCtx->instance)
    {
        NVIGI_LOG_ERROR("GPT inference instance not provided");
        return kResultInvalidParameter;
    }
    
    if (ctx.feature != execCtx->instance->getFeatureId(execCtx->instance->data))
    {
        NVIGI_LOG_ERROR("Invalid inference instance - expecting GPT %u got %u", ctx.feature, execCtx->instance->getFeatureId(execCtx->instance->data));
        return kResultInvalidParameter;
    }

    using namespace nvigi::gpt;

    auto instance = (nvigi::gpt::InferenceContext*)(execCtx->instance->data);

    if (NVIGI_FAILED(result, instance->state.load()))
    {
        NVIGI_LOG_ERROR("Instance is in invalid state and it must be destroyed and recreated");
        return result;
    }

    // CRITICAL: Wait for any previous async job to complete BEFORE any session operations.
    // If we don't wait here, session_clear or session_add_messages could corrupt the KV cache
    // state while the previous generation is still running.
    if (instance->asyncJob.valid())
    {
        NVIGI_LOG_VERBOSE("Waiting for previous async job before starting new evaluation...");
        instance->cancelled.store(true);
        
        const int maxIterations = instance->getTimeoutIterations();
        int iterations = 0;
        while (iterations < maxIterations && 
               instance->asyncJob.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready)
        {
            instance->pollCtx.releaseResults(nvigi::kInferenceExecutionStateDone);
            iterations++;
        }
        
        if (iterations >= maxIterations)
        {
            NVIGI_LOG_ERROR("Previous async generation did not complete within %.1fs timeout", instance->maxTimeoutSeconds);
            instance->cancelled.store(false);
            return kResultTimedOut;
        }
        
        instance->asyncJob.get();
        instance->cancelled.store(false);
        NVIGI_LOG_VERBOSE("Previous async job completed, proceeding with new evaluation");
    }

    // Read input slots
    const nvigi::InferenceDataText* systemSlot{};
    const nvigi::InferenceDataText* userSlot{};
    const nvigi::InferenceDataText* assistantSlot{};
    const nvigi::InferenceDataText* jsonSlot{};
    execCtx->inputs->findAndValidateSlot(kGPTDataSlotSystem, &systemSlot);
    execCtx->inputs->findAndValidateSlot(kGPTDataSlotUser, &userSlot);
    execCtx->inputs->findAndValidateSlot(kGPTDataSlotAssistant, &assistantSlot);
    execCtx->inputs->findAndValidateSlot(kGPTDataSlotJSON, &jsonSlot);
    
    if (!userSlot && !systemSlot && !assistantSlot)
    {
        NVIGI_LOG_ERROR("Expecting inference input(s) of type 'nvigi::InferenceDataText' - either system, user and/or assistant inputs should be provided");
        return kResultInvalidParameter;
    }

#if GGML_USE_CUBLAS
    // IMPORTANT: Note that we are not using RuntimeContextScope here, server has callbacks to manage CUDA context scope for us. We just need to apply the scheduling mode to the streams before generation.
    if (instance->cudaContext.usingCiG && ctx.icig->getVersion() >= 2)
    {
        // Apply the global priority to all streams
        nvigi::Result cuerr = ctx.icig->cudaApplyGlobalGpuInferenceSchedulingMode(instance->cuda_streams.data(), instance->cuda_streams.size());
        if (cuerr != kResultOk)
        {
            if (cuerr == kResultDriverOutOfDate)
            {
                NVIGI_LOG_WARN_ONCE("Could not set relative priority of CUDA compute and graphics because the driver is out of date\n");
            }
            else
            {
                NVIGI_LOG_ERROR("Could not set relative priority of CUDA compute and graphics because a CUDA error occurred.\n");
            }
        }
    }
#endif

    // Extract custom JSON for pass-through to server (optional, for advanced parameters)
    std::string customJson = jsonSlot ? jsonSlot->getUTF8Text() : "";
    
    std::string system = systemSlot ? systemSlot->getUTF8Text() : "";
    std::string user = userSlot ? userSlot->getUTF8Text() : "";
    std::string assistant = assistantSlot ? assistantSlot->getUTF8Text() : "";
    
    // Check for image input (VLM)
    const nvigi::InferenceDataImage* imageSlot{};
    execCtx->inputs->findAndValidateSlot(kGPTDataSlotImage, &imageSlot);
    
    // Build the image data URL if we have image data and VLM is enabled
    std::string imageDataUrl;
    if (imageSlot != nullptr)
    {
        if (!instance->hasMultimodal)
        {
            NVIGI_LOG_WARN_ONCE("Image input provided but model does not have a multimodal projector (mmproj). Image data will be ignored.");
        }
        else
        {
            // Extract raw RGB data from the image slot
            auto* cpuData = castTo<CpuData>(imageSlot->bytes);
            if (cpuData && cpuData->buffer && cpuData->sizeInBytes > 0 &&
                imageSlot->w > 0 && imageSlot->h > 0 && imageSlot->c >= 3)
            {
                NVIGI_LOG_VERBOSE("Encoding VLM image (%dx%d, %d channels, %zu bytes) as BMP data URL",
                    imageSlot->w, imageSlot->h, imageSlot->c, cpuData->sizeInBytes);
                imageDataUrl = buildImageDataUrl(
                    static_cast<const uint8_t*>(cpuData->buffer),
                    imageSlot->w, imageSlot->h, imageSlot->c);
                NVIGI_LOG_VERBOSE("Image data URL size: %zu chars", imageDataUrl.size());
            }
            else
            {
                NVIGI_LOG_WARN("Invalid image data in kGPTDataSlotImage (buffer=%p, size=%zu, w=%d, h=%d, c=%d)",
                    cpuData ? cpuData->buffer : nullptr,
                    cpuData ? cpuData->sizeInBytes : 0,
                    imageSlot->w, imageSlot->h, imageSlot->c);
            }
        }
    }

    // Check runtime parameters for chat mode (interactive = chat, non-interactive = instruct)
    auto runtime = findStruct<GPTRuntimeParameters>(execCtx->runtimeParameters);
    bool chatMode = runtime ? runtime->interactive : true;
    
    // Apply prefix/suffix to user input if provided (backward compatible with old CLI approach)
    if (runtime && !user.empty())
    {
        if (runtime->prefix && strlen(runtime->prefix) > 0)
        {
            user = std::string(runtime->prefix) + user;
        }
        if (runtime->suffix && strlen(runtime->suffix) > 0)
        {
            user = user + std::string(runtime->suffix);
        }
    }

    // Update configurable timeout if provided (v6+)
    if (runtime && runtime->getVersion() >= 6 && runtime->maxTimeoutSeconds > 0.0f)
    {
        instance->maxTimeoutSeconds = runtime->maxTimeoutSeconds;
    }
    
    // Check if persistent KV cache is requested (v2+ sampler parameter)
    auto sampler = findStruct<GPTSamplerParameters>(execCtx->runtimeParameters);
    bool persistentKV = sampler && sampler->getVersion() >= 2 && sampler->persistentKVCache;
    
    // Handle session state based on mode
    if (systemSlot)
    {
        if (persistentKV)
        {
            NVIGI_LOG_VERBOSE("New system prompt provided but persistentKVCache=true, keeping session context");
        }
        else
        {
            NVIGI_LOG_VERBOSE("New system prompt provided, clearing session context");
            llama_server_session_clear(instance->serverHandle, instance->sessionId);
            instance->sessionHasContent = false;
            instance->systemPromptTokens = 0;
        }
        instance->inChatMode = chatMode;
    }
    // Note: In instruct mode, we DON'T clear here when receiving a user-only message.
    // The session should retain the system prompt that was cached in a previous call.
    // Clearing happens only when a NEW system prompt arrives (above) which starts a fresh instruction.
    
    // Ensure we have context space for the new prompt and response (chat mode only)
    if (chatMode && instance->sessionHasContent)
    {
        int reservedTokens = instance->maxTokens + 200; // Reserve for user input + response
        int userKeep = runtime ? runtime->tokensToKeep : 0;
        if (!ensureContextSpace(instance, reservedTokens, userKeep))
        {
            NVIGI_LOG_ERROR("Failed to ensure context space");
            return kResultInvalidState;
        }
    }
    
    // Build the full set of runtime+sampler params to pass with every server call.
    // The old CLI implementation applied ALL params on every evaluate (users can change between calls).
    json evalParams = buildFullParamsJson(runtime, sampler, instance->maxTokens, instance->inChatMode);
    
    // Helper to wrap a messages array with ALL current params for the server
    auto buildMessagesJson = [&evalParams](const json& messages) -> std::string {
        json wrapper = evalParams;
        wrapper["messages"] = messages;
        return wrapper.dump();
    };
    
    // Determine if we should generate a response
    // Generate when there's a user message, image input, or assistant prefix for constrained generation
    bool shouldGenerate = !user.empty() || !assistant.empty() || !imageDataUrl.empty();
    
    // Check if prompt is pre-templatized (v3+) - host already applied the chat template
    bool preTemplatized = runtime && runtime->getVersion() >= 3 && runtime->promptPretemplatized;
    
    // When prompt is pre-templatized, bypass the chat template and add raw text directly.
    // The host is responsible for correct formatting including special tokens.
    if (preTemplatized && shouldGenerate)
    {
        std::string rawPrompt;
        if (!system.empty()) rawPrompt += system;
        if (!user.empty()) rawPrompt += user;
        if (!assistant.empty()) rawPrompt += assistant;
        
        if (!rawPrompt.empty())
        {
            bool addBos = !instance->sessionHasContent;
            int tokensAdded = llama_server_session_add_raw_text(
                instance->serverHandle, instance->sessionId, rawPrompt.c_str(), addBos);
            
            if (tokensAdded < 0)
            {
                NVIGI_LOG_ERROR("Failed to add pre-templatized prompt to session");
                return kResultInvalidState;
            }
            
            instance->sessionHasContent = true;
            NVIGI_LOG_VERBOSE("Added pre-templatized prompt (%d tokens, add_bos=%s)", 
                              tokensAdded, addBos ? "true" : "false");
        }
        
        // Skip the normal system/user/assistant message construction below
        // and go directly to generation
    }
    else
    {
    
    // Step 1: Add system prompt if provided (without generation prompt)
    // This prepares the context without triggering generation
    if (!system.empty())
    {
        nvtx3::scoped_range r{"ggmlEvaluate Add system prompt"};
        json systemMessages = json::array();
        systemMessages.push_back({{"role", "system"}, {"content", system}});
        std::string systemStr = buildMessagesJson(systemMessages);
        
        // Track tokens for system prompt
        int preLen = llama_server_session_get_context_length(instance->serverHandle, instance->sessionId);
        
        // add_generation_prompt=false - just store in context, don't prepare for response
        int tokensAdded = llama_server_session_add_messages(
            instance->serverHandle,
            instance->sessionId,
            systemStr.c_str(),
            false  // No assistant prefix - we're just caching the system prompt
        );
        
        if (tokensAdded < 0)
        {
            NVIGI_LOG_ERROR("Failed to add system prompt to session");
            return kResultInvalidState;
        }
        
        // Track system prompt tokens for context truncation
        int postLen = llama_server_session_get_context_length(instance->serverHandle, instance->sessionId);
        instance->systemPromptTokens = (postLen > preLen) ? (postLen - preLen) : tokensAdded;
        
        instance->sessionHasContent = true;
        NVIGI_LOG_VERBOSE("Added system prompt to session (%d tokens)", instance->systemPromptTokens);
    }
    
    // Step 2: Add user message if provided (with generation prompt)
    // For VLM: when image data is available, build multimodal content blocks
    if (!user.empty() || !imageDataUrl.empty())
    {
        nvtx3::scoped_range r{ "ggmlEvaluate Add user message" };
        json userMessages = json::array();
        
        if (!imageDataUrl.empty())
        {
            // Multimodal message: content is an array of content blocks (OpenAI format)
            // [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:..."}}]
            //
            // Backward compatibility with <image> marker:
            // The old VLM implementation allowed hosts to place "<image>" in the user text to control
            // where the image appears in the prompt. We support this by splitting the text at "<image>"
            // and interleaving text/image content blocks. If no "<image>" marker is found, the image
            // is appended after the text (matching the old default behavior for Nemovision models).
            // Only one image per prompt is supported - additional "<image>" markers are stripped.
            json contentBlocks = json::array();
            json imageBlock = {
                {"type", "image_url"},
                {"image_url", {{"url", imageDataUrl}}}
            };
            
            if (!user.empty())
            {
                const std::string imageMarker = "<image>";
                size_t markerPos = user.find(imageMarker);
                
                if (markerPos != std::string::npos)
                {
                    // Host specified image placement - split text around <image>
                    std::string textBefore = user.substr(0, markerPos);
                    std::string textAfter = user.substr(markerPos + imageMarker.size());
                    
                    // Strip any additional <image> markers (only 1 image supported)
                    size_t pos;
                    while ((pos = textAfter.find(imageMarker)) != std::string::npos)
                    {
                        textAfter.erase(pos, imageMarker.size());
                    }
                    
                    if (!textBefore.empty())
                        contentBlocks.push_back({{"type", "text"}, {"text", textBefore}});
                    contentBlocks.push_back(imageBlock);
                    if (!textAfter.empty())
                        contentBlocks.push_back({{"type", "text"}, {"text", textAfter}});
                    
                    NVIGI_LOG_VERBOSE("Image placed at <image> marker position in user text");
                }
                else
                {
                    // No marker - image goes after text (default for Nemovision and similar models)
                    contentBlocks.push_back({{"type", "text"}, {"text", user}});
                    contentBlocks.push_back(imageBlock);
                }
            }
            else
            {
                // Image only, no text
                contentBlocks.push_back(imageBlock);
            }
            
            userMessages.push_back({{"role", "user"}, {"content", contentBlocks}});
            NVIGI_LOG_VERBOSE("Built multimodal user message with %zu content blocks", contentBlocks.size());
        }
        else
        {
            // Text-only message
            userMessages.push_back({{"role", "user"}, {"content", user}});
        }
        
        std::string userStr = buildMessagesJson(userMessages);
        
        //NVIGI_LOG_VERBOSE("Server JSON %s", userStr.c_str());

        // add_generation_prompt=true if no assistant prefix, false if we have one
        // (assistant prefix means we're doing constrained generation)
        bool addGenPrompt = assistant.empty();
        
        int tokensAdded = llama_server_session_add_messages(
            instance->serverHandle,
            instance->sessionId,
            userStr.c_str(),
            addGenPrompt
        );
        
        if (tokensAdded < 0)
        {
            NVIGI_LOG_ERROR("Failed to add user message to session");
            return kResultInvalidState;
        }
        
        instance->sessionHasContent = true;
        NVIGI_LOG_VERBOSE("Added user message to session (%d tokens, add_generation_prompt=%s)", 
                          tokensAdded, addGenPrompt ? "true" : "false");
    }
    
    // Step 3: Add assistant prefix if provided (for constrained generation)
    if (!assistant.empty())
    {
        nvtx3::scoped_range r{ "ggmlEvaluate Add assistant prefix" };
        json assistantMessages = json::array();
        assistantMessages.push_back({{"role", "assistant"}, {"content", assistant}});
        std::string assistantStr = buildMessagesJson(assistantMessages);
        
        // add_generation_prompt=false - the assistant content IS the start of generation
        int tokensAdded = llama_server_session_add_messages(
            instance->serverHandle,
            instance->sessionId,
            assistantStr.c_str(),
            false  // No additional assistant prefix needed
        );
        
        if (tokensAdded < 0)
        {
            NVIGI_LOG_ERROR("Failed to add assistant prefix to session");
            return kResultInvalidState;
        }
        
        instance->sessionHasContent = true;
        NVIGI_LOG_VERBOSE("Added assistant prefix to session (%d tokens)", tokensAdded);
    }
    
    } // end of else (non-pretemplatized path)
    
    // If only system prompt was provided (no user message), don't generate - just return
    // This allows the host to set up the context without triggering a response
    if (!shouldGenerate)
    {
        NVIGI_LOG_VERBOSE("System prompt cached, awaiting user message before generation");
        
        // For async mode, we need to signal completion via an async task
        // We can't call deliverResponse directly as it would block the main thread
        if (async)
        {
            // Wait for any previous async job to complete first
            if (instance->asyncJob.valid())
            {
                // Release any pending results to unblock the previous async task
                // Keep trying in case of race condition with task startup
                instance->cancelled.store(true);
                const int maxIterations = instance->getTimeoutIterations();
                int iterations = 0;
                while (iterations < maxIterations && 
                       instance->asyncJob.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready)
                {
                    instance->pollCtx.releaseResults(nvigi::kInferenceExecutionStateDone);
                    iterations++;
                }
                
                if (iterations >= maxIterations)
                {
                    NVIGI_LOG_ERROR("Previous async generation did not complete within timeout");
                    instance->cancelled.store(false);
                    return kResultTimedOut;
                }
                
                instance->asyncJob.get();
                instance->cancelled.store(false);
            }
            
            // Launch a minimal async task that signals completion
            instance->asyncJob = std::async(std::launch::async, 
                [instance, execCtx]() -> nvigi::Result {
                    // Deliver empty response with Done state to signal completion
                    deliverResponse("", "", execCtx, instance, nvigi::kInferenceExecutionStateDone);
                    return kResultOk;
                });
        }
        
        return kResultOk;
    }

    // Reset cancellation flag
    instance->cancelled.store(false);

    // Execute request using session-based generation
    nvigi::Result result = kResultOk;
    
    if (async)
    {
        // Wait for any previous async job to complete
        if (instance->asyncJob.valid())
        {
            // Release any pending results to unblock the previous async task
            // Keep trying in case of race condition with task startup
            instance->cancelled.store(true);
            const int maxIterations = instance->getTimeoutIterations();
            int iterations = 0;
            while (iterations < maxIterations && 
                   instance->asyncJob.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready)
            {
                instance->pollCtx.releaseResults(nvigi::kInferenceExecutionStateDone);
                iterations++;
            }
            
            if (iterations >= maxIterations)
            {
                // Previous async job did not complete - server slot may still be busy
                // Cannot safely start a new operation
                NVIGI_LOG_ERROR("Previous async generation did not complete within timeout - cannot start new operation");
                instance->cancelled.store(false);
                return kResultTimedOut;
            }
            
            instance->asyncJob.get();
            instance->cancelled.store(false);
        }
        
        // Launch async session request (capture customJson by value for thread safety)
        instance->asyncJob = std::async(std::launch::async, 
            [instance, execCtx, customJson]() -> nvigi::Result {
                return sendSessionRequest(instance, execCtx, true, customJson);
            });
    }
    else
    {
        // Blocking session request
        result = sendSessionRequest(instance, execCtx, false, customJson);
    }

    return result;
}

nvigi::Result gptGetResults(nvigi::InferenceExecutionContext* execCtx, bool wait, nvigi::InferenceExecutionState* state)
{
    if (!execCtx || !execCtx->instance)
        return kResultInvalidParameter;

    auto instance = static_cast<gpt::InferenceContext*>(execCtx->instance->data);
    return instance->pollCtx.getResults(wait, state);
}

nvigi::Result gptReleaseResults(nvigi::InferenceExecutionContext* execCtx, nvigi::InferenceExecutionState state)
{
    if (!execCtx || !execCtx->instance)
        return kResultInvalidParameter;

    auto instance = static_cast<gpt::InferenceContext*>(execCtx->instance->data);
    return instance->pollCtx.releaseResults(state);
}

nvigi::Result gptCancelAsyncEvaluation(nvigi::InferenceExecutionContext* execCtx)
{
    if (!execCtx || !execCtx->instance)
        return kResultInvalidParameter;

    auto instance = static_cast<gpt::InferenceContext*>(execCtx->instance->data);
    
    // Check if async job is actually running
    if (!instance->asyncJob.valid())
    {
        NVIGI_LOG_WARN("cancelAsyncEvaluation called but no async evaluation is running");
        return kResultNoImplementation;
    }
    
    // Set cancellation flag
    instance->cancelled.store(true);
    
    // Release any pending results to unblock the background thread
    const int maxIterations = instance->getTimeoutIterations();
    int iterations = 0;
    while (iterations < maxIterations && 
           instance->asyncJob.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready)
    {
        instance->pollCtx.releaseResults(nvigi::kInferenceExecutionStateDone);
        iterations++;
    }
    
    if (iterations < maxIterations)
    {
        instance->asyncJob.get();
        return kResultOk;
    }
    else
    {
        NVIGI_LOG_WARN("Async job timed out during cancellation after 60 seconds");
        return kResultTimedOut;
    }
}


//=============================================================================
// Session state (KV cache save/restore) - IGPTSessionState implementation
//=============================================================================

static size_t ggmlSessionSeqGetDataSize(const nvigi::InferenceInstance* instance)
{
    if (!instance || !instance->data) 
        return 0;
    auto* ctx = static_cast<const nvigi::gpt::InferenceContext*>(instance->data);
    if (!ctx->serverHandle || ctx->sessionId < 0) 
        return 0;
    return llama_server_session_seq_get_data_size(ctx->serverHandle, ctx->sessionId);
}

static size_t ggmlSessionSeqGetData(const nvigi::InferenceInstance* instance,
    unsigned char* buffer, size_t buffer_size, int* out_n_past)
{
    if (!instance || !instance->data || !buffer || !out_n_past) 
        return 0;
    auto* ctx = static_cast<const nvigi::gpt::InferenceContext*>(instance->data);
    if (!ctx->serverHandle || ctx->sessionId < 0) 
        return 0;
    return llama_server_session_seq_get_data(ctx->serverHandle, ctx->sessionId,
        buffer, buffer_size, out_n_past);
}

static bool ggmlSessionSeqSetData(const nvigi::InferenceInstance* instance,
    const unsigned char* buffer, size_t buffer_size, int n_past)
{
    if (!instance || !instance->data || !buffer || buffer_size==0)
        return false;
    auto* ctx = static_cast<const nvigi::gpt::InferenceContext*>(instance->data);
    if (!ctx->serverHandle || ctx->sessionId < 0) 
        return false;
    return llama_server_session_seq_set_data(ctx->serverHandle, ctx->sessionId,
        buffer, buffer_size, n_past);
}

//=============================================================================
// Exception Handling Wrappers
//=============================================================================

namespace gpt
{
nvigi::Result createInstance(const nvigi::NVIGIParameter* params, nvigi::InferenceInstance** instance)
{
    NVIGI_CATCH_EXCEPTION(ggmlCreateInstance(params, instance));
}
nvigi::Result destroyInstance(const nvigi::InferenceInstance* instance)
{
    NVIGI_CATCH_EXCEPTION(ggmlDestroyInstance(instance));
}
nvigi::Result getCapsAndRequirements(nvigi::NVIGIParameter** modelInfo, const nvigi::NVIGIParameter* params)
{
    NVIGI_CATCH_EXCEPTION(ggmlGetCapsAndRequirements(modelInfo, params));
}
nvigi::Result evaluate(nvigi::InferenceExecutionContext* execCtx)
{
    NVIGI_CATCH_EXCEPTION(ggmlEvaluate(execCtx, false));
}
nvigi::Result evaluateAsync(nvigi::InferenceExecutionContext* execCtx)
{
    NVIGI_CATCH_EXCEPTION(ggmlEvaluate(execCtx, true));
}
nvigi::Result getResults(nvigi::InferenceExecutionContext* execCtx, bool wait, nvigi::InferenceExecutionState* state)
{
    NVIGI_CATCH_EXCEPTION(gptGetResults(execCtx, wait, state));
}

nvigi::Result releaseResults(nvigi::InferenceExecutionContext* execCtx, nvigi::InferenceExecutionState state)
{
    NVIGI_CATCH_EXCEPTION(gptReleaseResults(execCtx, state));
}

nvigi::Result cancelAsyncEvaluation(nvigi::InferenceExecutionContext* execCtx)
{
    NVIGI_CATCH_EXCEPTION(gptCancelAsyncEvaluation(execCtx));
}
} // gpt

//=============================================================================
// Plugin Entry Points
//=============================================================================

Result nvigiPluginGetInfo(framework::IFramework* framework, nvigi::plugin::PluginInfo** _info)
{
    auto& ctx = (*gpt::getContext());
    
    if (!plugin::internalPluginSetup(framework)) return kResultInvalidState;

    // Internal API, we know that incoming pointer is always valid
    auto& info = plugin::getContext()->info;
    *_info = &info;

    info.id = gpt::getFeatureId(nullptr);
    info.description = "Server-based backend for LLM inference";
    info.author = "NVIDIA";
    info.build = GIT_BRANCH_AND_LAST_COMMIT;
    info.interfaces = { plugin::getInterfaceInfo<IGeneralPurposeTransformer>(),
                   plugin::getInterfaceInfo<nvigi::IGPTSessionState>() };

    // Always the same OS requirements
    info.minOS = { NVIGI_DEF_MIN_OS_MAJOR, NVIGI_DEF_MIN_OS_MINOR, NVIGI_DEF_MIN_OS_BUILD };

    // Default to no GPU requirements for now
    info.minGPUArch = {};
    info.minDriver = {};

    if (!framework::getInterface(framework, nvigi::core::framework::kId, &ctx.isystem))
    {
        NVIGI_LOG_ERROR("Missing interface from core framework");
        return kResultInvalidState;
    }

    // Check if we have an NV adapter available and if so, set the minimum GPU architecture and driver version
    bool hasNVAdapter = false;
    const nvigi::system::SystemCaps* caps = ctx.isystem->getSystemCaps();
    for (uint32_t i = 0; i < caps->adapterCount; i++)
    {
        const nvigi::system::Adapter* adapter = caps->adapters[i];
        if (adapter->vendor == VendorId::eNVDA)
        {
            hasNVAdapter = true;
            break;
        }
    }

#ifdef GGML_USE_CUBLAS
    info.minDriver = { NVIGI_CUDA_MIN_DRIVER_MAJOR, NVIGI_CUDA_MIN_DRIVER_MINOR, NVIGI_CUDA_MIN_DRIVER_BUILD };
    info.minGPUArch = { NVIGI_CUDA_MIN_GPU_ARCH };
    info.requiredVendor = VendorId::eNVDA;

    if (hasNVAdapter && caps->driverVersion.major < 580)
    {
        NVIGI_LOG_WARN_ONCE("CUDA backend recommends driver version 580 or higher");
    }
#elif defined(GGML_USE_D3D12)
    info.requiredVendor = VendorId::eAny;

    if (hasNVAdapter)
    {
        info.minGPUArch = { NVIGI_CUDA_MIN_GPU_ARCH };
        info.minDriver = { NVIGI_D3D12_MIN_DRIVER_MAJOR, NVIGI_D3D12_MIN_DRIVER_MINOR, NVIGI_D3D12_MIN_DRIVER_BUILD };
        info.requiredVendor = VendorId::eNVDA;
    }
#elif defined(GGML_USE_VULKAN)
    info.requiredVendor = VendorId::eAny;
#else
    info.requiredVendor = nvigi::VendorId::eNone;
#endif

    // Must release, as we may not have the chance to release it later if we are not registered
    framework::releaseInterface(plugin::getContext()->framework, nvigi::core::framework::kId, ctx.isystem);
    ctx.isystem = nullptr;

    return kResultOk;
}

Result nvigiPluginRegister(framework::IFramework* framework)
{
    if (!plugin::internalPluginSetup(framework)) return kResultInvalidState;

    auto& ctx = (*gpt::getContext());

    ctx.feature = gpt::getFeatureId(nullptr);

#if GGML_USE_CUBLAS
    if (!framework::getInterface(plugin::getContext()->framework, plugin::hwi::cuda::kId, &ctx.icig))
    {
        NVIGI_LOG_ERROR("Missing interface from 'nvigi.plugin.hwi.cuda'");
        return kResultInvalidState;
    }
#elif defined(GGML_USE_D3D12)
    if (!framework::getInterface(framework, nvigi::core::framework::kId, &ctx.isystem))
    {
        NVIGI_LOG_ERROR("Missing core interface 'nvigi::system::ISystem'");
        return kResultMissingInterface;
    }
#endif

    ctx.api.createInstance = gpt::createInstance;
    ctx.api.destroyInstance = gpt::destroyInstance;
    ctx.api.getCapsAndRequirements = gpt::getCapsAndRequirements;
 
    framework->addInterface(ctx.feature, &ctx.api, 0);

    // Add polled interface
    ctx.polledApi.getResults = gpt::getResults;
    ctx.polledApi.releaseResults = gpt::releaseResults;
    framework->addInterface(ctx.feature, &ctx.polledApi, 0);

    // Add session state interface (KV cache save/restore, per llama sequence / slot)
    ctx.sessionStateApi.seq_get_data_size = ggmlSessionSeqGetDataSize;
    ctx.sessionStateApi.seq_get_data = ggmlSessionSeqGetData;
    ctx.sessionStateApi.seq_set_data = ggmlSessionSeqSetData;
    framework->addInterface(ctx.feature, &ctx.sessionStateApi, 0);

    return kResultOk;
}

Result nvigiPluginDeregister()
{
    auto& ctx = (*gpt::getContext());

    ai::freeCommonCapsAndRequirements(ctx.capsData);

#if GGML_USE_CUBLAS
    framework::releaseInterface(plugin::getContext()->framework, plugin::hwi::cuda::kId, ctx.icig);
    ctx.icig = nullptr;
#elif defined(GGML_USE_D3D12)
    framework::releaseInterface(plugin::getContext()->framework, plugin::hwi::d3d12::kId, ctx.iscg);
    ctx.iscg = nullptr;
    framework::releaseInterface(plugin::getContext()->framework, nvigi::core::framework::kId, ctx.isystem);
    ctx.isystem = nullptr;
#endif

    llama_server_backend_free();

    return kResultOk;
}

// The only exported function - gateway to all functionality
NVIGI_EXPORT void* nvigiPluginGetFunction(const char* functionName)
{
    // Core API
    NVIGI_EXPORT_FUNCTION(nvigiPluginGetInfo);
    NVIGI_EXPORT_FUNCTION(nvigiPluginRegister);
    NVIGI_EXPORT_FUNCTION(nvigiPluginDeregister);
    
    return nullptr;
}

}
