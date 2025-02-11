// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

// BEGIN NVIDIA EDITORS NOTES
// 
// Anyone modifying this file, pay special attention to the generate function and take care to how it is modified.
// This is _mostly_ a block copy of the LlamaCPP file main.cpp located in the llama-cli project.
// Initialization and shutdown has been moved, and a few modifications have been made to stream results back appropriately
// for NVIGI.  
// 
// In the interests of making future merges easier, if any modifications are needed, wrap them in
// // Begin NVIDIA Modification
// ...modifications...
// // End NVIDIA Modification
// This should help ensure functionality is not lost or accidentally broken during future code updates.
//
// END NVIDIA EDITORS NOTES

#pragma once

#include <inttypes.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <regex>

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#elif GGML_USE_CLBLAST
#include "ggml-opencl.h"
#endif

// Begin NVIDIA Modification
#include "source/core/nvigi.log/log.h"
#define LOG(fmt,...)
#define LOG_DBG(fmt,...)
#define LOG_CNT(fmt,...)
#define LOG_INF(fmt,...)
#define LOG_ERR NVIGI_LOG_ERROR
#define LOG_WRN NVIGI_LOG_WARN
// End NVIDIA Modification

#include "llama.h"
#include "common.h"
//#include "log.h"
#include "sampling.h"
#include "clip.h"
#include "llava.h"

#include "external/json/source/nlohmann/json.hpp"
using json = nlohmann::json;

//#ifdef NVIGI_GPT_GFN_NVCF

//! Unfortunately GGML gpt_params structure requires this method so reimplementing
int32_t get_num_physical_cores()
{
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

//#endif

namespace nvigi
{
namespace gpt
{

#ifndef NVIGI_GPT_GFN_NVCF

struct InferenceContextSync
{
    std::promise<void> runningWithMutexLockedPromise;
    std::future<nvigi::Result> job;
    std::mutex mtx;
    std::condition_variable cvInput;
    std::condition_variable cvDone;
    std::string input;
    const unsigned char* rgb_data;
    int rgb_width;
    int rgb_height;
    std::atomic<bool> runningChat = false;
    std::atomic<bool> newInput = false;
    std::atomic<bool> silenceOutput = false;
    nvigi::InferenceExecutionContext* execCtx;
};


using internalCallback = std::function<nvigi::InferenceExecutionState(nvigi::InferenceExecutionContext* execCtx, int32_t token, const std::string& response, nvigi::InferenceExecutionState state)>;

// Begin NVIDIA Modification
// Added common_params as an input parameter instead of a global static
static std::string chat_add_and_format(common_params* g_params, struct llama_model* model, std::vector<common_chat_msg>& chat_msgs, const std::string& role, const std::string& content) {
    common_chat_msg new_msg{ role, content };
    auto formatted = common_chat_format_single(model, g_params->chat_template, chat_msgs, new_msg, role == "user");
    chat_msgs.push_back({ role, content });
    LOG_DBG("formatted: '%s'\n", formatted.c_str());
    return formatted;
}
// End NVIDIA Modification

//! Local generate function, not needed when running on cloud
int generate(InferenceContextSync* sync, llama_model* model, llama_context* ctx, clip_ctx* clip_context, common_params& params, internalCallback callback)
{
    std::unique_lock lock(sync->mtx);

    if (sync->runningChat.load())
    {
        // In chat mode, signal that we are running and mutex is locked
        sync->runningWithMutexLockedPromise.set_value();
    }

    auto& sparams = params.sparams;

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    common_sampler* smpl = nullptr;

    std::vector<common_chat_msg> chat_msgs;

    // Begin NVIDIA Modification
    // Originally a LOG, but that was sending spurious output to the console, so changing to LOG_INF, which I think it should be anyways
    LOG_INF("n_ctx: %d\n", n_ctx);
    // End NVIDIA Modification

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }

    // print chat template example in conversation mode
    if (params.conversation) {
        if (params.enable_chat_template) {
            LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(model, params.chat_template).c_str());
        } else {
            LOG_INF("%s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
        }
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }
    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    // Begin NVIDIA Modification
    // Commenting out trans-session saving for now.
    //if (!path_session.empty()) {
    //    LOG_INF("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
    //    if (!file_exists(path_session)) {
    //        LOG_INF("%s: session file does not exist, will create.\n", __func__);
    //    } else if (file_is_empty(path_session)) {
    //        LOG_INF("%s: The session file is empty. A new session will be initialized.\n", __func__);
    //    } else {
    //        // The file exists and is not empty
    //        session_tokens.resize(n_ctx);
    //        size_t n_token_count_out = 0;
    //        if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
    //            LOG_ERR("%s: failed to load session file '%s'\n", __func__, path_session.c_str());
    //            return 1;
    //        }
    //        session_tokens.resize(n_token_count_out);
    //        LOG_INF("%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
    //    }
    //}
    // End NVIDIA Modification

    const bool add_bos = llama_add_bos_token(model);
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_add_eos_token(model));
    }

    LOG_DBG("n_ctx: %d, add_bos: %d\n", n_ctx, add_bos);

    std::vector<llama_token> embd_inp;

    {
        // Begin NVIDIA Modification
        // chat_add_and_format modified to take params
        auto prompt = (params.conversation && params.enable_chat_template && !params.prompt.empty())
            ? chat_add_and_format(&params, model, chat_msgs, "system", params.prompt) // format the system prompt in conversation mode
            : params.prompt;
        // End NVIDIA Modification

        if (params.interactive_first || !params.prompt.empty() || session_tokens.empty()) {
            LOG_DBG("tokenize the prompt\n");
            embd_inp = common_tokenize(ctx, prompt, true, true);
        } else {
            LOG_DBG("use session tokens\n");
            embd_inp = session_tokens;
        }

        LOG_DBG("prompt: \"%s\"\n", prompt.c_str());
        LOG_DBG("tokens: %s\n", string_from(ctx, embd_inp).c_str());
    }

    // Should not run without any tokens
    if (embd_inp.empty()) {
        if (add_bos) {
            embd_inp.push_back(llama_token_bos(model));
            LOG_WRN("embd_inp was considered empty and bos was added: %s\n", string_from(ctx, embd_inp).c_str());
        } else {
            LOG_ERR("input is empty\n");
            return -1;
        }
    }

    // Tokenize negative prompt
    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            LOG_INF("%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            LOG_INF("%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            LOG_WRN("%s: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            LOG_INF("%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }

        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_cache_seq_rm(ctx, -1, n_matching_session_tokens, -1);
    }

    LOG_DBG("recalculate the cached logits (check): embd_inp.size() %zu, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu\n",
         embd_inp.size(), n_matching_session_tokens, embd_inp.size(), session_tokens.size());

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        LOG_DBG("recalculate the cached logits (do): session_tokens.resize( %zu )\n", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }

    if (params.conversation) {
        params.interactive_first = true;
    }

    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG_INF("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", embd_inp[i], common_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (params.n_keep > add_bos) {
            LOG_INF("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_CNT("%s", common_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_CNT("'\n");
        }
        LOG_INF("\n");
    }

    // Begin NVIDIA Modification
    // disabling Ctrl+C handling
//    // ctrl+C handling
//    {
//#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
//        struct sigaction sigint_action;
//        sigint_action.sa_handler = sigint_handler;
//        sigemptyset (&sigint_action.sa_mask);
//        sigint_action.sa_flags = 0;
//        sigaction(SIGINT, &sigint_action, NULL);
//#elif defined (_WIN32)
//        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
//            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
//        };
//        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
//#endif
//    }
    // End NVIDIA Modification

    if (params.interactive) {
        LOG_INF("%s: interactive mode on.\n", __func__);

        if (!params.antiprompt.empty()) {
            for (const auto & antiprompt : params.antiprompt) {
                LOG_INF("Reverse prompt: '%s'\n", antiprompt.c_str());
                if (params.verbose_prompt) {
                    auto tmp = common_tokenize(ctx, antiprompt, false, true);
                    for (int i = 0; i < (int) tmp.size(); i++) {
                        LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                    }
                }
            }
        }

        if (params.input_prefix_bos) {
            LOG_INF("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_INF("Input prefix: '%s'\n", params.input_prefix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_prefix, true, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }

        if (!params.input_suffix.empty()) {
            LOG_INF("Input suffix: '%s'\n", params.input_suffix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_suffix, false, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }
    }

    smpl = common_sampler_init(model, sparams);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        return 1;
    }

    LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl));
    LOG_INF("sampler params: \n%s\n", sparams.print().c_str());
    LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl).c_str());

    LOG_INF("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
    int ga_i = 0;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
      //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
      //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG_INF("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_INF("\n");

    // Begin NVIDIA Modifications
    // Was a global static in original code, can be a local scoped variable here.
    bool is_interacting = false;
    bool need_insert_eot = false;
    // End NVIDIA Modifications

    if (params.interactive) {
        const char * control_message;
        if (params.multiline_input) {
            control_message = " - To return control to the AI, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to the AI.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_INF("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_INF(       " - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_INF(       "%s\n", control_message);

        is_interacting = params.interactive_first;
    }

    bool is_antiprompt        = false;
    // Begin NVIDIA Modifications
    // NVIGI doesn't echo back the input as the llama-cli does.
    bool input_echo           = false; 
    // End NVIDIA Modifications
    bool display              = true;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    // Begin NVIDIA Modifications
    // We don't need to store these tokens into global variables.
    std::vector<int>   input_tokens;
    std::vector<int>   output_tokens;
    std::ostringstream output_ss; 
    // End NVIDIA Modifications
    std::ostringstream assistant_ss; // for storing current assistant message, used in conversation mode

    // the first thing we will do is to output the prompt, so set color accordingly
    // Begin NVIDIA Modifications
    // NVIGI is not meant specifically for console application, so setting display modes right here is unnecessary
    // console::set_display(console::prompt);
    // End NVIDIA Modifications
    display = params.display_prompt;

    std::vector<llama_token> embd;

    // Begin NVIDIA Modifications
    // 
    // image_marker_embd is a marker that we will use to check where the image has been placed in the prompt.  It's a list of integers (token_ids)
    // 
    // In gpt.cpp, we replace "<image>" with " NVIGI_IMG ".  Note the spaces in our replacement string. 
    // This is to ensure that "NVIGI_IMG" _always_ tokenizes the same way regardless of what is before and after.
    // Problematic Examples just using <image>:
    // "<image>", "\n<image>" might have different tokenizations despite looking similar to a human
    // Examples:
    // "<image>" => "<", "image", ">"       => [131,   25231, 132]
    // "\n<image>" => "\n<", "image", ">".  => [31012, 25231, 132]
    // 
    // Despite looking the same to a human, this means using just "<image>" would not match the case of "\n<image>" even though it clearly should.
    // Even replacing "<image>", with " <image> ", might not be enough for consistent tokenization, as "\n <", might be common enough to have a distinct token from " <"
    // Further, I can't just check for "image" (25231, ignoring the angle brackets) as it's too common a word
    // 
    // Although this is less likely to happen with NVIGI_IMG, we surround it with known tokens we don't care to match (spaces in this case) to ensure tokenization is the same every time.
    // This allows us to do a quick integer comparison of the tokens to see when we match.
    // Now we don't care what precedes or succeeds <image>, as we will replace it with " NVIGI_IMG " and then only search for "NVIGI_IMG" (no spaces) to see where the marker is.
    // 
    // We tokenize here to do it once per generate call instead of once per input.
    std::vector<llama_token> image_marker_embd = common_tokenize(ctx, "NVIGI_IMG", false, true);
    // End NVIDIA Modifications

    // tokenized antiprompts
    std::vector<std::vector<llama_token>> antiprompt_ids;

    antiprompt_ids.reserve(params.antiprompt.size());
    for (const std::string & antiprompt : params.antiprompt) {
        antiprompt_ids.emplace_back(::common_tokenize(ctx, antiprompt, false, true));
    }

    if (llama_model_has_encoder(model)) {
        int enc_input_size = embd_inp.size();
        llama_token * enc_input_buf = embd_inp.data();

        if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == -1) {
            decoder_start_token_id = llama_token_bos(model);
        }

        embd_inp.clear();
        embd_inp.push_back(decoder_start_token_id);
    }

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (!embd.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                // Begin NVIDIA Modifications
                // NVIGI is not meant specifically for console application, so setting display modes right here is unnecessary
                // console::set_display(console::error);
                LOG_WRN("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                // console::set_display(console::reset);
                // End NVIDIA Modifications
            }

            if (ga_n == 1) {
                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches

                if (n_past + (int) embd.size() >= n_ctx) {
                    if (!params.ctx_shift){
                        LOG_DBG("\n\n%s: context full and context shift is disabled => stopping\n", __func__);
                        break;
                    }

                    if (params.n_predict == -2) {
                        LOG_DBG("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                        break;
                    }

                    const int n_left    = n_past - params.n_keep;
                    const int n_discard = n_left/2;

                    LOG_DBG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                            n_past, n_left, n_ctx, params.n_keep, n_discard);

                    llama_kv_cache_seq_rm (ctx, 0, params.n_keep            , params.n_keep + n_discard);
                    llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    LOG_DBG("after swap: n_past = %d\n", n_past);

                    LOG_DBG("embd: %s\n", string_from(ctx, embd).c_str());

                    LOG_DBG("clear session path\n");
                    path_session.clear();
                }
            } else {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    LOG_DBG("\n");
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib*bd, ga_i + ib*bd, n_past + ib*bd);
                    LOG_DBG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n, (ga_i + ib*bd)/ga_n, (ga_i + ib*bd + ga_w)/ga_n);
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib*bd + ga_w, n_past + ib*bd, dd, ga_i + ib*bd + ga_w + dd, n_past + ib*bd + dd);

                    llama_kv_cache_seq_add(ctx, 0, ga_i,                n_past,              ib*bd);
                    llama_kv_cache_seq_div(ctx, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                    llama_kv_cache_seq_add(ctx, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

                    n_past -= bd;

                    ga_i += ga_w/ga_n;

                    LOG_DBG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
                }
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

			// Begin NVIDIA Modification 
			// Breaking up the embd inst segments seperated by NVIGI_IMG.  NVIGI_IMGs are spliced into the 
			// tokenization

            std::vector<std::pair<int,size_t>> segments;

            int image_start_index = -1;
            int match_index = 0;

            if (sync->rgb_data != nullptr && clip_context != nullptr)
            {
                // look for our image marker in embd
                for (int i = 0; i < (int)embd.size(); i++) {
                    if (image_marker_embd[match_index] == embd[i])
                        match_index++;
                    else
                        match_index = 0;

                    if (match_index == image_marker_embd.size())
                    {
                        image_start_index = (i + 1) - (int)image_marker_embd.size();
                        int image_stop_index = image_start_index + (int)image_marker_embd.size();
                        
                        // add the text segment _before_ the image, only if text exists before the image marker.
                        if ( image_start_index > 0 )
                            segments.push_back(std::make_pair(0, image_start_index));

                        // add the image segment
                        segments.push_back(std::make_pair(image_start_index, image_marker_embd.size()));

                        // add the text segement _after_ the image, it text exists after the image marker
                        if (( embd.size() - image_stop_index ) > 0 )
                            segments.push_back(std::make_pair(image_stop_index, embd.size() - image_stop_index));

                        // we only handle one image per prompt right now, so break out once we've found this.
                        break;
                    }
                }
            }

            if (image_start_index == -1)
            {
                segments.push_back(std::make_pair(0, embd.size()));
            }

            for (int seg_idx = 0; seg_idx < segments.size(); seg_idx++)
            {
                std::pair<int,size_t> seg_start_size = segments[seg_idx];
                int seg_start_index = seg_start_size.first;
                size_t seg_size = seg_start_size.second;
                if (seg_start_index != image_start_index)
                {
                    for (int i = seg_start_index; i < seg_start_index + seg_size; i += params.n_batch) {
                        int n_eval = (int)seg_size - ( i - seg_start_index );
                        if (n_eval > params.n_batch) {
                            n_eval = params.n_batch;
                        }

                        LOG_DBG("eval: %s\n", string_from(ctx, embd).c_str());

                        if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
                            LOG_ERR("%s : failed to eval\n", __func__);
                            return 1;
                        }

                        n_past += n_eval;

                        LOG_DBG("n_past = %d\n", n_past);
                        // Display total tokens alongside total time
                        if (params.n_print > 0 && n_past % params.n_print == 0) {
                            LOG_DBG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                        }
                    }

                }
                else
                {
                    NVIGI_LOG_INFO("decoding image");

                    if (clip_context != nullptr && sync->rgb_data != nullptr && sync->rgb_height > 0 && sync->rgb_width > 0 )
                    {
                        llava_image_embed image_embed;
                        clip_image_u8* clip_img = clip_image_u8_init();
                        build_clip_img_from_data(sync->rgb_data, sync->rgb_width, sync->rgb_height, clip_img);

                        bool image_embed_result = llava_image_embed_make_with_clip_img(clip_context, params.cpuparams.n_threads, clip_img, &(image_embed.embed), &(image_embed.n_image_pos));
                        clip_image_u8_free(clip_img);
                        if (image_embed_result)
                        {
                            llava_eval_image_embed(ctx, &image_embed, params.n_batch, &n_past);
                            free(image_embed.embed);
                        }
                        // in the case of a failed embedding, we do nothing, so answers shouldn't adhere to any picture or will be hallucinated.
                    }
                }
            }
			// End NVIDIA Modification

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

                LOG_DBG("saved session to %s\n", path_session.c_str());
            }

            // Begin NVIDIA Comment
            // Commenting here as this is important - THIS LINE is where generation of new tokens takes place
            // This is incased in a giant while loop that starts a few hundred lines up, gets here, generates a token, then goes
            // through the while loop again, skipping most of it, to get back here to generate the next token.
            // 
            // When you think of "generate a next word" - this is the line that does it.
            // End NVIDIA Comment
            const llama_token id = common_sampler_sample(smpl, ctx, -1);

            common_sampler_accept(smpl, id, /* accept_grammar= */ true);

            // LOG_DBG("last: %s\n", string_from(ctx, smpl->prev.to_vector()).c_str());

            embd.push_back(id);

            // Begin NVIDIA Modifications
            // echo this to console  but only if not system prompt processing (setting up the chat)
            input_echo = !sync->silenceOutput.load();
            // End NVIDIA Modifications

            // decrement remaining sampling budget
            --n_remain;

            LOG_DBG("n_remain: %d\n", n_remain);
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                common_sampler_accept(smpl, embd_inp[n_consumed], /* accept_grammar= */ false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo && display) {
            for (auto id : embd) {
                const std::string token_str = common_token_to_piece(ctx, id, params.special);

                // Begin NVIDIA Modifications
                // Console/Stream Output
                //LOG("%s", token_str.c_str());
                if (input_echo)
                {
                    auto _res = callback(sync->execCtx, id, token_str, nvigi::kInferenceExecutionStateDataPending);
                    if (_res == nvigi::kInferenceExecutionStateCancel || _res == nvigi::kInferenceExecutionStateInvalid)
                    {
                        // This is actually OK, not our error
                        return 0;
                    }
                }
                // End NVIDIA Modifications

                // Record Displayed Tokens To Log
                // Note: Generated tokens are created one by one hence this check
                if (embd.size() > 1) {
                    // Incoming Requested Tokens
                    input_tokens.push_back(id);
                } else {
                    // Outgoing Generated Tokens
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
        }

        // reset color to default if there is no pending user input
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            // Begin NVIDIA Modifications
            // NVIGI is not meant specifically for console application, so setting display modes right here is unnecessary
            // console::set_display(console::reset);
            // End NVIDIA Modifications
            display = true;
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens
                llama_token last_token = common_sampler_last(smpl);
                for (std::vector<llama_token> ids : antiprompt_ids) {
                    if (ids.size() == 1 && last_token == ids[0]) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                if (is_antiprompt) {
                    LOG_DBG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            // deal with end of generation tokens in interactive mode
            if (llama_token_is_eog(model, common_sampler_last(smpl))) {
                LOG_DBG("found an EOG token\n");

                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = common_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    if (params.enable_chat_template) {
                        // Begin NVIDIA Modifications
                        // passing in params instead of using a global static
                        chat_add_and_format(&params, model, chat_msgs, "assistant", assistant_ss.str());
                        // End NVIDIA Modifications
                    }
                    is_interacting = true;
                    LOG("\n");
                }
            }

            // if current token is not EOG, we add it to current assistant message
            if (params.conversation) {
                const auto id = common_sampler_last(smpl);
                assistant_ss << common_token_to_piece(ctx, id, false);
            }

            if (n_past > 0 && is_interacting) {
                LOG_DBG("waiting for user input\n");

                if (params.conversation) {
                    LOG("\n> ");
                }

                if (params.input_prefix_bos) {
                    LOG_DBG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_token_bos(model));
                }

                std::string buffer;
                if (!params.input_prefix.empty() && !params.conversation) {
                    LOG_DBG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    LOG("%s", params.input_prefix.c_str());
                }

                // color user input only
                // Begin NVIDIA Modifications
                // NVIGI doesn't deal directly with the console.
                // console::set_display(console::user_input);
                // End NVIDIA Modifications

                display = params.display_prompt;

                // Begin NVIDIA Modifications
                // Custom code used to signal to the user when data is available.
                sync->cvDone.notify_one();

                // inform user that we are done with our turn
                auto _res = callback(sync->execCtx, 0, "", nvigi::kInferenceExecutionStateDone);
                if (_res == nvigi::kInferenceExecutionStateCancel || _res == nvigi::kInferenceExecutionStateInvalid)
                {
                    // This is OK, not an error, we got interrupted
                    return 0;
                }
                // unlock and wait for input from the user or cancellation
                sync->cvInput.wait(lock, [sync] {return !sync->runningChat.load() || sync->newInput.load(); });
                // regained lock again
                sync->newInput.store(false);
                if (!sync->runningChat.load())
                {
                    return 0;
                }
                // Any time we get new input, n_remain should be reset to avoid inconsistent response cutoffs
                n_remain = params.n_predict;
                // In chat mode, signal that we are running and mutex is locked
                sync->runningWithMutexLockedPromise.set_value();
                buffer = sync->input;
                sync->input.clear();
                // End NVIDIA Modifications

                // Begin NVIDIA Modifications
                // 
                // NVIGI doesn't deal with the console.
                //std::string line;
                //bool another_line = true;
                //do {
                //    another_line = console::readline(line, params.multiline_input);
                //    buffer += line;
                //} while (another_line);

                // done taking input, reset color
                // console::set_display(console::reset);
                // 
                // End NVIDIA Modifications
                display = true;

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    // append input suffix if any
                    if (!params.input_suffix.empty() && !params.conversation) {
                        LOG_DBG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        LOG("%s", params.input_suffix.c_str());
                    }

                    LOG_DBG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    if (params.escape) {
                        string_process_escapes(buffer);
                    }

                    bool format_chat = params.conversation && params.enable_chat_template;
                    // Begin NVIDIA Modifications
                    // chat_add_and_format takes an input params instead of using a global static.
                    std::string user_inp = format_chat
                        ? chat_add_and_format(&params, model, chat_msgs, "user", std::move(buffer))
                        : std::move(buffer);
                    // End NVIDIA Modifications

                    // TODO: one inconvenient of current chat template implementation is that we can't distinguish between user input and special tokens (prefix/postfix)
                    const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = common_tokenize(ctx, user_inp,            false, format_chat);
                    const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);

                    LOG_DBG("input tokens: %s\n", string_from(ctx, line_inp).c_str());

                    // if user stop generation mid-way, we must add EOT to finish model's last response
                    if (need_insert_eot && format_chat) {
                        llama_token eot = llama_token_eot(model);
                        embd_inp.push_back(eot == -1 ? llama_token_eos(model) : eot);
                        need_insert_eot = false;
                    }

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        output_tokens.push_back(token);
                        output_ss << common_token_to_piece(ctx, token);
                    }

                    // reset assistant message
                    assistant_ss.str("");

                    // Begin NVIDIA Modification
                    // This should be removed from TOT LlamaCPP, but right now, it makes no sense to subtract from n_remain (number of remaining prediction tokens) the lenght of the input.  Removing.
                    // n_remain -= line_inp.size();
                    // End NVIDIA Modification
                    LOG_DBG("n_remain: %d\n", n_remain);
                } else {
                    LOG_DBG("empty line, passing control back\n");
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0) {
                if (is_interacting) {
                    common_sampler_reset(smpl);
                }
                is_interacting = false;
            }
        }

        // end of generation
        if (!embd.empty() && llama_token_is_eog(model, embd.back()) && !(params.interactive)) {
            LOG(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    // Begin NVIDIA Modification
    // This clears the conversation cache.  This allows us to have different conversations.
    llama_kv_cache_clear(ctx);
    // End NVIDIA Modification

    // Begin NVIDIA Modification
    // Commenting out trans-session saving for now.
    //if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
    //    LOG("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
    //    llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    //}
    // End NVIDIA Modification

    // Begin NVIDIA Modification
    // Additional timing related storage for later retrieval
#ifndef NVIGI_PRODUCTION
    
    callback(sync->execCtx, 0, "", nvigi::kInferenceExecutionStateDone);

    //common_perf_print(ctx, smpl);
    const auto smpl_timings = llama_perf_sampler(smpl->chain);

    const auto timings = llama_perf_context(ctx);
    double t_end_ms = 1e-3 * ggml_time_us();

    NVIGI_LOG_INFO("timings:sample %s", extra::format("{} ms / {} runs ({} ms/token, {} tokens/second)", smpl_timings.t_sample_ms, smpl_timings.n_sample, smpl_timings.t_sample_ms / smpl_timings.n_sample, 1e3 / smpl_timings.t_sample_ms * smpl_timings.n_sample).c_str());
    NVIGI_LOG_INFO("timings:prompt %s", extra::format("{} ms / {} tokens ({} ms/token, {} tokens/second)", timings.t_p_eval_ms, timings.n_p_eval, timings.t_p_eval_ms / timings.n_p_eval, 1e3 / timings.t_p_eval_ms * timings.n_p_eval).c_str());
    NVIGI_LOG_INFO("timings:eval   %s", extra::format("{} ms / {} runs ({} ms/token, {} tokens/second)", timings.t_eval_ms, timings.n_eval, timings.t_eval_ms / timings.n_eval, 1e3 / timings.t_eval_ms * timings.n_eval).c_str());
    NVIGI_LOG_INFO("timings:total  %s", extra::format("{}ms", (t_end_ms - timings.t_start_ms)).c_str());
#else
    callback(sync->execCtx, 0, "", nvigi::kInferenceExecutionStateDone);
#endif
    // End NVIDIA Modification

    // Begin NVIDIA Modification
    // Sampler has to die here, as perf reporting requires it earlier.  We create/delete every generate as runtime parameters could theoretically 
    // change the sampler parameters per evaluate call so we can't store sampler as an instance parameter.  
    // See gpt.cpp ggmlEvaluate runtimeParameters
    common_sampler_free(smpl);

    // We free the context and model in ggmlDestroyInstance instead.
    // backend is freed in nvigiPluginDeregister
    // nvigi doesn't use ggml_threadpools
    //llama_free(ctx);
    //llama_free_model(model);
    // 
    //llama_backend_free();
    //
    //ggml_threadpool_free(threadpool);
    //ggml_threadpool_free(threadpool_batch);

    // End NVIDIA Modification

    return 0;
}
#endif

}
}