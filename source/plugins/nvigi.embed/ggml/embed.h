// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

// BEGIN NVIDIA EDITORS NOTES
// 
// Anyone modifying this file, pay special and take care to how it is modified.
// This is _mostly_ a block copy of the LlamaCPP file embedding.cpp located in the llama-embedding project.
// Most of this example in llama-embedding embedding.cpp is in the main method.  Some code was seperated out
// like tokenize to make it more viable in a real product.  Still attempts are made to keep the same variable names
// spacing, layout, etc, when possible, for easier merging in the future.
// 
// In the interests of making future merges easier, if any modifications are needed, wrap them in
// // Begin NVIDIA Modification
// ...modifications...
// // End NVIDIA Modification
// This should help ensure functionality is not lost or accidentally broken during future code updates.
//
// END NVIDIA EDITORS NOTES

#pragma once

#include <functional>

#ifdef GGML_USE_CUBLA
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

#include "common.h"
#include "llama.h"
//#include "log.h"
#include "external/json/source/nlohmann/json.hpp"
#include <ctime>

#ifdef GGML_USE_CUBLAS
#include "source/core/nvigi.api/nvigi_cuda.h"
#include "ggml-cuda.h"
#endif

using json = nlohmann::json;
//! Unfortunately GGML gpt_params structure requires this method so reimplementing
int32_t get_num_physical_cores()
{
	unsigned int n_threads = std::thread::hardware_concurrency();
	return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

namespace nvigi
{
	namespace embed
	{
		// Begin NVIDIA Modification
		// Memory callbacks
#ifdef GGML_USE_CUBLAS
		void setCudaMallocReportCallback(PFun_nvigiCudaReportCallback callback, void* userContext)
		{
			ggml_backend_cuda_set_malloc_report_callback(callback, userContext);
		}

		void setCudaFreeReportCallback(PFun_nvigiCudaReportCallback callback, void* userContext)
		{
			ggml_backend_cuda_set_free_report_callback(callback, userContext);
		}

		void setCudaMallocCallback(PFun_nvigiCudaMallocCallback callback, void* userContext)
		{
			ggml_backend_cuda_set_malloc_callback(callback, userContext);
		}

		void setCudaFreeCallback(PFun_nvigiCudaFreeCallback callback, void* userContext)
		{
			ggml_backend_cuda_set_free_callback(callback, userContext);
		}
#endif // GGML_USE_CUBLAS
		// End NVIDIA Modification

		// Begin NVIDIA Modification
		// NVidia custom functions to sanitize input data.

		// Check if a character is a valid ASCII character
		static bool isValidASCII(char ch) {
			return (ch >= 0 && ch <= 127); // ASCII range (valid UTF-8 single byte)
		}

		// Check if a string contains non-UTF-8 characters
		static bool containsNonUTF8(const std::string& input) {
			std::string output;
			for (char ch : input) {
				// If non-UTF-8 characters we return true
				if (!isValidASCII(ch)) {
					return true;
				}
			}
			return false;
		}

		// End NVIDIA Modification

		// Used to separate prompts inside a same string
		static std::vector<std::string> split_lines(const std::string& s, const std::string& separator = "\n") {
			std::vector<std::string> lines;
			size_t start = 0;
			size_t end = s.find(separator);

			while (end != std::string::npos) {
				lines.push_back(s.substr(start, end - start));
				start = end + separator.length();
				end = s.find(separator, start);
			}

			lines.push_back(s.substr(start)); // Add the last part

			return lines;
		}

		// Add all tokens of a sequence to the batch
		static void batch_add_seq(llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id) {
			size_t n_tokens = tokens.size();
			for (size_t i = 0; i < n_tokens; i++) {
				common_batch_add(batch, tokens[i], i, { seq_id }, true);
			}
		}

		// Go from tokens to embedding
		static void batch_decode(llama_context* ctx, llama_batch& batch, float* output, int n_seq, int n_embd, int embd_norm) {
			const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
			const struct llama_model* model = llama_get_model(ctx);

			// clear previous kv_cache values (irrelevant for embeddings)
			llama_kv_self_clear(ctx);

			// run model
			LOG_INF("%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
			if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
				// encoder-only model
				if (llama_encode(ctx, batch) < 0) {
					LOG_ERR("%s : failed to encode\n", __func__);
				}
			} else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
				// decoder-only model
				if (llama_decode(ctx, batch) < 0) {
					LOG_ERR("%s : failed to decode\n", __func__);
				}
			}

			for (int i = 0; i < batch.n_tokens; i++) {
				if (!batch.logits[i]) {
					continue;
				}

				const float* embd = nullptr;
				int embd_pos = 0;

				if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
					// try to get token embeddings
					embd = llama_get_embeddings_ith(ctx, i);
					embd_pos = i;
					GGML_ASSERT(embd != NULL && "failed to get token embeddings");
				} else {
					// try to get sequence embeddings - supported only when pooling_type is not NONE
					embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
					embd_pos = batch.seq_id[i][0];
					GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
				}

				float* out = output + embd_pos * n_embd;
				common_embd_normalize(embd, out, n_embd, embd_norm);
			}
		}

		// Apply tokenizer to each prompt
		nvigi::Result tokenize(llama_context* ctx, llama_model* model, common_params& params, std::vector<std::vector<int32_t>> &inputs) {
			// split the prompt into lines
			std::vector<std::string> prompts = split_lines(params.prompt, params.embd_sep);

			// max batch size
			const uint64_t n_batch = params.n_batch;
			GGML_ASSERT(params.n_batch >= params.n_ctx);

			// tokenize the prompts and trim
			// Begin NVIDIA Modifications
			// Our inputs are passed in, so we don't need a local variable here.  
			// The for loop does some additional sanitization not found in the original code (i.e. containsNonUTF8 )
			//std::vector<std::vector<int32_t>> inputs;
			for (size_t i_prompt = 0; i_prompt < prompts.size(); ++i_prompt) {
				const auto& prompt = prompts[i_prompt];
				if (containsNonUTF8(prompt))
				{
					NVIGI_LOG_ERROR("Prompt %d contains non utf8 character", i_prompt);
					return kResultNonUtf8;
				}

				auto inp = common_tokenize(ctx, prompt, true, true);
				if (inp.size() > n_batch) {
					LOG_ERR("%s: number of tokens in input line (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                            __func__, (long long int) inp.size(), (long long int) n_batch);
					return nvigi::kResultMaxTokensReached;
				}

				inputs.push_back(inp);
			}
			// End NVIDIA Modifications

			// check if the last token is SEP
			// it should be automatically added by the tokenizer when 'tokenizer.ggml.add_eos_token' is set to 'true'
			for (auto& inp : inputs) {
				//if (inp.empty() || inp.back() != llama_token_sep(model)) 
				{
					LOG_WRN("%s: last token in the prompt is not SEP\n", __func__);
					LOG_WRN("%s: 'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header\n", __func__);
				}
			}

			// tokenization stats
			if (params.verbose_prompt) {
				for (int i = 0; i < (int)inputs.size(); i++) {
					LOG_INF("%s: prompt %d: '%s'\n", __func__, i, prompts[i].c_str());
					LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, inputs[i].size());
					for (int j = 0; j < (int)inputs[i].size(); j++) {
						LOG("%6d -> '%s'\n", inputs[i][j], common_token_to_piece(ctx, inputs[i][j]).c_str());
					}
					LOG("\n\n");
				}
			}

			return kResultOk;
		}

		nvigi::Result embed(llama_context* ctx, llama_model* model, common_params& params, std::vector<float>& embeddings)
		{
			params.embedding = true;
			// For non-causal models, batch size must be equal to ubatch size
			params.n_ubatch = params.n_batch;

			//print_build_info();

			if (params.sampling.seed == LLAMA_DEFAULT_SEED) {
				params.sampling.seed = static_cast<decltype(params.sampling.seed)>(time(NULL));
			}

			LOG_INF("seed  = %u", params.sparams.seed);

			std::mt19937 rng(params.sampling.seed);

			// load the model
			if (model == NULL) {
				LOG_ERR("%s: unable to load model\n", __func__);
				return kResultInvalidParameter;
			}

			const int n_ctx_train = llama_model_n_ctx_train(model);
			const int n_ctx = llama_n_ctx(ctx);

			const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

			if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
				LOG_ERR("%s: computing embeddings in encoder-decoder models is not supported\n", __func__);
			}

			if (n_ctx > n_ctx_train) {
				LOG_WRN("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                         __func__, n_ctx_train, n_ctx);
			}

			// print system information
			{
				LOG_INF("\n");
				LOG_INF("%s\n", common_params_get_system_info(params).c_str());
			}

			// max batch size
			const int32_t n_batch = params.n_batch;
			GGML_ASSERT(params.n_batch >= params.n_ctx);

			// tokenize the prompts and trim
			std::vector<std::vector<int32_t>> inputs;
			nvigi::Result res = tokenize(ctx, model, params, inputs);
			if (res != kResultOk)
				return res;

			// initialize batch
			// Begin NVIDIA Modification
			// original code was assigning size_t to an int, causing a compiler warning.  this fixes that.
			const int n_prompts = static_cast<int>(inputs.size());
			// End NVIDIA Modification
			struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

			// count number of embeddings
			int n_embd_count = 0;
			if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
				for (int k = 0; k < n_prompts; k++) {
					n_embd_count += inputs[k].size();
				}
			} else {
				n_embd_count = n_prompts;
			}

			// allocate output
			const int n_embd = llama_model_n_embd(model);
			// Begin NVIDIA Modification
			// we take embedding as input, compared to LlamaCPP embedding.cpp which creates a local here.
			embeddings.resize(n_embd_count * n_embd);
			// End NVIDIA Modification
			float* emb = embeddings.data();

			// break into batches
			int e = 0; // number of embeddings already stored
			int s = 0; // number of prompts in current batch
			for (int k = 0; k < n_prompts; k++) {
				// clamp to n_batch tokens
				auto& inp = inputs[k];

				const uint64_t n_toks = inp.size();

				// encode if at capacity
				if (batch.n_tokens + n_toks > n_batch) {
					float* out = emb + e * n_embd;
					batch_decode(ctx, batch, out, s, n_embd, params.embd_normalize);
					e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
					s = 0;
					common_batch_clear(batch);
				}

				// add to batch
				batch_add_seq(batch, inp, s);
				s += 1;
			}

			// final batch
			float* out = emb + e * n_embd;
			batch_decode(ctx, batch, out, s, n_embd, params.embd_normalize);
			llama_batch_free(batch);
			// Begin NVIDIA Modifications
			// We free the context and model in ggmlDestroyInstance instead.
			// backend is freed in nvigiPluginDeregister
			// llama_free(ctx);
			// llama_free_model(model);
			// llama_backend_free();
			// End NVIDIA Modification

			return kResultOk;
		}

		int get_embed_size(llama_model* model)
		{
			if (model == nullptr)
				return 0;
			else
				return llama_model_n_embd(model);
		}

	}

}