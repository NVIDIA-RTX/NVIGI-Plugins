// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "external/json/source/nlohmann/json.hpp"
#include "external_trt/TRTModel.h"
#include "source/plugins/nvigi.tts/asqflow/trt/language_normalizer.h"
#include <cstdint>
#include <cuda_fp16.h>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <nvigi_result.h>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace nvigi
{
namespace asqflow
{

enum class InferenceState
{
    RUNNING,
    FINISHED,
    ERROR_HAPPENED
};

using InternalCallback =
    std::function<void(const std::vector<int16_t> &audio, const std::string textNormalized, InferenceState state)>;

struct InferenceParameters
{
    std::string textPrompt;
    std::string transcriptTarget;
    std::filesystem::path targetSpectrogramPath;
    float speed = 1.0;
    int minSizeChunk = 100;  // Runtime parameter for minimum chunk size
    int maxSizeChunk = 200;  // Runtime parameter for maximum chunk size
    int seed = -725171668;   // Runtime parameter for random seed
    std::atomic<bool>* cancelled = nullptr;  // Pointer to cancellation flag for early termination
};

struct CreationParameters
{
    bool warmUpModels = true;
	std::string extendedPhonemesDictPath = "";
};

class ASqFlow
{

  public:
    /**
     * Constructs an instance of the ASqFlow class.
     *
     * @param dpModelPath The path to the duration predictor model.
     * @param generatorModelPath The path to the generator model.
     * @param vocoderModelPath The path to the vocoder model.
     * @param g2pModelPath The path to the grapheme-to-phoneme model.
     * @param cmuDictPath The path to the CMU pronunciation dictionary.
     * @param configData The configuration data.
     * @param params The creation parameters.
     *
     * @throws ErrorType If an error occurs during construction.
     */
    ASqFlow(const std::string &dpModelPath, const std::string &generatorModelPath, const std::string &vocoderModelPath,
           const std::string &g2pModelPath, const std::string &phonemesDictPathDefault, const json &configData,
           const CreationParameters &params);
    ~ASqFlow();

    /** Evaluates the ASqFlow model. Do all the necessary steps :
        - Normalize text
        - Split Sentences
        - Grapheme to phonemes
        - Get phonemes ids
        - Run duration predictor model
        - Run generator model
        - Run vocodder model
        Output audio is always returned at a sampling rate of 22050
    */
    nvigi::Result evaluate(const InferenceParameters &inferenceParams, InternalCallback callback, cudaStream_t cudaStream);

    /** Initializes the timers which stores execution time for all the steps. */
    void initializeTimers();
    std::map<std::string, long long> timers; // store time taken at each step

    int32_t getMaxAuxStreamCount();
    void setAuxStreams(cudaStream_t* streams, size_t streamCount);

  private:
    std::unique_ptr<nvigi::asqflow::LanguageNormalizer> textNormalizer;
    const json &configData;

    TRTModel dpModel;
    TRTModel generatorModel;
    TRTModel vocoderModel;

    unsigned int numFeaturesSpectrogram = 80;
    const int nTimeSteps = 32;
    // we notice that the beggning of each audio is just a weird sound, most probablue due to the target mel
    // spectrogram. We ignore the first nMelIgnors values
    int nMelIgnores = 0;
    // Denormalizing mel spectrogram parameters
    const float melOffset = float(-5.884);
    const float melScale = float(2.261);

    nvigi::Result runDpModel(std::vector<int> &phonemesEncoding, std::vector<__half> &melSpectogramRef,
                             unsigned int &outputDpVal, cudaStream_t cudaStream);
    nvigi::Result runGeneratorModel(std::vector<int> &phonemesEncoding, std::vector<__half> &melSpectogramRef,
                                    unsigned int &yDpPrediction, std::vector<__half> &outputMelSpectrogram, 
                                    cudaStream_t cudaStream, int seed = -725171668);
    nvigi::Result runVocoderModel(std::vector<__half> &melSpectrogram, std::vector<int16_t> &outputAudio, cudaStream_t cudaStream);
};

} // namespace asqflow
} // namespace nvigi
