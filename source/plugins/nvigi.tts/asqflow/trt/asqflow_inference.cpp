// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "source/plugins/nvigi.tts/asqflow/trt/asqflow_inference.h"
#include "external/json/source/nlohmann/json.hpp"
#include "external_trt/TRTTensor.h"
#include "language_normalizer.h"
#include "nvigi_core/source/core/nvigi.log/log.h"
#include "onnxruntime_cxx_api.h"
#include "source/plugins/nvigi.tts/nvigi_tts.h"
#include <NvInferRuntimeBase.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define CHECK(expr)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(expr))                                                                                                   \
        {                                                                                                              \
            std::cerr << "Error: CHECK failed in " << __FILE__ << " at line " << __LINE__ << ": " << #expr             \
                      << " returned false." << std::endl;                                                              \
        }                                                                                                              \
    } while (0)

using shape_vec_t = std::vector<size_t>;
using shape_map_t = std::unordered_map<std::string, shape_vec_t>;
using buffer_map_t = std::unordered_map<std::string, void *>;

void storeExecutionTime(std::map<std::string, long long> &timers,
                        const std::chrono::time_point<std::chrono::high_resolution_clock> &start,
                        const std::chrono::time_point<std::chrono::high_resolution_clock> &end,
                        const std::string &stepName)
{

    if (timers.find(stepName) != timers.end())
    {
        timers[stepName] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    else
    {
        timers[stepName] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
}

/**
 * Reads a spectrogram from a file and converts it to a vector of __half.
 *
 * @param pathSpectrogram The path to the spectrogram file.
 * @param outputSpectrogram The vector to store the spectrogram.
 */

void readSpectrogram(const std::string pathSpectrogram, std::vector<__half> &outputSpectrogram)
{
    // Open the file in binary mode
    std::ifstream inputFile(pathSpectrogram, std::ios::binary);

    // Check if the file is open
    if (!inputFile.is_open())
    {
        NVIGI_LOG_ERROR("Failed to open the spectrogram file : '%s'", pathSpectrogram.c_str());
        return;
    }

    // Go to the end of the file to determine its size
    inputFile.seekg(0, std::ios::end);
    std::streamsize file_size = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    // Calculate the number of float elements in the file
    std::size_t num_elements = file_size / sizeof(float);

    outputSpectrogram.clear();
    outputSpectrogram.reserve(num_elements);

    std::vector<float> outputSpectrogramFloat;
    outputSpectrogramFloat.resize(num_elements);
    // Read the file data into the vector
    if (!inputFile.read(reinterpret_cast<char *>(outputSpectrogramFloat.data()), file_size))
    {
        NVIGI_LOG_ERROR("Error reading the file '%s'", pathSpectrogram.c_str());
        return;
    }

    // Convert to float16
    std::transform(std::begin(outputSpectrogramFloat), std::end(outputSpectrogramFloat),
                   std::back_inserter(outputSpectrogram), [](float value) { return __float2half(value); });

    // Close the file
    inputFile.close();
}

namespace nvigi
{

namespace asqflow
{

ASqFlow::ASqFlow(const std::string &dpModelPath, const std::string &generatorModelPath,
               const std::string &vocoderModelPath, const std::string &g2pModelPath, const std::string &phonemesDictPathDefault,
               const json &configData, const CreationParameters &params)
    : textNormalizer(std::make_unique<EnglishNormalizer>(g2pModelPath, phonemesDictPathDefault, configData, params.extendedPhonemesDictPath)),
      configData(configData)
{

    // Due to challenges in getting the DP model to work with TRT, 
    // it has been replaced with a straightforward formula that provides results comparable to the model.
    //NVIGI_LOG_INFO("Loading Duration Predictor model ...");
    //dpModel.init(dpModelPath);

    NVIGI_LOG_INFO("Loading generator model ...");
    generatorModel.init(generatorModelPath);

    NVIGI_LOG_INFO("Loading vocoder model ...");
    vocoderModel.init(vocoderModelPath);

    if (configData.contains("n_feat_spectrogram"))
    {
        numFeaturesSpectrogram = (float)configData["n_feat_spectrogram"];
    }
}

ASqFlow::~ASqFlow()
{
}

void ASqFlow::initializeTimers()
{
    timers.clear();
}

/**
 * Run the duration predictor model
 * This model takes as input :
 * - phonemes encoding
 * - Mel spectrogram of the target speaker
 *
 * Note : This model is not used anymore, we replaced it with a simple formula
 */
nvigi::Result ASqFlow::runDpModel(std::vector<int> &phonemesEncoding, std::vector<__half> &melSpectogramRef,
                                 unsigned int &outputDpVal, cudaStream_t cudaStream)
{

    const char *inputNames[] = {"x_dp", "p", "language_id"};

    int languageId = 3; // en_us

    // Define shapes
    std::vector<int> nbPhonemes = {(int)phonemesEncoding.size()}; // x_tst_lenghts
    shape_vec_t phonemesDims = {1, phonemesEncoding.size()};
    unsigned int melSpectrogramLen = melSpectogramRef.size() / numFeaturesSpectrogram;
    shape_vec_t melSpectrogramDims = {1, numFeaturesSpectrogram, melSpectrogramLen};
    shape_vec_t scalarsDim = {1};
    shape_map_t inputShapeMap = {{"x_dp", phonemesDims}, {"p", melSpectrogramDims}, {"language_id", scalarsDim}};

    // device tensors for TRT inference

    // Phones tensor
    auto phonemesTensor = TRTTensor(phonemesEncoding.data(), // initialized with host ptr
                                    phonemesDims, nvinfer1::DataType::kINT32);

    // mel spectrogram tensor
    auto melSpectrogramTensor = TRTTensor(melSpectogramRef.data(), // initialized with host ptr
                                          melSpectrogramDims, nvinfer1::DataType::kHALF);

    // language_id tensor
    auto languageIdTensor = TRTTensor(&languageId, // initialized with host ptr
                                      scalarsDim, nvinfer1::DataType::kINT32);

    auto outputTensor = TRTTensor(scalarsDim, nvinfer1::DataType::kINT32);

    buffer_map_t ioBufferMap = {{"x_dp", phonemesTensor.getDevicePtr()},
                                {"p", melSpectrogramTensor.getDevicePtr()},
                                {"language_id", languageIdTensor.getDevicePtr()},
                                {"y_lengths_dp", outputTensor.getDevicePtr()}};

    // TRT inference
    try
    {
        // inference setup: set input shapes & IO buffer address
        CHECK(dpModel.setInputShapes(inputShapeMap));
        CHECK(dpModel.setIOBuffers(ioBufferMap));

        // synchronous inference
        CHECK(dpModel.execute(cudaStream));

        // copy output to host and dump as npy array

        CHECK(outputTensor.copyToHost(&outputDpVal));
        NVIGI_LOG_INFO("DP predicted : %d", outputDpVal);
    }
    catch (const std::exception &e)
    {
        NVIGI_LOG_ERROR("Error while doing duration predictor inference: %s ", e.what());
        return kResultErrorDpModel;
    }
    catch (...)
    {
        NVIGI_LOG_ERROR("Error while doing inference");
        return kResultErrorDpModel;
    }

    return kResultOk;
}

/**
* Run inference for the generator model which predicts output mel spectrogram
* The model takes as input :
* - Phonemes ids (x)
* - reference mel spectogram (p)
* - Prediction of the duration predictor model (y_dp)
* - the language ids (lang_ids)

*/
nvigi::Result ASqFlow::runGeneratorModel(std::vector<int> &phonemesEncoding, std::vector<__half> &melSpectogramRef,
                                        unsigned int &yDpPrediction, std::vector<__half> &outputMelSpectrogram,
                                        cudaStream_t cudaStream, int seed)
{

    const char *inputNames[] = {"x", "p", "y", "language_id", "i_timestep"};
    unsigned int languageId = 3; // en_us
    unsigned int iTimeStep = 0;
    unsigned int melSpectrogramRefLen = (int)(melSpectogramRef.size() / numFeaturesSpectrogram);

    // initialize y with random value of size 1*n_feats*yDpPrediction
    // std::random_device rd;
    // std::mt19937 gen(rd());
    
    // Use random seed if seed is -1, otherwise use the provided seed
    std::mt19937 gen;
    if (seed == -1) {
        std::random_device rd;
        gen.seed(rd());
        NVIGI_LOG_INFO("Using random seed for generator model");
    } else {
        gen.seed(seed);
        NVIGI_LOG_INFO("Using fixed seed %d for generator model", seed);
    }
    
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<__half> y(1 * numFeaturesSpectrogram * (yDpPrediction + melSpectrogramRefLen));
    // Fill vector with random normal values
    for (auto &val : y)
    {
        val = __float2half(dist(gen));
    }

    // Define shapes
    std::vector<int> nbPhonemes = {(int)phonemesEncoding.size()}; // x_tst_lenghts
    shape_vec_t phonemesDims = {1, phonemesEncoding.size()};
    shape_vec_t melSpectrogramDims = {1, numFeaturesSpectrogram, melSpectrogramRefLen};
    shape_vec_t yInDim = {1, numFeaturesSpectrogram, (yDpPrediction + melSpectrogramRefLen)};
    shape_vec_t scalarsDim = {1};
    shape_map_t inputShapeMap = {
        {"x", phonemesDims},         {"p", melSpectrogramDims},  {"y", yInDim},
        {"language_id", scalarsDim}, {"i_timestep", scalarsDim},

    };

    // device tensors for TRT inference

    // Phones tensor
    auto phonemesTensor = TRTTensor(phonemesEncoding.data(), // initialized with host ptr
                                    phonemesDims, nvinfer1::DataType::kINT32);

    auto melSpectrogramRefTensor = TRTTensor(melSpectogramRef.data(), // initialized with host ptr
                                             melSpectrogramDims, nvinfer1::DataType::kHALF);

    // Y int tensor
    auto yInTensor = TRTTensor(y.data(), // initialized with host ptr
                               yInDim, nvinfer1::DataType::kHALF);

    // language_id tensor
    auto languageIdTensor = TRTTensor(&languageId, // initialized with host ptr
                                      scalarsDim, nvinfer1::DataType::kINT32);

    outputMelSpectrogram.resize(numFeaturesSpectrogram * (yDpPrediction - nMelIgnores));
    std::vector<__half> yOut((yDpPrediction + melSpectrogramRefLen) * numFeaturesSpectrogram);

    // yout tensor
    auto yOutTensor = TRTTensor(yOut.data(), // initialized with host ptr
                                yInDim, nvinfer1::DataType::kHALF);

    // TRT inference
    try
    {
        CHECK(generatorModel.setInputShapes(inputShapeMap));
        // Prepare output names and execute inference
        for (int i = 0; i < nTimeSteps; i++)
        {
            // update iTimeStep
            iTimeStep = i;

            // i_time_step
            auto iTimeStepTensor = TRTTensor(&iTimeStep, // initialized with host ptr
                                             scalarsDim, nvinfer1::DataType::kINT32);

            buffer_map_t ioBufferMap = {{"x", phonemesTensor.getDevicePtr()},
                                        {"p", melSpectrogramRefTensor.getDevicePtr()},
                                        {"y", yInTensor.getDevicePtr()},
                                        {"language_id", languageIdTensor.getDevicePtr()},
                                        {"i_timestep", iTimeStepTensor.getDevicePtr()},
                                        {"output", yOutTensor.getDevicePtr()}};
            CHECK(generatorModel.setIOBuffers(ioBufferMap));

            // synchronous inference
            CHECK(generatorModel.execute(cudaStream));

            CHECK(yOutTensor.copyDeviceToDevice(yInTensor.getDevicePtr()));
        }

        // copy output to host
        CHECK(yOutTensor.copyToHost(yOut.data()));

        unsigned int indexOutputMel = 0;
        unsigned int yLen = yDpPrediction + melSpectrogramRefLen;
        for (int i = 0; i < y.size(); i++)
        {

            // The first `melSpectrogramRefLen` elements correspond to the reference Mel spectrogram.
            // It is observed that the beginning of each audio contains unusual noise, likely caused by the target Mel
            // spectrogram. To address this, we ignore the first `nMelIgnore` values, where `nMelIgnore` is determined
            // empirically.
            if ((i % yLen) < (melSpectrogramRefLen + nMelIgnores))
                continue;

            // We denormalize the mel spectrogram
            outputMelSpectrogram[indexOutputMel++] = yOut[i];
        }
    }
    catch (const std::exception &e)
    {
        NVIGI_LOG_ERROR("Error while doing generator inference: %s ", e.what());
        return kResultErrorDpModel;
    }
    catch (...)
    {
        NVIGI_LOG_ERROR("Error while doing generator inference");
        return kResultErrorDpModel;
    }

    return kResultOk;
}

/**
 * Vocoder model in charge of converting mel spectrogram to audio
 * Sampling rate output will be set to 22050Hz
 */
nvigi::Result ASqFlow::runVocoderModel(std::vector<__half> &melSpectrogram, std::vector<int16_t> &outputAudio, cudaStream_t cudaStream)
{

    const char *inputNames[] = {"audio_buffer"};

    if (melSpectrogram.size() % numFeaturesSpectrogram != 0)
    {
        NVIGI_LOG_ERROR("Mel spectrogram size should be multiple of numFeaturesSpectrogram");
        return kResultInvalidParameter;
    }
    shape_vec_t melSpectrogramDims = {1, numFeaturesSpectrogram, (int)melSpectrogram.size() / numFeaturesSpectrogram};
    // vocoder model is composed of 2 upsampling x4 and 4 upsampling x2 layers
    shape_vec_t outputShape = {1, 1, melSpectrogramDims[2] * (4 * 4) * (2 * 2 * 2 * 2)};

    shape_map_t inputShapeMap = {{"audio_buffer", melSpectrogramDims}

    };

    auto melSpectrogramTensor = TRTTensor(melSpectrogram.data(), // initialized with host ptr
                                          melSpectrogramDims, nvinfer1::DataType::kHALF);
    vocoderModel.setInputShapes(inputShapeMap);

    std::vector<__half> outputAudioFloat16(outputShape[2]);
    auto outputAudioTensor = TRTTensor(outputAudioFloat16.data(), // initialized with host ptr
                                       outputShape, nvinfer1::DataType::kHALF);

    buffer_map_t ioBufferMap = {{"audio_buffer", melSpectrogramTensor.getDevicePtr()},
                                {"output_signal", outputAudioTensor.getDevicePtr()}};

    // ONNX inference
    try
    {

        CHECK(vocoderModel.setIOBuffers(ioBufferMap));

        // synchronous inference
        CHECK(vocoderModel.execute(cudaStream));
        outputAudioTensor.copyToHost(outputAudioFloat16.data());

        outputAudio.clear();
        outputAudio.resize(outputAudioFloat16.size());
        // Convert to int16
        for (int i = 0; i < outputAudioFloat16.size(); i++)
        {
            const float &value = __half2float(outputAudioFloat16[i]);
            outputAudio[i] = ((int16_t)(value * 32767));
        }
    }
    catch (const std::exception &e)
    {
        NVIGI_LOG_ERROR("Error while doing vocoder inference: %s ", e.what());
        return kResultErrorDpModel;
    }
    catch (...)
    {
        NVIGI_LOG_ERROR("Error while doing vocoder inference");
        return kResultErrorDpModel;
    }
    return kResultOk;
}

/** ASqFlow inference inference
Steps :
 - Normalize text
 - Split Sentences
 - Grapheme to phonemes
 - Get phonemes ids
 - Run duration predictor model
 - Run generator model
 - Run vocodder model
Output audio is always returned at a sampling rate of 22050
*/
nvigi::Result ASqFlow::evaluate(const InferenceParameters &inferParams, InternalCallback callback, cudaStream_t cudaStream)
{
    auto start_total = std::chrono::high_resolution_clock::now();
    std::vector<int16_t> outputAudio;
    std::vector<__half> spectrogramTarget;

    // Check if the target spectrogram of this speaker is present
    if (!std::filesystem::exists(inferParams.targetSpectrogramPath))
    {
        NVIGI_LOG_ERROR("Target spectrogram has not be find at '%s",
                        inferParams.targetSpectrogramPath.string().c_str());
        callback(outputAudio, "", InferenceState::ERROR_HAPPENED);
        return kResultInvalidParameter;
    }
    // Read spectrograms
    readSpectrogram(inferParams.targetSpectrogramPath.string(), spectrogramTarget);
    unsigned int spectrogramLength = (spectrogramTarget.size() / numFeaturesSpectrogram);

    auto start = std::chrono::high_resolution_clock::now();

    // Normalize text + split texts
    std::string textNormalized;
    auto res = textNormalizer->normalizeText(inferParams.textPrompt, textNormalized);
    if (res != kResultOk)
    {
        callback(outputAudio, textNormalized, InferenceState::ERROR_HAPPENED);
        return res;
    }
    if (textNormalized.empty())
    {
        callback(outputAudio, textNormalized, InferenceState::FINISHED);
        return kResultOk;
    }
    // std::string textNormalizedTranscript = textNormalizer->normalizeText(inferParams.transcriptTarget);
    //  we assume that the trancription is already normalized
    std::string textNormalizedTranscript = inferParams.transcriptTarget;
    std::vector<std::string> sentencesSplit = textNormalizer->splitSentences(textNormalized, inferParams.minSizeChunk, inferParams.maxSizeChunk);
    auto end = std::chrono::high_resolution_clock::now();
    storeExecutionTime(timers, start, end, "Preprocessing step");

    // Process each chunk : G2p + Generator + Vocodor and send data to callback
    std::vector<float> outputBaseAudio;
    start = std::chrono::high_resolution_clock::now();
    int indexPiece = 0; 
    std::vector<std::wstring> full_phones;
    for (const std::string &textPieceNormalized : sentencesSplit)
    {

        std::vector<std::wstring> phones;
        std::vector<int> inputNetPhonesTextToGen;
        std::vector<int> inputNetPhonesTextToGenAndTranscript;
        unsigned int dpVal;
        size_t sizePhonemesTarget = 0;

        outputAudio.clear();

        try
        {
            // Graph to phonemes
            auto start_g2p = std::chrono::high_resolution_clock::now();
            NVIGI_LOG_INFO("Processing a chunk of %d characters", textPieceNormalized.size());
            auto res = textNormalizer->g2p(textPieceNormalized, phones);
            if (res != kResultOk)
            {
                callback(outputAudio, textPieceNormalized, InferenceState::ERROR_HAPPENED);
                return res;
            }
            textNormalizer->convertTextsToIds(phones, inputNetPhonesTextToGen);

            phones.clear();
            res = textNormalizer->g2p(textNormalizedTranscript, phones);
            if (res != kResultOk)
            {
                callback(outputAudio, textPieceNormalized, InferenceState::ERROR_HAPPENED);
                return res;
            }
            // Two space are needed between the transcription and the text to generate
            phones.push_back(L" ");
            phones.push_back(L" ");
            textNormalizer->convertTextsToIds(phones, inputNetPhonesTextToGenAndTranscript);
            sizePhonemesTarget = inputNetPhonesTextToGenAndTranscript.size();
            inputNetPhonesTextToGenAndTranscript.insert(inputNetPhonesTextToGenAndTranscript.end(),
                                                        inputNetPhonesTextToGen.begin(), inputNetPhonesTextToGen.end());

            auto end_g2p = std::chrono::high_resolution_clock::now();
            storeExecutionTime(timers, start_g2p, end_g2p, "G2p computation");
        }
        catch (const Ort::Exception &e)
        {
            NVIGI_LOG_ERROR("Error while converting graph to phonemes: %s", e.what());
            callback(outputAudio, textPieceNormalized, InferenceState::ERROR_HAPPENED);
            return kResultInvalidState;
        }

        // Duration predictor inference
        auto start_inf1 = std::chrono::high_resolution_clock::now();


        //auto res = runDpModel(inputNetPhonesTextToGen, spectrogramTarget, dpVal, cudaStream);
        // 
        // Due to challenges in getting the DP model to work with TRT,
        // we use a straightforward formula to predict duration (linear equation fitted on real data base on : 
        // number_of_phonemes, size_of_spectrogram_target, number_of_phonemes_target) .
        // This approach appears to yield results comparable to the model.
        float dpValFloat = 14.6779 * (spectrogramLength / sizePhonemesTarget) + 3.3106 * inputNetPhonesTextToGen.size() + 104.7347;

        // for short sentences, the above formula does not work well. 
        // We found empirically that the following formula works better.
        if (textPieceNormalized.size() < 50) {
            // rule of 3
			dpValFloat = (float)inputNetPhonesTextToGen.size() * (float)spectrogramLength / (float)sizePhonemesTarget;
            dpValFloat *= 1.5;
        }

        // We clip the dpvalue between 70 and 2000.
        // We have set this shape constraint when converting model to TRT engine
        dpValFloat *= 1/inferParams.speed;
        dpVal = std::min((unsigned int)dpValFloat, (unsigned int)2000);
        dpVal = std::max((unsigned int)70, dpVal);
        NVIGI_LOG_INFO("Duration predictor returned %d", dpVal);

        if (res != kResultOk)
        {
            callback(outputAudio, textPieceNormalized, InferenceState::ERROR_HAPPENED);
            return res;
        }
        if (dpVal == 0)
        {
            NVIGI_LOG_INFO("Duration predictor returned 0. No output generated");
            callback(outputAudio, textPieceNormalized, InferenceState::ERROR_HAPPENED);
            return kResultInvalidState;
        }

        auto end_inf1 = std::chrono::high_resolution_clock::now();
        storeExecutionTime(timers, start_inf1, end_inf1, "Dp Model inference");

        //// Generator model inference
        start_inf1 = std::chrono::high_resolution_clock::now();
        std::vector<__half> outputMelSpectrogram;
        res = runGeneratorModel(inputNetPhonesTextToGenAndTranscript, spectrogramTarget, dpVal, outputMelSpectrogram, cudaStream, inferParams.seed);
        if (res != kResultOk)
        {
            callback(outputAudio, textPieceNormalized, InferenceState::ERROR_HAPPENED);
            return res;
        }

        end_inf1 = std::chrono::high_resolution_clock::now();
        storeExecutionTime(timers, start_inf1, end_inf1, "Generator Model inference");

        // Vocoder inference
        start_inf1 = std::chrono::high_resolution_clock::now();
        std::vector<int16_t> outputAudio;
        res = runVocoderModel(outputMelSpectrogram, outputAudio, cudaStream);
        if (res != kResultOk)
        {
            callback(outputAudio, textPieceNormalized, InferenceState::ERROR_HAPPENED);
            return res;
        }
        end_inf1 = std::chrono::high_resolution_clock::now();
        storeExecutionTime(timers, start_inf1, end_inf1, "Vocoder Model inference");

        if (indexPiece == sentencesSplit.size() - 1)
        {
            auto end_total = std::chrono::high_resolution_clock::now();
            storeExecutionTime(timers, start_total, end_total, "Total Time");
            callback(outputAudio, textPieceNormalized, InferenceState::FINISHED);
        }
        else
        {
            callback(outputAudio, textPieceNormalized, InferenceState::RUNNING);
        }

        if (indexPiece == 0 && timers.find("Total time to process first chunk") == timers.end())
        {
            auto end_iter = std::chrono::high_resolution_clock::now();
            storeExecutionTime(timers, start_total, end_iter, "Total time to process first chunk");
        }

        indexPiece += 1;
    }
    end = std::chrono::high_resolution_clock::now();

    // If empty string as input, we return an empty audio vector as output
    if (sentencesSplit.size() == 0)
    {
        auto end_total = std::chrono::high_resolution_clock::now();
        storeExecutionTime(timers, start_total, end_total, "Total Time");
        std::vector<int16_t> outputAudio;
        callback(outputAudio, "", InferenceState::FINISHED);
    }

    return kResultOk;
}

int32_t ASqFlow::getMaxAuxStreamCount()
{
    //int32_t dpStreamCount = dpModel.getAuxStreamCount();
    int32_t dpStreamCount = 0; // dpModel is disabled for now
    int32_t generatorStreamCount = generatorModel.getAuxStreamCount();
    int32_t vocoderStreamCount = vocoderModel.getAuxStreamCount();

    return std::max(std::max(dpStreamCount, generatorStreamCount), vocoderStreamCount);
}

void ASqFlow::setAuxStreams(cudaStream_t* streams, size_t streamCount)
{
    // dpModel.setAuxStreams(streams, streamCount); // dpModel is disabled for now
    generatorModel.setAuxStreams(streams, streamCount);
    vocoderModel.setAuxStreams(streams, streamCount);
}

} // namespace asqflow
} // namespace nvigi