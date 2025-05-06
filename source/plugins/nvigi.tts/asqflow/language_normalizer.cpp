// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "onnxruntime_cxx_api.h"
#include "external/json/source/nlohmann/json.hpp"
#include "language_normalizer.h"
#include "nvigi_core/source/core/nvigi.log/log.h"
#include "source/plugins/nvigi.tts/nvigi_tts.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <numeric>
#include <regex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <exception>
#include <filesystem>
#include <locale>
#include <nvigi_result.h>
#include <stdexcept>
#include <stringapiset.h>
#include <WinNls.h>
#include <stdlib.h>
#include <onnxruntime_c_api.h>
#include <nvspeech-grammars.h>

using json = nlohmann::json;

/**
 * @brief Removes leading and trailing whitespace from a given string.
 *
 * @param str The input string to be trimmed.
 * @return std::string A new string with all leading and trailing whitespace removed.
 * If the input string contains only whitespace, an empty string is returned.
 *
 **/
std::string trim(const std::string &str)
{
    auto start = std::find_if(str.begin(), str.end(), [](unsigned char ch) { return !std::isspace(ch); });
    auto end = std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base();

    return (start < end) ? std::string(start, end) : std::string();
}

/**
 * @brief Splits a long text into smaller parts based on desired and maximum lengths,
 *        while attempting to preserve sentence integrity.
 *
 * @param inputText The input text to be split into smaller segments.
 * @param desiredLength The target length for each segment (default is 100 characters).
 *                      The function tries to make each segment close to this length.
 * @param maxLength The maximum allowed length for any segment (default is 200 characters).
 *                  Segments will not exceed this length.
 * @return std::vector<std::string> A vector of strings, where each string is a segment of
 *                                   the original text split according to the specified parameters.
 *
 * @details
 * - The function prioritizes splitting text near the desired length, but it will extend
 *   segments up to the maximum length if necessary to preserve sentence boundaries.
 * - Sentence preservation is achieved by avoiding splits in the middle of sentences whenever possible.
 */
std::vector<std::string> txtsplit(const std::string &inputText, int desiredLength = 100, int maxLength = 200)
{
    // Step 1: Normalize the text with regex substitutions
    std::string text = inputText;
    text = std::regex_replace(text, std::regex(R"(\n\n+)"), "\n");
    text = std::regex_replace(text, std::regex(R"(\s+)"), " ");
    text = std::regex_replace(text, std::regex(R"([""])"), "\"");
    text = std::regex_replace(text, std::regex(R"([,.?!])"), "$& ");
    text = std::regex_replace(text, std::regex(R"(\s+)"), " ");

    // Step 2: Variables for processing
    std::vector<std::string> rv;
    bool in_quote = false;
    std::string current = "";
    std::vector<int> split_pos;
    int pos = -1;
    int end_pos = text.size() - 1;

    // Helper functions: Seek, Peek, and Commit
    auto seek = [&](int delta) -> char {
        bool is_neg = delta < 0;
        for (int i = 0; i < std::abs(delta); ++i)
        {
            if (is_neg)
            {
                pos--;
                current.pop_back();
            }
            else
            {
                pos++;
                current.push_back(text[pos]);
            }
            if (text[pos] == '"')
            {
                in_quote = !in_quote;
            }
        }
        return text[pos];
    };

    auto peek = [&](int delta) -> char {
        int p = pos + delta;
        return (p >= 0 && p < text.size()) ? text[p] : '\0';
    };

    auto commit = [&]() {
        rv.push_back(current);
        current.clear();
        split_pos.clear();
    };

    // Step 3: Main loop for splitting
    while (pos < end_pos)
    {
        char c = seek(1);

        if (current.length() >= maxLength)
        {
            if (!split_pos.empty() && current.length() > (desiredLength / 2))
            {
                int d = pos - split_pos.back();
                seek(-d);
            }
            else
            {
                while (c != '!' && c != '?' && c != '.' && c != '\n' && c != ' ' && pos > 0 &&
                       current.length() > desiredLength)
                {
                    c = seek(-1);
                }
            }
            commit();
        }
        else if (!in_quote && (c == '!' || c == '?' || c == '\n' || (c == '.' || c == ',' && peek(1) == ' ')))
        {
            while (pos < end_pos && current.length() < maxLength &&
                   (peek(1) == '!' || peek(1) == '?' || peek(1) == '.'))
            {
                c = seek(1);
            }
            split_pos.push_back(pos);
            if (current.length() >= desiredLength)
            {
                commit();
            }
        }
        else if (in_quote && peek(1) == '"' && (peek(2) == ' ' || peek(2) == '\n'))
        {
            seek(2);
            split_pos.push_back(pos);
        }
    }

    rv.push_back(current);

    // Step 4: Trim and filter the results
    for (auto &s : rv)
    {
        s = trim(s);
    }
    rv.erase(std::remove_if(rv.begin(), rv.end(),
                            [](const std::string &s) {
                                return s.empty() || std::regex_match(s, std::regex(R"(^[\s\.,;:!?]*$)"));
                            }),
             rv.end());

    return rv;
}

// Convert  std::string (UTF-8) to std::wstring (UTF-16)
std::wstring stringToWstring(const std::string &str)
{
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    if (len == 0)
        return L"";

    std::wstring wstr(len - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], len);
    return wstr;
}

std::string wstringToString(const std::wstring &str)
{
    int len = WideCharToMultiByte(CP_UTF8, 0, str.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (len == 0)
        return "";

    std::string utf8(len - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, str.c_str(), -1, &utf8[0], len, nullptr, nullptr);
    return utf8;
}

namespace nvigi
{
namespace asqflow
{
// Based on the python implementation implemented in https://github.com/Kyubyong/g2p
// It is used when a word/token is not present in the dictionary
GraphemeToPhonemePred::GraphemeToPhonemePred()
{
    graphemes = {"<pad>", "<unk>", "</s>", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
                 "m",     "n",     "o",    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"};

    phonemes = {"<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2",
                "AO0",   "AO1",   "AO2", "AW0",  "AW1", "AW2", "AY0", "AY1", "AY2", "B",   "CH",  "D",   "DH",
                "EH0",   "EH1",   "EH2", "ER0",  "ER1", "ER2", "EY0", "EY1", "EY2", "F",   "G",   "HH",  "IH0",
                "IH1",   "IH2",   "IY0", "IY1",  "IY2", "JH",  "K",   "L",   "M",   "N",   "NG",  "OW0", "OW1",
                "OW2",   "OY0",   "OY1", "OY2",  "P",   "R",   "S",   "SH",  "T",   "TH",  "UH0", "UH1", "UH2",
                "UW",    "UW0",   "UW1", "UW2",  "V",   "W",   "Y",   "Z",   "ZH"};

    for (int idx = 0; idx < graphemes.size(); ++idx)
    {
        g2idx[graphemes[idx]] = idx;
        idx2g[idx] = graphemes[idx];
    }

    for (int idx = 0; idx < phonemes.size(); ++idx)
    {
        p2idx[phonemes[idx]] = idx;
        idx2p[idx] = phonemes[idx];
    }
}

void GraphemeToPhonemePred::encode(const std::string word, std::vector<int> &indices)
{

    // 1. Convert word to list of characters + add "</s>"
    std::vector<std::string> chars;
    for (char c : word)
    {
        chars.push_back(std::string(1, c)); // Add each char as a string
    }
    chars.push_back("</s>");

    // 2. Map characters to indices
    for (const auto &ch : chars)
    {
        auto it = g2idx.find(ch);
        if (it != g2idx.end())
        {
            indices.push_back(it->second);
        }
        else
        {
            indices.push_back(g2idx["<unk>"]);
        }
    }
}

void GraphemeToPhonemePred::decode(const std::vector<int> &indices, std::vector<std::string> &phonemes)
{

    for (const int &ind : indices)
    {
        auto it = idx2p.find(ind);
        if (it != idx2p.end())
        {
            if (it->second == "</s>") // End of prediction
                break;

            phonemes.push_back(it->second);
        }
        else
        {
            phonemes.push_back("<unk>");
        }
    }
}

EnglishNormalizer::EnglishNormalizer(const std::string &pathG2PModel, const std::string &pathEnglishDictDefault,
                                     const json &configData, const std::string& pathEnglishDictExtended)
    : env(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "ONNXModel")), session_g2p(nullptr), session_options(),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)), allocator(), io(nullptr)
{
    readPhonemesDict(pathEnglishDictDefault);

	if (!pathEnglishDictExtended.empty())
        readPhonemesDict(pathEnglishDictExtended);

    if (eng_dict.empty())
    {
        NVIGI_LOG_ERROR("Phoneme dictionary is empty");
        throw std::runtime_error("Phoneme dictionary is empty");
    }

    if (!configData.contains("symbols") || !configData["symbols"].is_array()) {
		NVIGI_LOG_ERROR("Symbols array is missing in the configuration file");
		throw std::runtime_error("Symbols array is missing in the configuration file");
    }

    int i = 0;
    for (const auto &symbol : configData["symbols"])
    {
        if (symbol.is_string()) {
            symbol_to_id_map[stringToWstring(symbol.get<std::string>())] = i;
        }
        i++;
    }

    if (configData.contains("add_bos_eos_to_text") && configData["add_bos_eos_to_text"].is_boolean())
    {
        add_bos_eos_to_text = (bool)configData["add_bos_eos_to_text"];
    }

    if (configData.contains("minSizeChunk") && configData["minSizeChunk"].is_number())
    {
        minSizeChunk = configData["minSizeChunk"];
    }
    if (configData.contains("maxSizeChunk") && configData["maxSizeChunk"].is_number())
    {
        maxSizeChunk = configData["maxSizeChunk"];
    }
    NVIGI_LOG_INFO("Using minSizeChunk: %d, maxSizeChunk: %d", minSizeChunk, maxSizeChunk);

    NVIGI_LOG_INFO("Using nemo text normalization, loading far archives ... ");
    std::wstring parentFolder = std::filesystem::path(pathG2PModel).parent_path().wstring();
    std::wstring tokenizerPath = parentFolder + L"/tokenize_and_classify.far";
    std::wstring verbalizerPath = parentFolder + L"/verbalize.far";
    try
    {
        textNormalizerFar = std::make_unique<TextNormalizerFar>(tokenizerPath.c_str(), verbalizerPath.c_str());
    }
    catch (...)
    {
        throw std::runtime_error("An issue happened when instantiating text normalization.\
The tokenize_and_classify.far or/and verbalize.far archive are/is maybe missing.\
Please place those files in the same directory as the G2P model.");
    }

    NVIGI_LOG_INFO("Warming up text normalizer ....");
    std::string textNormalized;
    auto res = normalizeUsingFar("hello here a warming up prompt", textNormalized);
    if (res != nvigi::kResultOk)
    {
        NVIGI_LOG_ERROR("An issue happened during warm up of advanced text normalization.\
Please check that tokenize_and_classify.far and verbalize.far are not corrupted.");
    }

#ifdef _WIN32
    // Convert std::string to std::wstring for Windows
    NVIGI_LOG_INFO("Loading g2p model ... ");
    std::wstring wide_path_g2p_model(pathG2PModel.begin(), pathG2PModel.end());
    session_g2p = std::make_unique<Ort::Session>(env, wide_path_g2p_model.c_str(), session_options);

#else
    // On Linux and macOS, use the regular string
    session_g2p = std::make_unique<Ort::Session>(env, pathG2PModel.c_str(), session_options);
#endif
}

/**
 *   Wrapper of textNormalizerFar->normalizeText with some additional preprocessing/postprocessing steps
 * Preprocessing :
 *   - Replace some symbols with space (-, *)
 *   - Trim
 * Postprocessing :
 * - lower case
 * - remove non ascii symbols
 **/
nvigi::Result EnglishNormalizer::normalizeUsingFar(const std::string &textPrompt, std::string &textNormalized)
{
    // Some preprocessing
    std::string textPreprocess = textPrompt;
    textPreprocess = trim(textPreprocess);
    if (textPreprocess.empty())
        return nvigi::kResultOk;

    const char *outNormalized = nullptr;
    try
    {
        int res = textNormalizerFar->normalizeText(textPreprocess.c_str(), outNormalized);
        if (res < 0)
        {
            textNormalized = std::string(outNormalized);
            return nvigi::kResultInvalidState;
        }
    }
    catch (const std::exception &err)
    {
        NVIGI_LOG_WARN("An error occurred during text normalization: %s\n for the prompt: %s", err.what(),
                       textPrompt.c_str());
        textNormalized = textPreprocess;
        return nvigi::kResultInvalidState;
    }
    catch (...)
    {
        NVIGI_LOG_WARN("An unknown error occurred during text normalization for the prompt: %s", textPrompt.c_str());
        textNormalized = textPreprocess;
        return nvigi::kResultInvalidState;
    }

    textNormalized = std::string(outNormalized);
    // Additional post-processing
    std::transform(textNormalized.begin(), textNormalized.end(), textNormalized.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // Keep only useful characters
    std::regex unwanted(R"([^a-z0-9.,;!?'\s])");
    textNormalized = std::regex_replace(textNormalized, unwanted, " ");

    // Replace multiple dots with a single dot to avoind long silence when encountering tree dots
    std::regex patternDots("\\.+");
    textNormalized = std::regex_replace(textNormalized, patternDots, ".");

    return nvigi::kResultOk;
}

nvigi::Result EnglishNormalizer::normalizeText(const std::string &text, std::string &textNormalized)
{

    nvigi::Result res = nvigi::kResultOk;
    res = normalizeUsingFar(text, textNormalized);
    if (res != nvigi::kResultOk)
    {
        NVIGI_LOG_WARN("An issue happened during text normalization. Please check the input prompt : %s", text.c_str());
        res = kResultOk;
    }
    NVIGI_LOG_INFO("Output after normalization: %s \n", textNormalized.c_str());
    return res;
}

std::vector<std::string> EnglishNormalizer::splitSentences(std::string &text)
{
    // Replace punctuation using std::regex and std::regex_replace
    text = std::regex_replace(text, std::regex("[。！？；]"), ".");
    text = std::regex_replace(text, std::regex("[，]"), ",");
    text = std::regex_replace(text, std::regex("[“”]"), "\"");
    text = std::regex_replace(text, std::regex("[‘’]"), "'");
    text = std::regex_replace(text, std::regex(R"([\<\>\(\)\[\]\"\«\»]+)"), "");

    // Call the txtsplit function to split text
    std::vector<std::string> split_result = txtsplit(text, minSizeChunk, maxSizeChunk);

    // Strip each sentence and filter out empty sentences
    std::vector<std::string> result;
    for (const auto &item : split_result)
    {
        std::string stripped_item = trim(item);
        if (!stripped_item.empty())
        {
            result.push_back(stripped_item);
        }
    }
    return result;
}

/**
 * Read .txt IPA dictionary
 */
void EnglishNormalizer::readPhonemesDict(std::string pathPhonemesDict)
{
    const int start_line = 49;
    std::ifstream file(pathPhonemesDict);

    if (!file.is_open())
    {
        NVIGI_LOG_ERROR("Unable to open phonemes dictionary file at %s", pathPhonemesDict.c_str());
        return;
    }
    file.imbue(std::locale(""));

    std::string line;
    int line_index = 1;

    // Example of one line : ISSUE'S  ˈ ɪ ʃ u ː z
    // (words are separated to phonemes by 2 spaces,
    // phones are separated to each other by one space
    while (std::getline(file, line))
    {
        size_t pos = 0, start = 0;
        std::string word;
        std::vector<std::wstring> phonemes;
        // Extract word from line
        pos = line.find('  ', start);
        if (pos == std::string::npos)
        {
            continue;
        }

        word = line.substr(start, pos - start);

        // Extract phonemes.
        std::wstring phonemes_wstr = stringToWstring(line.substr(pos + 2));
        while ((pos = phonemes_wstr.find(' ', start)) != std::string::npos)
        {
            phonemes.push_back(phonemes_wstr.substr(start, pos - start));
            start = pos + 1;
        }
        // extract last phoneme which does not end by a space
        phonemes.push_back(phonemes_wstr.substr(start, start + 1));

        eng_dict[word] = std::move(phonemes);

        line_index++;
    }
}

/**
 * Uses G2P model converted to onnx to predict the phonemes of an unkown word.
 */
nvigi::Result EnglishNormalizer::predictPhonemesFromWord(const std::string &word,
                                                         std::vector<std::string> &outputPhonemes)
{
    std::vector<int> indices_encoding_word;
    g2pPred.encode(word, indices_encoding_word);

    const char *input_names[] = {"word_enc_indices"};

    std::vector<int64_t> wordEncIndicesDims = {1, static_cast<int64_t>(indices_encoding_word.size())};
    Ort::Value wordEncIndicesTensor =
        Ort::Value::CreateTensor<int>(memory_info, indices_encoding_word.data(), indices_encoding_word.size(),
                                      wordEncIndicesDims.data(), wordEncIndicesDims.size());

    std::array<Ort::Value, 1> input_tensors = {std::move(wordEncIndicesTensor)};

    std::vector<Ort::Value> output_phonemes_tensors;
    try
    {
        const char *output_names[] = {"output"};
        output_phonemes_tensors = session_g2p->Run(
            Ort::RunOptions{nullptr}, input_names, input_tensors.data(), input_tensors.size(), output_names, 1);
    }
    catch (const Ort::Exception &e)
    {
        NVIGI_LOG_ERROR("Error while running G2p model: %s", e.what());
        return kResultErrorG2pModelASqFlow;
    }

    auto output_tensor_data = output_phonemes_tensors[0].GetTensorMutableData<int>();
    auto type_and_shape_info = output_phonemes_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = type_and_shape_info.GetShape();

    int64_t num_elements = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());

    std::vector<int> output_phonemes_ids(output_tensor_data, output_tensor_data + num_elements);
    g2pPred.decode(output_phonemes_ids, outputPhonemes);

    return kResultOk;
}

/** Uses CMU dictionnary to generate phonemes. When not found, uses a model to predict the pronounciation
 */
nvigi::Result EnglishNormalizer::g2p(const std::string &text, std::vector<std::wstring> &phones)
{
    // Regular expression to match words and punctuation separately
    std::regex wordRegex(R"(\w+(?:'\w+)?|[^\w\s])");
    std::sregex_iterator wordsBegin(text.begin(), text.end(), wordRegex);
    std::sregex_iterator wordsEnd;

    for (std::sregex_iterator it = wordsBegin; it != wordsEnd; ++it)
    {
        std::string w = it->str();
        std::string wordUpper = w;
        std::transform(wordUpper.begin(), wordUpper.end(), wordUpper.begin(), ::toupper);

        if (eng_dict.contains(wordUpper))
        {
            // extractIpaCharacters(eng_dict.at(wordUpper)[0], phones, phones.size() != 0);

            if (phones.size() != 0)
            {
                phones.push_back(L" ");
            }

            for (auto ph : eng_dict.at(wordUpper))
            {
                phones.push_back(ph);
            }

            // phones.push_back(ipa_characters);
        }
        else if (w == "," || w == "." || w == "?" || w == "!" || w == ";" || w == ":")
        {
            if (!phones.empty())
            {
                phones.push_back(stringToWstring(w));
            }
        }
        else
        {
            w = trim(w);
            std::vector<std::string> phns;
            auto res = predictPhonemesFromWord(w, phns);
            if (res != kResultOk)
            {
                return res;
            }

            // Convert to IPA

            for (const auto &ph : phns)
            {
                if (arpabet_to_ipa.contains(ph))
                {

                    phones.push_back(stringToWstring(arpabet_to_ipa.at(ph)));
                }
            }
        }
    }

    return kResultOk;
}

/**
 * Prepare data for network
 * Convert text (phones) to integer id
 */
void EnglishNormalizer::convertTextsToIds(const std::vector<std::wstring> &phones, std::vector<int> &out_phones)
{

    if (add_bos_eos_to_text)
    {
        out_phones.push_back(symbol_to_id_map[L"<bos>"]);
    }

    // Convert phones string to phones ids
    out_phones.reserve(phones.size());
    for (const auto &symbol : phones)
    {
        auto it = symbol_to_id_map.find(symbol);
        if (it != symbol_to_id_map.end())
        {
            out_phones.push_back(it->second);
        }

        else
        {
            // Loop over each character of the symbol
            for (const auto &car : symbol)
            {
                std::wstring wstr = {car};
                auto it = symbol_to_id_map.find(wstr);
                if (it != symbol_to_id_map.end())
                {
                    out_phones.push_back(it->second);
                }
            }
        }
    }

    if (add_bos_eos_to_text)
    {
        out_phones.push_back(symbol_to_id_map[L"<eos>"]);
    }
}
} // namespace asqflow
} // namespace nvigi
