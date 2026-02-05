// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#include "onnxruntime_cxx_api.h"
#include "source/plugins/nvigi.tts/asqflow/trt/symbols.h"
#include "external/json/source/nlohmann/json.hpp"
#include <nvigi.h>
#include <SimpleFarForTTS/nvspeech-grammars.h>
#include <array>
#include <memory>
#include <nvigi_result.h>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::json;
const std::unordered_map<std::string, std::string> _number_args = {
    {"zero", "zero"}, {"one", "one"}, {"andword", "and"}};

const std::regex DIGIT_GROUP(R"((\d))");
const std::regex TWO_DIGITS(R"((\d)(\d))");
const std::regex THREE_DIGITS(R"((\d)(\d)(\d))");
const std::regex THREE_DIGITS_WORD(R"((\d)(\d)(\d)(?=\D*\Z))");
const std::regex TWO_DIGITS_WORD(R"((\d)(\d)(?=\D*\Z))");
const std::regex ONE_DIGIT_WORD(R"((\d)(?=\D*\Z))");

namespace nvigi
{
namespace asqflow
{
// based on the python implementation implemented in https://github.com/Kyubyong/g2p
//  It is used when a word/token is not present in the dictionary
class GraphemeToPhonemePred
{
  public:
    std::unordered_map<std::string, int> p2idx;

    GraphemeToPhonemePred();

    void encode(const std::string word, std::vector<int> &indices);

    void decode(const std::vector<int> &indices, std::vector<std::string> &phonemes);

  private:
    std::vector<std::string> graphemes;
    std::vector<std::string> phonemes;
    std::unordered_map<std::string, int> g2idx;
    std::unordered_map<int, std::string> idx2g;
    std::unordered_map<int, std::string> idx2p;
};

class LanguageNormalizer
{
  public:
    virtual nvigi::Result normalizeText(const std::string &text, std::string &textNormalized) = 0;
    virtual nvigi::Result g2p(const std::string &text, std::vector<std::wstring> &phones) = 0;

    virtual void convertTextsToIds(const std::vector<std::wstring> &phones, std::vector<int> &out_phones) = 0;

    virtual std::vector<std::string> splitSentences(std::string &text, int minSizeChunk = 100, int maxSizeChunk = 200) = 0;
};

class EnglishNormalizer : public LanguageNormalizer
{
  public:
    EnglishNormalizer(const std::string& pathG2PModel, const std::string& pathEnglishDictDefault,
                      const json &configData, const std::string& pathEnglishDictExtended="");

    nvigi::Result normalizeText(const std::string &text, std::string &textNormalized) override;

    nvigi::Result g2p(const std::string &text, std::vector<std::wstring> &phones) override;

    void convertTextsToIds(const std::vector<std::wstring> &phones, std::vector<int> &out_phones) override;
    // Function equivalent to split_sentences_latin
    std::vector<std::string> splitSentences(std::string &text, int minSizeChunk = 100, int maxSizeChunk = 200) override;

    static const std::regex DIGIT_GROUP;
    static const std::regex TWO_DIGITS;
    static const std::regex THREE_DIGITS;
    static const std::regex THREE_DIGITS_WORD;
    static const std::regex TWO_DIGITS_WORD;
    static const std::regex ONE_DIGIT_WORD;

    static constexpr const std::array<const char *, 10> unit = {"zero", "one", "two",   "three", "four",
                                                                "five", "six", "seven", "eight", "nine"};
    static constexpr const std::array<const char *, 10> ten = {"",      "",      "twenty",  "thirty", "forty",
                                                               "fifty", "sixty", "seventy", "eighty", "ninety"};
    static constexpr const std::array<const char *, 10> teen = {
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
    static constexpr const std::array<const char *, 4> mill = {"", "thousand", "million", "billion"};
    static const std::unordered_map<const char *, const char *> _number_args;
    static int mill_count;

  private:
    const std::regex _comma_number_re{R"(([0-9][0-9\,]+[0-9]))"};
    const std::regex _decimal_number_re{R"(([0-9]+\.[0-9]+))"};
    const std::regex _currency_re{R"((�|\$|�)([0-9\,\.]*[0-9]+))"};
    const std::regex _ordinal_re{R"([0-9]+(st|nd|rd|th))"};
    const std::regex _number_re{R"(-?[0-9]+)"};
    const std::regex time_re{
        R"(\b((0?[0-9])|(1[0-1])|(1[2-9])|(2[0-3])):([0-5][0-9])\s*(a\.m\.|am|pm|p\.m\.|a\.m|p\.m)?\b)",
        std::regex::icase};
    const std::vector<std::pair<std::regex, std::string>> abbreviations_en = {
        {std::regex(R"(\bmrs\.)", std::regex::icase), "misess"},
        {std::regex(R"(\bmr\.)", std::regex::icase), "mister"},
        {std::regex(R"(\bdr\.)", std::regex::icase), "doctor"},
        {std::regex(R"(\bst\.)", std::regex::icase), "saint"},
        {std::regex(R"(\bco\.)", std::regex::icase), "company"},
        {std::regex(R"(\bjr\.)", std::regex::icase), "junior"},
        {std::regex(R"(\bmaj\.)", std::regex::icase), "major"},
        {std::regex(R"(\bgen\.)", std::regex::icase), "general"},
        {std::regex(R"(\bdrs\.)", std::regex::icase), "doctors"},
        {std::regex(R"(\brev\.)", std::regex::icase), "reverend"},
        {std::regex(R"(\blt\.)", std::regex::icase), "lieutenant"},
        {std::regex(R"(\bhon\.)", std::regex::icase), "honorable"},
        {std::regex(R"(\bsgt\.)", std::regex::icase), "sergeant"},
        {std::regex(R"(\bcapt\.)", std::regex::icase), "captain"},
        {std::regex(R"(\besq\.)", std::regex::icase), "esquire"},
        {std::regex(R"(\bltd\.)", std::regex::icase), "limited"},
        {std::regex(R"(\bcol\.)", std::regex::icase), "colonel"},
        {std::regex(R"(\bft\.)", std::regex::icase), "fort"}};

    // Create the replacement map (rep_map)
    const std::unordered_map<std::string, std::string> postPhRepMap = {
        {"：", ","}, {"；", ","}, {"，", ","}, {"。", "."},  {"！", "!"}, {"？", "?"},
        {"\n", "."}, {"·", ","},  {"、", ","}, {"...", "…"}, {"v", "V"}};

    const std::unordered_map<std::string, std::string> arpabet_to_ipa = {// Vowels - Monophthongs
                                                                         {"AO", "ɔ"},
                                                                         {"AO0", "ɔ"},
                                                                         {"AO1", "ˈɔ"},
                                                                         {"AO2", "ˌɔ"},
                                                                         {"AA", "ɑ"},
                                                                         {"AA0", "ɑ"},
                                                                         {"AA1", "ˈɑ"},
                                                                         {"AA2", "ˌɑ"},
                                                                         {"IY", "i"},
                                                                         {"IY0", "i"},
                                                                         {"IY1", "ˈi"},
                                                                         {"IY2", "ˌi"},
                                                                         {"UW", "u"},
                                                                         {"UW0", "u"},
                                                                         {"UW1", "ˈu"},
                                                                         {"UW2", "ˌu"},
                                                                         {"EH", "ɛ"},
                                                                         {"EH0", "ɛ"},
                                                                         {"EH1", "ˈɛ"},
                                                                         {"EH2", "ˌɛ"},
                                                                         {"IH", "ɪ"},
                                                                         {"IH0", "ɪ"},
                                                                         {"IH1", "ˈɪ"},
                                                                         {"IH2", "ˌɪ"},
                                                                         {"UH", "ʊ"},
                                                                         {"UH0", "ʊ"},
                                                                         {"UH1", "ˈʊ"},
                                                                         {"UH2", "ˌʊ"},
                                                                         {"AH", "ʌ"},
                                                                         {"AH0", "ə"},
                                                                         {"AH1", "ˈʌ"},
                                                                         {"AH2", "ˌʌ"},
                                                                         {"AE", "æ"},
                                                                         {"AE0", "æ"},
                                                                         {"AE1", "ˈæ"},
                                                                         {"AE2", "ˌæ"},
                                                                         {"AX", "ə"},
                                                                         {"AX0", "ə"},
                                                                         {"AX1", "ˈə"},
                                                                         {"AX2", "ˌə"},

                                                                         // Vowels - Diphthongs
                                                                         {"EY", "eɪ"},
                                                                         {"EY0", "eɪ"},
                                                                         {"EY1", "ˈeɪ"},
                                                                         {"EY2", "ˌeɪ"},
                                                                         {"AY", "aɪ"},
                                                                         {"AY0", "aɪ"},
                                                                         {"AY1", "ˈaɪ"},
                                                                         {"AY2", "ˌaɪ"},
                                                                         {"OW", "oʊ"},
                                                                         {"OW0", "oʊ"},
                                                                         {"OW1", "ˈoʊ"},
                                                                         {"OW2", "ˌoʊ"},
                                                                         {"AW", "aʊ"},
                                                                         {"AW0", "aʊ"},
                                                                         {"AW1", "ˈaʊ"},
                                                                         {"AW2", "ˌaʊ"},
                                                                         {"OY", "ɔɪ"},
                                                                         {"OY0", "ɔɪ"},
                                                                         {"OY1", "ˈɔɪ"},
                                                                         {"OY2", "ˌɔɪ"},

                                                                         // Consonants - Stops
                                                                         {"P", "p"},
                                                                         {"B", "b"},
                                                                         {"T", "t"},
                                                                         {"D", "d"},
                                                                         {"K", "k"},
                                                                         {"G", "g"},

                                                                         // Consonants - Affricates
                                                                         {"CH", "tʃ"},
                                                                         {"JH", "dʒ"},

                                                                         // Consonants - Fricatives
                                                                         {"F", "f"},
                                                                         {"V", "v"},
                                                                         {"TH", "θ"},
                                                                         {"DH", "ð"},
                                                                         {"S", "s"},
                                                                         {"Z", "z"},
                                                                         {"SH", "ʃ"},
                                                                         {"ZH", "ʒ"},
                                                                         {"HH", "h"},

                                                                         // Consonants - Nasals
                                                                         {"M", "m"},
                                                                         {"N", "n"},
                                                                         {"NG", "ŋ"},

                                                                         // Consonants - Liquids
                                                                         {"L", "l"},
                                                                         {"R", "ɹ"},

                                                                         // Vowels - R-colored vowels
                                                                         {"ER", "ɜɹ"},
                                                                         {"ER0", "ɜɹ"},
                                                                         {"ER1", "ˈɜɹ"},
                                                                         {"ER2", "ˌɜɹ"},
                                                                         {"AXR", "ər"},
                                                                         {"AXR0", "ər"},
                                                                         {"AXR1", "ˈər"},
                                                                         {"AXR2", "ˌər"},

                                                                         // Consonants - Semivowels
                                                                         {"Y", "j"},
                                                                         {"W", "w"}};

    const std::set<std::wstring> ipa_vowels = {// Monophthongs
                                               L"ɔ", L"ɑ", L"i", L"u", L"ɛ", L"ɪ", L"ʊ", L"ʌ", L"ə", L"æ",

                                               // Diphthongs
                                               L"eɪ", L"aɪ", L"oʊ", L"aʊ", L"ɔɪ",

                                               // R-colored vowels
                                               L"ɜɹ", L"ər"};

    // Define a regex pattern to match IPA symbols, including stress marks and
    // multi - symbol characters
    const std::wregex ipa_pattern{
        L"((?:ˈ|ˌ)?(?:[ɪʊɛɔɒəɜɹɐʏøɞɯɰɨʉʌœɑɒɝaeiouy]+|[a-zɸβθðʃʒɲŋɖʂʐɽɳɭɴɟʄɡɢʙɰʍʜʝχʁʔʕħʛʧʤɕʑɬɮʎɥɹɻʧʤɾ]|["
        L"iɪʏʊeøɤɔæɑʌəɒɑɞɐɵʉɨʊəɔɞɜɵøɛæəɑɪɛʉɤɜɐəɜɯɝae]|.+))"};

    std::string pathEnglishDict = "";
    std::unordered_map<std::string, std::vector<std::wstring>> eng_dict;

    Symbols symbols;
    json configData;
    std::unordered_map<std::wstring, int> symbol_to_id_map;

    bool add_bos_eos_to_text = false;

    std::unique_ptr<TextNormalizerFar> textNormalizerFar;

    std::unique_ptr<Ort::Session> session_g2p;
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::MemoryInfo memory_info;
    Ort::IoBinding io;
    Ort::AllocatorWithDefaultOptions allocator;

    GraphemeToPhonemePred g2pPred;

    nvigi::Result normalizeUsingFar(const std::string &textPrompt, std::string &textNormalized);

    void readPhonemesDict(std::string pathPhonemesDict);

    nvigi::Result predictPhonemesFromWord(const std::string &word, std::vector<std::string> &outputPhonemes);
};
} // namespace asqflow
} // namespace nvigi
