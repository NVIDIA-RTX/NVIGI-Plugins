// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#include <algorithm> // for sorting and set
#include <iostream>
#include <map>
#include <string>
#include <vector>

class Symbols
{
  public:
    Symbols()
    {
        pu_symbols.push_back("SP");
        pu_symbols.push_back("UNK");

        // Combine all symbols
        std::vector<std::string> normal_symbols;
        normal_symbols.insert(normal_symbols.end(), zh_symbols.begin(), zh_symbols.end());
        normal_symbols.insert(normal_symbols.end(), ja_symbols.begin(), ja_symbols.end());
        normal_symbols.insert(normal_symbols.end(), en_symbols.begin(), en_symbols.end());
        normal_symbols.insert(normal_symbols.end(), kr_symbols.begin(), kr_symbols.end());
        normal_symbols.insert(normal_symbols.end(), es_symbols.begin(), es_symbols.end());
        normal_symbols.insert(normal_symbols.end(), fr_symbols.begin(), fr_symbols.end());
        normal_symbols.insert(normal_symbols.end(), de_symbols.begin(), de_symbols.end());
        normal_symbols.insert(normal_symbols.end(), ru_symbols.begin(), ru_symbols.end());

        std::sort(normal_symbols.begin(), normal_symbols.end());
        normal_symbols.erase(std::unique(normal_symbols.begin(), normal_symbols.end()), normal_symbols.end());

        std::vector<std::string> symbols = {pad};
        symbols.insert(symbols.end(), normal_symbols.begin(), normal_symbols.end());
        symbols.insert(symbols.end(), pu_symbols.begin(), pu_symbols.end());

        // Find silent phoneme IDs
        std::vector<int> sil_phonemes_ids;
        for (const auto &pu_symbol : pu_symbols)
        {
            auto it = std::find(symbols.begin(), symbols.end(), pu_symbol);
            if (it != symbols.end())
            {
                sil_phonemes_ids.push_back(std::distance(symbols.begin(), it));
            }
        }
    }

    // Symbols and punctuation
    std::vector<std::string> punctuation = {"!", "?", "...", ",", ".", "'", "-", "¿", "¡"};
    std::vector<std::string> pu_symbols = punctuation;

    std::string pad = "_";

    // Chinese symbols
    std::vector<std::string> zh_symbols = {
        "E",  "En", "a", "ai",  "an", "ang", "ao",   "b",   "c",  "ch", "d",   "e",    "ei", "en",  "eng", "er",   "f",
        "g",  "h",  "i", "i0",  "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong", "ir", "iu",  "j",   "k",    "l",
        "m",  "n",  "o", "ong", "ou", "p",   "q",    "r",   "s",  "sh", "t",   "u",    "ua", "uai", "uan", "uang", "ui",
        "un", "uo", "v", "van", "ve", "vn",  "w",    "x",   "y",  "z",  "zh",  "AA",   "EE", "OO"};
    int num_zh_tones = 6;

    // Japanese symbols
    std::vector<std::string> ja_symbols = {"N", "a",  "a:", "b",  "by", "ch", "d", "dy", "e",  "e:", "f",
                                           "g", "gy", "h",  "hy", "i",  "i:", "j", "k",  "ky", "m",  "my",
                                           "n", "ny", "o",  "o:", "p",  "py", "q", "r",  "ry", "s",  "sh",
                                           "t", "ts", "ty", "u",  "u:", "w",  "y", "z",  "zy"};
    int num_ja_tones = 1;

    // English symbols
    std::vector<std::string> en_symbols = {"aa", "ae", "ah", "ao", "aw", "ay", "b",  "ch", "d", "dh", "eh", "er", "ey",
                                           "f",  "g",  "hh", "ih", "iy", "jh", "k",  "l",  "m", "n",  "ng", "ow", "oy",
                                           "p",  "r",  "s",  "sh", "t",  "th", "uh", "uw", "V", "w",  "y",  "z",  "zh"};
    int num_en_tones = 4;

    // Korean symbols
    std::vector<std::string> kr_symbols = {"ᄌ", "ᅥ",  "ᆫ",  "ᅦ",  "ᄋ", "ᅵ", "ᄅ", "ᅴ",  "ᄀ", "ᅡ",  "ᄎ", "ᅪ",
                                           "ᄑ", "ᅩ",  "ᄐ", "ᄃ", "ᅢ",  "ᅮ", "ᆼ",  "ᅳ",  "ᄒ", "ᄆ", "ᆯ",  "ᆷ",
                                           "ᄂ", "ᄇ", "ᄉ", "ᆮ",  "ᄁ", "ᅬ", "ᅣ",  "ᄄ", "ᆨ",  "ᄍ", "ᅧ",  "ᄏ",
                                           "ᆸ",  "ᅭ",  "(",  "ᄊ", ")",  "ᅲ", "ᅨ",  "ᄈ", "ᅱ",  "ᅯ",  "ᅫ",  "ᅰ",
                                           "ᅤ",  "~",  "\\", "[",  "]",  "/", "^",  ":",  "ㄸ", "*"};
    int num_kr_tones = 1;

    // Spanish symbols
    std::vector<std::string> es_symbols = {"N", "Q", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                                           "n", "o", "p", "s", "t", "u", "v", "w", "x", "y", "z", "ɑ", "æ", "ʃ",
                                           "ʑ", "ç", "ɯ", "ɪ", "ɔ", "ɛ", "ɹ", "ð", "ə", "ɫ", "ɥ", "ɸ", "ʊ", "ɾ",
                                           "ʒ", "θ", "β", "ŋ", "ɦ", "ɡ", "r", "ɲ", "ʝ", "ɣ", "ʎ", "ˈ", "ˌ", "ː"};
    int num_es_tones = 1;

    // French symbols
    std::vector<std::string> fr_symbols = {"\u0303", "œ", "ø", "ʁ", "ɒ", "ʌ", "ɜ", "ɐ"};
    int num_fr_tones = 1;

    // German symbols
    std::vector<std::string> de_symbols = {"ʏ", "̩"};
    int num_de_tones = 1;

    // Russian symbols
    std::vector<std::string> ru_symbols = {"ɭ", "ʲ", "ɕ", "\"", "ɵ", "^", "ɬ"};
    int num_ru_tones = 1;

    // Combine all tones
    int num_tones = num_zh_tones + num_ja_tones + num_en_tones + num_kr_tones + num_es_tones + num_fr_tones +
                    num_de_tones + num_ru_tones;

    // Language maps
    std::map<std::string, int> language_id_map = {{"ZH", 0}, {"JP", 1}, {"EN", 2}, {"ZH_MIX_EN", 3},
                                                  {"KR", 4}, {"ES", 5}, {"SP", 5}, {"FR", 6}};
    int num_languages = language_id_map.size();

    std::map<std::string, int> language_tone_start_map = {
        {"ZH", 0},
        {"ZH_MIX_EN", 0},
        {"JP", num_zh_tones},
        {"EN", num_zh_tones + num_ja_tones},
        {"KR", num_zh_tones + num_ja_tones + num_en_tones},
        {"ES", num_zh_tones + num_ja_tones + num_en_tones + num_kr_tones},
        {"SP", num_zh_tones + num_ja_tones + num_en_tones + num_kr_tones},
        {"FR", num_zh_tones + num_ja_tones + num_en_tones + num_kr_tones + num_es_tones}};
};
