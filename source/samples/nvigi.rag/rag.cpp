// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#ifdef NVIGI_WINDOWS
#include <conio.h>
#else
#include <linux/limits.h>
#endif

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <regex>
#include <thread>

namespace fs = std::filesystem;

#if NVIGI_WINDOWS
#include <windows.h>
#endif

#include <nvigi.h>
#include "nvigi_embed.h"
#include "nvigi_gpt.h"
#include "nvigi_types.h"

#if NVIGI_LINUX
#include <unistd.h>
#include <dlfcn.h>
using HMODULE = void*;
#define GetProcAddress dlsym
#define FreeLibrary dlclose
#define LoadLibraryA(lib) dlopen(lib, RTLD_LAZY)
#define LoadLibraryW(lib) dlopen(nvigi::extra::toStr(lib).c_str(), RTLD_LAZY)

#define sscanf_s sscanf
#define strcpy_s(a,b,c) strcpy(a,c)
#define strcat_s(a,b,c) strcat(a,c)
#define memcpy_s(a,b,c,d) memcpy(a,c,d)
typedef struct __LUID
{
    unsigned long LowPart;
    long HighPart;
} 	LUID;
#endif

#define DECLARE_NVIGI_CORE_FUN(F) PFun_##F* ptr_##F
#define GET_NVIGI_CORE_FUN(F) ptr_##F = (PFun_##F*)GetProcAddress(lib, #F)
DECLARE_NVIGI_CORE_FUN(nvigiInit);
DECLARE_NVIGI_CORE_FUN(nvigiShutdown);
DECLARE_NVIGI_CORE_FUN(nvigiLoadInterface);
DECLARE_NVIGI_CORE_FUN(nvigiUnloadInterface);

typedef std::vector< std::string > StringVec;
typedef std::vector< std::vector< float > > VectorStore;
typedef std::pair< size_t, float > IndexScore;
typedef std::vector< std::pair<size_t, float> > IndexScoreVec;

int max_position_embeddings = 0;    // Maximum context length that the model can process
int embedding_size = 0;             // Embedding output size of the model
const uint32_t vram = 1024 * 12;    // maximum vram available in GB
std::string modelDir = "";
HMODULE lib = nullptr;

inline std::string getExecutablePath()
{
#ifdef NVIGI_LINUX
    char exePath[PATH_MAX] = {};
    readlink("/proc/self/exe", exePath, sizeof(exePath));
    std::string searchPathW = exePath;
    searchPathW.erase(searchPathW.rfind('/'));
    return searchPathW + "/";
#else
    CHAR pathAbsW[MAX_PATH] = {};
    GetModuleFileNameA(GetModuleHandleA(NULL), pathAbsW, ARRAYSIZE(pathAbsW));
    std::string searchPathW = pathAbsW;
    searchPathW.erase(searchPathW.rfind('\\'));
    return searchPathW + "\\";
#endif
}

void loggingCallback(nvigi::LogType type, const char* msg)
{
#ifdef NVIGI_WINDOWS
    OutputDebugStringA(msg);
#endif
    std::cout << msg;
}

template<typename T>
bool unloadInterface(nvigi::PluginID feature, T*& _interface)
{
    if (_interface == nullptr ) 
        return false;

    nvigi::Result result = ptr_nvigiUnloadInterface(feature, _interface);
    if (result == nvigi::kResultOk)
    {
        _interface = nullptr;
    }
    else
    {
        loggingCallback(nvigi::LogType::eError, "Failed to unload interface");
        return false;
    }

    return true;
}

bool initIGI()
{
#ifndef DEBUG
#ifdef NVIGI_WINDOWS
    FILE* f{};
    freopen_s(&f, "NUL", "w", stderr);
#else
    freopen("dev/nul", "w", stderr);
#endif
#endif 

    auto exePathUtf8 = getExecutablePath();
#ifdef NVIGI_WINDOWS
    auto libPath = exePathUtf8 + "/nvigi.core.framework.dll";
#else
    auto libPath = exePathUtf8 + "/nvigi.core.framework.so";
#endif
    lib = LoadLibraryA(libPath.c_str());
    if (lib == nullptr)
    {
        loggingCallback(nvigi::LogType::eError, "Could not load NVIGI core library");
        return false;
    }

    GET_NVIGI_CORE_FUN(nvigiInit);
    GET_NVIGI_CORE_FUN(nvigiShutdown);
    GET_NVIGI_CORE_FUN(nvigiLoadInterface);
    GET_NVIGI_CORE_FUN(nvigiUnloadInterface);

    if (ptr_nvigiInit == nullptr || ptr_nvigiShutdown == nullptr ||
        ptr_nvigiLoadInterface == nullptr || ptr_nvigiUnloadInterface == nullptr)
    {
        loggingCallback(nvigi::LogType::eError, "Could not load NVIGI core library");
        return false;
    }

    const char* paths[] =
    {
        exePathUtf8.c_str()
    };

    nvigi::Preferences pref{};
    pref.logLevel = nvigi::LogLevel::eVerbose;
    pref.showConsole = false;
    pref.numPathsToPlugins = 1;
    pref.utf8PathsToPlugins = paths;
    pref.logMessageCallback = nullptr;
    pref.utf8PathToLogsAndData = exePathUtf8.c_str();

    auto result = ptr_nvigiInit(pref, nullptr, nvigi::kSDKVersion);
    if (result != nvigi::kResultOk)
    {
        loggingCallback(nvigi::LogType::eError, "NVIGI init failed");
        return false;
    }

    return true;
}

int32_t shutdownIGI()
{
    nvigi::Result result = ptr_nvigiShutdown();
    if (result != nvigi::kResultOk)
    {
        loggingCallback(nvigi::LogType::eError, "Error in NVIGI shutdown");
        return -1;
    }

    FreeLibrary(lib);

    return 0;
}

nvigi::InferenceInstance* createEmbedInstance(nvigi::IEmbed* iembed)
{
    nvigi::EmbedCreationParameters embedParams{};
    nvigi::CommonCreationParameters embedCommon{};
    
    embedParams.chain(embedCommon);
    embedCommon.utf8PathToModels = modelDir.c_str();
    embedCommon.numThreads = 1;
    embedCommon.vramBudgetMB = vram;
    embedCommon.modelGUID = "{5D458A64-C62E-4A9C-9086-2ADBF6B241C7}"; // e5-large-unsupervised

    // Query capabilities/models list and find the model we are interested in to use that index to find what embedding size our output data should be in.
    nvigi::EmbedCapabilitiesAndRequirements* info{};
    getCapsAndRequirements(iembed, embedParams, &info);
    if (info == nullptr)
        return nullptr;

    for (int i = 0; i < info->common->numSupportedModels; ++i)
    {
        if (strcmp(info->common->supportedModelGUIDs[i], embedCommon.modelGUID) == 0)
        {
            embedding_size = (int)info->embedding_numel[i];
            max_position_embeddings = info->max_position_embeddings[i];
            break;
        }
    }

    if (embedding_size == 0 || max_position_embeddings == 0)
        return nullptr;

    nvigi::InferenceInstance* instance{};
    auto result = iembed->createInstance(embedParams, &instance);

    return instance;
}

void destroyEmbedInstance(nvigi::IEmbed* iembed, nvigi::InferenceInstance** embedInstance)
{
    if ((embedInstance == nullptr) || (*embedInstance == nullptr))
        return;

    nvigi::Result result = iembed->destroyInstance(*embedInstance);
    if (result == nvigi::kResultOk)
        *embedInstance = nullptr;
    else
        loggingCallback(nvigi::LogType::eError, "Failed to destroy embed instance");
}

nvigi::InferenceInstance* createGPTInstance(nvigi::IGeneralPurposeTransformer* igpt)
{
    nvigi::GPTCreationParameters gptParams{};
    nvigi::CommonCreationParameters gptCommon{};
    gptCommon.utf8PathToModels = modelDir.c_str();
    gptCommon.numThreads = 16;
    gptCommon.vramBudgetMB = vram;
    gptParams.contextSize = 4096;
    gptCommon.modelGUID = "{8E31808B-C182-4016-9ED8-64804FF5B40D}"; // nemotron4-mini-instruct v0.1.3
    gptCommon.chain(gptParams);

    nvigi::CommonCapabilitiesAndRequirements* info{};
    getCapsAndRequirements(igpt, gptCommon, &info);
    if (info == nullptr)
        return nullptr;

    nvigi::InferenceInstance* instance{};
    auto result = igpt->createInstance(gptCommon, &instance);

    return instance;
}

void destroyGPTInstance(nvigi::IGeneralPurposeTransformer* igpt, nvigi::InferenceInstance** gptInstance)
{
    if ((gptInstance == nullptr) || (*gptInstance == nullptr))
        return;

    nvigi::Result result = igpt->destroyInstance(*gptInstance);
    if (result == nvigi::kResultOk)
        *gptInstance = nullptr;
    else
        loggingCallback(nvigi::LogType::eError, "Failed to destroy gpt instance");
}

void getCompletion(nvigi::InferenceInstance* instance, const std::string& prompt, std::string& answer)
{
    answer = "";

    nvigi::CpuData text(prompt.length() + 1, (void*)prompt.c_str());
    nvigi::InferenceDataText data(text);
    char buffer[1024];
    nvigi::CpuData text1(1024, (void*)buffer);
    nvigi::InferenceDataText data1(text1);
    std::vector<nvigi::InferenceDataSlot> inSlots = { {nvigi::kGPTDataSlotUser, &data} };
    std::vector<nvigi::InferenceDataSlot> outSlots = { {nvigi::kGPTDataSlotResponse, &data1} };
    nvigi::InferenceDataSlotArray inputs = { inSlots.size(), inSlots.data() };
    nvigi::InferenceDataSlotArray outputs = { outSlots.size(), outSlots.data() };

    nvigi::GPTRuntimeParameters runtime{};
    runtime.seed = -1;
    runtime.tokensToPredict = 200;
    runtime.interactive = false;

    struct UserDataBlock
    {
        std::atomic<bool> done = false;
        std::string response;
        bool terminator_found = false;
    };
    UserDataBlock userData;

    nvigi::InferenceExecutionContext ctx{};
    ctx.instance = instance;
    ctx.callbackUserData = &userData;
    ctx.callback = [](const nvigi::InferenceExecutionContext* ctx, nvigi::InferenceExecutionState state, void* userData)->nvigi::InferenceExecutionState
        {
            UserDataBlock* userDataBlock = static_cast<UserDataBlock*>(userData);
            if (ctx)
            {
                auto slots = ctx->outputs;
                const nvigi::InferenceDataText* text{};
                slots->findAndValidateSlot(nvigi::kGPTDataSlotResponse, &text);
                auto response = std::string((const char*)text->getUTF8Text());

                if (response == "</s>")
                {
                    // For Nemotron, the </s> character denotes end of stream.  
                    // Must wait for state to be kInferenceExecutionStateDone before evaluation is finished though.
                    userDataBlock->terminator_found = true;
                }

                if (state == nvigi::kInferenceExecutionStateDone)
                    response += "\n\n";

                if ( !userDataBlock->terminator_found )
                {
                    userDataBlock->response += response;
                    loggingCallback(nvigi::LogType::eInfo, response.c_str());
                }
            }
            userDataBlock->done.store(state == nvigi::kInferenceExecutionStateDone);
            return state;
        };

    ctx.inputs = &inputs;
    ctx.outputs = &outputs;
    ctx.runtimeParameters = runtime;

    if (ctx.instance->evaluate(&ctx) != nvigi::kResultOk)
    {
        loggingCallback(nvigi::LogType::eError, "GPT evaluate failed");
    }
    // ctx is held in this scope, so we can't let it go out of scope while the LLM is evaluating.
    while (!userData.done);

    answer = userData.response;
}

// Check if a character is a valid ASCII character
static bool isValidASCII(char ch) 
{
    return (ch >= 0 && ch <= 127); // ASCII range (valid UTF-8 single byte)
}

// Remove non-UTF-8 characters inside a string
static std::string removeNonUTF8(const std::string& input) 
{
    std::string output;
    for (char ch : input) 
    {
        // Remove non-UTF-8 characters
        if (isValidASCII(ch)) 
        {
            output += ch;
        }
        else 
        {
            // Optional - this could be rather noisy in the logs
            // loggingCallback(nvigi::LogType::eWarn, "Non utf8 character has been removed");
        }
    }
    return output;
}

// Count the number of prompts inside a string. The separator indicates how to differenciate prompts
static int countLines(const std::string& s, const std::string& separator = "\n") 
{
    size_t start = 0;
    size_t end = s.find(separator);

    int counter_lines = 0;

    while (end != std::string::npos) 
    {
        ++counter_lines;
        start = end + separator.length();
        end = s.find(separator, start);
    }

    // Add the last part
    ++counter_lines; 

    return counter_lines;
}

void cosSimScore(const std::vector<float>& vec1, const VectorStore& embeddings, IndexScoreVec& out_scores)
{
    out_scores.clear();
    auto calc_norm = [](const std::vector<float>& vec)->float
        {
            float sum = 0.0f;
            for (float elem : vec)
            {
                sum += elem * elem;
            }
            return std::sqrt(sum);
        };

    float vec1_norm = calc_norm(vec1);
    if (vec1_norm == 0.0f)
        vec1_norm = std::numeric_limits<float>::epsilon();

    size_t index = 0;
    for (VectorStore::const_iterator it = embeddings.begin(); it != embeddings.end(); ++it)
    {
        const std::vector<float>& vec2 = *it;
        float vec2_norm = calc_norm(vec2);
        if (vec2_norm == 0.0f)
            vec2_norm = std::numeric_limits<float>::epsilon();

        float dotProduct = 0.0f;
        for (int i = 0; i < vec1.size(); i++) {
            dotProduct += vec1[i] * vec2[i];
        }

        float cos_sim = dotProduct / (vec1_norm * vec2_norm);

        out_scores.push_back(std::make_pair(index, cos_sim));
        index++;
    }
}

bool loadText(const std::string& filepath, std::string& dest)
{
    std::ifstream file(filepath);
    if (!file) 
    {
        std::cerr << "Error opening file!" << std::endl;
        return false;
    }

    // Read the entire file into a string
    std::ostringstream ss;
    ss << file.rdbuf();
    dest = ss.str();

    return true;
}

void embed(nvigi::IEmbed* iembed, nvigi::InferenceInstance* instance, const std::string& input, std::vector<float>& output_embeddings)
{
    int n_prompts = countLines(input, nvigi::prompts_sep);
    output_embeddings.clear();

    output_embeddings.resize(n_prompts * embedding_size);

    nvigi::CpuData text(input.length() + 1, (void*)input.c_str());
    nvigi::InferenceDataText input_prompt(text);

    nvigi::CpuData cpu_data;
    cpu_data.sizeInBytes = output_embeddings.size() * sizeof(float);
    cpu_data.buffer = output_embeddings.data();
    nvigi::InferenceDataByteArray output_param(cpu_data);
    std::vector<nvigi::InferenceDataSlot> inSlots = { {nvigi::kEmbedDataSlotInText, &input_prompt} };
    std::vector<nvigi::InferenceDataSlot> outSlots = { {nvigi::kEmbedDataSlotOutEmbedding, &output_param} };
    nvigi::InferenceDataSlotArray inputs = { inSlots.size(), inSlots.data() };
    nvigi::InferenceDataSlotArray outputs = { outSlots.size(), outSlots.data() };

    nvigi::InferenceExecutionContext ctx{};
    ctx.instance = instance;
    std::atomic<bool> done = false;
    ctx.callbackUserData = &done;
    ctx.callback = [](const nvigi::InferenceExecutionContext* ctx, nvigi::InferenceExecutionState state, void* userData)->nvigi::InferenceExecutionState 
        { 
            std::atomic<bool>* done = static_cast<std::atomic<bool>*>(userData);
            done->store(state == nvigi::kInferenceExecutionStateDone);
            return state; 
        };
    ctx.inputs = &inputs;
    ctx.outputs = &outputs;
    nvigi::Result res = ctx.instance->evaluate(&ctx);
    if (res != nvigi::kResultOk)
    {
        loggingCallback(nvigi::LogType::eError, "Embed evaluate failed");
    }

    while (!done);
}

void splitString(const std::string& src, const std::string& delimiter, std::vector<std::string>& dest)
{
    size_t start = 0, end, delimLength = delimiter.length();
    while ((end = src.find(delimiter, start)) != std::string::npos)
    {
        dest.push_back(src.substr(start, end - start));
        start = end + delimLength;
    }
    dest.push_back(src.substr(start));
}

size_t replaceSubstring(std::string& str, const std::string& from, const std::string& to)
{
    size_t num_replaced = 0;
    size_t startPos = 0;
    while ((startPos = str.find(from, startPos)) != std::string::npos)
    {
        str.replace(startPos, from.length(), to);
        startPos += to.length(); // Move past the last replaced substring
        ++num_replaced;
    }
    return num_replaced;
}

void createTextEmbeddings(nvigi::IEmbed* iembed, nvigi::InferenceInstance* instance, const std::string& textfile, VectorStore& vector_store, StringVec& text_corpus)
{
    std::string text;
    if (!loadText(textfile, text))
    {
        std::string errorMsg = std::string("Failed to load ") + textfile;
        loggingCallback(nvigi::LogType::eError, errorMsg.c_str() );
    }

    text = removeNonUTF8(text);
    splitString(text, "\n\n", text_corpus);

    std::string prepped_text = text;
    size_t num_replaced = replaceSubstring(prepped_text, "\n\n", nvigi::prompts_sep);
    size_t num_entries = num_replaced + 1;

    std::vector<float> output_embeddings;
    embed(iembed, instance, prepped_text, output_embeddings);

    float* full_embedding = output_embeddings.data();

    for (size_t i = 0; i < num_entries; ++i)
    {
        vector_store.push_back(std::vector<float>(full_embedding + i * embedding_size, full_embedding + (i + 1) * embedding_size));
    }
}

void retrieveContext(nvigi::IEmbed* iembed, nvigi::InferenceInstance* instance, const std::string& input_prompt, VectorStore& vector_store, StringVec& text_corpus, size_t top_n, std::string& out_context)
{
    out_context = "";
    std::vector<float> player_prompt_embeddings;
    std::string sanitized_prompt = removeNonUTF8(input_prompt);
    embed(iembed, instance, input_prompt, player_prompt_embeddings);

    IndexScoreVec scores;
    cosSimScore(player_prompt_embeddings, vector_store, scores);
    std::sort(scores.begin(), scores.end(), [](const IndexScore& a, const IndexScore& b) { return a.second > b.second; });
    for (size_t i = 0; i < top_n; ++i)
    {
        size_t context_index = scores[i].first;
        out_context += text_corpus[context_index];
        out_context += "\n\n";
    }
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        loggingCallback(nvigi::LogType::eError, "nvigi.rag <path to models> <text file>");
        return -1;
    }
    modelDir = argv[1];
    std::string textFilePath = argv[2];

    if (!initIGI())
        return -1;

    //////////////////////////////////////////////////////////////////////////////
    //! Init Plugin Interfaces and Instances

    //! Embed Interface
    nvigi::IEmbed* iembed{};
    if (NVIGI_FAILED(result, nvigiGetInterfaceDynamic(nvigi::plugin::embed::ggml::cuda::kId, &iembed, ptr_nvigiLoadInterface)))
    {
        loggingCallback(nvigi::LogType::eError, "Could not query Embed interface");
        return -1;
    }

    //! Embed Instance
    nvigi::InferenceInstance* embedInstance = createEmbedInstance(iembed);
    if (embedInstance == nullptr)
    {
        loggingCallback(nvigi::LogType::eError, "Could not create Embed instance");
        return -1;
    }

    //! GPT Interface
    nvigi::IGeneralPurposeTransformer* igpt{};
    if (NVIGI_FAILED(result, nvigiGetInterfaceDynamic(nvigi::plugin::gpt::ggml::cuda::kId, &igpt, ptr_nvigiLoadInterface)))
    {
        loggingCallback(nvigi::LogType::eError, "Could not query GPT interface");
        return -1;
    }

    //! GPT Instance
    nvigi::InferenceInstance* gptInstance = createGPTInstance(igpt);
    if (gptInstance == nullptr)
    {
        loggingCallback(nvigi::LogType::eError, "Could not create GPT instance");
        return -1;
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Rag Basic Flow

    //! Embed text corpus

    // the numeric embeddings of each text section
    VectorStore text_embedding;
    // the textual passages that map to each embedding
    StringVec text_corpus;
    createTextEmbeddings(iembed, embedInstance, textFilePath, text_embedding, text_corpus);

    std::cout << "\nAsk your questions of the document.  Type 'exit' by itself to end the program\n";

    //! Set the system prompt
    std::string system_prompt = "You are a helpful AI assistant.  Please answer the questions from the user given the context provided.  If you are unsure or the context does not have the answer to the question, say so rather than giving a wrong answer";

    //! Get the user's prompt
    std::string user_prompt;
    std::cout << "\nUser: ";
    std::getline(std::cin, user_prompt);

    //! Loop until the user wants to exit the conversation.
    while (user_prompt != "exit")
    {
        //! Get the TopN context from the text that match the user prompt
        std::string context = "";
        size_t top_n = 5;
        retrieveContext(iembed, embedInstance, user_prompt, text_embedding, text_corpus, top_n, context);

        //! This prompt template is built around Nemotron4-mini-instruct.  Modify it as needed for other LLMs
        //! Note white space and new lines are important to correct usage of the prompt template.  Pay attention.
        std::string prompt_template =
            "<extra_id_0>System\n"
            + system_prompt + "\n"
            "<context>\n" + context + "\n</context>\n"
            "<extra_id_1>User\n"
            + user_prompt + "\n"
            "<extra_id_1>Assistant\n";

        //! Pass the prompt to the LLM for completion.
        std::string answer = "";
        getCompletion(gptInstance, prompt_template, answer);

        std::cout << "\nUser: ";
        std::getline(std::cin, user_prompt);
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Shutdown NVIGI

    destroyGPTInstance(igpt, &gptInstance);

    destroyEmbedInstance(iembed, &embedInstance);

    unloadInterface(nvigi::plugin::gpt::ggml::cuda::kId, igpt);

    unloadInterface(nvigi::plugin::embed::ggml::cuda::kId, iembed);

    return shutdownIGI();
}