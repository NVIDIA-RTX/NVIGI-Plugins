// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <iostream>
#include <thread>
#include <memory>
#include <queue>
#include <type_traits>
#include <mutex>
#include <atomic>
#include <chrono>


#include <map>

namespace fs = std::filesystem;

#if NVIGI_WINDOWS
#include <windows.h>
#include <source/utils/nvigi.dsound/player.h>
// D3D12 Agility SDK exports
extern "C" __declspec(dllexport) UINT         D3D12SDKVersion = 615;
extern "C" __declspec(dllexport) const char* D3D12SDKPath = ".\\D3D12\\";
#endif

#include <nvigi.h>
#include "nvigi_asr_whisper.h"
#include "nvigi_gpt.h"
#include "nvigi_tts.h"
#include "nvigi_cloud.h"
#include <source/utils/nvigi.dsound/recorder.h>
#include <nvigi_stl_helpers.h>

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
#define GET_NVIGI_CORE_FUN(lib, F) ptr_##F = (PFun_##F*)GetProcAddress(lib, #F)
DECLARE_NVIGI_CORE_FUN(nvigiInit);
DECLARE_NVIGI_CORE_FUN(nvigiShutdown);
DECLARE_NVIGI_CORE_FUN(nvigiLoadInterface);
DECLARE_NVIGI_CORE_FUN(nvigiUnloadInterface);

inline std::vector<int16_t> read(const char* fname)
{
    try
    {
        fs::path p(fname);
        size_t file_size = fs::file_size(p);
        std::vector<int16_t> ret_buffer(file_size / 2);
#ifdef NVIGI_LINUX
        std::fstream file(fname, std::ios::binary | std::ios::in);
#else
        std::fstream file(fname, std::ios::binary | std::ios::in);
#endif
        file.read((char*)ret_buffer.data(), file_size);
        return ret_buffer;
    }
    catch (...)
    {
    }
    return {};
}

class CommandLineParser {
public:
    struct Command {
        std::string short_name;
        std::string long_name;
        std::string description;
        std::string default_value;
        std::string value;
        bool required = false;
    };

    void add_command(const std::string& short_name, const std::string& long_name,
        const std::string& description, const std::string& default_value = "",
        bool required = false) {
        if (!short_name.empty())
        {
            commands[short_name] = { short_name, long_name, description, default_value, default_value, required };
        }
        commands[long_name] = { short_name, long_name, description, default_value, default_value, required };
    }

    void parse(int argc, char* argv[]) {
        std::vector<std::string> args(argv + 1, argv + argc);
        for (size_t i = 0; i < args.size(); ++i) {
            std::string arg = args[i];

            if (arg[0] == '-') {
                std::string key = (arg[1] == '-') ? arg.substr(2) : arg.substr(1); // Long or short name
                auto it = commands.find(key);
                if (it == commands.end()) {
                    throw std::invalid_argument("Unknown command: " + arg);
                }

                Command& cmd = it->second;
                if (i + 1 < args.size() && args[i + 1][0] != '-') {
                    cmd.value = args[++i]; // Take the next argument as the value
                    auto altKey = cmd.long_name == key ? cmd.short_name : cmd.long_name;
                    commands[altKey].value = cmd.value;
                }
                else if (cmd.default_value.empty()) {
                    throw std::invalid_argument("Missing value for command: " + arg);
                }
            }
            else {
                throw std::invalid_argument("Unexpected argument format: " + arg);
            }
        }

        // Check required commands
        for (const auto& [key, cmd] : commands) {
            if (cmd.required && cmd.value == cmd.default_value) {
                throw std::invalid_argument("Missing required command: --" + cmd.long_name);
            }
        }
    }

    std::string get(const std::string& name) const {
        auto it = commands.find(name);
        if (it == commands.end()) {
            throw std::invalid_argument("Unknown command: " + name);
        }
        return it->second.value;
    }

    bool has(const std::string& name) const {
        auto it = commands.find(name);
        if (it == commands.end()) {
            return false;
        }
        return !it->second.value.empty();
    }

    void print_help(const std::string& program_name) const {
        std::cout << "Usage: " << program_name << " [options]\n";
        for (const auto& [key, cmd] : commands) {
            if (key == cmd.long_name) { // Print each command only once
                std::string tmp;
                if (!cmd.short_name.empty())
                    tmp = "  -" + cmd.short_name + ", --" + cmd.long_name;
                else
                    tmp = "  --" + cmd.long_name;
                std::string spaces(std::max(0, 20 - (int)tmp.size()), ' ');
                std::cout << tmp << spaces << cmd.description << " (default: " << cmd.default_value << ")\n";
            }
        }
    }

private:
    std::map<std::string, Command> commands;
};

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

void loggingPrint(nvigi::LogType type, const char* msg)
{
#ifdef NVIGI_WINDOWS
    OutputDebugStringA(msg);
#endif
    std::cout << msg;
}

void loggingCallback(nvigi::LogType type, const char* msg)
{
#ifndef NVIGI_DEBUG
    if (type == nvigi::LogType::eError)
#endif
        loggingPrint(type, msg);
}

struct WAVHeader {
    // RIFF Chunk Descriptor
    char riff[4] = { 'R', 'I', 'F', 'F' };  // RIFF Header Magic
    uint32_t chunkSize;                     // RIFF Chunk Size
    char wave[4] = { 'W', 'A', 'V', 'E' };  // WAVE Header

    // "fmt" sub-chunk
    char fmt[4] = { 'f', 'm', 't', ' ' };   // FMT header
    uint32_t subchunk1Size = 16;          // Size of the fmt chunk
    uint16_t audioFormat = 1;             // Audio format (1 = PCM)
    uint16_t numChannels;                 // Number of channels
    uint32_t sampleRate;                  // Sampling Frequency in Hz
    uint32_t byteRate;                    // Bytes per second (sampleRate * numChannels * bitsPerSample/8)
    uint16_t blockAlign;                  // Bytes per sample (numChannels * bitsPerSample/8)
    uint16_t bitsPerSample;               // Bits per sample

    // "data" sub-chunk
    char data[4] = { 'd', 'a', 't', 'a' };  // DATA header
    uint32_t dataSize;                    // Size of the data section
};

// Write a WAV file
void writeWav(std::vector<int16_t> dataBuffer, std::string outputPath, int sampleRate, int bitsPerSample) {
    WAVHeader header;
    header.numChannels = 1;
    header.sampleRate = sampleRate;
    header.bitsPerSample = bitsPerSample;  // Assuming 16-bit samples
    header.byteRate = header.sampleRate * header.numChannels * (header.bitsPerSample / 8);
    header.blockAlign = header.numChannels * (header.bitsPerSample / 8);
    header.dataSize = static_cast<uint32_t>(dataBuffer.size() * sizeof(int16_t));
    header.chunkSize = 36 + header.dataSize;

    std::ofstream outFile(outputPath, std::ios::binary);
    if (!outFile) {
        std::cerr << "Could not open file for writing: " << outputPath << std::endl;
        return;
    }

    // Write the header
    outFile.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write the audio data
    outFile.write(reinterpret_cast<const char*>(dataBuffer.data()), (unsigned long)header.dataSize);

    outFile.close();
}


// Check if a character is a valid ASCII character
inline bool isValidASCII(char ch) {
    return (ch >= 0 && ch <= 127); // ASCII range (valid UTF-8 single byte)
}

// Function template limited to float and int16_t only. Debug function
// Save audio file to .wav and/or play audio
// If floating point, it will be converted to int16_t by multiplied to 32767
template <typename T, typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, int16_t>::value, int>::type = 0>
void savePlayAudioData(const std::vector<T> audio_data, const std::string output_path,
    const int sampling_rate, std::mutex & mtxPlayAudio,
    const bool playAudio = false, const bool saveWav = true) {


    std::vector<int16_t> audio_data_int16;
    if constexpr (std::is_same<T, float>::value) {
        for (const float& val : audio_data) {
            audio_data_int16.push_back(static_cast<int16_t>(val * 32767));
        }
    }
    else {
        audio_data_int16 = audio_data;
    }

    constexpr int bytesPerSample = 16;

#ifdef NVIGI_WINDOWS
    if (playAudio) {
        mtxPlayAudio.lock();
        nvigi::utils::Player player(bytesPerSample, sampling_rate);
        nvigi::utils::Buffer buffer(player,
            (const T* const)(audio_data_int16.data()),
            (DWORD)(audio_data_int16.size() * sizeof(int16_t)));
        buffer.Play();
        buffer.Wait();
        mtxPlayAudio.unlock();
    }
#endif

    if (saveWav) {
        writeWav(audio_data_int16, output_path, sampling_rate, bytesPerSample);
    }
}

// Remove non-UTF-8 characters inside a string
inline std::string removeNonUTF8(const std::string& input) {
    std::string output;
    int countNonUtf8 = 0;
    for (char ch : input) {
        // Remove non-UTF-8 characters
        if (isValidASCII(ch)) {
            output += ch;
        }
        else {
            countNonUtf8++;
        }
    }
    return output;
}

struct NVIGIAppCtx
{
    HMODULE coreLib{};
    nvigi::IAutoSpeechRecognition* iasr{};
    nvigi::InferenceInstance* asr{};
    nvigi::IGeneralPurposeTransformer* igpt{};
    nvigi::PluginID gptId{};
    nvigi::InferenceInstance* gpt{};
    nvigi::ITextToSpeech* itts{};
    nvigi::PluginID ttsId{};
    nvigi::InferenceInstance* tts{};
};

static NVIGIAppCtx nvigiCtx;
constexpr uint32_t n_threads = 4;

///////////////////////////////////////
//! NVIGI Init and Shutdown

int InitNVIGI(const std::string& pathToSDKUtf8)
{
    loggingPrint(nvigi::LogType::eInfo, "Initializing NVIGI\n");

#ifdef NVIGI_WINDOWS
    auto libPath = pathToSDKUtf8 + "/nvigi.core.framework.dll";
#else
    auto libPath = pathToSDKUtf8 + "/nvigi.core.framework.so";
#endif
    nvigiCtx.coreLib = LoadLibraryA(libPath.c_str());
    if (nvigiCtx.coreLib == nullptr)
    {
        loggingPrint(nvigi::LogType::eError, "Could not load NVIGI core library");
        return -1;
    }

    GET_NVIGI_CORE_FUN(nvigiCtx.coreLib, nvigiInit);
    GET_NVIGI_CORE_FUN(nvigiCtx.coreLib, nvigiShutdown);
    GET_NVIGI_CORE_FUN(nvigiCtx.coreLib, nvigiLoadInterface);
    GET_NVIGI_CORE_FUN(nvigiCtx.coreLib, nvigiUnloadInterface);

    if (ptr_nvigiInit == nullptr || ptr_nvigiShutdown == nullptr ||
        ptr_nvigiLoadInterface == nullptr || ptr_nvigiUnloadInterface == nullptr)
    {
        loggingPrint(nvigi::LogType::eError, "Could not load NVIGI core library");
        return -1;
    }

    const char* paths[] =
    {
        pathToSDKUtf8.c_str()
    };

    nvigi::Preferences pref{};
    pref.logLevel = nvigi::LogLevel::eVerbose;
    pref.showConsole = false;
    pref.numPathsToPlugins = 1;
    pref.utf8PathsToPlugins = paths;
    pref.logMessageCallback = loggingCallback; // avoid duplicating logs in the console
    pref.utf8PathToLogsAndData = pathToSDKUtf8.c_str();

    if (NVIGI_FAILED(result, ptr_nvigiInit(pref, nullptr, nvigi::kSDKVersion)))
    {
        loggingPrint(nvigi::LogType::eError, "NVIGI init failed");
        return -1;
    }

    loggingPrint(nvigi::LogType::eInfo, "Initializing NVIGI succeeded\n");
    return 0;
}

int ShutdownNVIGI()
{
    if (NVIGI_FAILED(result, ptr_nvigiShutdown()))
    {
        loggingPrint(nvigi::LogType::eError, "Error in 'nvigiShutdown'");
        return -1;
    }

    FreeLibrary(nvigiCtx.coreLib);

    return 0;
}

///////////////////////////////////////
//! ASR Init and Release

int InitASR(const std::string& modelDir, const std::string& guidASR, size_t vramBudgetMB)
{
    loggingPrint(nvigi::LogType::eInfo, "Initializing ASR\n");

    //! ASR Interface and Instance
    if (NVIGI_FAILED(result, nvigiGetInterfaceDynamic(nvigi::plugin::asr::ggml::cuda::kId, &nvigiCtx.iasr, ptr_nvigiLoadInterface)))
    {
        loggingPrint(nvigi::LogType::eError, "Could not query ASR interface");
        return -1;
    }

    nvigi::ASRWhisperCreationParameters asrParams{};
    nvigi::CommonCreationParameters asrCommon{};
    {
        //! IMPORTANT: Please see the GPT section below for more detailed explanation on how VRAM/modelGUID etc impact create instance
        asrCommon.utf8PathToModels = modelDir.c_str();
        asrCommon.numThreads = n_threads;
        asrCommon.vramBudgetMB = vramBudgetMB;
        asrCommon.modelGUID = guidASR.c_str();
        if (NVIGI_FAILED(result, asrCommon.chain(asrParams)))
        {
            loggingPrint(nvigi::LogType::eError, "ASR param chaining failed");
            return -1;
        }
    }

    if (NVIGI_FAILED(result, nvigiCtx.iasr->createInstance(asrCommon, &nvigiCtx.asr)))
    {
        loggingPrint(nvigi::LogType::eError, "Could not create ASR instance");
        return -1;
    }

    loggingPrint(nvigi::LogType::eInfo, "Initializing ASR succeeded\n");

    return 0;
}

int ReleaseASR()
{
    nvigiCtx.iasr->destroyInstance(nvigiCtx.asr);

    // Hard-coded to local
    if (NVIGI_FAILED(result, ptr_nvigiUnloadInterface(nvigi::plugin::asr::ggml::cuda::kId, nvigiCtx.iasr)))
    {
        loggingPrint(nvigi::LogType::eError, "Error in 'nvigiUnloadInterface'");
        return -1;
    }

    return 0;
}

///////////////////////////////////////
//! GPT Init and Release

int InitGPT(const std::string& modelDir, const std::string& gptMode, const std::string& cloudToken, const std::string& guidGPT, size_t vramBudgetMB)
{
    loggingPrint(nvigi::LogType::eInfo, "Initializing GPT\n");

    //! GPT Interface and Instance
    //! When using GPT cloud it does not matter which endpoint we are going to hit, they all use the same REST based backend
    nvigiCtx.gptId = gptMode == "local" ? nvigi::plugin::gpt::ggml::cuda::kId : nvigi::plugin::gpt::cloud::rest::kId;
    if (NVIGI_FAILED(result, nvigiGetInterfaceDynamic(nvigiCtx.gptId, &nvigiCtx.igpt, ptr_nvigiLoadInterface)))
    {
        loggingPrint(nvigi::LogType::eError, "Could not query GPT interface");
        return -1;
    }

    // Cloud specific
    nvigi::RESTParameters restParams{};
    // GPT specific
    nvigi::GPTCreationParameters gptParams{};
    gptParams.contextSize = 4096;
    // Common
    nvigi::CommonCreationParameters gptCommon{};
    gptCommon.utf8PathToModels = modelDir.c_str();
    gptCommon.numThreads = n_threads;
    // Chain together specific and common
    if (NVIGI_FAILED(result, gptCommon.chain(gptParams)))
    {
        loggingPrint(nvigi::LogType::eError, "GPT param chaining failed");
        return -1;
    }

    //! Obtain capabilities and requirements for GPT model(s)
    //! 
    //! Few options here:
    //! 
    //! LOCAL
    //! 
    //! * provide specific model GUID and VRAM budget and check if that particular model can run within the budget
    //! * provide null model GUID and VRAM budget to get a list of models that can run within the budget
    //! * provide null model GUID and 'infinite' (SIZE_MAX) VRAM budget to get a list of ALL models
    //! 
    //! CLOUD (VRAM ignored)
    //! 
    //! * provide specific model GUID to obtain CloudCapabilities which include URL and JSON request body for the endpoint used by the model
    //! * provide null model GUID to get a list of ALL models (CloudCapabilities in this case will NOT provide any info)
    //! 

    //! Here we are selection option #1 - specific model GUID and VRAM budget
    //! 
    //! To obtain all models we could do something like this:
    //! 
    //! gptCommon.modelGUID = nullptr;
    //! gptCommon.vramBudgetMB = SIZE_MAX;
    //! 
    gptCommon.modelGUID = guidGPT.c_str();
    gptCommon.vramBudgetMB = vramBudgetMB;

    nvigi::CommonCapabilitiesAndRequirements* caps{};
    if (NVIGI_FAILED(result, getCapsAndRequirements(nvigiCtx.igpt, gptCommon, &caps)))
    {
        loggingPrint(nvigi::LogType::eError, "'getCapsAndRequirements' failed");
        return -1;
    }

    //! We provided model GUID and VRAM budget so caps and requirements will contain just one model, assuming VRAM budget is sufficient or if cloud backend is selected!
    //! 
    //! NOTE: This will be >=1 if we provide null as modelGUID in common creation parameters
    if (caps->numSupportedModels != 1)
    {
        loggingPrint(nvigi::LogType::eError, "'getCapsAndRequirements' failed to find our model or model cannot run on system given the VRAM restrictions");
        return -1;
    }

    //! Cloud caps should be chained here if we are using cloud plugin
    //! 
    //! Note that instead of doing '&caps->_base' we use helper operator hence '*caps'
    auto* ccaps = findStruct<nvigi::CloudCapabilities>(*caps);
    if (ccaps)
    {
        //! Cloud parameters
        //! 
        //! IMPORTANT: --token MUST be provided on the command line

        if (cloudToken.empty())
        {
            loggingPrint(nvigi::LogType::eError, "--token parameter must be provided when using GPT cloud path");
            exit(1);
        }

        restParams.url = ccaps->url;
        restParams.authenticationToken = cloudToken.c_str();
        restParams.verboseMode = false; // Set to true to debug issues with connection protocol (if any)        
        if (NVIGI_FAILED(result, gptCommon.chain(restParams)))
        {
            loggingPrint(nvigi::LogType::eError, "REST param chaining failed");
            return -1;
        }
    }

    if (NVIGI_FAILED(result, nvigiCtx.igpt->createInstance(gptCommon, &nvigiCtx.gpt)))
    {
        loggingPrint(nvigi::LogType::eError, "Could not create GPT instance");
        return -1;
    }

    loggingPrint(nvigi::LogType::eInfo, "Initializing GPT succeeded\n");

    return 0;
}

int ReleaseGPT()
{
    nvigiCtx.igpt->destroyInstance(nvigiCtx.gpt);

    // Hard-coded to local
    if (NVIGI_FAILED(result, ptr_nvigiUnloadInterface(nvigiCtx.gptId, nvigiCtx.igpt)))
    {
        loggingPrint(nvigi::LogType::eError, "Error in 'nvigiUnloadInterface'");
        return -1;
    }

    return 0;
}

///////////////////////////////////////
//! TTS Init and Release

#ifdef NVIGI_WINDOWS
int InitTTS(const std::string& modelDir, 
    const std::string& extendedPhonemeDict, const std::string& guidTTS, size_t vramBudgetMB)
{
    loggingPrint(nvigi::LogType::eInfo, "Initializing TTS\n");

    //! TTS Interface and Instance
    //! Detect backend based on GUID
    //! GGML: {16EEB8EA-55A8-4F40-BECE-CE995AF44101}
    //! TRT:  {81320D1D-DF3C-4CFC-B9FA-4D3FF95FC35F}
    nvigiCtx.ttsId = (guidTTS == "{16EEB8EA-55A8-4F40-BECE-CE995AF44101}" || guidTTS == "{3D52FDC0-5B6D-48E1-B108-84D308818602}") ? 
        nvigi::plugin::tts::asqflow_ggml::cuda::kId : nvigi::plugin::tts::asqflow_trt::kId;
    if (NVIGI_FAILED(result, nvigiGetInterfaceDynamic(nvigiCtx.ttsId, &nvigiCtx.itts, ptr_nvigiLoadInterface)))
    {
        loggingPrint(nvigi::LogType::eError, "Could not query TTS interface");
        return -1;
    }

    nvigi::TTSCreationParameters ttsParams{};
    nvigi::TTSASqFlowCreationParameters asqfParams{};
    nvigi::CommonCreationParameters ttsCommon{};

   
    {
        asqfParams.extendedPhonemesDictPath = extendedPhonemeDict.c_str();

        ttsCommon.utf8PathToModels = modelDir.c_str();
        ttsCommon.numThreads = n_threads;
        ttsCommon.vramBudgetMB = vramBudgetMB;

        //! Model is the same regardless of the backend
        ttsCommon.modelGUID = guidTTS.c_str();
        if (NVIGI_FAILED(result, ttsCommon.chain(ttsParams)))
        {
            loggingPrint(nvigi::LogType::eError, "TTS param chaining failed");
            return -1;
        }
        if (NVIGI_FAILED(result, ttsCommon.chain(asqfParams)))
        {
            loggingPrint(nvigi::LogType::eError, "TTS ASquaredFlow param chaining failed");
            return -1;
        }
    }

    if (NVIGI_FAILED(result, nvigiCtx.itts->createInstance(ttsCommon, &nvigiCtx.tts)))
    {
        loggingPrint(nvigi::LogType::eError, "Could not create TTS instance");
        return -1;
    }

    loggingPrint(nvigi::LogType::eInfo, "Initializing TTS succeeded\n");

    return 0;
}

int ReleaseTTS()
{
    nvigiCtx.itts->destroyInstance(nvigiCtx.tts);
    // Can be GGML or TRT
    if (NVIGI_FAILED(result, ptr_nvigiUnloadInterface(nvigiCtx.ttsId, nvigiCtx.itts)))
    {
        loggingPrint(nvigi::LogType::eError, "Error in 'nvigiUnloadInterface'");
        return -1;
    }

    return 0;
}
#endif

///////////////////////////////////////
//! Full pipeline inference context

struct BasicCallbackCtx
{
    std::mutex callbackMutex;
    std::mutex ttsCallbackMutex;
    std::condition_variable asrCallbackCV;
    std::condition_variable gptCallbackCV;
    std::condition_variable ttsCallbackCV;
    std::atomic<nvigi::InferenceExecutionState> asrCallbackState = nvigi::kInferenceExecutionStateDataPending;
    std::atomic<nvigi::InferenceExecutionState> gptCallbackState = nvigi::kInferenceExecutionStateDataPending;
    std::atomic<nvigi::InferenceExecutionState> ttsCallbackState = nvigi::kInferenceExecutionStateDataPending;
    std::string asrOutput;
    std::string gptOutput;

    std::string ttsInput;
    nvigi::InferenceExecutionContext ttsExecCtx{};
    nvigi::InferenceDataTextSTLHelper dataTextTTS;
    size_t posLastSpace = 0; // used to handle TTS input chunks
    size_t posLastPeriod = 0; // used to handle TTS input chunks
    std::vector<std::int16_t> ttsOutput;
    std::queue< std::unique_ptr<std::thread>> playAudioThreads;
    std::mutex mtxPlayAudio;
    std::mutex mtxttsInput;
    std::chrono::high_resolution_clock::time_point startTimeToFirstAudio;

    nvigi::InferenceInstance* gptInstance{};
    nvigi::InferenceInstance* asrInstance{};
    nvigi::InferenceInstance* ttsInstance{};
    bool conversationInitialized = false;
};

///////////////////////////////////////
//! ASR inference

nvigi::InferenceExecutionState ASRInferenceDataCallback(const nvigi::InferenceExecutionContext* ctx, nvigi::InferenceExecutionState state, void* userData)
{
    auto cbkCtx = (BasicCallbackCtx*)userData;
    std::scoped_lock lck(cbkCtx->callbackMutex);

    // Outputs from ASR
    auto slots = ctx->outputs;
    const nvigi::InferenceDataText* text{};
    slots->findAndValidateSlot(nvigi::kASRWhisperDataSlotTranscribedText, &text);
    auto response = std::string(text->getUTF8Text());
    cbkCtx->asrOutput += response;

    cbkCtx->asrCallbackState.store(state);
    cbkCtx->asrCallbackCV.notify_one();

    return state;
}

int ASRInference(BasicCallbackCtx& cbkCtx, nvigi::InferenceDataAudioSTLHelper& audioData, std::string& gptInputText)
{
    //! ASR
    std::vector<nvigi::InferenceDataSlot> slots = { {nvigi::kASRWhisperDataSlotAudio, audioData} };
    nvigi::InferenceDataSlotArray inputs{ slots.size(), slots.data() };

    nvigi::InferenceExecutionContext asrExecCtx{};
    asrExecCtx.instance = nvigiCtx.asr;
    nvigi::ASRWhisperRuntimeParameters runtime{};
    runtime.sampling = nvigi::ASRWhisperSamplingStrategy::eGreedy;
    asrExecCtx.runtimeParameters = runtime;
    asrExecCtx.callback = &ASRInferenceDataCallback;
    asrExecCtx.callbackUserData = &cbkCtx;
    asrExecCtx.inputs = &inputs;
    cbkCtx.asrOutput = "";

    loggingPrint(nvigi::LogType::eInfo, "** Start ASR results\n");
    cbkCtx.asrCallbackState.store(nvigi::kInferenceExecutionStateDataPending);
    std::thread infer([&asrExecCtx]()
        {
            nvigiCtx.asr->evaluate(&asrExecCtx);
        });

    // Wait for the ASR to stop returning eDataPending in the callback
    {
        std::unique_lock lck(cbkCtx.callbackMutex);
        cbkCtx.asrCallbackCV.wait(lck, [&cbkCtx]() { return cbkCtx.asrCallbackState != nvigi::kInferenceExecutionStateDataPending; });
        if (cbkCtx.asrCallbackState != nvigi::kInferenceExecutionStateDone)
        {
            loggingPrint(nvigi::LogType::eError, "ASR Inference error!\n");
            return -1;
        }
    }
    infer.join();
    loggingPrint(nvigi::LogType::eInfo, (std::string("\nUser Speech: ") + cbkCtx.asrOutput + "\n").c_str());
    loggingPrint(nvigi::LogType::eInfo, "\n** End ASR results\n");
    gptInputText = cbkCtx.asrOutput;

    return 0;
}

///////////////////////////////////////
//! TTS inference

nvigi::InferenceExecutionState TTSInferenceDataCallback(const nvigi::InferenceExecutionContext* ctx, nvigi::InferenceExecutionState state, void* userData)
{
    auto cbkCtx = (BasicCallbackCtx*)userData;
    std::scoped_lock lck(cbkCtx->ttsCallbackMutex);

#ifdef NVIGI_WINDOWS
    if (state == nvigi::kInferenceExecutionStateDone || state == nvigi::kInferenceExecutionStateDataPending) {
        auto slots = ctx->outputs;
        std::vector<int16_t> temp;
        const nvigi::InferenceDataByteArray* outputAudioData{};
        slots->findAndValidateSlot(nvigi::kTTSDataSlotOutputAudio, &outputAudioData);


        if (cbkCtx->conversationInitialized) {
            nvigi::CpuData* cpuBuffer = castTo<nvigi::CpuData>(outputAudioData->bytes);

            for (int i = 0; i < cpuBuffer->sizeInBytes / 2; i++) {
                int16_t value = reinterpret_cast<const int16_t*>(cpuBuffer->buffer)[i];
                cbkCtx->ttsOutput.push_back(value);
                temp.push_back(value);
            }
            auto endTimeToFirstAudio = std::chrono::high_resolution_clock::now();

            // Fisrt audio
            if (temp.size() == cbkCtx->ttsOutput.size()) {
                auto timeToFirstAudio = std::chrono::duration_cast<std::chrono::milliseconds>(endTimeToFirstAudio - cbkCtx->startTimeToFirstAudio).count();
                loggingPrint(nvigi::LogType::eInfo, ("\nTime to first audio: " + std::to_string(timeToFirstAudio) + "ms\n").c_str());
            }

            cbkCtx->playAudioThreads.push(std::make_unique<std::thread>(std::thread(savePlayAudioData<int16_t>, temp, "", 22050, std::ref(cbkCtx->mtxPlayAudio), true, false)));
        }
    }
    cbkCtx->ttsCallbackState.store(state);
    cbkCtx->ttsCallbackCV.notify_one();
#endif

    return state;
}

int TTSInference(BasicCallbackCtx& cbkCtx, std::string& inputChunk, const bool waitAudioToFinish = false)
{
#ifdef NVIGI_WINDOWS
    //! TTS
    // Define Runtime parameters

    if (!inputChunk.empty())
    {
        cbkCtx.dataTextTTS = nvigi::InferenceDataTextSTLHelper(inputChunk);

        loggingPrint(nvigi::LogType::eInfo, "\n** Start TTS results\n");
        cbkCtx.ttsCallbackState = nvigi::kInferenceExecutionStateDataPending;

        nvigiCtx.tts->evaluate(&(cbkCtx.ttsExecCtx));
    }
    
    // Wait for the TTS to stop returning eDataPending in the callback
    {
        std::unique_lock lck(cbkCtx.ttsCallbackMutex);
        cbkCtx.ttsCallbackCV.wait(lck, [&cbkCtx]() { return cbkCtx.ttsCallbackState != nvigi::kInferenceExecutionStateDataPending; });
        if (cbkCtx.ttsCallbackState != nvigi::kInferenceExecutionStateDone)
        {
            loggingPrint(nvigi::LogType::eError, "TTS Inference error!\n");
            return -1;
        }
    }


    loggingPrint(nvigi::LogType::eInfo, "** End TTS results\n");
#endif

    return 0;
}


///////////////////////////////////////
//! GPT inference

nvigi::InferenceExecutionState GPTInferenceDataCallback(const nvigi::InferenceExecutionContext* ctx, nvigi::InferenceExecutionState state, void* userData)
{
    auto cbkCtx = (BasicCallbackCtx*)userData;
    std::scoped_lock lck(cbkCtx->callbackMutex);

    // Outputs from GPT
    auto slots = ctx->outputs;
    const nvigi::InferenceDataText* text{};
    slots->findAndValidateSlot(nvigi::kGPTDataSlotResponse, &text);
    auto response = std::string(text->getUTF8Text());
    if (cbkCtx->conversationInitialized)
    {
        cbkCtx->gptOutput += response;
        cbkCtx->mtxttsInput.lock();
        cbkCtx->ttsInput += response;
        cbkCtx->mtxttsInput.unlock();
        loggingPrint(nvigi::LogType::eInfo, response.c_str());

        if (cbkCtx->ttsInput != "") {

            // We try to process chunks between 128 and 256 chracters maximum and avoid cutting sentences
            bool isLastCharacterPeriod = (cbkCtx->ttsInput.back() == '\n' || cbkCtx->ttsInput.back() == '.' ||
                cbkCtx->ttsInput.back() == '!' || cbkCtx->ttsInput.back() == '?');

            if (isLastCharacterPeriod) {
                cbkCtx->posLastPeriod = cbkCtx->ttsInput.size() - 1;
            }
            else if (cbkCtx->ttsInput.back() == ' ') {
                cbkCtx->posLastSpace = cbkCtx->ttsInput.size() - 1;
            }

            if ((isLastCharacterPeriod && (cbkCtx->ttsInput.size() >= 128)) ||
                (cbkCtx->ttsInput.size() > 256) ||
                state == nvigi::kInferenceExecutionStateDone) {

                std::string chunkToProcess;
                cbkCtx->mtxttsInput.lock();
                if (state == nvigi::kInferenceExecutionStateDone || isLastCharacterPeriod || (cbkCtx->posLastPeriod == 0 && cbkCtx->posLastSpace == 0)) {
                    chunkToProcess = cbkCtx->ttsInput;
                    cbkCtx->ttsInput = "";
                }
                else if (cbkCtx->posLastPeriod != 0) {
                    chunkToProcess = cbkCtx->ttsInput.substr(0, cbkCtx->posLastPeriod + 1);
                    cbkCtx->ttsInput = cbkCtx->ttsInput.substr(cbkCtx->posLastPeriod + 1);
                }
                else if (cbkCtx->posLastSpace != 0) {
                    chunkToProcess = cbkCtx->ttsInput.substr(0, cbkCtx->posLastSpace + 1);
                    cbkCtx->ttsInput = cbkCtx->ttsInput.substr(cbkCtx->posLastSpace + 1);
                }
                cbkCtx->mtxttsInput.unlock();

                cbkCtx->posLastPeriod = 0;
                cbkCtx->posLastSpace = 0;
                cbkCtx->ttsCallbackState.store(nvigi::kInferenceExecutionStateDataPending);

                // Synchronous TTS inference. We wait for TTS to finish before resuming GPT
                TTSInference(*cbkCtx, chunkToProcess, false);
            }
        }
    }

    cbkCtx->gptCallbackState.store(state);
    cbkCtx->gptCallbackCV.notify_one();

    return state;
}

int GPTInference(BasicCallbackCtx& cbkCtx, std::string& gptInputText)
{
    //! GPT
    nvigi::InferenceExecutionContext gptExecCtx{};
    gptExecCtx.instance = nvigiCtx.gpt;
    gptExecCtx.callback = &GPTInferenceDataCallback;
    gptExecCtx.callbackUserData = &cbkCtx;
    cbkCtx.gptCallbackState.store(nvigi::kInferenceExecutionStateDataPending);
    cbkCtx.gptOutput = "";

    nvigi::GPTRuntimeParameters runtime{};
    runtime.seed = -1;
    runtime.tokensToPredict = 200;
    runtime.interactive = true;
    runtime.reversePrompt = "User: ";
    gptExecCtx.runtimeParameters = runtime;

    nvigi::InferenceDataTextSTLHelper text(gptInputText);
    std::vector<nvigi::InferenceDataSlot> slots = {
        {cbkCtx.conversationInitialized ? nvigi::kGPTDataSlotUser : nvigi::kGPTDataSlotSystem, text} };
    nvigi::InferenceDataSlotArray inputs = { slots.size(), slots.data() };
    gptExecCtx.inputs = &inputs;

    loggingPrint(nvigi::LogType::eInfo, "** Assistant:\n");
    cbkCtx.gptCallbackState.store(nvigi::kInferenceExecutionStateDataPending);
    std::thread infer([&gptExecCtx]()
        {
            nvigiCtx.gpt->evaluate(&gptExecCtx);
        });

    // Wait for the GPT to stop returning eDataPending in the callback
    {
        std::unique_lock lck(cbkCtx.callbackMutex);
        cbkCtx.gptCallbackCV.wait(lck, [&cbkCtx]() { return cbkCtx.gptCallbackState != nvigi::kInferenceExecutionStateDataPending; });
        if (cbkCtx.gptCallbackState != nvigi::kInferenceExecutionStateDone)
        {
            loggingPrint(nvigi::LogType::eError, "GPT Inference error!\n");
            return -1;
        }
    }
    infer.join();

    return 0;
}


///////////////////////////////////////
//! Full-sequence inference

int RunInference(bool& hasAudio, nvigi::InferenceDataAudioSTLHelper& audioData, const std::string targetPathSpectrogram, std::string& gptInputText, bool conversationInitialized)
{
    BasicCallbackCtx cbkCtx{};
    cbkCtx.conversationInitialized = conversationInitialized;
    cbkCtx.gptInstance = nvigiCtx.gpt;
    cbkCtx.asrInstance = nvigiCtx.asr;
    cbkCtx.ttsInstance = nvigiCtx.tts;

    // TTS instance is created, We can set up the inference context since we need to run TTS while GPT is running.
    nvigi::TTSASqFlowRuntimeParameters  runtime{};
    cbkCtx.ttsExecCtx.instance = nvigiCtx.tts;
    cbkCtx.ttsExecCtx.callback = &TTSInferenceDataCallback;
    cbkCtx.ttsExecCtx.callbackUserData = &cbkCtx;
    cbkCtx.ttsExecCtx.runtimeParameters = runtime;
    nvigi::InferenceDataTextSTLHelper inputPathTargetSpectrogram(targetPathSpectrogram);

    // Initialize data bugger with empty string
    std::string inputChunk = "";
    cbkCtx.dataTextTTS = nvigi::InferenceDataTextSTLHelper(inputChunk);
    std::vector<nvigi::InferenceDataSlot> slots = { {nvigi::kTTSDataSlotInputText, cbkCtx.dataTextTTS},
                                                    {nvigi::kTTSDataSlotInputTargetSpectrogramPath, inputPathTargetSpectrogram } };
    nvigi::InferenceDataSlotArray inputs = { slots.size(), slots.data() };
    cbkCtx.ttsExecCtx.inputs = &inputs;

    // Initialize timer time to first audio
    cbkCtx.startTimeToFirstAudio = std::chrono::high_resolution_clock::now();
    if (hasAudio)
    {
        if (ASRInference(cbkCtx, audioData, gptInputText))
            return -1;
        hasAudio = false;
    }

    if (GPTInference(cbkCtx, gptInputText))
        return -1;

    // TTS inference is executed within GPT inference

    // If an audio is playing, wait for it to finish and destroy the corresponding threads
    while (true) {
        std::lock_guard<std::mutex> lock(cbkCtx.callbackMutex);
        std::unique_ptr<std::thread> thread;
        {
            if (cbkCtx.playAudioThreads.empty())
                break;
            thread = std::move(cbkCtx.playAudioThreads.front());
            cbkCtx.playAudioThreads.pop();
        }

        if (thread->joinable()) {
            thread->join();
        }
    }
    cbkCtx.ttsOutput.clear();
    cbkCtx.ttsOutput.shrink_to_fit();

    return 0;
}

int main(int argc, char** argv)
{
    // Block the llama output, so it does not pollute the app's console output
#ifdef NVIGI_WINDOWS
    FILE* f{};
    freopen_s(&f, "NUL", "w", stderr);
#else
    freopen("dev/nul", "w", stderr);
#endif

    CommandLineParser parser;
    parser.add_command("s", "sdk", " sdk location, if none provided assuming exe location", "");
    parser.add_command("m", "models", " model repo location", "", true);
    parser.add_command("", "targetPathSpectrogram", " target path of the spectrogram of the voice you want to clone", "", true);
    parser.add_command("", "extendedPhonemeDict", " path to the extendend phonemes dictionary for ASqFlow TTS model", "", false);
    parser.add_command("a", "audio", " audio file location", "", false); // used only for Linux
    parser.add_command("", "gpt", " gpt mode, 'local' or 'cloud' (model GUID determines cloud endpoint)", "local");
    parser.add_command("", "gpt-guid", " gpt model guid in registry format", "{01F43B70-CE23-42CA-9606-74E80C5ED0B6}");
    parser.add_command("", "asr-guid", " asr model guid in registry format", "{5CAD3A03-1272-4D43-9F3D-655417526170}");
    parser.add_command("", "tts-guid", " tts model guid in registry format (GGML: {16EEB8EA-55A8-4F40-BECE-CE995AF44101} TRT: {81320D1D-DF3C-4CFC-B9FA-4D3FF95FC35F})", "{16EEB8EA-55A8-4F40-BECE-CE995AF44101}");
    parser.add_command("t", "token", " authorization token for the cloud provider", "");
    parser.add_command("", "vram", " the amount of vram to use in MB", "8192");

    try {
        parser.parse(argc, argv);
    }
    catch (std::exception& e)
    {
        printf("%s\n\n", e.what());
        parser.print_help("nvigi.basic");
        exit(1);
    }

    auto pathToSDKArgument = parser.get("sdk");
    auto pathToSDKUtf8 = pathToSDKArgument.empty() ? getExecutablePath() : pathToSDKArgument;

    // Mandatory so we know that they are provided
    std::string modelDir = parser.get("models");
    std::string targetPathSpectrogram = parser.get("targetPathSpectrogram");

    // Defaults
    auto extendedPhonemeDict = parser.get("extendedPhonemeDict");
    auto audioFile = parser.get("audio");
    size_t vramBudgetMB = (size_t)atoi(parser.get("vram").c_str());

#ifdef NVIGI_LINUX
    auto wav = read(audioFile.c_str());
    if (wav.empty())
    {
        loggingPrint(nvigi::LogType::eError, "Could not load input WAV file");
        return -1;
    }
#endif

    //////////////////////////////////////////////////////////////////////////////
    //! Init NVIGI
    if (InitNVIGI(pathToSDKUtf8))
        return -1;

    //////////////////////////////////////////////////////////////////////////////
    //! Init Plugin Interfaces and Instances
    //! 
    {
        auto guidASR = parser.get("asr-guid");
        if (InitASR(modelDir, guidASR, vramBudgetMB))
            return -1;
    }

    {
        auto guidGPT = parser.get("gpt-guid");
        auto gptMode = parser.get("gpt");
        auto cloudToken = parser.get("token");
        if (InitGPT(modelDir, gptMode, cloudToken, guidGPT, vramBudgetMB))
            return -1;
    }

#ifdef NVIGI_WINDOWS
    {
        auto guidTTS = parser.get("tts-guid");
        if (InitTTS(modelDir, extendedPhonemeDict, guidTTS, vramBudgetMB))
            return -1;
    }
#endif
    {
        //////////////////////////////////////////////////////////////////////////////
        //! Run inference
        //! 
        bool running = true;
        bool hasAudio = false;
        bool conversationInitialized = false;
        std::string gptInputText = "This is a transcript of a dialog between a user and a helpful AI assistant.\
 Generate only medium size answers and avoid describing what you are doing physically.\
 Avoid using specific words that are not part of the dictionary.\n";

#ifdef NVIGI_WINDOWS
        nvigi::InferenceDataAudioSTLHelper audioData;
#else
        nvigi::InferenceDataAudioSTLHelper audioData(wav);
#endif

        do
        {
            if (RunInference(hasAudio, audioData, targetPathSpectrogram, gptInputText, conversationInitialized))
                return -1;

            conversationInitialized = true;

#if NVIGI_WINDOWS
            loggingPrint(nvigi::LogType::eInfo, "\n** Please continue the converation (enter with no text to start recording your query, 'q' or 'quit' to exit, any other text to type your query\n>:");
#else
            loggingPrint(nvigi::LogType::eInfo, "\n** Please continue the converation (enter with no text to use the wav file for prompt, 'q' or 'quit' to exit, any other text to type your query\n>:");
#endif

            std::getline(std::cin, gptInputText);
            if (gptInputText == "q" || gptInputText == "Q" || gptInputText == "quit")
            {
                loggingPrint(nvigi::LogType::eInfo, "Exiting - user request\n");
                running = false;
            }
            else if (gptInputText == "")
            {
#if NVIGI_WINDOWS
                // Record audio
                nvigi::utils::RecordingInfo* ri = nvigi::utils::startRecordingAudio();
                loggingPrint(nvigi::LogType::eInfo, "Recording in progress: ask your question or comment and press enter to stop recording\n");
                std::getline(std::cin, gptInputText);
                gptInputText = "";

                nvigi::utils::stopRecordingAudio(ri, &audioData);
                hasAudio = true;
#endif
            }
            else
            {
                // Use the given getline result as the text
            }
        } while (running);
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Shutdown NVIGI
    //! 
    if (ReleaseASR())
        return -1;

    if (ReleaseGPT())
        return -1;

#if NVIGI_WINDOWS
    if (ReleaseTTS())
        return -1;
#endif

    if (ShutdownNVIGI())
        return -1;

    return 0;
}
