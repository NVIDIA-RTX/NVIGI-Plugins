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
#include <iostream>
#include <thread>
#include <memory>
#include <queue>
#include <type_traits>
#include <mutex>
#include <atomic>

#include <map>

namespace fs = std::filesystem;

#if NVIGI_WINDOWS
#include <windows.h>
#include <source/utils/nvigi.dsound/player.h>
#endif

#include <nvigi.h>
#include "nvigi_asr_whisper.h"
#include "nvigi_gpt.h"
#include "nvigi_cloud.h"
#include <source/utils/nvigi.dsound/recorder.h>

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

inline std::vector<uint8_t> read(const char* fname)
{
    try
    {
        fs::path p(fname);
        size_t file_size = fs::file_size(p);
        std::vector<uint8_t> ret_buffer(file_size);
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
                if(!cmd.short_name.empty())
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

void loggingCallback(nvigi::LogType type, const char* msg)
{
#ifdef NVIGI_WINDOWS
    OutputDebugStringA(msg);
#endif
    std::cout << msg;
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
    header.dataSize = dataBuffer.size() * sizeof(int16_t);
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
                        const int sampling_rate, std::mutex &mtxPlayAudio,
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
        printf("%s has been saved \n", output_path.c_str());
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
    if (countNonUtf8 > 0) {
		printf("\n%d non-utf8 characters have been removed \n", countNonUtf8);
	}
    return output;
}

int main(int argc, char** argv)
{
    CommandLineParser parser;
    parser.add_command("s", "sdk", "sdk location, if none provided assuming exe location", "");
    parser.add_command("m", "models", "model repo location", "", true);
    parser.add_command("a", "audio", "audio file location", "", false); // used only for Linux
    parser.add_command("", "gpt", "gpt mode, 'local' or 'cloud' (model GUID determines cloud endpoint)", "local");
    parser.add_command("", "gpt-guid", "gpt model guid in registry format", "{01F43B70-CE23-42CA-9606-74E80C5ED0B6}");
    parser.add_command("", "asr-guid", "asr model guid in registry format", "{5CAD3A03-1272-4D43-9F3D-655417526170}");
    parser.add_command("t", "token", "authorization token for the cloud provider", "");
    parser.add_command("", "vram", "the amount of vram to use in MB", "8192");

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
#ifdef NVIGI_WINDOWS
    auto libPath = pathToSDKUtf8 + "/nvigi.core.framework.dll";
#else
    auto libPath = pathToSDKUtf8 + "/nvigi.core.framework.so";
#endif
    HMODULE lib = LoadLibraryA(libPath.c_str());
    if (lib == nullptr)
    {
        loggingCallback(nvigi::LogType::eError, "Could not load NVIGI core library");
        return -1;
    }

    GET_NVIGI_CORE_FUN(nvigiInit);
    GET_NVIGI_CORE_FUN(nvigiShutdown);
    GET_NVIGI_CORE_FUN(nvigiLoadInterface);
    GET_NVIGI_CORE_FUN(nvigiUnloadInterface);

    if (ptr_nvigiInit == nullptr || ptr_nvigiShutdown == nullptr ||
        ptr_nvigiLoadInterface == nullptr || ptr_nvigiUnloadInterface == nullptr)
    {
        loggingCallback(nvigi::LogType::eError, "Could not load NVIGI core library");
        return -1;
    }

    const char* paths[] =
    {
        pathToSDKUtf8.c_str()
    };

    // Mandatory so we know that they are provided
    std::string modelDir = parser.get("models");

    // Defaults
    auto audioFile = parser.get("audio");
    auto guidASR = parser.get("asr-guid");
    auto guidGPT = parser.get("gpt-guid");
    auto cloudToken = parser.get("token");
    size_t vramBudgetMB = (size_t)atoi(parser.get("vram").c_str());

#ifdef NVIGI_LINUX
    auto wav = read(audioFile.c_str());
    if (wav.empty())
    {
        loggingCallback(nvigi::LogType::eError, "Could not load input WAV file");
        return -1;
    }
#endif

    //////////////////////////////////////////////////////////////////////////////
    //! Init NVIGI
    //! 

    nvigi::Preferences pref{};
    pref.logLevel = nvigi::LogLevel::eVerbose;
    pref.showConsole = true;
    pref.numPathsToPlugins = 1;
    pref.utf8PathsToPlugins = paths;
    pref.logMessageCallback = pref.showConsole ? (nvigi::PFun_LogMessageCallback*)nullptr : loggingCallback; // avoid duplicating logs in the console
    pref.utf8PathToLogsAndData = pathToSDKUtf8.c_str();

    uint32_t n_threads = 4;

    if (NVIGI_FAILED(result, ptr_nvigiInit(pref, nullptr, nvigi::kSDKVersion)))
    {
        loggingCallback(nvigi::LogType::eError, "NVIGI init failed");
        return -1;
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Init Plugin Interfaces and Instances
    //! 

    //! ASR Interface and Instance
    nvigi::IAutoSpeechRecognition* iasr{};
    if (NVIGI_FAILED(result, nvigiGetInterfaceDynamic(nvigi::plugin::asr::ggml::cuda::kId, &iasr,ptr_nvigiLoadInterface)))
    {
        loggingCallback(nvigi::LogType::eError, "Could not query ASR interface");
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
        asrCommon.chain(asrParams);
    }

    nvigi::InferenceInstance* asr{};
    if(NVIGI_FAILED(result, iasr->createInstance(asrCommon, &asr)))
    {
        loggingCallback(nvigi::LogType::eError, "Could not create ASR instance");
        return -1;
    }

    //! GPT Interface and Instance
    nvigi::IGeneralPurposeTransformer* igpt{};
    auto gptMode = parser.get("gpt");
    //! When using GPT cloud it does not matter which endpoint we are going to hit, they all use the same REST based backend
    nvigi::PluginID gptId = gptMode == "local" ? nvigi::plugin::gpt::ggml::cuda::kId : nvigi::plugin::gpt::cloud::rest::kId;
    if (NVIGI_FAILED(result, nvigiGetInterfaceDynamic(gptId, &igpt, ptr_nvigiLoadInterface)))
    {
        loggingCallback(nvigi::LogType::eError, "Could not query GPT interface");
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
    gptCommon.chain(gptParams);

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
    if (NVIGI_FAILED(result, getCapsAndRequirements(igpt, gptCommon, &caps)))
    {
        loggingCallback(nvigi::LogType::eError, "'getCapsAndRequirements' failed");
        return -1;
    }

    //! We provided model GUID and VRAM budget so caps and requirements will contain just one model, assuming VRAM budget is sufficient or if cloud backend is selected!
    //! 
    //! NOTE: This will be >=1 if we provide null as modelGUID in common creation parameters
    if (caps->numSupportedModels != 1)
    {
        loggingCallback(nvigi::LogType::eError, "'getCapsAndRequirements' failed to find our model or model cannot run on system given the VRAM restrictions");
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
            printf("--token parameter must be provided when using GPT cloud path");
            exit(1);
        }

        restParams.url = ccaps->url;
        restParams.authenticationToken = cloudToken.c_str();
        restParams.verboseMode = false; // Set to true to debug issues with connection protocol (if any)        
        gptCommon.chain(restParams);
    }
    
    nvigi::InferenceInstance* gpt{};
    if (NVIGI_FAILED(result, igpt->createInstance(gptCommon, &gpt)))
    {
        loggingCallback(nvigi::LogType::eError, "Could not create GPT instance");
        return -1;
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Run inference
    //! 

    struct BasicCallbackCtx
    {
        std::mutex callbackMutex;
        std::condition_variable asrCallbackCV;
        std::condition_variable gptCallbackCV;
        std::atomic<nvigi::InferenceExecutionState> asrCallbackState = nvigi::kInferenceExecutionStateDataPending;
        std::atomic<nvigi::InferenceExecutionState> gptCallbackState = nvigi::kInferenceExecutionStateDataPending;
        std::string asrOutput;
        std::string gptOutput;

        nvigi::InferenceInstance* gptInstance{};
        nvigi::InferenceInstance* asrInstance{};
        bool conversationInitialized = false;
    };
    BasicCallbackCtx cbkCtx{};
    cbkCtx.gptInstance = gpt;
    cbkCtx.asrInstance = asr;

    bool running = true;
    nvigi::CpuData _audio{};
    nvigi::InferenceDataAudio audioData{ _audio };
    bool hasAudio = false;
    std::string gptInputText = "This is a transcript of a dialog between a user and a helpful AI assistant.\n";

    do
    {
        auto dataCallback = [](const nvigi::InferenceExecutionContext* ctx, nvigi::InferenceExecutionState state, void* userData)->nvigi::InferenceExecutionState
            {
                auto cbkCtx = (BasicCallbackCtx*)userData;
                std::scoped_lock lck(cbkCtx->callbackMutex);

                if (ctx->instance == cbkCtx->gptInstance)
                {
                    // Outputs from GPT
                    auto slots = ctx->outputs;
                    const nvigi::InferenceDataText* text{};
                    slots->findAndValidateSlot(nvigi::kGPTDataSlotResponse, &text);
                    auto response = std::string(text->getUTF8Text());
                    if (cbkCtx->conversationInitialized)
                    {
                        cbkCtx->gptOutput += response;
                        loggingCallback(nvigi::LogType::eInfo, response.c_str());
                    }

                    cbkCtx->gptCallbackState.store(state);
                    cbkCtx->gptCallbackCV.notify_one();
                }
                else if (ctx->instance == cbkCtx->asrInstance)
                {
                    // Outputs from ASR
                    auto slots = ctx->outputs;
                    const nvigi::InferenceDataText* text{};
                    slots->findAndValidateSlot(nvigi::kASRWhisperDataSlotTranscribedText, &text);
                    auto response = std::string(text->getUTF8Text());
                    cbkCtx->asrOutput += response;

                    cbkCtx->asrCallbackState.store(state);
                    cbkCtx->asrCallbackCV.notify_one();
                }
                else
                {
                    assert(!"unknown instance");
                }

                return state;
            };

        //! ASR
        if (hasAudio)
        {
            std::vector<nvigi::InferenceDataSlot> slots = { {nvigi::kASRWhisperDataSlotAudio, &audioData} };
            nvigi::InferenceDataSlotArray inputs{ slots.size(), slots.data() };

            nvigi::InferenceExecutionContext asrExecCtx{};
            asrExecCtx.instance = asr;
            nvigi::ASRWhisperRuntimeParameters runtime{};
            runtime.sampling = nvigi::ASRWhisperSamplingStrategy::eGreedy;
            asrExecCtx.runtimeParameters = runtime;
            asrExecCtx.callback = dataCallback;
            asrExecCtx.callbackUserData = &cbkCtx;
            asrExecCtx.inputs = &inputs;
            cbkCtx.asrOutput = "";

            loggingCallback(nvigi::LogType::eInfo, "** Start ASR results\n");
            cbkCtx.asrCallbackState.store(nvigi::kInferenceExecutionStateDataPending);
            std::thread infer([asr, &asrExecCtx]()
                {
                    asr->evaluate(&asrExecCtx);
                });

            // Wait for the ASR to stop returning eDataPending in the callback
            {
                std::unique_lock lck(cbkCtx.callbackMutex);
                cbkCtx.asrCallbackCV.wait(lck, [&cbkCtx]() { return cbkCtx.asrCallbackState != nvigi::kInferenceExecutionStateDataPending; });
            }
            infer.join();
            loggingCallback(nvigi::LogType::eInfo, (std::string("\nUser Speech: ") + cbkCtx.asrOutput + "\n").c_str());
            loggingCallback(nvigi::LogType::eInfo, "\n** End ASR results\n");
            gptInputText = cbkCtx.asrOutput;
            hasAudio = false;
        }

        //! GPT
        {
            nvigi::InferenceExecutionContext gptExecCtx{};
            gptExecCtx.instance = gpt;
            gptExecCtx.callback = dataCallback;
            gptExecCtx.callbackUserData = &cbkCtx;
            cbkCtx.gptCallbackState.store(nvigi::kInferenceExecutionStateDataPending);
            cbkCtx.gptOutput = "";

            nvigi::GPTRuntimeParameters runtime{};
            runtime.seed = -1;
            runtime.tokensToPredict = 200;
            runtime.interactive = true;
            runtime.reversePrompt = "User: ";
            gptExecCtx.runtimeParameters = runtime;

            nvigi::CpuData _text(gptInputText.length() + 1, (const void*)gptInputText.c_str());
            nvigi::InferenceDataText text(_text);
            std::vector<nvigi::InferenceDataSlot> slots = { 
                {cbkCtx.conversationInitialized ? nvigi::kGPTDataSlotUser : nvigi::kGPTDataSlotSystem, &text} };
            nvigi::InferenceDataSlotArray inputs = { slots.size(), slots.data() };
            gptExecCtx.inputs = &inputs;

            loggingCallback(nvigi::LogType::eInfo, "** Assistant:\n");
            cbkCtx.gptCallbackState.store(nvigi::kInferenceExecutionStateDataPending);
            std::thread infer([gpt, &gptExecCtx]()
                {
                    gpt->evaluate(&gptExecCtx);
                });

            // Wait for the GPT to stop returning eDataPending in the callback
            {
                std::unique_lock lck(cbkCtx.callbackMutex);
                cbkCtx.gptCallbackCV.wait(lck, [&cbkCtx]() { return cbkCtx.gptCallbackState != nvigi::kInferenceExecutionStateDataPending; });
            }
            infer.join();
            cbkCtx.conversationInitialized = true;
        }

#if NVIGI_WINDOWS
        loggingCallback(nvigi::LogType::eInfo, "\n** Please continue the converation (enter with no text to start recording your query, 'q' or 'quit' to exit, any other text to type your query\n>:");
#else
        loggingCallback(nvigi::LogType::eInfo, "\n** Please continue the converation (enter with no text to use the wav file for prompt, 'q' or 'quit' to exit, any other text to type your query\n>:");
#endif

        std::getline(std::cin, gptInputText);
        if (gptInputText == "q" || gptInputText == "Q" || gptInputText == "quit")
        {
            loggingCallback(nvigi::LogType::eInfo, "Exiting - user request\n");
            running = false;
        }
        else if (gptInputText == "")
        {
#if NVIGI_WINDOWS
            // Record audio
            nvigi::utils::RecordingInfo* ri = nvigi::utils::startRecordingAudio();
            loggingCallback(nvigi::LogType::eInfo, "Recording in progress: ask your question or comment and press enter to stop recording\n");
            std::getline(std::cin, gptInputText);
            gptInputText = "";

            nvigi::utils::stopRecordingAudio(ri, &audioData);
            hasAudio = true;
#else
            _audio.sizeInBytes = wav.size();
            _audio.buffer = wav.data();
#endif
        }
        else
        {
            // Use the given getline result as the text
        }
    } while (running);


    //////////////////////////////////////////////////////////////////////////////
    //! Shutdown NVIGI
    //! 

    igpt->destroyInstance(gpt);
    iasr->destroyInstance(asr);




    // Hard-coded to local
    if (NVIGI_FAILED(result, ptr_nvigiUnloadInterface(nvigi::plugin::asr::ggml::cuda::kId, iasr)))
    {
        loggingCallback(nvigi::LogType::eError, "Error in 'nvigiUnloadInterface'");
        return -1;
    }

    // Can be cloud or local
    if (NVIGI_FAILED(result, ptr_nvigiUnloadInterface(gptId, igpt)))
    {
        loggingCallback(nvigi::LogType::eError, "Error in 'nvigiUnloadInterface'");
        return -1;
    }

    if(NVIGI_FAILED(result, ptr_nvigiShutdown()))
    {
        loggingCallback(nvigi::LogType::eError, "Error in 'nvigiShutdown'");
        return -1;
    }

    FreeLibrary(lib);

    return 0;


    return 0;
}