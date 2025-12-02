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
#include <future>
#include <iostream>
#include <sstream>
#include <regex>
#include <thread>
#include <format>

namespace fs = std::filesystem;

#if NVIGI_WINDOWS
#include <windows.h>
#endif

#include <nvigi.h>
#include "nvigi_gpt.h"
#include "nvigi_types.h"
#include <nvigi_stl_helpers.h>

#include "external/json/source/nlohmann/json.hpp"
#define CURL_STATICLIB
#include "external/libcurl/include/curl/curl.h"

using json = nlohmann::json;

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

typedef std::vector< std::string > StringVec;
typedef std::vector< std::vector< float > > VectorStore;
typedef std::pair< size_t, float > IndexScore;
typedef std::vector< std::pair<size_t, float> > IndexScoreVec;

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
        loggingPrint(nvigi::LogType::eError, "Failed to unload interface");
        return false;
    }

    return true;
}

struct NVIGIAppCtx
{
    HMODULE coreLib{};

    nvigi::IGeneralPurposeTransformer* igpt{};
    nvigi::InferenceInstance* gptInst{};
};



///////////////////////////////////////
//! NVIGI Init and Shutdown

int InitNVIGI(NVIGIAppCtx& nvigiCtx, const std::string& pathToSDKUtf8)
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

int ShutdownNVIGI(NVIGIAppCtx& nvigiCtx)
{
    if (NVIGI_FAILED(result, ptr_nvigiShutdown()))
    {
        loggingPrint(nvigi::LogType::eError, "Error in 'nvigiShutdown'");
        return -1;
    }

    FreeLibrary(nvigiCtx.coreLib);

    return 0;
}

int InitGPT(NVIGIAppCtx& nvigiCtx, const std::string& modelDir)
{
    loggingPrint(nvigi::LogType::eInfo, "Initializing GPT\n");

    //! GPT Interface
    if (NVIGI_FAILED(result, nvigiGetInterfaceDynamic(nvigi::plugin::gpt::ggml::cuda::kId, &nvigiCtx.igpt, ptr_nvigiLoadInterface)))
    {
        loggingPrint(nvigi::LogType::eError, "Could not query GPT interface");
        return -1;
    }

    nvigi::GPTCreationParameters gptParams{};
    nvigi::CommonCreationParameters gptCommon{};
    gptCommon.utf8PathToModels = modelDir.c_str();
    gptCommon.numThreads = 16;
    gptCommon.vramBudgetMB = vram;
    gptParams.contextSize = 4096;
    gptCommon.modelGUID = "{545F7EC2-4C29-499B-8FC8-61720DF3C626}"; // Qwen3-8B-Q4_K_M
    if (NVIGI_FAILED(result, gptCommon.chain(gptParams)))
    {
        loggingPrint(nvigi::LogType::eError, "GPT param chaining failed");
        return -1;
    }

    nvigi::CommonCapabilitiesAndRequirements* info{};
    getCapsAndRequirements(nvigiCtx.igpt, gptCommon, &info);
    if (info == nullptr)
        return -1;

    auto result = nvigiCtx.igpt->createInstance(gptCommon, &nvigiCtx.gptInst);

    loggingPrint(nvigi::LogType::eInfo, "Initializing GPT succeeded\n");

    return 0;
}

int ReleaseGPT(NVIGIAppCtx& nvigiCtx)
{
    if (nvigiCtx.gptInst == nullptr)
        return -1;

    if (NVIGI_FAILED(result, nvigiCtx.igpt->destroyInstance(nvigiCtx.gptInst)))
    {
        loggingPrint(nvigi::LogType::eError, "Failed to destroy GPT instance");
        return -1;
    }

    if (!unloadInterface(nvigi::plugin::gpt::ggml::cuda::kId, nvigiCtx.igpt))
    {
        loggingPrint(nvigi::LogType::eError, "Failed to release GPT interface");
        return -1;
    }

    return 0;
}

void GetCompletion(NVIGIAppCtx& nvigiCtx, const std::string& system_prompt, const std::string& user_prompt, std::string& answer)
{
    answer = "";

    nvigi::InferenceDataTextSTLHelper system_data( system_prompt );
    nvigi::InferenceDataTextSTLHelper user_data( user_prompt );

    std::vector<nvigi::InferenceDataSlot> inSlots;
    if (!system_prompt.empty())
        inSlots.push_back({ nvigi::kGPTDataSlotSystem, system_data });
    inSlots.push_back({ nvigi::kGPTDataSlotUser, user_data });

    nvigi::InferenceDataSlotArray inputs = { inSlots.size(), inSlots.data() };

    nvigi::GPTRuntimeParameters runtime{};
    runtime.seed = -1;
    runtime.tokensToPredict = 200;
    runtime.interactive = true;
    runtime.reversePrompt = "User: ";

    struct UserDataBlock
    {
        std::atomic<bool> done = false;
        std::string response;
        bool terminator_found = false;
    };
    UserDataBlock userData;

    nvigi::InferenceExecutionContext ctx{};
    ctx.instance = nvigiCtx.gptInst;
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
                }
            }
            userDataBlock->done.store(state == nvigi::kInferenceExecutionStateDone);
            return state;
        };

    ctx.inputs = &inputs;
    ctx.runtimeParameters = runtime;

    if (ctx.instance->evaluate(&ctx) != nvigi::kResultOk)
    {
        loggingPrint(nvigi::LogType::eError, "GPT evaluate failed");
    }
    // ctx is held in this scope, so we can't let it go out of scope while the LLM is evaluating.
    while (!userData.done);

    answer = userData.response;
}


///////////////////////////////////////
//! JsonFunctionCreator
//! 
//! Helper class to aid in the creation of proper json functions for tool calling.
//! Makes typos less likely and improves readability of function definition in code.
//! 
class JsonFunctionCreator
{
public:
    typedef std::vector< std::string > StringVec;
    struct ParameterType
    {
        std::string name;
        std::string type;
        std::string description;

        StringVec valid_values; // can be empty
    };

private:
    std::string m_function_name;
    std::string m_function_description;
    StringVec m_required;

    typedef std::vector<ParameterType> ParamTypeVec;
    ParamTypeVec m_parameters;

public:
    void setFunctionName(std::string function_name)
    {
        m_function_name = function_name;
    }

    void setFunctionDesc(std::string desc)
    {
        m_function_description = desc;
    }

    void addParameter(std::string name, std::string type, std::string description, bool required, StringVec* valid_values = nullptr )
    {
        if (valid_values == nullptr)
            m_parameters.push_back(ParameterType{ name, type, description, {} });
        else
            m_parameters.push_back(ParameterType{ name, type, description, *valid_values });

        if (required)
            m_required.push_back(name);
    }

    std::string getString()
    {
        json properties = json::object();
        for (const auto& param : m_parameters) {
            json prop;
            prop["type"] = param.type;
            prop["description"] = param.description;
            if (!param.valid_values.empty())
                prop["enum"] = param.valid_values;
            properties[param.name] = prop;
        }

        json params_obj;
        params_obj["type"] = "object";
        if (!m_parameters.empty())
            params_obj["properties"] = properties;
        if (!m_required.empty())
            params_obj["required"] = m_required;

        json function_obj;
        function_obj["name"] = m_function_name;
        function_obj["description"] = m_function_description;
        function_obj["parameters"] = params_obj;

        json schema;
        schema["type"] = "function";
        schema["function"] = function_obj;

        return schema.dump(4);
    }
};

///////////////////////////////////////
//! ToolBaseClass
//! 
//! Base class for all tools to derive from
//! 

class ToolBaseClass
{
public:
    ToolBaseClass(NVIGIAppCtx& nvigiCtx) : m_nvigiCtx(nvigiCtx) {}
    virtual ~ToolBaseClass() {}

    virtual std::string getJsonStr() = 0;
    virtual std::string getName() = 0;
    virtual bool validateJson(const json& parameters) = 0;
    virtual std::string call(const json& parameters) = 0;

protected:
    NVIGIAppCtx& m_nvigiCtx;
};

///////////////////////////////////////
//! CurrentTempTool
//! 
//! This tool is hardcoded to give temperatures for Durham, Austin and Santa Clara
//! It uses minimal dependencies as to be very clear about what exactly is being sent
//! to the LLM at different points in time.
//! 
class CurrentTempTool : public ToolBaseClass
{
private:
    const float k_absolute_zero_fahrenheit = -459.67f;

    //! The actual function call we have written for the AI to use.
    float getCurrentTemperature(std::string location)
    {
        // this is absolute zero in fahrenheit. 
        float temperature = k_absolute_zero_fahrenheit;
        
        // lowercase location so it's easier to match
        std::transform( location.begin(), location.end(), location.begin(), [](unsigned char c) { return std::tolower(c); });

		// perform a substring look up to deal with different versions of the city coming in.  Hopefully this captures permuations such as
		// city, state, country or city, state or city or CITY, STATE, COUNTRY, etc
        if (location.find( "durham" ) != std::string::npos )
        {
            temperature = 68.0f;
        }
        else if (location.find("austin") != std::string::npos)
        {
            temperature = 99.0f;
        }
        else if (location.find("santa clara") != std::string::npos)
        {
            temperature = 80;
        }

        return temperature;
    }

public:
    CurrentTempTool(NVIGIAppCtx& nvigiCtx) : ToolBaseClass(nvigiCtx) {}

    //! In this example, we use no helper utility, and just write out the JSON in long hand.
    //! This is for illustrative purposes to understand exactly what would be fed into the AI.
    virtual std::string getJsonStr() override
    {
        std::string func_def_json =
        "{\n"
            "\"type\": \"function\",\n"
            "\"function\" : {\n"
                "\"name\": \"get_current_temperature\",\n"
                "\"description\" : \"Get current temperature at a location.\",\n"
                "\"parameters\" : {\n"
                    "\"type\": \"object\",\n"
                    "\"properties\" : {\n"
                        "\"location\": {\n"
                            "\"type\": \"string\",\n"
                            "\"description\" : \"The location to get the temperature for, in the format \\\"City, State, Country\\\".  Do not appreviate the city, state or country. list them in ascii and lower case.\"\n"
                        "},\n"
                        "\"unit\" : {\n"
                            "\"type\": \"string\",\n"
                            "\"enum\" : [\n"
                                "\"celsius\",\n"
                                "\"fahrenheit\"\n"
                            "],\n"
                            "\"description\" : \"The unit to return the temperature in. Defaults to \\\"celsius\\\".\"\n"
                        "}\n"
                    "},\n"
                    "\"required\": [\n"
                        "\"location\"\n"
                    "]\n"
                "}\n"
            "}\n"
        "}\n";

        return func_def_json;
    }

    //! The name that the AI will use to call this function.  Also referenced in the getJsonStr.
    virtual std::string getName() override
    {
        return "get_current_temperature";
    }

    //! validation code to make sure that the function call has the parameters required. 
    //! More validation may be needed depending on use case.
    virtual bool validateJson(const json& parameters) override
    {
        bool json_wellformed = parameters.contains("name") && parameters.contains("arguments") &&
            parameters["arguments"].contains("location");

        return json_wellformed;
    }

    //! The call into this function, given a json parameter object.
    virtual std::string call(const json& parameters) override
    {
        std::string ai_response_to_tool_result;

        if (!validateJson(parameters))
        {
            // Not only will the error be logged, but the AI will be informed about it as well. This may allow the AI to follow up with a proper correction
            // without needing to fail completely, and still allowing a seamless response to the user.
            loggingPrint(nvigi::LogType::eError, "LLM sent back valid JSON, WITH correct function name, but incorrect function arguments:");
            loggingPrint(nvigi::LogType::eError, parameters.dump().c_str());
            GetCompletion(m_nvigiCtx, "", "The tool call json was malformed.  It must contain a name, arguments, and one of those arguments must be location", ai_response_to_tool_result);
            return ai_response_to_tool_result;
        }

        // You must extract out and validate your arguments.  Keep your arguments simple, otherwise, the more complex
        // they are, the more leeway the LLM has generate them in an unanticipated manner.
        std::string location = parameters["arguments"]["location"];
        bool in_fahrenheit = true;
        if (parameters["arguments"].contains("unit"))
        {
            std::string unit_str = parameters["arguments"]["unit"];
            if (unit_str == "celsius")
            {
                in_fahrenheit = false;
            }
        }

        float temperature = getCurrentTemperature(location);

        // sentinel value that is returned in cases for city we don't have.
        if ( temperature > k_absolute_zero_fahrenheit)
        {
            // convert to celcius if needed
            std::string return_unit = "F";
            if (!in_fahrenheit)
            {
                temperature = (temperature - 32.0f) * (5.0f / 9.0f);
                return_unit = "C";
            }

            // The tool response can be anything really, and the LLM will attempt to parse it reasonably.
            // You can experiment with creating an output schema that is given to the LLM in the function tool 
            // declaration to see if that produces better responses.
            std::string tool_response = std::format("<tool_response>\n{{ \"temperature\": {}, \"unit\" : \"{}\", \"location\" : \"{}\" }}\n</tool_response>", temperature, return_unit, location);
            GetCompletion(m_nvigiCtx, "", tool_response, ai_response_to_tool_result);
        }
        else
        {
            // With errors, the AI can explain the problem to the user.
            GetCompletion(m_nvigiCtx, "", "Tell the user that you don't have information for that city, only Austin, Texas, Durham, North Carolina and Santa Clara, California", ai_response_to_tool_result);
        }

        return ai_response_to_tool_result;
    }
};

///////////////////////////////////////
//! WikiSearchTool
//! 
//! This tool uses libCurl to make wikipedia searches.  It won't pull down the full article, but will simply 
//! pull the top search results
//! 
class WikiSearchTool : public ToolBaseClass
{
private:
    //! Helper method for libCurl usage
    static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp) 
    {
        std::string* output = static_cast<std::string*>(userp);
        size_t totalSize = size * nmemb;
        output->append(static_cast<char*>(contents), totalSize);
        return totalSize;
    }

    //! if available, pulls text from the extract of a title and returns it to the LLM as reference for a potential answer.
    std::string getWikipediaSummary(CURL* curl, std::string title)
    {
        std::string response_data;

        // Optionally reset if you've done a different transfer with this handle
        curl_easy_reset(curl);

        char* encoded_title = curl_easy_escape(curl, title.c_str(), 0);

        std::string summary_url = std::format("https://en.wikipedia.org/api/rest_v1/page/summary/{}", encoded_title);

        // Must identify yourself as a bot and provide appropriate contact information otherwise you risk being blocked.  
        // If this happens, the calling model will likely stop calling this wikipedia function completely
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "NVIGIFCallBot/1.3 (https://developer.nvidia.com/rtx/in-game-inferencing) libcurl/7.8 task:education");

        curl_easy_setopt(curl, CURLOPT_URL, summary_url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WikiSearchTool::writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);

        CURLcode res = curl_easy_perform(curl);
        curl_free(encoded_title);

        if (res != CURLE_OK)
        {
            std::stringstream strstr;
            strstr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
            loggingPrint(nvigi::LogType::eError, strstr.str().c_str());
            return "";
        }
        else
        {
            // Parse response_data JSON for "extract" field
            // Use your favorite JSON library for this part
            json page_info = json::parse(response_data);
            if (page_info.contains("extract"))
            {
                // Note the extract only contains the first few sentences of the opening summary of the page, so some
                // Critical information (like most recent updates which tend to be towards the end), might be cut off here, 
                // but this should serve to illustrate usage.
                return page_info["extract"];
            }
            else
            {
                // fallback to just returning the whole json and letting the LLM figure it out.
                return page_info.dump(4);
            }
        }
    }

    //! Does a simple wikipedia search given the search term.
    std::string searchWikipedia(std::string search)
    {
        std::string response_data;

        CURL* curl = curl_easy_init();
        if (curl) 
        {
            // curl_easy_escape allocates memory that must be freed later.
            char* encoded_search = curl_easy_escape(curl, search.c_str(), 0);
            std::string wikipedia_search_url = std::format("https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={}&format=json", encoded_search);

            // Must identify yourself as a bot and provide appropriate contact information otherwise you risk being blocked.  
            // If this happens, the calling model will likely stop calling this wikipedia function completely.
            curl_easy_setopt(curl, CURLOPT_USERAGENT, "NVIGIFCallBot/1.3 (https://developer.nvidia.com/rtx/in-game-inferencing) libcurl/7.8 task:education");

            curl_easy_setopt(curl, CURLOPT_URL, wikipedia_search_url.c_str() );
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WikiSearchTool::writeCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);

            CURLcode res = curl_easy_perform(curl);

            // free the encoded_search easy_escape string.
            curl_free(encoded_search);

            if (res != CURLE_OK)
            {
                std::stringstream strstr;
                strstr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
                loggingPrint(nvigi::LogType::eError, strstr.str().c_str());
            }
            else
            {
                try
                {
                    json search_results = json::parse(response_data);
                    if (search_results.contains("query") &&
                        search_results["query"].contains("search") &&
                        search_results["query"]["search"].is_array() &&
                        !search_results["query"]["search"].empty() &&
                        search_results["query"]["search"][0].contains("title"))
                    {
                        std::string search_title = search_results["query"]["search"][0]["title"];
                        response_data = getWikipediaSummary(curl, search_title);
                    }
                    // otherwise, just return the search results we have, so keep response_data as is.
                }
                catch (std::exception e)
                {
                    std::stringstream strstr;
                    strstr << e.what();
                    loggingPrint(nvigi::LogType::eError, strstr.str().c_str());
                }
            }
        }
        else {
            std::stringstream strstr;
            strstr << "Failed to initialize curl." << std::endl;
            loggingPrint(nvigi::LogType::eError, strstr.str().c_str());
        }

        // clean up curl allocation.
        curl_easy_cleanup(curl);

        return response_data;
    }

public:
    WikiSearchTool(NVIGIAppCtx& nvigiCtx) : ToolBaseClass(nvigiCtx) {}

    //! In this example, the JsonFunctionCreator creates the json string based on the function name, 
    //! description and parameters making the code much clearer.  It also avoids typos that result from 
    //! typing out the json long hand, at the expense of abstracting a bit what the underlying data looks like.
    virtual std::string getJsonStr() override
    {
        JsonFunctionCreator func_creator;
        func_creator.setFunctionName(getName());
        func_creator.setFunctionDesc("Performs a search on wikipedia.");
        func_creator.addParameter("search_query", "string", "The topic to search for on wikipedia", true);

        std::string func_def_json = func_creator.getString();

        return func_def_json;
    }

    //! The name that the AI will use to call this function.  Also referenced in the getJsonStr.
    virtual std::string getName() override
    {
        return "search_wikipedia";
    }

    //! validation code to make sure that the function call has the parameters required. 
    //! More validation may be needed depending on use case.
    virtual bool validateJson(const json& parameters) override
    {
        bool json_wellformed = parameters.contains("name") &&
            parameters.contains("arguments") &&
            parameters["arguments"].contains("search_query");

        return json_wellformed;
    }

    //! The call into this function, given a json parameter object.
    virtual std::string call(const json& parameters) override
    {
        std::string ai_response_to_tool_result;

        if (!validateJson(parameters))
        {
            // Not only will the error be logged, but the AI will be informed about it as well. This may allow the AI to follow up with a proper correction
            // without needing to fail completely, and still allowing a seamless response to the user.
            loggingPrint(nvigi::LogType::eError, "LLM sent back valid JSON, WITH correct function name, but incorrect function arguments:");
            loggingPrint(nvigi::LogType::eError, parameters.dump().c_str());
            GetCompletion(m_nvigiCtx, "", "The tool call json was malformed.  It must contain a name, arguments, and one of those arguments must be search_query", ai_response_to_tool_result);
            return ai_response_to_tool_result;
        }

        // You must extract out and validate your arguments.  Keep your arguments simple, otherwise, the more complex
        // they are, the more leeway the AI has generate them in an unanticipated manner.
        std::string search_query = parameters["arguments"]["search_query"];
        std::string search_response = searchWikipedia(search_query);

        if (! search_response.empty())
        {
            // Here, we use json to make a cleaner response than needing to hardcode the schema.
            // Note, that there is no fixed output schema required here, the AI will simply attempt to parse what
            // is given it as best it can.
            json tool_response = {
                {"search_query", search_query},
                {"search_response", search_response}
            };
            std::string tool_response_string = tool_response.dump(4);
            GetCompletion(m_nvigiCtx, "", tool_response_string, ai_response_to_tool_result);
        }
        else
        {
            // With errors, the AI can explain the problem to the user, possibly attempting a best guess.
            GetCompletion(m_nvigiCtx, "", "wiki search failed. Explain to the user that you were unable to search wiki and then try your best to answer the question", ai_response_to_tool_result);
        }

        return ai_response_to_tool_result;
    }
};

typedef std::vector<ToolBaseClass*> ToolVec;

//! iterates through all tools given the json_str and attempt to call the correct one and provide an answer
void callTool(NVIGIAppCtx& nvigiCtx, const ToolVec& tools, const std::string& json_str, std::string& answer, bool tools_enabled)
{
    try
    {
        json tool_params = json::parse(json_str);

        if (tool_params.contains("name"))
        {
            std::string function_name = tool_params["name"];

            if (!tools_enabled)
            {
                // Tools turned off, but AI has attempted a tool call.  Explain to the AI that tools have been disabled and try it's best.
                std::string tool_unavailable_prompt = "Explain that tool calls have been disabled and attempt to answer the user's question as best you can";
                GetCompletion(nvigiCtx, "", tool_unavailable_prompt, answer);
            }
            else
            {
                // Tools turned on, so see if there is an appropriate tool to call.
                bool called = false;
                for (const auto& tool : tools)
                {
                    if (_stricmp(tool->getName().c_str(), function_name.c_str()) == 0)
                    {
                        answer = tool->call(tool_params);
                        called = true;
                        break;
                    }
                }

                //! The LLM can hallucinate unsupported functions.  In the case a tool was called, but 
                //! not one of the names we support, remind the AI of which tools are available.
                if (!called)
                {
                    std::string tools_names_str;
                    for (const auto& tool : tools)
                    {
                        tools_names_str += tool->getName();
                        tools_names_str += "\n";
                    }
                    answer = "Unable to find a tool called " + function_name + ". The only tools available to be called are: " + tools_names_str;
                }
            }
        }
        else
        {
            loggingPrint(nvigi::LogType::eError, "LLM sent back valid JSON, but JSON has no 'name' keyword to indicate which function to call:");
            loggingPrint(nvigi::LogType::eError, answer.c_str());
            answer = "Unable to find tool without a name keyword";
        }
    }
    catch (const json::parse_error& e)
    {
        loggingPrint(nvigi::LogType::eError, "JSON parse error:");
        loggingPrint(nvigi::LogType::eError, e.what());
        answer = "malformed json returned";
    }
}

int main(int argc, char** argv)
{
    NVIGIAppCtx nvigiCtx;
    // Block the llama output, so it does not pollute the app's console output
#ifdef NVIGI_WINDOWS
    FILE* f{};
    freopen_s(&f, "NUL", "w", stderr);
#else
    freopen("dev/nul", "w", stderr);
#endif

    if (argc != 2)
    {
        loggingPrint(nvigi::LogType::eError, "nvigi.fcall <path to models>");
        return -1;
    }
    modelDir = argv[1];
    auto pathToSDKUtf8 = getExecutablePath();

    if (InitNVIGI(nvigiCtx, pathToSDKUtf8))
        return -1;

    //////////////////////////////////////////////////////////////////////////////
    //! Init Plugin Interfaces and Instances

    //! GPT Instance
    if (InitGPT(nvigiCtx, modelDir))
    {
        loggingPrint(nvigi::LogType::eError, "Could not create GPT instance");
        return -1;
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Function Calling Flow

    // User Instructions
    std::cout << "\n\nIn this sample, you may ask about the temperature in Durham, Austin or Santa Clara.\n"
        "If you have an internet connection, you may also ask about topics that might be found in a typical wikipedia search.\n\n"
        "Example: Can you tell me about the 2025 Russian earthquake?\n\n"
        "Example: What is the temperature in Austin?\n\n"
        "Commands:\n"
        "exit : exits this program\n"
        "disable_tools : disable the ability for the AI to call tools\n"
        "enable_tools : enables the ability for the AI to call tools [DEFAULT ON]\n\n";
    
    //! Tools we have defined.  Placed in an array for easy access.
    CurrentTempTool temp_tool(nvigiCtx);
    WikiSearchTool wiki_tool(nvigiCtx);

    ToolVec tools{ &temp_tool, &wiki_tool };

    //! Get the json string for all the tools we support.  We will inject this into the tool prompt
    std::string tools_str;
    for (const auto& tool : tools) 
    {
        tools_str += tool->getJsonStr();
        tools_str += "\n";
    }

    //! Set the system prompt
    std::string system_prompt = "You are a helpful AI assistant.";

    //! Fill out the tool instructions. This may change per model - check your model and adjust accordingly.
    std::string full_tool_prompt = std::format("\n\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        "{}\n"
        "</tools>\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        "{{\"name\": <function-name>, \"arguments\" : <args-json-object>}}\n"
        "</tool_call>\n"
        , tools_str );
    std::string full_system_prompt = system_prompt + full_tool_prompt;

    //! Get the user's prompt
    std::string user_prompt;
    std::cout << "\nUser: ";
    std::getline(std::cin, user_prompt);

    //! whether or not tools are enabled.  User togglable.
    bool tools_enabled = true;

    //! Loop until the user wants to exit the conversation.
    while (user_prompt != "exit")
    {
        if (user_prompt == "disable_tools")
        {
            tools_enabled = false;
            full_system_prompt = system_prompt;
            std::cout << "Assistant: Tools Disabled" << std::endl;
        }
        else if (user_prompt == "enable_tools")
        {
            tools_enabled = true;
            full_system_prompt = system_prompt + full_tool_prompt;
            std::cout << "Assistant: Tools Enabled" << std::endl;
        }
        else
        {
            //! Pass the prompt to the LLM for completion.
            std::string answer = "";
            GetCompletion(nvigiCtx, full_system_prompt, user_prompt, answer);

            //! clearing out the system prompt as having a new system prompt implicitly clears the conversation cache.
            full_system_prompt = "";

            //! The LLM might decide to call multiple tools before it reports back a result to the user.
            //! So we loop until we finish matching all tool_call requests.  Consider limiting number of tool_calls as 
            //! a precaution to avoid unending loops.  Omitted here for clarity.
            bool regex_match = true;
            while (regex_match)
            {
                std::regex re("<tool_call>([\\s\\S]*?)</tool_call>");
                std::smatch match;
                if (std::regex_search(answer, match, re))
                {
                    std::string json_str = match[1].str();

                    //! iterate through all tools given the json_str and attempt to call the correct one.
                    callTool(nvigiCtx, tools, json_str, answer, tools_enabled);
                }
                else
                {
                    regex_match = false;
                }
            }

            // For QWen reasoning models, remove all the thinking sections.
            std::regex pattern(R"(<think>[\s\S]*?</think>)", std::regex::icase);
            std::string cleaned_answer = std::regex_replace(answer, pattern, "");

            // At times Qwen may only return </think> instead of <think>...</think>, especially in non-reasoning mode on smaller models.
            // This captures that case.
            if (cleaned_answer.find("</think>") != std::string::npos)
            {
                std::regex end_think_pattern(R"(</think>)", std::regex::icase);
                cleaned_answer = std::regex_replace(cleaned_answer, end_think_pattern, "");
            }

            std::cout << "Assistant: " << cleaned_answer << std::endl;
        }

        std::cout << "\nUser: ";
        std::getline(std::cin, user_prompt);
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Shutdown NVIGI

    if (ReleaseGPT(nvigiCtx))
        return -1;

    return ShutdownNVIGI(nvigiCtx);
}