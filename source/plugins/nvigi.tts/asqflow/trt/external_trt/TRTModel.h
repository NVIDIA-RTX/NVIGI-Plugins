// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#include "nvigi_core/source/core/nvigi.log/log.h"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <cuda_runtime_api.h>
#include <format>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <NvInferImpl.h>
#include <exception>
#include <iosfwd>
#include <string>
#include <cstdint>
#include <cassert>
#include <utility>
#include <stdexcept>
#include <nvigi_cuda.h>
#include <cuda.h>

#define CHECK_CUDA_AND_RETURN(call)                                                                                    \
    {                                                                                                                  \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            NVIGI_LOG_ERROR("CUDA Error: %s at %s : %s", cudaGetErrorString(err), __FILE__, __LINE__);                 \
            return false;                                                                                              \
        }                                                                                                              \
        return true;                                                                                                   \
    }

#define CHECK_CUDA(call)                                                                                               \
    {                                                                                                                  \
                                                                                                                       \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
                                                                                                                       \
            NVIGI_LOG_ERROR("CUDA Error: %s at %s : %s", cudaGetErrorString(err), __FILE__, __LINE__);                 \
        }                                                                                                              \
    }

using shape_vec_t = std::vector<size_t>;

inline nvinfer1::Dims shapeToDims(shape_vec_t const &vec)
{
    nvinfer1::Dims dims{vec.size(), {}};
    std::copy_n(vec.begin(), dims.nbDims, std::begin(dims.d));
    return dims;
}


inline shape_vec_t DimsToShape(nvinfer1::Dims dims)
{
    shape_vec_t shapeVec;
    // nvinfer1::Dims dims{ vec.size(), {} };
    for (int i = 0; i < dims.nbDims; i++)
    {
        shapeVec.push_back((size_t)dims.d[i]);
    }
    return shapeVec;
}

//! Implements the TensorRT IStreamReader to allow deserializing an engine directly from the plan file.
class FileStreamReader final : public nvinfer1::IStreamReader
{
  public:
    bool open(std::string filepath)
    {
        mFile.open(filepath, std::ios::binary);
        return mFile.is_open();
    }

    void close()
    {
        if (mFile.is_open())
        {
            mFile.close();
        }
    }

    ~FileStreamReader() final
    {
        close();
    }

    int64_t read(void *dest, int64_t bytes) final
    {
        if (!mFile.good())
        {
            return -1;
        }
        mFile.read(static_cast<char *>(dest), bytes);
        return mFile.gcount();
    }

    void reset()
    {
        assert(mFile.good());
        mFile.seekg(0);
    }

    bool isOpen() const
    {
        return mFile.is_open();
    }

  private:
    std::ifstream mFile;
};

class LoggerTRT : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Filter out INFO messages if needed
        if (severity == Severity::kINFO) return;

		NVIGI_LOG_INFO("[TensorRT] %s: %s", severityToString(severity), msg);
    }

private:
    const char* severityToString(Severity severity) {
        switch (severity) {
        case Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
        case Severity::kERROR: return "ERROR";
        case Severity::kWARNING: return "WARNING";
        case Severity::kINFO: return "INFO";
        case Severity::kVERBOSE: return "VERBOSE";
        default: return "UNKNOWN";
        }
    }
};

// By default, TensorRT (TRT) uses cuMemAllocAsync-cuMemPool for memory allocation,
// which is currently incompatible with CIG.
// To resolve this, we will attach a custom GPU allocator to TRT that exclusively uses cuMemAlloc.
class GPUAllocator final : public nvinfer1::IGpuAllocator
{

    // Allocate memory on the GPU with specified size and alignment
    void *allocate(uint64_t size, uint64_t alignment, uint32_t /*flags*/) noexcept override
    {
        try
        {
            // Validate alignment - must be a power of 2
            if (alignment == 0 || (alignment & (alignment - 1)) != 0)
            {
                NVIGI_LOG_ERROR("Error: Memory alignment must be a power of 2.");
                return nullptr;
            }

            // Allocate memory
            void *memory = nullptr;
            CHECK_CUDA(cudaMalloc(&memory, size));
            
            if(!memory)
            {
                return nullptr; // Return nullptr if allocation fails
            }

            // Ensure memory is properly aligned
            if (reinterpret_cast<uintptr_t>(memory) % alignment != 0)
            {
                NVIGI_LOG_ERROR("Error: Allocated memory is not properly aligned.");
                CHECK_CUDA(cudaFree(memory)); // Free the incorrectly aligned memory
                return nullptr;
            }

            return memory;
        }
        catch (const std::exception &ex)
        {
            NVIGI_LOG_ERROR("Exception in allocate: %s", ex.what());
            return nullptr;
        }
    }

    bool deallocate(void *const memory) noexcept
    {
        if (!memory)
        {
            return false; // return false instead of dealinglocate memory that isn't there
        }

        CHECK_CUDA_AND_RETURN(cudaFree(memory));
        return true;
    }

    void *reallocate(void * /*baseAddr*/, uint64_t /*alignment*/, uint64_t /*newSiz*/) noexcept
    {
        return nullptr;
    }
};

class TRTModel
{
  public:
    bool init(const std::string &engineFilePath)
    {
        bool success = loadEngine(engineFilePath);
        return success;
    }

    ~TRTModel()
    {
        // members will delete themselves on exit safely
        mContext.reset();
        mEngine.reset();
        mGpuAllocator.reset();        
    }

    // Load engine from file, create engine and context
    bool loadEngine(const std::string &engineFilePath)
    {
        FileStreamReader engineData;
        if (!engineData.open(engineFilePath))
        {
            NVIGI_LOG_ERROR("Failed to load engine file!");
            return false;
        }

        mGpuAllocator.reset(new GPUAllocator());

        nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(mLogger);
        if (!runtime)
        {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }

        // Get the current CUDA context
        CUcontext context;
        auto result = cuCtxGetCurrent(&context);
        if (result != CUDA_SUCCESS) {
            NVIGI_LOG_ERROR("Failed to get current CUDA context");
            return -1;
        }
        // Query the CIG enabled limit
        size_t cigEnabled;
        result = cuCtxGetLimit(&cigEnabled, CU_LIMIT_CIG_ENABLED);
        if (result != CUDA_SUCCESS) {
            NVIGI_LOG_ERROR("Failed to get CIG enabled limit");
            return -1;
        }
        // Print the result
        NVIGI_LOG_INFO("CIG Enabled Limit: %d",cigEnabled);

        runtime->setEngineHostCodeAllowed(true);
        runtime->setGpuAllocator(mGpuAllocator.get());
        mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData));
        delete runtime;
        if (!mEngine)
        {
            throw std::runtime_error("Failed to deserialize the engine");
        }

        mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        if (!mContext)
        {
            throw std::runtime_error("Failed to create execution context");
        }

        // save IO names when loading a new engine
        setIONames();
        return true;
    }

    void setIONames()
    {
        mIONamesInOrder.clear();
        auto nIO = mEngine->getNbIOTensors();
        mIONamesInOrder.resize(nIO);
        for (int i = 0; i < nIO; ++i)
        {
            auto tensorName = mEngine->getIOTensorName(i);
            auto tensorIOMode = mEngine->getTensorIOMode(tensorName);
            mIONamesInOrder[i] = {tensorName, tensorIOMode};
        }
    }

    std::vector<std::pair<std::string, nvinfer1::TensorIOMode>> getIONamesInOrder()
    {
        return mIONamesInOrder;
    }

    // Set IO device pointers
    // map should contain input AND output names, but safe to have other items too
    bool setIOBuffers(const std::unordered_map<std::string, void *> &ioBuffers)
    {
        for (auto &io : mIONamesInOrder)
        {
            auto &ioName = io.first;
            if (ioBuffers.find(ioName) == ioBuffers.end())
            {
                NVIGI_LOG_ERROR("Invalid Input: Couldn't find IO key - %s", ioName.c_str());
                return false;
            }
            mIOBuffers.insert_or_assign(ioName, ioBuffers.at(ioName));
        }
        return true;
    }

    // Set input buffer pointers
    bool setInputShapes(const std::unordered_map<std::string, shape_vec_t> &inputShapes)
    {
        for (auto &io : mIONamesInOrder)
        {
            // skip when not input
            if (io.second != nvinfer1::TensorIOMode::kINPUT)
            {
                continue;
            }

            auto &inputName = io.first;
            if (inputShapes.find(inputName) == inputShapes.end())
            {
                NVIGI_LOG_ERROR("Invalid Input Shape: Couldn't find input key - %s", inputName);
                return false;
            }
            mInputShapes.insert_or_assign(inputName, inputShapes.at(inputName));
            nvinfer1::Dims dims = shapeToDims(inputShapes.at(inputName));
            mContext->setInputShape(inputName.c_str(), dims);
        }
        return true;
    }

    // Run sync inference using executeV2
    bool execute(cudaStream_t stream)
    {
        for (auto& io : mIONamesInOrder)
        {
            auto& ioName = io.first;
            mContext->setTensorAddress(ioName.c_str(), mIOBuffers.at(ioName));
        }

        bool retval = mContext->enqueueV3(stream);
        cudaStreamSynchronize(stream);

        return retval;
    }

    int32_t getAuxStreamCount()
    {
        return mEngine->getNbAuxStreams();
    }

    void setAuxStreams(cudaStream_t* streams, size_t nbStreams)
    {
        mContext->setAuxStreams(streams, int32_t(nbStreams));
    }

  private:
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
    std::unique_ptr<GPUAllocator> mGpuAllocator;
    LoggerTRT mLogger;


    // idx in vector will match idx in TRT engine, for ease of use in executeV2()
    // io mode is saved as well to tell input/output
    std::vector<std::pair<std::string, nvinfer1::TensorIOMode>> mIONamesInOrder;
    std::unordered_map<std::string, void *> mIOBuffers;
    std::unordered_map<std::string, shape_vec_t> mInputShapes;
};
