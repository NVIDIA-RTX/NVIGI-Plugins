// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#include <NvInfer.h>
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                                                               \
    {                                                                                                                  \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            std::cout << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err)           \
                      << std::endl;                                                                                    \
        }                                                                                                              \
    }

inline size_t trtTypeSize(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kINT64:
        return 8u;
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT:
        return 4U;
    case nvinfer1::DataType::kHALF:
        return 2U;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8:
        return 1U;
    }
    return 0;
}

using shape_vec_t = std::vector<size_t>;
using datatype_t = nvinfer1::DataType;

// GPU tensor for TensorRT
class TRTTensor
{
  public:
    // Constructor 1: create non-initialized device tensor
    TRTTensor(const shape_vec_t &shape, datatype_t type)
        : mShape(shape), mType(type), mNbElements(computNbElements()), mSize(computeSize()), mDevicePtr(nullptr)
    {
        allocateTensor();
    }

    // Constructor 2: initialize device data with a host pointer
    TRTTensor(void *hostPtr, const shape_vec_t &shape, datatype_t type)
        : mShape(shape), mType(type), mNbElements(computNbElements()), mSize(computeSize()), mDevicePtr(nullptr)
    {
        copyFromHost(hostPtr, mSize);
    }

    // Destructor: free the device memory
    ~TRTTensor()
    {
        if (mDevicePtr)
        {
            CUDA_CHECK(cudaFree(mDevicePtr));
        }
    }

    const shape_vec_t &getShape() const
    {
        return mShape;
    }
    datatype_t getType() const
    {
        return mType;
    }
    size_t getSize() const
    {
        return mSize;
    }
    void *getDevicePtr() const
    {
        return mDevicePtr;
    }
    size_t getNbElements() const
    {
        return mNbElements;
    }

    bool copyToHost(void *hostPtr)
    {
        if (mDevicePtr == nullptr)
        {
            std::cerr << "Error: Device memory is not allocated!" << std::endl;
            return false;
        }
        // Copy memory from the device to the host on our stream
        CUDA_CHECK(cudaMemcpy(hostPtr, mDevicePtr, mSize, cudaMemcpyDeviceToHost));
        return true;
    }

    bool copyDeviceToDevice(void* mDevicePtrDest)
    {
        if (mDevicePtr == nullptr)
        {
            std::cerr << "Error: Device memory is not allocated!" << std::endl;
            return false;
        }

        if (mDevicePtrDest == nullptr)
        {
            std::cerr << "Error: Device dst memory is not allocated!" << std::endl;
            return false;
        }

        // Copy memory from the device to the host on our stream
        CUDA_CHECK(cudaMemcpy(mDevicePtrDest, mDevicePtr, mSize, cudaMemcpyDeviceToDevice));
        return true;
    }

  private:
    shape_vec_t mShape; // Shape of the tensor (e.g., {1, 3, 224, 224} for a 4D tensor)
    datatype_t mType;   // TensorRT Data type of the tensor
    size_t mNbElements; // Number of elements
    size_t mSize;       // Size of the tensor in bytes
    void *mDevicePtr;   // Pointer to the GPU memory

    size_t computNbElements()
    {
        size_t numElements = 1;
        for (auto dim : mShape)
        {
            numElements *= dim;
        }
        return numElements;
    }

    size_t computeSize()
    {
        size_t numElements = computNbElements();

        // Get the size of each element based on the data type
        size_t elementSize = trtTypeSize(mType);
        return numElements * elementSize;
    }

    // Allocate memory for the tensor based on shape and data type
    void allocateTensor()
    {

        // Allocate GPU memory
        CUDA_CHECK(cudaMalloc(&mDevicePtr, mSize));
    }

    bool copyFromHost(void *hostPtr, size_t nBytes)
    {
        if (mDevicePtr == nullptr)
        {
            CUDA_CHECK(cudaMalloc(&mDevicePtr, nBytes));
        }
        // Copy memory from the host to the device on our stream
        CUDA_CHECK(cudaMemcpy(mDevicePtr, hostPtr, mSize, cudaMemcpyHostToDevice));
        return true;
    }
};
