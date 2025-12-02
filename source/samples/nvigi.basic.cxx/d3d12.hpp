// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <atomic>

namespace nvigi {
namespace d3d12 {

using Microsoft::WRL::ComPtr;

// Forward declarations for D3D12 callback types
using PFun_D3D12CreateCommittedResource = ID3D12Resource*(*)(
    ID3D12Device*,
    const D3D12_HEAP_PROPERTIES*,
    D3D12_HEAP_FLAGS,
    const D3D12_RESOURCE_DESC*,
    D3D12_RESOURCE_STATES,
    const D3D12_CLEAR_VALUE*,
    void*
);
using PFun_D3D12DestroyResource = void(*)(ID3D12Resource*, void*);

struct D3D12Config {
    ID3D12Device* device{nullptr};
    ID3D12CommandQueue* command_queue{nullptr};
    
    // Memory allocation callbacks
    PFun_D3D12CreateCommittedResource create_committed_resource_callback{nullptr};
    PFun_D3D12DestroyResource destroy_resource_callback{nullptr};
    void* create_resource_user_context{nullptr};
    void* destroy_resource_user_context{nullptr};
    
    D3D12Config& set_device(ID3D12Device* dev) {
        device = dev;
        return *this;
    }
    D3D12Config& set_queue(ID3D12CommandQueue* queue) {
        command_queue = queue;
        return *this;
    }
    D3D12Config& set_create_committed_resource_callback(PFun_D3D12CreateCommittedResource callback) {
        create_committed_resource_callback = callback;
        return *this;
    }
    D3D12Config& set_destroy_resource_callback(PFun_D3D12DestroyResource callback) {
        destroy_resource_callback = callback;
        return *this;
    }
    D3D12Config& set_create_resource_user_context(void* context) {
        create_resource_user_context = context;
        return *this;
    }
    D3D12Config& set_destroy_resource_user_context(void* context) {
        destroy_resource_user_context = context;
        return *this;
    }
};

// Callback types for memory allocation tracking
using PFun_CreateCommittedResource = ID3D12Resource*(
    ID3D12Device* device,
    const D3D12_HEAP_PROPERTIES* pHeapProperties,
    D3D12_HEAP_FLAGS HeapFlags,
    const D3D12_RESOURCE_DESC* pDesc,
    D3D12_RESOURCE_STATES InitialResourceState,
    const D3D12_CLEAR_VALUE* pOptimizedClearValue,
    void* userContext
);

using PFun_DestroyResource = void(
    ID3D12Resource* pResource,
    void* userContext
);

// Tracking counters (can be used in callbacks)
inline std::atomic<size_t> g_resource_count{0};
inline std::atomic<size_t> g_total_allocation_bytes{0};

struct DeviceAndQueue {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> compute_queue;
};

class D3D12Helper {
public:
    static DeviceAndQueue create_best_compute_device() {
        ComPtr<IDXGIFactory6> factory;
        if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
            throw std::runtime_error("Failed to create DXGI factory");
        }

        // First try to find a high-performance GPU with compute support
        DeviceAndQueue result = create_device_for_adapter_preference(
            factory.Get(), 
            DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE
        );

        if (!result.device) {
            // Fall back to any available adapter
            result = enumerate_and_create_device(factory.Get());
        }

        if (!result.device) {
            throw std::runtime_error("No suitable D3D12 device found");
        }

        return result;
    }

private:
    static DeviceAndQueue create_device_for_adapter_preference(
        IDXGIFactory6* factory,
        DXGI_GPU_PREFERENCE preference
    ) {
        DeviceAndQueue result{};
        UINT adapter_index = 0;
        ComPtr<IDXGIAdapter4> adapter;

        while (factory->EnumAdapterByGpuPreference(
            adapter_index++,
            preference,
            IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND) {
            
            DXGI_ADAPTER_DESC3 desc{};
            adapter->GetDesc3(&desc);

            // Skip software adapters
            if (desc.Flags & DXGI_ADAPTER_FLAG3_SOFTWARE) {
                continue;
            }

            // Try to create device for this adapter
            if (SUCCEEDED(D3D12CreateDevice(
                adapter.Get(),
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&result.device)))) {
                
                // Create compute queue
                if (create_compute_queue(result.device.Get(), &result.compute_queue)) {
                    return result;
                }
            }
        }

        return DeviceAndQueue{};
    }

    static DeviceAndQueue enumerate_and_create_device(IDXGIFactory1* factory) {
        DeviceAndQueue result{};
        UINT adapter_index = 0;
        ComPtr<IDXGIAdapter1> adapter;

        while (factory->EnumAdapters1(adapter_index++, &adapter) != DXGI_ERROR_NOT_FOUND) {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);

            // Skip software adapters
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                continue;
            }

            // Try to create device
            if (SUCCEEDED(D3D12CreateDevice(
                adapter.Get(),
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&result.device)))) {
                
                // Create compute queue
                if (create_compute_queue(result.device.Get(), &result.compute_queue)) {
                    return result;
                }
            }
        }

        return DeviceAndQueue{};
    }

    static bool create_compute_queue(
        ID3D12Device* device,
        ID3D12CommandQueue** queue
    ) {
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        queue_desc.NodeMask = 0;

        return SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(queue)));
    }
};

// Default implementation of memory tracking callbacks
inline ID3D12Resource* default_create_committed_resource(
    ID3D12Device* device,
    const D3D12_HEAP_PROPERTIES* pHeapProperties,
    D3D12_HEAP_FLAGS HeapFlags,
    const D3D12_RESOURCE_DESC* pDesc,
    D3D12_RESOURCE_STATES InitialResourceState,
    const D3D12_CLEAR_VALUE* pOptimizedClearValue,
    void* userContext
) {
    ID3D12Resource* resource = nullptr;
    HRESULT hr = device->CreateCommittedResource(
        pHeapProperties,
        HeapFlags,
        pDesc,
        InitialResourceState,
        pOptimizedClearValue,
        IID_PPV_ARGS(&resource)
    );
    
    if (SUCCEEDED(hr) && resource) {
        g_resource_count++;
        
        // Track allocation size if resource was created
        D3D12_RESOURCE_DESC desc = resource->GetDesc();
        D3D12_RESOURCE_ALLOCATION_INFO allocInfo = device->GetResourceAllocationInfo(0, 1, &desc);
        g_total_allocation_bytes += allocInfo.SizeInBytes;
    }
    
    return resource;
}

inline void default_destroy_resource(
    ID3D12Resource* pResource,
    void* userContext
) {
    if (pResource) {
        g_resource_count--;
        pResource->Release();
    }
}

} // namespace d3d12
} // namespace nvigi
