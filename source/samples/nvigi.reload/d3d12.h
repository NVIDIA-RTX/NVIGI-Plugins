// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace nvigi
{
using namespace Microsoft::WRL;

struct D3D12ContextInfo {
    ComPtr <IDXGIAdapter3> adapter{};
    ComPtr <IDXGIFactory2> factory{};
    ComPtr<ID3D12Device> device{};
    ComPtr<ID3D12CommandQueue> d3d_direct_queue{};
    ComPtr<ID3D12CommandQueue> d3d_compute_queue{};
    ComPtr<ID3D12CommandQueue> d3d_copy_queue{};

    static D3D12ContextInfo* CreateD3D12Device()
    {
        D3D12ContextInfo* p = new D3D12ContextInfo;
        UINT dxgi_factory_flags = 0;

#ifdef _DEBUG
        {
            // Enable the debug layer (requires the Graphics Tools "optional feature").
            // NOTE: Enabling the debug layer after device creation will invalidate the active device.
            {
                ComPtr<ID3D12Debug> debug_controller;
                if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller))))
                {
                    debug_controller->EnableDebugLayer();

                    // Enable additional debug layers.
                    dxgi_factory_flags |= DXGI_CREATE_FACTORY_DEBUG;
                }
            }
        }
#endif

        HRESULT hres = CreateDXGIFactory2(dxgi_factory_flags, IID_PPV_ARGS(&p->factory));

        unsigned int adapter_no = 0;
        while (SUCCEEDED(hres))
        {
            ComPtr<IDXGIAdapter> p_adapter;
            hres = p->factory->EnumAdapters(adapter_no, &p_adapter);

            if (SUCCEEDED(hres))
            {

                DXGI_ADAPTER_DESC a_desc;
                p_adapter->GetDesc(&a_desc);

                // NVDA adapter
                if (a_desc.VendorId == 0x10DE)
                {
					IDXGIAdapter3* p_adapter3 = nullptr;
                    p_adapter->QueryInterface(&p_adapter3);
                    p->adapter = p_adapter3;
                    break;
                }
            }

            adapter_no++;
        }

        if (!p->adapter)
        {
            std::cerr << "error: No NV adapter found!" << std::endl;
            delete p;
            return nullptr;
        }

        if (!SUCCEEDED(D3D12CreateDevice(
            p->adapter.Get(),
            D3D_FEATURE_LEVEL_12_2,
            IID_PPV_ARGS(&p->device))
        ))
        {
            std::cerr << "error: failed to create a D32D12 device" << std::endl;
            delete p;
            return nullptr;
        }

        // create command queue for the device
        D3D12_COMMAND_QUEUE_DESC commandqueue_desc;
        commandqueue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        commandqueue_desc.NodeMask = 0;
        commandqueue_desc.Priority = 0;
        commandqueue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

        if (!SUCCEEDED(p->device->CreateCommandQueue(&commandqueue_desc, __uuidof(ID3D12CommandQueue), (void**)&p->d3d_direct_queue)))
        {
            std::cerr << "error: failed to create a D32D12 direct queue" << std::endl;
            delete p;
            return nullptr;
        }

        commandqueue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;

        if (!SUCCEEDED(p->device->CreateCommandQueue(&commandqueue_desc, __uuidof(ID3D12CommandQueue), (void**)&p->d3d_compute_queue)))
        {
            std::cerr << "error: failed to create a D32D12 compute queue" << std::endl;
            delete p;
            return nullptr;
        }

        commandqueue_desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;

        if (!SUCCEEDED(p->device->CreateCommandQueue(&commandqueue_desc, __uuidof(ID3D12CommandQueue), (void**)&p->d3d_copy_queue)))
        {
            std::cerr << "error: failed to create a D32D12 compute queue" << std::endl;
            delete p;
            return nullptr;
        }

        active_instance = p;
        return p;
    }

    void WaitForIdle()
    {
        if (d3d_direct_queue)
        {
            ID3D12Fence* d3d_fence{};
            device->CreateFence(1, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d_fence));
            d3d_fence->Signal(0); // Reset value on the CPU
            d3d_direct_queue->Signal(d3d_fence, UINT64(-1)); // Signal it on the GPU
            d3d_fence->SetEventOnCompletion(UINT64(-1), nullptr);
            d3d_fence->Release();
        }
        if (d3d_compute_queue)
        {
            ID3D12Fence* d3d_fence{};
            device->CreateFence(1, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d_fence));
            d3d_fence->Signal(0); // Reset value on the CPU
            d3d_compute_queue->Signal(d3d_fence, UINT64(-1)); // Signal it on the GPU
            d3d_fence->SetEventOnCompletion(UINT64(-1), nullptr);
            d3d_fence->Release();
        }
        if (d3d_copy_queue)
        {
            ID3D12Fence* d3d_fence{};
            device->CreateFence(1, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d_fence));
            d3d_fence->Signal(0); // Reset value on the CPU
            d3d_copy_queue->Signal(d3d_fence, UINT64(-1)); // Signal it on the GPU
            d3d_fence->SetEventOnCompletion(UINT64(-1), nullptr);
            d3d_fence->Release();
        }
    }

    ~D3D12ContextInfo()
    {
        WaitForIdle();
        d3d_direct_queue = nullptr;
        d3d_compute_queue = nullptr;
        d3d_copy_queue = nullptr;
        device = nullptr;
        adapter = nullptr;
        factory = nullptr;

        active_instance = nullptr;
    }

    static D3D12ContextInfo* GetActiveInstance() { return active_instance; }

    private:
        D3D12ContextInfo() {}

        static D3D12ContextInfo* active_instance;
};
D3D12ContextInfo* D3D12ContextInfo::active_instance = nullptr;
}