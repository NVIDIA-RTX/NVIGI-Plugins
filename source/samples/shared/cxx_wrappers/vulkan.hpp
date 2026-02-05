// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <cstdio>

namespace nvigi {
namespace vulkan {

#ifndef VK_NULL_HANDLE
#define VK_NULL_HANDLE nullptr
#endif

// Forward declarations for callback types
using PFun_VkAllocateMemory = VkResult(*)(VkDevice, VkDeviceSize, uint32_t, VkDeviceMemory*);
using PFun_VkFreeMemory = void(*)(VkDevice, VkDeviceMemory);

struct VulkanConfig {
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue compute_queue{VK_NULL_HANDLE};
    VkQueue transfer_queue{VK_NULL_HANDLE};
    
    // Memory allocation callbacks
    PFun_VkAllocateMemory allocate_memory_callback{nullptr};
    PFun_VkFreeMemory free_memory_callback{nullptr};

    VulkanConfig& set_device(VkDevice dev) {
        device = dev;
        return *this;
    }
    
    VulkanConfig& set_queue(VkQueue queue) {
        compute_queue = queue;
        return *this;
    }
    VulkanConfig& set_instance(VkInstance inst) {
        instance = inst;
        return *this;
    }
    VulkanConfig& set_physical_device(VkPhysicalDevice phys_dev) {
        physical_device = phys_dev;
        return *this;
    }
    VulkanConfig& set_transfer_queue(VkQueue queue) {
        transfer_queue = queue;
        return *this;
    }
    VulkanConfig& set_allocate_memory_callback(PFun_VkAllocateMemory callback) {
        allocate_memory_callback = callback;
        return *this;
    }
    VulkanConfig& set_free_memory_callback(PFun_VkFreeMemory callback) {
        free_memory_callback = callback;
        return *this;
    }
};


// Callback types for memory allocation tracking
using PFun_AllocateMemory = VkResult(
    VkDevice device,
    VkDeviceSize size,
    uint32_t memoryTypeIndex,
    VkDeviceMemory* outMemory
);

using PFun_FreeMemory = void(
    VkDevice device,
    VkDeviceMemory memory
);

// Tracking counters (can be used in callbacks)
inline std::atomic<size_t> g_memory_allocation_count{0};
inline std::atomic<size_t> g_total_allocation_bytes{0};

// Detailed tracking for leak detection
inline std::mutex g_memory_map_mutex;
inline std::map<VkDeviceMemory, VkDeviceSize> g_memory_map;

struct VulkanObjects {
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue compute_queue{VK_NULL_HANDLE};
    VkQueue transfer_queue{VK_NULL_HANDLE};
    uint32_t compute_queue_family_index{UINT32_MAX};
    uint32_t transfer_queue_family_index{UINT32_MAX};

    ~VulkanObjects() {
        // Note: Cleanup should be done manually by the user
        // We don't cleanup here to avoid double-free issues
    }
};

class VulkanHelper {
public:
    static VulkanObjects create_best_compute_device() {
        VulkanObjects result{};

        // Create Vulkan instance
        result.instance = create_instance();
        if (!result.instance) {
            throw std::runtime_error("Failed to create Vulkan instance");
        }

        // Enumerate physical devices
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(result.instance, &device_count, nullptr);
        
        if (device_count == 0) {
            vkDestroyInstance(result.instance, nullptr);
            throw std::runtime_error("No Vulkan physical devices found");
        }

        std::vector<VkPhysicalDevice> physical_devices(device_count);
        vkEnumeratePhysicalDevices(result.instance, &device_count, physical_devices.data());

        // Score and select best device
        VkPhysicalDevice best_device = VK_NULL_HANDLE;
        uint32_t best_compute_queue_family = UINT32_MAX;
        uint32_t best_transfer_queue_family = UINT32_MAX;
        int best_score = -1;

        for (const auto& physical_device : physical_devices) {
            uint32_t compute_queue_family = UINT32_MAX;
            uint32_t transfer_queue_family = UINT32_MAX;
            
            find_queue_families(physical_device, compute_queue_family, transfer_queue_family);
            
            if (compute_queue_family == UINT32_MAX) {
                continue; // No compute queue support
            }

            int score = score_physical_device(physical_device);
            if (score > best_score) {
                best_score = score;
                best_device = physical_device;
                best_compute_queue_family = compute_queue_family;
                best_transfer_queue_family = transfer_queue_family;
            }
        }

        if (best_device == VK_NULL_HANDLE) {
            vkDestroyInstance(result.instance, nullptr);
            throw std::runtime_error("No suitable Vulkan device with compute support found");
        }

        result.physical_device = best_device;
        result.compute_queue_family_index = best_compute_queue_family;
        result.transfer_queue_family_index = best_transfer_queue_family;

        // Create logical device and queues
        result.device = create_logical_device(result.physical_device, best_compute_queue_family, best_transfer_queue_family);
        if (!result.device) {
            vkDestroyInstance(result.instance, nullptr);
            throw std::runtime_error("Failed to create Vulkan logical device");
        }

        // Get compute queue
        vkGetDeviceQueue(result.device, best_compute_queue_family, 0, &result.compute_queue);
        
        // Get transfer queue (if available)
        if (best_transfer_queue_family != UINT32_MAX) {
            vkGetDeviceQueue(result.device, best_transfer_queue_family, 0, &result.transfer_queue);
        }

        return result;
    }

    static void cleanup(VulkanObjects& objects) {
        if (objects.device) {
            vkDestroyDevice(objects.device, nullptr);
            objects.device = VK_NULL_HANDLE;
        }
        if (objects.instance) {
            vkDestroyInstance(objects.instance, nullptr);
            objects.instance = VK_NULL_HANDLE;
        }
    }

private:
    static VkInstance create_instance() {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "NVIGI Application";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "NVIGI";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        create_info.enabledLayerCount = 0;
        create_info.enabledExtensionCount = 0;

        VkInstance instance;
        if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }

        return instance;
    }

    static void find_queue_families(VkPhysicalDevice physical_device, uint32_t& compute_queue_family, uint32_t& transfer_queue_family) {
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());

        compute_queue_family = UINT32_MAX;
        transfer_queue_family = UINT32_MAX;

        // Find queue families
        for (uint32_t i = 0; i < queue_family_count; i++) {
            // Find dedicated compute queue (without graphics)
            if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT && !(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                compute_queue_family = i;
            }
            
            // Find dedicated transfer queue (without compute or graphics)
            if (queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT && !(queue_families[i].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT))) {
                transfer_queue_family = i;
            }
        }
        
        // If no dedicated compute queue, find any queue with compute support
        if (compute_queue_family == UINT32_MAX) {
            for (uint32_t i = 0; i < queue_family_count; i++) {
                if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    compute_queue_family = i;
                    break;
                }
            }
        }
    }

    static int score_physical_device(VkPhysicalDevice physical_device) {
        VkPhysicalDeviceProperties device_properties;
        VkPhysicalDeviceFeatures device_features;
        vkGetPhysicalDeviceProperties(physical_device, &device_properties);
        vkGetPhysicalDeviceFeatures(physical_device, &device_features);

        int score = 0;

        // Discrete GPUs have a significant performance advantage
        if (device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 1000;
        }

        // Maximum possible size of textures/memory affects graphics capability
        score += device_properties.limits.maxImageDimension2D;

        return score;
    }

    static std::vector<const char*> filter_available_extensions(
        VkPhysicalDevice physical_device,
        const std::vector<const char*>& desired_extensions)
    {
        // Enumerate available device extensions
        uint32_t extension_count = 0;
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
        
        std::vector<VkExtensionProperties> available_extensions(extension_count);
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, available_extensions.data());
        
        // Filter desired extensions to only those available
        std::vector<const char*> enabled_extensions;
        for (const char* desired : desired_extensions) {
            bool found = std::find_if(available_extensions.begin(), available_extensions.end(),
                [desired](const VkExtensionProperties& ext) {
                    return strcmp(ext.extensionName, desired) == 0;
                }) != available_extensions.end();
            
            if (found) {
                enabled_extensions.push_back(desired);
            }
        }
        
        return enabled_extensions;
    }

    static VkDevice create_logical_device(VkPhysicalDevice physical_device, uint32_t compute_queue_family_index, uint32_t transfer_queue_family_index) {
        float queue_priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
        
        // Create compute queue
        VkDeviceQueueCreateInfo compute_queue_create_info{};
        compute_queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        compute_queue_create_info.queueFamilyIndex = compute_queue_family_index;
        compute_queue_create_info.queueCount = 1;
        compute_queue_create_info.pQueuePriorities = &queue_priority;
        queue_create_infos.push_back(compute_queue_create_info);
        
        // Create transfer queue if available and different from compute queue
        if (transfer_queue_family_index != UINT32_MAX && transfer_queue_family_index != compute_queue_family_index) {
            VkDeviceQueueCreateInfo transfer_queue_create_info{};
            transfer_queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            transfer_queue_create_info.queueFamilyIndex = transfer_queue_family_index;
            transfer_queue_create_info.queueCount = 1;
            transfer_queue_create_info.pQueuePriorities = &queue_priority;
            queue_create_infos.push_back(transfer_queue_create_info);
        }

        VkPhysicalDeviceFeatures device_features{};

        // Desired extensions - will be filtered to only those available
        std::vector<const char*> desired_extensions =
        {
            "VK_EXT_pipeline_robustness",
            "VK_KHR_maintenance4",
            "VK_EXT_subgroup_size_control",
            "VK_KHR_16bit_storage",
            "VK_KHR_shader_float16_int8",
            "VK_KHR_cooperative_matrix",
            "VK_NV_cooperative_matrix2",
            "VK_NV_shader_sm_builtins",
            "VK_KHR_shader_integer_dot_product",
            "VK_KHR_pipeline_executable_properties"
        };

        // Filter to only available extensions
        std::vector<const char*> enabled_extensions = filter_available_extensions(physical_device, desired_extensions);

        VkDeviceCreateInfo device_create_info{};
        device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_create_info.pQueueCreateInfos = queue_create_infos.data();
        device_create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
        device_create_info.pEnabledFeatures = &device_features;
        device_create_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
        device_create_info.ppEnabledExtensionNames = enabled_extensions.data();
        device_create_info.enabledLayerCount = 0;

        VkDevice device;
        if (vkCreateDevice(physical_device, &device_create_info, nullptr, &device) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }

        return device;
    }
};

// Default implementation of memory tracking callbacks
inline VkResult default_allocate_memory(
    VkDevice device,
    VkDeviceSize size,
    uint32_t memoryTypeIndex,
    VkDeviceMemory* outMemory
) {
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    VkResult result = vkAllocateMemory(device, &allocInfo, nullptr, outMemory);
    
    if (result == VK_SUCCESS) {
        g_memory_allocation_count++;
        g_total_allocation_bytes += size;
        
        // Track in map for leak detection
        {
            std::lock_guard<std::mutex> lock(g_memory_map_mutex);
            g_memory_map[*outMemory] = size;
        }
    }
    
    return result;
}

inline void default_free_memory(
    VkDevice device,
    VkDeviceMemory memory
) {
    if (memory != VK_NULL_HANDLE) {
        // Remove from tracking map
        {
            std::lock_guard<std::mutex> lock(g_memory_map_mutex);
            auto it = g_memory_map.find(memory);
            if (it != g_memory_map.end()) {
                g_total_allocation_bytes -= it->second;
                g_memory_map.erase(it);
            }
            // Decrement counter inside mutex to maintain consistency with map
            g_memory_allocation_count--;
        }
        
        vkFreeMemory(device, memory, nullptr);
    }
}

// Reset tracking counters (call at start of test)
inline void reset_tracking() {
    g_memory_allocation_count.store(0);
    g_total_allocation_bytes.store(0);
    std::lock_guard<std::mutex> lock(g_memory_map_mutex);
    g_memory_map.clear();
}

// Check for leaks and return true if leaks detected
inline bool check_leaks() {
    bool hasLeaks = false;
    
    std::lock_guard<std::mutex> lock(g_memory_map_mutex);
    if (!g_memory_map.empty()) {
        hasLeaks = true;
        fprintf(stderr, "!!! VK MEMORY LEAK DETECTED !!! %zu allocations not freed\n", 
            g_memory_map.size());
        for (const auto& [mem, size] : g_memory_map) {
            fprintf(stderr, "  Leaked VkDeviceMemory: %p, Size: %llu bytes (%llu KB)\n", 
                mem, (unsigned long long)size, (unsigned long long)(size / 1024));
        }
    }
    
    if (g_memory_allocation_count.load() != 0) {
        hasLeaks = true;
        fprintf(stderr, "!!! VK COUNT MISMATCH !!! %zu allocations not freed\n", 
            g_memory_allocation_count.load());
    }
    
    return hasLeaks;
}

} // namespace vulkan
} // namespace nvigi

