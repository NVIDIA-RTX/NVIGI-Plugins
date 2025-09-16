// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <donut/app/DeviceManager_VK.h>
#include <donut/core/log.h>
#include "../nvigi/NVIGIContext.h"

#if DONUT_WITH_VULKAN

class DeviceManagerOverride_VK : public DeviceManager_VK
{
public:
    DeviceManagerOverride_VK();

protected:
    bool createDeviceCustom();
    bool CreateDevice() override;
};

#define CHECK(a) if (!(a)) { return false; }

using namespace donut;
using namespace donut::app;

static constexpr uint32_t kComputeQueueIndex = 0;
static constexpr uint32_t kGraphicsQueueIndex = 0;
static constexpr uint32_t kPresentQueueIndex = 0;
static constexpr uint32_t kTransferQueueIndex = 0;


static std::vector<const char*> stringSetToVector(const std::unordered_set<std::string>& set)
{
    std::vector<const char*> ret;
    for (const auto& s : set)
    {
        ret.push_back(s.c_str());
    }

    return ret;
}

template <typename T>
static std::vector<T> setToVector(const std::unordered_set<T>& set)
{
    std::vector<T> ret;
    for (const auto& s : set)
    {
        ret.push_back(s);
    }

    return ret;
}


DeviceManagerOverride_VK::DeviceManagerOverride_VK()
{
    // Initialize base extensions that this class always needs
    optionalExtensions.device.insert(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    optionalExtensions.device.insert(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
    optionalExtensions.device.insert(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    optionalExtensions.device.insert(VK_EXT_PIPELINE_ROBUSTNESS_EXTENSION_NAME);
    optionalExtensions.device.insert(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    optionalExtensions.device.insert(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME);
    optionalExtensions.device.insert(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME);
    optionalExtensions.device.insert(VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME);
    optionalExtensions.device.insert(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
    optionalExtensions.device.insert(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
    
    // Add CIG (Compute in Graphics) extensions if CIG is enabled
    if (NVIGIContext::Get().IsCIGEnabled())
    {
        optionalExtensions.instance.insert(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        optionalExtensions.device.insert(VK_NV_EXTERNAL_COMPUTE_QUEUE_EXTENSION_NAME);
        donut::log::info("DeviceManagerOverride_VK: Added CIG extensions for Vulkan CIG support");
    }

    donut::log::info("DeviceManagerOverride_VK: Initialized with cooperative matrix and advanced features support");
}

bool DeviceManagerOverride_VK::createDeviceCustom()
{
    // figure out which optional extensions are supported
    auto deviceExtensions = m_VulkanPhysicalDevice.enumerateDeviceExtensionProperties();
    for(const auto& ext : deviceExtensions)
    {
        const std::string name = ext.extensionName;
        if (optionalExtensions.device.find(name) != optionalExtensions.device.end())
        {
            if (name == VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME && m_DeviceParams.headlessDevice)
                continue;
            enabledExtensions.device.insert(name);
        }

        if (m_DeviceParams.enableRayTracingExtensions && m_RayTracingExtensions.find(name) != m_RayTracingExtensions.end())
        {
            enabledExtensions.device.insert(name);
        }
    }

    if (!m_DeviceParams.headlessDevice)
    {
        enabledExtensions.device.insert(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    const vk::PhysicalDeviceProperties physicalDeviceProperties = m_VulkanPhysicalDevice.getProperties();
    m_RendererString = std::string(physicalDeviceProperties.deviceName.data());

    bool accelStructSupported = false;
    bool rayPipelineSupported = false;
    bool rayQuerySupported = false;
    bool meshletsSupported = false;
    bool vrsSupported = false;
    bool interlockSupported = false;
    bool barycentricSupported = false;
    bool storage16BitSupported = false;
    bool synchronization2Supported = false;
    bool maintenance4Supported = false;
    bool aftermathSupported = false;
    bool clusterAccelerationStructureSupported = false;
    bool mutableDescriptorTypeSupported = false;
    bool cooperativeMatrixSupported = false;
    bool cooperativeMatrix2Supported = false;
    bool shaderFloat16Int8Supported = false;
    bool subgroupSizeControlSupported = false;
    bool externalComputeQueueSupported = false;

    log::message(m_DeviceParams.infoLogSeverity, "Enabled Vulkan device extensions:");
    for (const auto& ext : enabledExtensions.device)
    {
        log::message(m_DeviceParams.infoLogSeverity, "    %s", ext.c_str());

        if (ext == VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
            accelStructSupported = true;
        else if (ext == VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
            rayPipelineSupported = true;
        else if (ext == VK_KHR_RAY_QUERY_EXTENSION_NAME)
            rayQuerySupported = true;
        else if (ext == VK_NV_MESH_SHADER_EXTENSION_NAME)
            meshletsSupported = true;
        else if (ext == VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)
            vrsSupported = true;
        else if (ext == VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME)
            interlockSupported = true;
        else if (ext == VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)
            barycentricSupported = true;
        else if (ext == VK_KHR_16BIT_STORAGE_EXTENSION_NAME)
            storage16BitSupported = true;
        else if (ext == VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
            synchronization2Supported = true;
        else if (ext == VK_KHR_MAINTENANCE_4_EXTENSION_NAME)
            maintenance4Supported = true;
        else if (ext == VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME)
            m_SwapChainMutableFormatSupported = true;
        else if (ext == VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME)
            aftermathSupported = true;
        else if (ext == VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME)
            clusterAccelerationStructureSupported = true;
        else if (ext == VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME)
            mutableDescriptorTypeSupported = true;
        else if (ext == VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)
            cooperativeMatrixSupported = true;
        else if (ext == VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME)
            cooperativeMatrix2Supported = true;
        else if (ext == VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)
            shaderFloat16Int8Supported = true;
        else if (ext == VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)
            subgroupSizeControlSupported = true;
        else if (ext == VK_NV_EXTERNAL_COMPUTE_QUEUE_EXTENSION_NAME)
            externalComputeQueueSupported = true;
    }

#define APPEND_EXTENSION(condition, desc) if (condition) { (desc).pNext = pNext; pNext = &(desc); }  // NOLINT(cppcoreguidelines-macro-usage)
    void* pNext = nullptr;

    vk::PhysicalDeviceFeatures2 physicalDeviceFeatures2;
    // Determine support for Buffer Device Address, the Vulkan 1.2 way
    auto bufferDeviceAddressFeatures = vk::PhysicalDeviceBufferDeviceAddressFeatures();
    // Determine support for maintenance4
    auto maintenance4Features = vk::PhysicalDeviceMaintenance4Features();
    // Determine support for aftermath
    auto aftermathPhysicalFeatures = vk::PhysicalDeviceDiagnosticsConfigFeaturesNV();

    // Put the user-provided extensiofaftermathFeaturesn structure at the end of the chain
    pNext = m_DeviceParams.physicalDeviceFeatures2Extensions;
    APPEND_EXTENSION(true, bufferDeviceAddressFeatures);
    APPEND_EXTENSION(maintenance4Supported, maintenance4Features);
    APPEND_EXTENSION(aftermathSupported, aftermathPhysicalFeatures);

    physicalDeviceFeatures2.pNext = pNext;
    m_VulkanPhysicalDevice.getFeatures2(&physicalDeviceFeatures2);

    std::unordered_set<int> uniqueQueueFamilies = {
        m_GraphicsQueueFamily };

    if (!m_DeviceParams.headlessDevice)
        uniqueQueueFamilies.insert(m_PresentQueueFamily);

    if (m_DeviceParams.enableComputeQueue)
        uniqueQueueFamilies.insert(m_ComputeQueueFamily);

    if (m_DeviceParams.enableCopyQueue)
        uniqueQueueFamilies.insert(m_TransferQueueFamily);

    float priority = 1.f;
    std::vector<vk::DeviceQueueCreateInfo> queueDesc;
    queueDesc.reserve(uniqueQueueFamilies.size());
    for(int queueFamily : uniqueQueueFamilies)
    {
        queueDesc.push_back(vk::DeviceQueueCreateInfo()
                                .setQueueFamilyIndex(queueFamily)
                                .setQueueCount(1)
                                .setPQueuePriorities(&priority));
    }

    auto accelStructFeatures = vk::PhysicalDeviceAccelerationStructureFeaturesKHR()
        .setAccelerationStructure(true);
    auto rayPipelineFeatures = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR()
        .setRayTracingPipeline(true)
        .setRayTraversalPrimitiveCulling(true);
    auto rayQueryFeatures = vk::PhysicalDeviceRayQueryFeaturesKHR()
        .setRayQuery(true);
    auto meshletFeatures = vk::PhysicalDeviceMeshShaderFeaturesNV()
        .setTaskShader(true)
        .setMeshShader(true);
    auto interlockFeatures = vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT()
        .setFragmentShaderPixelInterlock(true);
    auto barycentricFeatures = vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR()
        .setFragmentShaderBarycentric(true);
    auto vrsFeatures = vk::PhysicalDeviceFragmentShadingRateFeaturesKHR()
        .setPipelineFragmentShadingRate(true)
        .setPrimitiveFragmentShadingRate(true)
        .setAttachmentFragmentShadingRate(true);
    auto vulkan13features = vk::PhysicalDeviceVulkan13Features()
        .setSynchronization2(synchronization2Supported)
        .setMaintenance4(maintenance4Features.maintenance4)
        .setComputeFullSubgroups(subgroupSizeControlSupported)
        .setSubgroupSizeControl(subgroupSizeControlSupported);
    auto aftermathFeatures = vk::DeviceDiagnosticsConfigCreateInfoNV()
        .setFlags(vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableResourceTracking
            | vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderDebugInfo
            | vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderErrorReporting);
    auto clusterAccelerationStructureFeatures = vk::PhysicalDeviceClusterAccelerationStructureFeaturesNV()
        .setClusterAccelerationStructure(true);
    auto mutableDescriptorTypeFeatures = vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT()
        .setMutableDescriptorType(true);
    auto cooperativeMatrixFeatures = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR()
        .setCooperativeMatrix(true);
        
    auto cooperativeMatrix2Features = vk::PhysicalDeviceCooperativeMatrix2FeaturesNV()
        .setCooperativeMatrixTensorAddressing(true)
        .setCooperativeMatrixFlexibleDimensions(true)
        .setCooperativeMatrixPerElementOperations(true)
        .setCooperativeMatrixReductions(true)
        .setCooperativeMatrixConversions(true)
        .setCooperativeMatrixBlockLoads(true)
        .setCooperativeMatrixWorkgroupScope(true);
    
    // External compute queue for CIG support
    VkExternalComputeQueueDeviceCreateInfoNV externalComputeQueueCreateInfo{};
    if (externalComputeQueueSupported && NVIGIContext::Get().IsCIGEnabled())
    {
        externalComputeQueueCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DEVICE_CREATE_INFO_NV;
        externalComputeQueueCreateInfo.pNext = nullptr;
        externalComputeQueueCreateInfo.reservedExternalQueues = 1; // Reserve one external compute queue for CIG
        donut::log::info("DeviceManagerOverride_VK: Configured for %d reserved external compute queues", 
                         externalComputeQueueCreateInfo.reservedExternalQueues);
    }
    
    pNext = nullptr;
    APPEND_EXTENSION(accelStructSupported, accelStructFeatures)
    APPEND_EXTENSION(rayPipelineSupported, rayPipelineFeatures)
    APPEND_EXTENSION(rayQuerySupported, rayQueryFeatures)
    APPEND_EXTENSION(meshletsSupported, meshletFeatures)
    APPEND_EXTENSION(vrsSupported, vrsFeatures)
    APPEND_EXTENSION(interlockSupported, interlockFeatures)
    APPEND_EXTENSION(barycentricSupported, barycentricFeatures)
    APPEND_EXTENSION(clusterAccelerationStructureSupported, clusterAccelerationStructureFeatures)
    APPEND_EXTENSION(mutableDescriptorTypeSupported, mutableDescriptorTypeFeatures)
    APPEND_EXTENSION(cooperativeMatrixSupported, cooperativeMatrixFeatures)
    APPEND_EXTENSION(cooperativeMatrix2Supported, cooperativeMatrix2Features)
    APPEND_EXTENSION(physicalDeviceProperties.apiVersion >= VK_API_VERSION_1_3, vulkan13features)
    APPEND_EXTENSION(physicalDeviceProperties.apiVersion < VK_API_VERSION_1_3 && maintenance4Supported, maintenance4Features);
    //APPEND_EXTENSION(physicalDeviceProperties.apiVersion >= VK_API_VERSION_1_4, pipelineRobustnessFeatures);
    

#if DONUT_WITH_AFTERMATH
    if (aftermathPhysicalFeatures.diagnosticsConfig && m_DeviceParams.enableAftermath)
        APPEND_EXTENSION(aftermathSupported, aftermathFeatures);
#endif
#undef APPEND_EXTENSION

    auto deviceFeatures = vk::PhysicalDeviceFeatures()
        .setShaderImageGatherExtended(true)
        .setSamplerAnisotropy(true)
        .setTessellationShader(true)
        .setTextureCompressionBC(true)
        .setGeometryShader(true)
        .setImageCubeArray(true)
        .setShaderInt16(true)
        .setFillModeNonSolid(true)
        .setFragmentStoresAndAtomics(true)
        .setDualSrcBlend(true)
        .setVertexPipelineStoresAndAtomics(true)
        .setShaderInt64(true)
        .setShaderStorageImageWriteWithoutFormat(true)
        .setShaderStorageImageReadWithoutFormat(true);

    // Add a Vulkan 1.1 structure with default settings to make it easier for apps to modify them
    auto vulkan11features = vk::PhysicalDeviceVulkan11Features()
        .setStorageBuffer16BitAccess(true)
        .setPNext(pNext);

    auto vulkan12features = vk::PhysicalDeviceVulkan12Features()
        .setDescriptorIndexing(true)
        .setRuntimeDescriptorArray(true)
        .setDescriptorBindingPartiallyBound(true)
        .setDescriptorBindingVariableDescriptorCount(true)
        .setTimelineSemaphore(true)
        .setShaderSampledImageArrayNonUniformIndexing(true)
        .setBufferDeviceAddress(bufferDeviceAddressFeatures.bufferDeviceAddress)
        .setShaderSubgroupExtendedTypes(true)
        .setScalarBlockLayout(true)
        .setShaderFloat16(shaderFloat16Int8Supported)
        .setVulkanMemoryModel(true)
        .setVulkanMemoryModelDeviceScope(true)
        .setStorageBuffer8BitAccess(true)
        .setShaderInt8(shaderFloat16Int8Supported)
        .setPNext(&vulkan11features);
    
    auto vulkan14features = vk::PhysicalDeviceVulkan14Features()
        .setPipelineRobustness(true)
        .setPNext(&vulkan12features);

    auto layerVec = stringSetToVector(enabledExtensions.layers);
    auto extVec = stringSetToVector(enabledExtensions.device);

    // Setup pNext chain for device creation, optionally including external compute queue info for CIG
    void* deviceCreatePNext = &vulkan14features;
    if (externalComputeQueueSupported && NVIGIContext::Get().IsCIGEnabled())
    {
        externalComputeQueueCreateInfo.pNext = deviceCreatePNext;
        deviceCreatePNext = &externalComputeQueueCreateInfo;
    }

    auto deviceDesc = vk::DeviceCreateInfo()
        .setPQueueCreateInfos(queueDesc.data())
        .setQueueCreateInfoCount(uint32_t(queueDesc.size()))
        .setPEnabledFeatures(&deviceFeatures)
        .setEnabledExtensionCount(uint32_t(extVec.size()))
        .setPpEnabledExtensionNames(extVec.data())
        .setEnabledLayerCount(uint32_t(layerVec.size()))
        .setPpEnabledLayerNames(layerVec.data())
        .setPNext(deviceCreatePNext);

    if (m_DeviceParams.deviceCreateInfoCallback)
        m_DeviceParams.deviceCreateInfoCallback(deviceDesc);
    
    const vk::Result res = m_VulkanPhysicalDevice.createDevice(&deviceDesc, nullptr, &m_VulkanDevice);
    if (res != vk::Result::eSuccess)
    {
        log::error("Failed to create a Vulkan physical device, error code = %s", nvrhi::vulkan::resultToString(VkResult(res)));
        return false;
    }

    m_VulkanDevice.getQueue(m_GraphicsQueueFamily, kGraphicsQueueIndex, &m_GraphicsQueue);
    if (m_DeviceParams.enableComputeQueue)
        m_VulkanDevice.getQueue(m_ComputeQueueFamily, kComputeQueueIndex, &m_ComputeQueue);
    if (m_DeviceParams.enableCopyQueue)
        m_VulkanDevice.getQueue(m_TransferQueueFamily, kTransferQueueIndex, &m_TransferQueue);
    if (!m_DeviceParams.headlessDevice)
        m_VulkanDevice.getQueue(m_PresentQueueFamily, kPresentQueueIndex, &m_PresentQueue);

    VULKAN_HPP_DEFAULT_DISPATCHER.init(m_VulkanDevice);

    // remember the bufferDeviceAddress feature enablement
    m_BufferDeviceAddressSupported = vulkan12features.bufferDeviceAddress;

    log::message(m_DeviceParams.infoLogSeverity, "Created Vulkan device: %s", m_RendererString.c_str());

    return true;
}

bool DeviceManagerOverride_VK::CreateDevice()
{
    if (m_DeviceParams.enableDebugRuntime)
    {
        installDebugCallback();
    }

    // add device extensions requested by the user
    for (const std::string& name : m_DeviceParams.requiredVulkanDeviceExtensions)
    {
        enabledExtensions.device.insert(name);
    }
    for (const std::string& name : m_DeviceParams.optionalVulkanDeviceExtensions)
    {
        optionalExtensions.device.insert(name);
    }

    if (!m_DeviceParams.headlessDevice)
    {
        // Need to adjust the swap chain format before creating the device because it affects physical device selection
        if (m_DeviceParams.swapChainFormat == nvrhi::Format::SRGBA8_UNORM)
            m_DeviceParams.swapChainFormat = nvrhi::Format::SBGRA8_UNORM;
        else if (m_DeviceParams.swapChainFormat == nvrhi::Format::RGBA8_UNORM)
            m_DeviceParams.swapChainFormat = nvrhi::Format::BGRA8_UNORM;

        CHECK(createWindowSurface())
    }
    CHECK(pickPhysicalDevice())
        CHECK(findQueueFamilies(m_VulkanPhysicalDevice))
        CHECK(createDeviceCustom())

        auto vecInstanceExt = stringSetToVector(enabledExtensions.instance);
    auto vecLayers = stringSetToVector(enabledExtensions.layers);
    auto vecDeviceExt = stringSetToVector(enabledExtensions.device);

    nvrhi::vulkan::DeviceDesc deviceDesc;
    deviceDesc.errorCB = &DefaultMessageCallback::GetInstance();
    deviceDesc.instance = m_VulkanInstance;
    deviceDesc.physicalDevice = m_VulkanPhysicalDevice;
    deviceDesc.device = m_VulkanDevice;
    deviceDesc.graphicsQueue = m_GraphicsQueue;
    deviceDesc.graphicsQueueIndex = m_GraphicsQueueFamily;
    if (m_DeviceParams.enableComputeQueue)
    {
        deviceDesc.computeQueue = m_ComputeQueue;
        deviceDesc.computeQueueIndex = m_ComputeQueueFamily;
    }
    if (m_DeviceParams.enableCopyQueue)
    {
        deviceDesc.transferQueue = m_TransferQueue;
        deviceDesc.transferQueueIndex = m_TransferQueueFamily;
    }
    deviceDesc.instanceExtensions = vecInstanceExt.data();
    deviceDesc.numInstanceExtensions = vecInstanceExt.size();
    deviceDesc.deviceExtensions = vecDeviceExt.data();
    deviceDesc.numDeviceExtensions = vecDeviceExt.size();
    deviceDesc.bufferDeviceAddressSupported = m_BufferDeviceAddressSupported;
#if DONUT_WITH_AFTERMATH
    deviceDesc.aftermathEnabled = m_DeviceParams.enableAftermath;
#endif
    deviceDesc.vulkanLibraryName = m_DeviceParams.vulkanLibraryName;
    deviceDesc.logBufferLifetime = m_DeviceParams.logBufferLifetime;

    m_NvrhiDevice = nvrhi::vulkan::createDevice(deviceDesc);

    if (m_DeviceParams.enableNvrhiValidationLayer)
    {
        m_ValidationLayer = nvrhi::validation::createValidationLayer(m_NvrhiDevice);
    }

#if DONUT_WITH_STREAMLINE
    StreamlineIntegration::VulkanInfo vulkanInfo;
    vulkanInfo.vkDevice = m_VulkanDevice;
    vulkanInfo.vkInstance = m_VulkanInstance;
    vulkanInfo.vkPhysicalDevice = m_VulkanPhysicalDevice;
    vulkanInfo.computeQueueIndex = kComputeQueueIndex;
    vulkanInfo.computeQueueFamily = m_ComputeQueueFamily;
    vulkanInfo.graphicsQueueIndex = kGraphicsQueueIndex;
    vulkanInfo.graphicsQueueFamily = m_GraphicsQueueFamily;

    StreamlineIntegration::Get().InitializeDeviceVK(m_NvrhiDevice, vulkanInfo);
#endif

    return true;
}

donut::app::DeviceManager* CreateVK()
{
    return new DeviceManagerOverride_VK();
}

#endif // DONUT_WITH_VULKAN
