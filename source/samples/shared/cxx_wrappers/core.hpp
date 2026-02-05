// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <nvigi.h>
#include <string>
#include <memory>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <iostream>
#include <iomanip>

namespace nvigi {

// C++ wrapper for AdapterSpec
class Adapter {
public:
    explicit Adapter(const AdapterSpec* spec) : m_spec(spec) {}
    
    LUID getId() const { return m_spec->id; }
    VendorId getVendor() const { return m_spec->vendor; }
    const char* getVendorName() const { return vendorIdToString(m_spec->vendor); }
    size_t getDedicatedMemoryInMB() const { return m_spec->dedicatedMemoryInMB; }
    Version getDriverVersion() const { return m_spec->driverVersion; }
    uint32_t getArchitecture() const { return m_spec->architecture; }
    
private:
    const AdapterSpec* m_spec;
};

// C++ wrapper for PluginSpec
class Plugin {
public:
    explicit Plugin(const PluginSpec* spec) : m_spec(spec) {}
    
    PluginID getId() const { return m_spec->id; }
    const char* getName() const { return m_spec->pluginName; }
    Version getVersion() const { return m_spec->pluginVersion; }
    Version getAPI() const { return m_spec->pluginAPI; }
    Version getRequiredOSVersion() const { return m_spec->requiredOSVersion; }
    Version getRequiredAdapterDriverVersion() const { return m_spec->requiredAdapterDriverVersion; }
    VendorId getRequiredAdapterVendor() const { return m_spec->requiredAdapterVendor; }
    uint32_t getRequiredAdapterArchitecture() const { return m_spec->requiredAdapterArchitecture; }
    Result getStatus() const { return m_spec->status; }
    
    bool supportsInterface(const UID& interfaceUID) const {
        for (size_t i = 0; i < m_spec->numSupportedInterfaces; ++i) {
            if (m_spec->supportedInterfaces[i] == interfaceUID) {
                return true;
            }
        }
        return false;
    }
    
    std::vector<UID> getSupportedInterfaces() const {
        std::vector<UID> interfaces;
        interfaces.reserve(m_spec->numSupportedInterfaces);
        for (size_t i = 0; i < m_spec->numSupportedInterfaces; ++i) {
            interfaces.push_back(m_spec->supportedInterfaces[i]);
        }
        return interfaces;
    }
    
private:
    const PluginSpec* m_spec;
};

// C++ wrapper for PluginAndSystemInformation
class SystemInfo {
public:
    explicit SystemInfo(const PluginAndSystemInformation* info) : m_info(info) {}
    
    size_t getNumPlugins() const { return m_info ? m_info->numDetectedPlugins : 0; }
    size_t getNumAdapters() const { return m_info ? m_info->numDetectedAdapters : 0; }
    
    Plugin getPlugin(size_t index) const {
        if (!m_info || index >= m_info->numDetectedPlugins) {
            throw std::out_of_range("Plugin index out of range");
        }
        return Plugin(m_info->detectedPlugins[index]);
    }

    Plugin getPlugin(const PluginID& idx) const {
        for (Plugin& p: getPlugins())
            if (p.getId() == idx)
				return p;
        throw std::out_of_range("Plugin not found");
    }
    
    Adapter getAdapter(size_t index) const {
        if (!m_info || index >= m_info->numDetectedAdapters) {
            throw std::out_of_range("Adapter index out of range");
        }
        return Adapter(m_info->detectedAdapters[index]);
    }
    
    std::vector<Plugin> getPlugins() const {
        std::vector<Plugin> plugins;
        if (m_info) {
            plugins.reserve(m_info->numDetectedPlugins);
            for (size_t i = 0; i < m_info->numDetectedPlugins; ++i) {
                plugins.emplace_back(Plugin(m_info->detectedPlugins[i]));
            }
        }
        return plugins;
    }
    
    std::vector<Adapter> getAdapters() const {
        std::vector<Adapter> adapters;
        if (m_info) {
            adapters.reserve(m_info->numDetectedAdapters);
            for (size_t i = 0; i < m_info->numDetectedAdapters; ++i) {
                adapters.emplace_back(Adapter(m_info->detectedAdapters[i]));
            }
        }
        return adapters;
    }
    
    Version getOSVersion() const { return m_info ? m_info->osVersion : Version{}; }
    SystemFlags getFlags() const { return m_info ? m_info->flags : SystemFlags::eNone; }
    bool isHWSchedulingEnabled() const { return m_info ? nvigi::isHWSchedulingEnabled(m_info) : false; }
    
    Result getPluginStatus(const PluginID& id) const {
        return m_info ? nvigi::getPluginStatus(m_info, id) : kResultItemNotFound;
    }
    
    Result isPluginExportingInterface(const PluginID& id, const UID& interfaceUID) const {
        return m_info ? nvigi::isPluginExportingInterface(m_info, id, interfaceUID) : kResultItemNotFound;
    }
    
    bool isValid() const { return m_info != nullptr; }
    
    // Print system information to console
    void print(std::ostream& os = std::cout) const {
        if (!m_info) {
            os << "No system information available\n";
            return;
        }
        
        os << "\n";
        os << "================================================================================\n";
        os << "                         NVIGI SYSTEM INFORMATION                               \n";
        os << "================================================================================\n";
        os << "\n";
        
        // OS Information
        auto osVer = getOSVersion();
        os << "Operating System:\n";
        os << "  Version: " << osVer.major << "." << osVer.minor << "." << osVer.build << "\n";
        os << "\n";
        
        // System Flags
        os << "System Flags:\n";
        os << "  Hardware Scheduling: " << (isHWSchedulingEnabled() ? "Enabled" : "Disabled") << "\n";
        os << "\n";
        
        // Adapters
        os << "Graphics Adapters: " << getNumAdapters() << "\n";
        if (getNumAdapters() > 0) {
            os << "--------------------------------------------------------------------------------\n";
            for (size_t i = 0; i < getNumAdapters(); ++i) {
                auto adapter = getAdapter(i);
                os << "  [" << i << "] " << adapter.getVendorName() << "\n";
                os << "      Vendor ID: 0x" << std::hex << std::setfill('0') << std::setw(4) 
                   << static_cast<uint32_t>(adapter.getVendor()) << std::dec << "\n";
                os << "      VRAM: " << adapter.getDedicatedMemoryInMB() << " MB\n";
                
                if (adapter.getVendor() == VendorId::eNVDA) {
                    auto driverVer = adapter.getDriverVersion();
                    os << "      Driver: " << driverVer.major << "." << driverVer.minor << "." << driverVer.build << "\n";
                    os << "      Architecture: " << adapter.getArchitecture() << "\n";
                }
                
                if (i < getNumAdapters() - 1) {
                    os << "\n";
                }
            }
            os << "--------------------------------------------------------------------------------\n";
        }
        os << "\n";
        
        // Plugins
        os << "Detected Plugins: " << getNumPlugins() << "\n";
        if (getNumPlugins() > 0) {
            os << "--------------------------------------------------------------------------------\n";
            for (size_t i = 0; i < getNumPlugins(); ++i) {
                auto plugin = getPlugin(i);
                auto ver = plugin.getVersion();
                auto status = plugin.getStatus();
                
                os << "  [" << i << "] " << plugin.getName() << "\n";
                os << "      Version: " << ver.major << "." << ver.minor << "." << ver.build << "\n";
                
                // Status with ASCII indicators
                os << "      Status: ";
                if (status == kResultOk) {
                    os << "[OK] Available";
                } else if (status == kResultNoImplementation) {
                    os << "[ ] No Implementation";
                } else if (status == kResultNoSupportedHardwareFound) {
                    os << "[X] Not Supported";
                } else if (status == kResultDriverOutOfDate) {
                    os << "[!] Driver Version Too Old";
                } else if (status == kResultOSOutOfDate) {
                    os << "[!] OS Version Too Old";
                } else if (status == kResultInsufficientResources) {
                    os << "[X] Insufficient Resources";
                } else if (status == kResultPluginOutOfDate) {
                    os << "[!] Plugin Out Of Date";
                } else if (status == kResultMissingDynamicLibraryDependency) {
                    os << "[X] Missing Dependency";
                } else {
                    os << "[X] Error (code: " << static_cast<int>(status) << ")";
                }
                os << "\n";
                
                // Required OS version
                auto reqOS = plugin.getRequiredOSVersion();
                if (reqOS.major > 0 || reqOS.minor > 0) {
                    os << "      Required OS: " << reqOS.major << "." << reqOS.minor << "." << reqOS.build << "\n";
                }
                
                // Required adapter info
                auto reqVendor = plugin.getRequiredAdapterVendor();
                if (reqVendor != VendorId::eNone && reqVendor != VendorId::eAny) {
                    os << "      Required GPU: " << vendorIdToString(reqVendor);
                    auto reqDriver = plugin.getRequiredAdapterDriverVersion();
                    if (reqDriver.major > 0 || reqDriver.minor > 0) {
                        os << " (Driver " << reqDriver.major << "." << reqDriver.minor << "." << reqDriver.build << "+)";
                    }
                    os << "\n";
                }
                
                // Supported interfaces
                auto interfaces = plugin.getSupportedInterfaces();
                if (!interfaces.empty()) {
                    os << "      Supported Interfaces: " << interfaces.size() << "\n";
                }
                
                if (i < getNumPlugins() - 1) {
                    os << "\n";
                }
            }
            os << "--------------------------------------------------------------------------------\n";
        }
        os << "\n";
    }
    
private:
    const PluginAndSystemInformation* m_info;
};

class Core {
public:
    struct Config {
        std::string_view sdkPath;
        LogLevel logLevel = LogLevel::eOff;
        bool showConsole = true;
        
        // Additional paths to plugins (optional, can be empty to only use sdkPath)
        std::vector<std::string_view> additionalPluginPaths;
        
        // Path to logs and data (nullptr/empty to disable file logging)
        std::string_view pathToLogsAndData;
        
        // Log message callback (optional)
        PFun_LogMessageCallback* logMessageCallback = nullptr;
        
        // Preference flags (optional)
        PreferenceFlags flags = {};
        
        // Path to shared dependencies (optional, if not provided assumes dependencies are next to plugins)
        std::string_view pathToDependencies;
    };

    explicit Core(const Config& config) {
        // If sdkPath is empty, assume exe location
        auto sdkPath = config.sdkPath;
        if (sdkPath.empty()) {
            char buffer[MAX_PATH];
            GetModuleFileNameA(NULL, buffer, MAX_PATH);
            std::string::size_type pos = std::string(buffer).find_last_of("\\/");
            if (pos != std::string::npos) {
                sdkPath = std::string(buffer).substr(0, pos);
            }
            else {
                throw std::runtime_error("Could not determine executable path");
            }
        }
        
        // Load core library
        auto pathToSDK = std::string(sdkPath) + "/nvigi.core.framework.dll";
        m_library = std::unique_ptr<void, LibraryDeleter>(
            LoadLibraryA(pathToSDK.c_str()),
            LibraryDeleter{}
        );
        
        if (!m_library) {
            throw std::runtime_error("Could not load NVIGI core library");
        }

        // Load core functions
        m_init = GetFunction<PFun_nvigiInit>("nvigiInit");
        m_shutdown = GetFunction<PFun_nvigiShutdown>("nvigiShutdown");
        m_loadInterface = GetFunction<PFun_nvigiLoadInterface>("nvigiLoadInterface");
        m_unloadInterface = GetFunction<PFun_nvigiUnloadInterface>("nvigiUnloadInterface");

        // Initialize NVIGI - build plugin paths array
        std::vector<const char*> pluginPaths;
        pluginPaths.push_back(sdkPath.data());
        for (const auto& path : config.additionalPluginPaths) {
            if (!path.empty()) {
                pluginPaths.push_back(path.data());
            }
        }

        Preferences pref{};
        pref.logLevel = config.logLevel;
        pref.showConsole = config.showConsole;
        pref.utf8PathsToPlugins = pluginPaths.data();
        pref.numPathsToPlugins = static_cast<uint32_t>(pluginPaths.size());
        pref.utf8PathToLogsAndData = config.pathToLogsAndData.empty() ? nullptr : config.pathToLogsAndData.data();
        pref.logMessageCallback = config.logMessageCallback;
        pref.flags = config.flags;
        pref.utf8PathToDependencies = config.pathToDependencies.empty() ? nullptr : config.pathToDependencies.data();

        PluginAndSystemInformation* pluginInfo = nullptr;
        if (NVIGI_FAILED(result, m_init(pref, &pluginInfo, kSDKVersion))) {
            throw std::runtime_error("Failed to initialize NVIGI");
        }
        
        m_systemInfo = SystemInfo(pluginInfo);
    }

    ~Core() {
        if (m_shutdown) {
            m_shutdown();
        }
    }

    // Non-copyable
    Core(const Core&) = delete;
    Core& operator=(const Core&) = delete;

    // Movable
    Core(Core&&) noexcept = default;
    Core& operator=(Core&&) noexcept = default;

    // Access to core functions
    auto loadInterface() const { return m_loadInterface; }
    auto unloadInterface() const { return m_unloadInterface; }
    
    // Access to system and plugin information
    const SystemInfo& getSystemInfo() const { return m_systemInfo; }

private:
    template<typename T>
    T* GetFunction(const char* name) {
        auto fn = reinterpret_cast<T*>(GetProcAddress(static_cast<HMODULE>(m_library.get()), name));
        if (!fn) {
            throw std::runtime_error(std::string("Failed to load function: ") + name);
        }
        return fn;
    }

    struct LibraryDeleter {
        void operator()(void* handle) const {
            if (handle) {
                FreeLibrary(static_cast<HMODULE>(handle));
            }
        }
    };

    std::unique_ptr<void, LibraryDeleter> m_library;
    PFun_nvigiInit* m_init{nullptr};
    PFun_nvigiShutdown* m_shutdown{nullptr};
    PFun_nvigiLoadInterface* m_loadInterface{nullptr};
    PFun_nvigiUnloadInterface* m_unloadInterface{nullptr};
    SystemInfo m_systemInfo{nullptr};
};

} // namespace nvigi
