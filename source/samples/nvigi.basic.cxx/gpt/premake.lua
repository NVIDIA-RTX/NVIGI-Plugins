if os.istarget("windows") then

group "samples"

project "nvigi.basic.gpt.cxx"
    kind "ConsoleApp"
    targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
    objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
    
    dependson {"gitVersion"}

    files {
        "./**.h",
        "./**.hpp",
        "./**.cpp",
        "./**.rc",
        coresdkdir .. "source/utils/**.h",
        -- Shared wrapper headers
        ROOT .. "source/samples/shared/cxx_wrappers/**.hpp",
    }

    includedirs {
        coresdkdir .. "source/core/nvigi.api", 
        coresdkdir .. "source/utils/nvigi.ai",
        ROOT .. "source/plugins/nvigi.gpt",
        ROOT .. "external/vulkanSDK/include",
        -- Shared wrappers directory
        ROOT .. "source/samples/shared"
    }
        

    filter {"system:windows"}
        vpaths { ["impl"] = {"./**.h","./**.hpp", "./**.cpp", }}
        vpaths { ["shared"] = {ROOT .. "source/samples/shared/cxx_wrappers/**.hpp"}}
        vpaths { ["utils"] = {ROOT .. "source/utils/**.h",ROOT .. "source/utils/**.cpp", }}
        libdirs { externaldir .. "vulkanSDK/Lib" }
        links { "d3d12.lib", "dxgi.lib", "vulkan-1.lib" }
    filter {}    

group ""

end