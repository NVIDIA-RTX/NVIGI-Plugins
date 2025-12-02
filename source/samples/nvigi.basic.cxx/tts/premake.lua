if os.istarget("windows") then

group "samples"

project "nvigi.basic.tts.cxx"
    kind "ConsoleApp"
    targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
    objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
    
    dependson {"gitVersion"}

    files {
        "./**.h",
        "./**.hpp",
        "../**.hpp",
        "./**.cpp",
        "./**.rc",
        coresdkdir .. "source/utils/**.h",
    }

    includedirs {
        coresdkdir .. "source/core/nvigi.api", 
        coresdkdir .. "source/utils/nvigi.ai",
        coresdkdir .. "source/utils",
        ROOT .. "source/plugins/nvigi.tts",
        ROOT .. "external/vulkanSDK/include",
        ".."
    }
        

    filter {"system:windows"}
        vpaths { ["impl"] = {"./**.h","./**.hpp", "./**.cpp", }}
        vpaths { ["shared"] = {"../**.hpp"}}
        vpaths { ["utils"] = {ROOT .. "source/utils/**.h",ROOT .. "source/utils/**.cpp", }}
        libdirs { externaldir .. "vulkanSDK/Lib" }
        links { "d3d12.lib", "dxgi.lib", "vulkan-1.lib", "dsound.lib" }
    filter {}    

group ""

end

