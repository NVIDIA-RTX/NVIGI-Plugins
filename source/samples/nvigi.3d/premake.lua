if os.istarget("windows") then

group "samples"

-- donut_dir = ROOT.."source/samples/nvigi.3d/opt-local-donut/_package/donut"
donut_dir = externaldir.."donut"
local corePath = _SCRIPT_DIR

project "nvigi.3d"
	kind "WindowedApp"
	targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	
	dependson {"gitVersion"}

	--floatingpoint "Fast"

	files {
		"./src/**.h",
		"./src/**.cpp",
		"./src/**.rc",
		coresdkdir .. "source/utils/**.h",
	}

	defines {
		"USE_DX12",
		"USE_VULKAN",
		"NOMINMAX",
		"DONUT_WITH_DX11",
		"DONUT_WITH_DX12",
		"DONUT_WITH_VULKAN",
		"VK_USE_PLATFORM_WIN32_KHR",
		"DONUT_WITH_AFTERMATH=0",
		"DONUT_WITH_STREAMLINE=0",
		"MINMAX",
		"_CRT_SECURE_NO_WARNINGS",
		"DONUT_WITH_LZ4",
		"DONUT_WITH_MINIZ",
		"MINIZ_STATIC_DEFINE",
		"DONUT_WITH_TASKFLOW",
		"DONUT_WITH_TINYEXR",
		"DONUT_WITH_STATIC_SHADERS=0"
	}

	filter {"configurations:Debug"}
		defines {
			"_ITERATOR_DEBUG_LEVEL=1"
		}
	filter {}

	includedirs {coresdkdir .. "source/core/nvigi.api", coresdkdir .. "source/utils/nvigi.ai"}
	-- allows the sample to compile using include directives that match a packaged SDK
	includedirs {
		ROOT .. "source/plugins/nvigi.asr",
		ROOT .. "source/plugins/nvigi.gpt",
		ROOT .. "source/plugins/nvigi.tts",
		ROOT .. "nvigi_core/source/plugins/nvigi.hwi/cuda",
		ROOT .. "nvigi_core/source/plugins/nvigi.hwi/common"
	}

	includedirs {
		donut_dir.."/include",
		donut_dir.."/nvrhi/include",
		donut_dir.."/thirdparty/glfw/include",
		donut_dir.."/thirdparty/imgui",
		donut_dir.."/thirdparty/taskflow",
		donut_dir.."/nvrhi/thirdparty/DirectX-Headers/include",
		donut_dir.."/nvrhi/thirdparty/Vulkan-Headers/include"
	}

	libdirs {
		donut_dir.."/%{cfg.buildcfg}",
		donut_dir.."/nvrhi/%{cfg.buildcfg}",
		donut_dir.."/thirdparty/%{cfg.buildcfg}",
		donut_dir.."/thirdparty/glfw/src/%{cfg.buildcfg}",
		donut_dir.."/thirdparty/miniz/%{cfg.buildcfg}",
		donut_dir.."/thirdparty/jsoncpp/src/lib_json/%{cfg.buildcfg}",
		donut_dir.."/ShaderMake/%{cfg.buildcfg}",
	}

	links { 
		"donut_core.lib",
		"donut_render.lib",
		"donut_app.lib",
		"donut_engine.lib",
		"nvrhi.lib", 
		"nvrhi_d3d11.lib", 
		"nvrhi_d3d12.lib", 
		"nvrhi_vk.lib",
		"ShaderMakeBlob",
		"glfw3",
		"jsoncpp",
		"miniz",
		"imgui",
		"lz4",
		"dxgi",
		"dxguid",
		"d3d11",
		"d3d12"
	}
	filter {"system:windows"}
		vpaths { ["impl"] = {"./src/**.h","./src/**.cpp", }}
		vpaths { ["utils"] = {ROOT .. "source/utils/**.h",ROOT .. "source/utils/**.cpp", }}
		links { "dsound.lib", "winmm.lib" }
	filter {"system:linux"}
	filter {}

	postbuildcommands {
	  '{COPYDIR} %{donut_dir}/shaders %{cfg.buildtarget.directory}/shaders',
	  '{COPYDIR} %{ROOT}/data/nvigi.test/nvigi.tts/asqflow/mel_spectrograms_targets %{cfg.buildtarget.directory}/tts'
	}

group ""

end
