group "plugins/tts"

project "nvigi.plugin.tts.asqflow-ggml.vk"
	if premakeX64Targets() then
		filter { "platforms:x64" }
			kind "SharedLib"
		filter {}
	end

	pluginBasicSetup("tts/asqflow/ggml")

	defines {
		"GGML_USE_VULKAN=1",
		"GGML_SCHED_MAX_COPIES=4",
		"K_QUANTS_PER_ITERATION=2",
    }

	files {
		"../*.h",
		"./*.h",
		"./*.cpp",
	}

	filter {"system:windows", "platforms:x64"}
		vectorextensions "AVX2"
		files {
			externaldir .."asqflow.cpp/include/**.h"
		}
		includedirs {
			externaldir .."asqflow.cpp/include",
			externaldir .."asqflow.cpp/ggml/include",
			externaldir .."vulkanSDK/include",
			coresdkdir .. "source/utils/nvigi.ai",
			ROOT .. "source/plugins/nvigi.tts/asqflow/ggml/",
			externaldir .."SimpleFarForTTS/",
		}
		vpaths { ["ggml"] = {
			externaldir .."asqflow.cpp/include/**.h"
		}}
		libdirs {
			externaldir .."vulkanSDK/Lib",
			externaldir .."asqflow.cpp/vulkan/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
			externaldir .."SimpleFarForTTS/lib/Release"
		}
	filter {}


	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"} 
	filter{}

	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}

	links {"ggml-vulkan","ggml-base","ggml-cpu","asqflow","ggml", "vulkan-1.lib", "winmm"}
	links {"RivaNormalizer.lib"}

project "nvigi.plugin.tts.asqflow-ggml.cuda"
	if premakeX64Targets() then
		filter { "platforms:x64" }
			kind "SharedLib"
		filter {}
	end

	pluginBasicSetup("tts/asqflow/ggml")

	defines {
		"GGML_USE_CUDA=1",
		"GGML_USE_CUBLAS=1",
		"GGML_CUDA_F16=1",
		"GGML_CUDA_DMMV_X=32",
		"GGML_CUDA_DMMV_Y=1",
		"K_QUANTS_PER_ITERATION=2",
		"GGML_USE_K_QUANTS",
		"GGML_CUDA_NO_VMM"
    }

	files {
		"../*.h",
		"./*.h",
		"./*.cpp",
	}

	filter {"system:windows", "platforms:x64"}
		vectorextensions "AVX2"
		files {
			externaldir .."asqflow.cpp/include/**.h"
		}
		includedirs {
			externaldir .."asqflow.cpp/include",
			externaldir .."asqflow.cpp/ggml/include",
			externaldir .."cuda/include",
			coresdkdir .. "source/utils/nvigi.ai",
			coresdkdir .. "source/utils/nvigi.ai",
			ROOT .. "source/plugins/nvigi.tts/asqflow/ggml",
			externaldir .."SimpleFarForTTS/",
		}
		vpaths { ["ggml"] = {
			externaldir .."asqflow.cpp/include/**.h"
		}}
		libdirs {
			externaldir .."cuda//lib/x64",
			externaldir .."asqflow.cpp/cuda/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
			externaldir .."SimpleFarForTTS/lib/Release"
		}
	filter {}


	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"} 
	filter{}

	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}

	links {"cuda.lib", "cublas.lib"}
	links {"ggml-cuda","ggml-base","ggml-cpu","asqflow","ggml", "winmm"}
	links {"RivaNormalizer.lib"}

	add_cuda_dependencies()

if os.istarget("windows") then
end


group ""
