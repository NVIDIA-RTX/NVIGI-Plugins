group "plugins/asr"
project "nvigi.plugin.asr.ggml.cuda"
	if premakeX64Targets() then
		filter { "platforms:x64" }
			kind "SharedLib"
		filter {}
	end
		
	pluginBasicSetup("asr/ggml")

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
			externaldir .."whisper.cpp/include/**.h"
		}
		includedirs {
			externaldir .."whisper.cpp/include",
			externaldir .."whisper.cpp/ggml/include",
			externaldir .."cuda/include",
		}

		vpaths { ["ggml"] = {
			externaldir .."whisper.cpp/include/**.h"
		}}
		libdirs {
			externaldir .."cuda//lib/x64",
			externaldir .."whisper.cpp/cuda/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
		}
	filter {}


	includedirs {
		coresdkdir .. "source/utils/nvigi.ai", 
		ROOT .. "source/plugins/nvigi.asr/ggml/external"
	}

	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"}
	filter{}

	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
		
	links {"ggml-cuda","ggml-base","ggml-cpu","whisper","ggml", "cuda", "cublas", "winmm"}
	add_cuda_dependencies()

project "nvigi.plugin.asr.ggml.vk"
	if premakeX64Targets() then
		filter { "platforms:x64" }
			kind "SharedLib"
		filter {}
	end
		
	pluginBasicSetup("asr/ggml")

	defines {
		"GGML_USE_VULKAN=1",
		"K_QUANTS_PER_ITERATION=2",
		"GGML_USE_K_QUANTS",
    }

	files {
		"../*.h",
		"./*.h",
		"./*.cpp",
	}

	filter {"system:windows", "platforms:x64"}
		vectorextensions "AVX2"
		files {
			externaldir .."whisper.cpp/include/**.h"
		}
		includedirs {
			externaldir .."whisper.cpp/include",
			externaldir .."whisper.cpp/ggml/include",
			coresdkdir .. "source/utils/nvigi.ai",
			ROOT .. "source/plugins/nvigi.asr/ggml/external",
			externaldir .. "vulkanSDK/include"
		}
		vpaths { ["ggml"] = {
			externaldir .."whisper.cpp/include/**.h"
		}}
		libdirs {
			externaldir .."vulkanSDK/Lib",
			externaldir .."whisper.cpp/vulkan/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
		}
	filter {}


	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"}
	filter{}

	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
		
	links {"whisper","ggml", "ggml-base","ggml-cpu","ggml-vulkan", "winmm", "vulkan-1"}
group ""

if os.istarget("windows") then

group "plugins/asr"

project "nvigi.plugin.asr.ggml.cpu"
	if premakeX64Targets() then
		filter { "platforms:x64" }
			kind "SharedLib"
		filter {}
	end
		
	pluginBasicSetup("asr/ggml")

	defines {
		"K_QUANTS_PER_ITERATION=2",
		"GGML_USE_K_QUANTS"
    }

	files {
		"../*.h",
		"./*.h",
		"./*.cpp",
	}

	filter {"system:windows", "platforms:x64"}
		vectorextensions "AVX2"
		files {
			externaldir .."whisper.cpp/include/**.h"
		}
		includedirs {
			externaldir .."whisper.cpp/include",
			externaldir .."whisper.cpp/ggml/include"
		}

		vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
		vpaths { ["ggml"] = {
			externaldir .."whisper.cpp/include/**.h"
		}}
		libdirs {
			externaldir .."whisper.cpp/cpu/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
		}
	filter {}


	filter {"system:windows","configurations:Production"}
		defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"}
	filter{}

	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}

	links {"whisper","ggml", "ggml-base","ggml-cpu","winmm"}	


group ""
end
