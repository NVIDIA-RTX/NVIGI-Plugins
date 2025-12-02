group "plugins/tts"

project "nvigi.plugin.tts.asqflow-ggml.vk"
	pluginBasicSetup("tts/asqflow/ggml")
	
	defines {
		"GGML_USE_VULKAN=1",
		"GGML_SCHED_MAX_COPIES=4",
		"K_QUANTS_PER_ITERATION=2",
    }

	vectorextensions "AVX2"
	
	files {
		"../*.h",
		"./*.h",
		"./*.cpp",
		externaldir .."asqflow.cpp/include/**.h"
	}
	
	includedirs {
		externaldir .."asqflow.cpp/include", 
		externaldir .."asqflow.cpp/ggml/include", 
		externaldir .."vulkanSDK/include",
		coresdkdir .. "source/utils/nvigi.ai", 
		ROOT .. "source/plugins/nvigi.tts/asqflow/ggml/"}

	-- some 3rd party libs (ggml) do not compile without it in production mode when security flags are set
	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"} 
	filter{}

	filter {"system:windows"}
		vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
		vpaths { ["ggml"] = {
			externaldir .."asqflow.cpp/include/**.h"
		}}

		libdirs {
			externaldir .."vulkanSDK/Lib",
			externaldir .."asqflow.cpp/vulkan/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
			externaldir .."SimpleFarForTTS/lib/Release"
		}
		
		links {"ggml-vulkan","ggml-base","ggml-cpu","asqflow","ggml", "vulkan-1.lib", "winmm"}
		includedirs {externaldir .."SimpleFarForTTS/"}
		links {"RivaNormalizer.lib"}

	
	filter {"system:linux"}
		libdirs {externaldir .."asqflow.cpp/vulkan/lib"}
		libdirs {externaldir .."SimpleFarForTTS/lib/Release"}
		libdirs {externaldir .."vulkanSDK/lib/"}
		includedirs {externaldir .."SimpleFarForTTS/"}
		linkoptions {"-fopenmp"}
		links {"asqflow", "ggml", "ggml-cpu","ggml-base","ggml-vulkan", "vulkan", "dl", "pthread", "rt"}
		-- RivaNormalizer must come after asqflow because asqflow depends on it
		links {"RivaNormalizer"}

	filter {}

project "nvigi.plugin.tts.asqflow-ggml.cuda"
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

	vectorextensions "AVX2"
	
	files {
		"../*.h",
		"./*.h",
		"./*.cpp",
		externaldir .."asqflow.cpp/include/**.h"
	}
	
	includedirs {
		externaldir .."asqflow.cpp/include", 
		externaldir .."asqflow.cpp/ggml/include", 
		externaldir .."cuda/include", 
		coresdkdir .. "source/utils/nvigi.ai", 
		ROOT .. "source/plugins/nvigi.tts/asqflow/ggml"}

	-- some 3rd party libs (ggml) do not compile without it in production mode when security flags are set
	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"} 
	filter{}

	filter {"system:windows"}
		vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
		vpaths { ["ggml"] = {
			externaldir .."asqflow.cpp/include/**.h"
		}}
		
		libdirs {
			externaldir .."cuda//lib/x64",
			externaldir .."asqflow.cpp/cuda/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
			externaldir .."SimpleFarForTTS/lib/Release"
		}

		links {"cuda.lib", "cublas.lib"}
		links {"ggml-cuda","ggml-base","ggml-cpu","asqflow","ggml", "winmm"}
		includedirs {externaldir .."SimpleFarForTTS/"}
		links {"RivaNormalizer.lib"}

	
	filter {"system:linux"}
		libdirs {
			externaldir .."cuda/lib64",
			externaldir .."cuda/lib64/stubs",
		}
		libdirs {externaldir .."asqflow.cpp/cuda/lib"}
		libdirs {externaldir .."SimpleFarForTTS/lib/Release"}
		includedirs {externaldir .."SimpleFarForTTS/"}
		linkoptions {"-fopenmp", "-lcublas"}
		links {"asqflow", "ggml", "ggml-cpu","ggml-base","ggml-cuda", "cuda", "cublas", "dl", "pthread", "rt"}
		-- RivaNormalizer must come after asqflow because asqflow depends on it
		links {"RivaNormalizer"}

	filter {}

	add_cuda_dependencies()

if os.istarget("windows") then
end


group ""