group "plugins/asr"
project "nvigi.plugin.asr.ggml.cuda"
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

	vectorextensions "AVX2"
	
	files {
		"../*.h",
		"./*.h",
		"./*.cpp",
		externaldir .."whisper.cpp/include/**.h"
	}
	
	includedirs {
		externaldir .."whisper.cpp/include", 
		externaldir .."whisper.cpp/ggml/include", 
		externaldir .."cuda/include", 
		coresdkdir .. "source/utils/nvigi.ai", 
		ROOT .. "source/plugins/nvigi.asr/ggml/external"}

	-- some 3rd party libs (ggml) do not compile without it in production mode when security flags are set
	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"} 
	filter{}

	filter {"system:windows"}
		vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
		vpaths { ["ggml"] = {
			externaldir .."whisper.cpp/include/**.h"
		}}
		

		libdirs {
			externaldir .."cuda//lib/x64",
			externaldir .."whisper.cpp/cuda/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
		}
		links {"ggml-cuda","ggml-base","ggml-cpu","whisper","ggml", "cuda", "cublas", "winmm"}
	
	filter {"system:linux"}
		libdirs {externaldir .."cuda/lib64", externaldir .."cuda/lib64/stubs", externaldir .."whisper.cpp/cuda/lib64"}
		linkoptions {"-fopenmp", "-lcublas"}
		links {"whisper", "common", "ggml", "ggml-cpu","ggml-base","ggml-cuda", "cuda", "cublas", "dl", "pthread", "rt"}
	filter {}

	add_cuda_dependencies()

project "nvigi.plugin.asr.ggml.vk"
	pluginBasicSetup("asr/ggml")
		
	defines {
		"GGML_USE_VULKAN=1",
		"K_QUANTS_PER_ITERATION=2",
		"GGML_USE_K_QUANTS",		
    }

	vectorextensions "AVX2"
	
	files {
		"../*.h",
		"./*.h",
		"./*.cpp",
		externaldir .."whisper.cpp/include/**.h"
	}
	
	includedirs {
		externaldir .."whisper.cpp/include", 
		externaldir .."whisper.cpp/ggml/include", 
		coresdkdir .. "source/utils/nvigi.ai", 
		ROOT .. "source/plugins/nvigi.asr/ggml/external",
		externaldir .. "vulkanSDK/include"
	}

	-- some 3rd party libs (ggml) do not compile without it in production mode when security flags are set
	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"} 
	filter{}

	filter {"system:windows"}
		vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
		vpaths { ["ggml"] = {
			externaldir .."whisper.cpp/include/**.h"
		}}
		
		libdirs {
			externaldir .."vulkanSDK/Lib",
			externaldir .."whisper.cpp/vk/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
		}

		links {"whisper","ggml", "ggml-base","ggml-cpu","ggml-vulkan", "winmm", "vulkan-1"}
	
	filter {"system:linux"}
		libdirs {externaldir .."whisper.cpp/vk/lib64"}
		libdirs {externaldir .."vulkanSDK/lib/"}
		linkoptions {"-fopenmp"}
		links {"whisper", "common", "ggml", "ggml-cpu","ggml-base","ggml-vulkan","vulkan","dl", "pthread", "rt"}
	filter {}

group ""

if os.istarget("windows") then

group "plugins/asr"

project "nvigi.plugin.asr.ggml.cpu"
	pluginBasicSetup("asr/ggml")
	
	defines {
		"K_QUANTS_PER_ITERATION=2",
		"GGML_USE_K_QUANTS"
    }

	vectorextensions "AVX2"
	
	-- some 3rd party libs (ggml) do not compile without it in production mode when security flags are set
	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"} 
	filter{}
	
	files {
		"../*.h",
		"./*.h",
		"./*.cpp",
		externaldir .."whisper.cpp/include/**.h"
	}

	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
	vpaths { ["ggml"] = {
		externaldir .."whisper.cpp/include/**.h"		
	}}

	includedirs {externaldir .."whisper.cpp/include", externaldir .."whisper.cpp/ggml/include" }
	
	filter {"system:windows"}
		libdirs {
			externaldir .."whisper.cpp/cpu/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
		}
		links {"whisper","ggml", "ggml-base","ggml-cpu","winmm"}	
	filter {"system:linux"}
		libdirs {externaldir .."whisper.cpp/cpu/lib64"}
		links {"ggml-base","ggml-cpu", "whisper","ggml", "dl", "pthread", "rt"}
	filter {}


group ""
end