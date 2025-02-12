group "plugins/asr"
project "nvigi.plugin.asr.ggml.cuda"
	kind "SharedLib"
	targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	
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
	
	includedirs {externaldir .."whisper.cpp/include", externaldir .."whisper.cpp/ggml/include", externaldir .."cuda/include", coresdkdir .. "source/utils/nvigi.ai", ROOT .. "source/plugins/nvigi.asr/ggml/external"}

	-- some 3rd party libs (ggml) do not compile without it in production mode when security flags are set
	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"} 
	filter{}

	filter {"system:windows"}
		vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
		vpaths { ["ggml"] = {
			externaldir .."whisper.cpp/include/**.h"
		}}
		

		libdirs {externaldir .."cuda/lib/x64"}
		links {"whisper.lib","ggml","common", "cuda.lib", "cublas.lib", "winmm.lib"}

		filter {"configurations:Debug"}
			libdirs {externaldir .."whisper.cpp/cuda/lib/Debug"}
		filter {"configurations:not Debug"}
			libdirs {externaldir .."whisper.cpp/cuda/lib/NotDebug"}
	
	filter {"system:linux"}
		libdirs {externaldir .."cuda/lib64", externaldir .."cuda/lib64/stubs", externaldir .."whisper.cpp/cuda/lib64"}
		links {"whisper","ggml","common", "cuda", "cublas", "dl", "pthread", "rt"}
	filter {}

	add_cuda_dependencies()

group ""

if os.istarget("windows") then

group "plugins/asr"

project "nvigi.plugin.asr.ggml.cpu"
	kind "SharedLib"
	targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	
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
		links {"winmm.lib","whisper","ggml","common"}
		filter {"configurations:Debug"}
			libdirs {externaldir .."whisper.cpp/cpu/lib/Debug"}
		filter {"configurations:not Debug"}
			libdirs {externaldir .."whisper.cpp/cpu/lib/NotDebug"}	
	filter {"system:linux"}
		libdirs {externaldir .."whisper.cpp/cpu/lib64"}
		links {"whisper","ggml","common", "dl", "pthread", "rt"}
	filter {}

	
group ""
end