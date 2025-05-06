group "plugins/tts"

project "nvigi.plugin.tts.asqflow.trt"
	kind "SharedLib"	
	-- kind "ConsoleApp"
	targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")		
	pluginBasicSetup("tts/asqflow")

	files { 
		"./**.h",
		"./**.cpp",			
	}
	defines {
		"USE_TRT=1"
	}

	vpaths { ["impl"] = {"./**.h", "./**.cpp" }}


	
	includedirs {externaldir .."cuda/include", externaldir .."cuda/extras/CUPTI/include"}
	includedirs {coresdkdir .. "source/utils/nvigi.ai"}
	includedirs {ROOT .. "source/plugins/nvigi.tts/asqflow"}


	filter {"system:windows"}

		includedirs {externaldir .."microsoft.ml.onnxruntime.directml.1.20.1/build/native/include", externaldir .."SimpleFarForTTS/", externaldir .."tensorrt/include/"}
		libdirs {externaldir .."microsoft.ml.onnxruntime.directml.1.20.1/runtimes/win-x64/native", externaldir .."SimpleFarForTTS/lib/Release", externaldir .."tensorrt/lib"}
		links {"onnxruntime.lib", "RivaNormalizer.lib"}

		libdirs {externaldir .."cuda//lib/x64"}
		links {"cuda.lib", "cublas.lib"}
		libdirs {externaldir .."cuda/lib/x64", externaldir .."cuda/extras/CUPTI/lib64"}
		vpaths { ["impl"] = {"./**.h","./**.cpp", }}
		vpaths { ["utils"] = {ROOT .. "source/utils/**.h",ROOT .. "source/utils/**.cpp", }}
		links { "WS2_32.lib", "d3d12.lib", "dxgi.lib", "dxguid.lib", "cuda.lib", "cupti.lib", "dsound.lib", "winmm.lib", "nvinfer_10.lib"}
	
	filter {"system:linux"}
		libdirs {
			externaldir .."cuda/lib64",
			externaldir .."cuda/lib64/stubs",
		}
	filter {}

	add_cuda_dependencies()

group ""