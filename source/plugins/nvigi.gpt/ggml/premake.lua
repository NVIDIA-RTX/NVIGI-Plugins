group "plugins/gpt"

local external_dir_llamacpp = externaldir .. "llama.cpp"

project "nvigi.plugin.gpt.ggml.vk"
	kind "SharedLib"
	targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	
	pluginBasicSetup("gpt/ggml")
	
	defines {
		"GGML_USE_VULKAN=1",
		"GGML_SCHED_MAX_COPIES=4",
		"K_QUANTS_PER_ITERATION=2",
    }

	vectorextensions "AVX2"
	
	files {
		"../*.h",
		"*.h",
		"*.cpp"	
	}
	includedirs {
		coresdkdir .. "source/utils/nvigi.ai", ROOT .. "source/plugins/nvigi.gpt/ggml/",
		external_dir_llamacpp .. "/common", 
		external_dir_llamacpp .. "/ggml/include", 
		external_dir_llamacpp .. "/include",
		external_dir_llamacpp .. "/tools/mtmd/", -- Begin VLM Addition/End VLM Addition
		externaldir .. "vulkanSDK/include"
	}
	
	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
	
	-- some 3rd party libs (ggml) do not compile without it in production mode when security flags are set
	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"} 
	filter{}

	libdirs {externaldir .."vulkanSDK/Lib"}

	filter {"system:windows"}

		filter {"system:windows", "configurations:Debug"}
			links {
				(external_dir_llamacpp .. "/vulkan/lib/Debug/common.lib"),
				(external_dir_llamacpp .. "/vulkan/lib/Debug/ggml.lib"), 
				(external_dir_llamacpp .. "/vulkan/lib/Debug/llama.lib"),
				(external_dir_llamacpp .. "/vulkan/lib/Debug/llava_static.lib"), -- Begin VLM Addition/End VLM Addition
				(external_dir_llamacpp .. "/vulkan/lib/Debug/mtmd_static.lib"), -- Begin VLM Addition/End VLM Addition
				(external_dir_llamacpp .. "/vulkan/lib/Debug/ggml-cpu.lib"),
				(external_dir_llamacpp .. "/vulkan/lib/Debug/ggml-base.lib"),
				(external_dir_llamacpp .. "/vulkan/lib/Debug/ggml-vulkan.lib"),
				"vulkan-1.lib"
			}

		filter {"system:windows", "configurations:not Debug"}
			links {
				(external_dir_llamacpp .. "/vulkan/lib/NotDebug/common.lib"),
				(external_dir_llamacpp .. "/vulkan/lib/NotDebug/ggml.lib"), 
				(external_dir_llamacpp .. "/vulkan/lib/NotDebug/llama.lib"),
				(external_dir_llamacpp .. "/vulkan/lib/NotDebug/llava_static.lib"),  -- Begin VLM Addition/End VLM Addition
				(external_dir_llamacpp .. "/vulkan/lib/NotDebug/mtmd_static.lib"),  -- Begin VLM Addition/End VLM Addition
				(external_dir_llamacpp .. "/vulkan/lib/NotDebug/ggml-cpu.lib"),
				(external_dir_llamacpp .. "/vulkan/lib/NotDebug/ggml-base.lib"),
				(external_dir_llamacpp .. "/vulkan/lib/NotDebug/ggml-vulkan.lib"),
				"vulkan-1.lib"
			}

	filter {"system:linux"}
		libdirs {
			external_dir_llamacpp .. "/vulkan/lib64/",
			libdirs {externaldir .."vulkanSDK/lib/"}
		}
		linkoptions {"-fopenmp"}
		links {"llama", "common", "ggml", "llava_static", "mtmd_static", "ggml-cpu","ggml-vulkan","ggml-base","dl", "pthread", "rt", "vulkan"}  -- Begin VLM Addition -- added mtmd_static -- End VLM Addition

	filter {}	

project "nvigi.plugin.gpt.ggml.cuda"
	kind "SharedLib"
	targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	
	pluginBasicSetup("gpt/ggml")
	
	defines {
		"GGML_USE_CUBLAS=1",
		"GGML_USE_CUDA=1",
		"GGML_SCHED_MAX_COPIES=4",
		"GGML_CUDA_USE_GRAPHS",
		"GGML_CUDA_DMMV_X=32",
		"GGML_CUDA_MMV_Y=1",
		"GGML_CUDA_F16",
		"K_QUANTS_PER_ITERATION=2",
		"GGML_CUDA_PEER_MAX_BATCH_SIZE=128",
		"GGML_CUDA_NO_VMM"
    }

	vectorextensions "AVX2"
	
	files {
		"../*.h",
		"*.h",
		"*.cpp"	
	}
	includedirs {
		externaldir .."cuda/include",
		coresdkdir .. "source/utils/nvigi.ai", ROOT .. "source/plugins/nvigi.gpt/ggml/",
		external_dir_llamacpp .. "/common", 
		external_dir_llamacpp .. "/ggml/include", 
		external_dir_llamacpp .. "/include",
		external_dir_llamacpp .. "/tools/mtmd/",  -- Begin VLM Addition/End VLM Addition
	}
	
	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}

	-- some 3rd party libs (ggml) do not compile without it in production mode when security flags are set
	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"} 
	filter{}

	filter {"system:windows"}

		libdirs {
			externaldir .."cuda//lib/x64",
		}

		links {"cuda.lib", "cublas.lib"}


		filter {"system:windows", "configurations:Debug"}
			links {
				(external_dir_llamacpp .. "/cuda/lib/Debug/common.lib"),
				(external_dir_llamacpp .. "/cuda/lib/Debug/ggml.lib"), 
				(external_dir_llamacpp .. "/cuda/lib/Debug/llama.lib"),
				(external_dir_llamacpp .. "/cuda/lib/Debug/llava_static.lib"), -- Begin VLM Addition/End VLM Addition
				(external_dir_llamacpp .. "/cuda/lib/Debug/mtmd_static.lib"),  -- Begin VLM Addition/End VLM Addition
				(external_dir_llamacpp .. "/cuda/lib/Debug/ggml-cpu.lib"),
				(external_dir_llamacpp .. "/cuda/lib/Debug/ggml-cuda.lib"),
				(external_dir_llamacpp .. "/cuda/lib/Debug/ggml-base.lib")
			}

		filter {"system:windows", "configurations:not Debug"}
			links {
				(external_dir_llamacpp .. "/cuda/lib/NotDebug/common.lib"),
				(external_dir_llamacpp .. "/cuda/lib/NotDebug/ggml.lib"), 
				(external_dir_llamacpp .. "/cuda/lib/NotDebug/llama.lib"),
				(external_dir_llamacpp .. "/cuda/lib/NotDebug/llava_static.lib"),  -- Begin VLM Addition/End VLM Addition
				(external_dir_llamacpp .. "/cuda/lib/NotDebug/mtmd_static.lib"),  -- Begin VLM Addition/End VLM Addition
				(external_dir_llamacpp .. "/cuda/lib/NotDebug/ggml-cpu.lib"),
				(external_dir_llamacpp .. "/cuda/lib/NotDebug/ggml-cuda.lib"),
				(external_dir_llamacpp .. "/cuda/lib/NotDebug/ggml-base.lib")
			}


	filter {"system:linux"}
		libdirs {
			externaldir .."cuda/lib64",
			externaldir .."cuda/lib64/stubs",
			external_dir_llamacpp .. "/cuda/lib64/",
		}
		linkoptions {"-fopenmp", "-lcublas"}
		links {"llama", "common", "ggml", "llava_static", "mtmd_static", "ggml-cpu","ggml-base","ggml-cuda", "cuda", "cublas", "dl", "pthread", "rt"}  -- Begin VLM Addition -- added mtmd_static -- End VLM Addition

	filter {}

	add_cuda_dependencies()

if os.istarget("windows") then

project "nvigi.plugin.gpt.ggml.cpu"
	kind "SharedLib"
	targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	
	pluginBasicSetup("gpt/ggml")
	
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
		"./*.cpp"		
	}

	includedirs {
		coresdkdir .. "source/utils/nvigi.ai", 
		ROOT .. "source/plugins/nvigi.gpt/ggml/",
		external_dir_llamacpp .. "/common", 
		external_dir_llamacpp .. "/ggml/include", 
		external_dir_llamacpp .. "/include",
		external_dir_llamacpp .. "/tools/mtmd/",                  -- Begin VLM Addition/End VLM Addition
	}

	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}

	filter {"system:windows", "configurations:Debug"}
		links {
			(external_dir_llamacpp .. "/cpu/lib/Debug/common.lib"),
			(external_dir_llamacpp .. "/cpu/lib/Debug/ggml.lib"), 
			(external_dir_llamacpp .. "/cpu/lib/Debug/llama.lib"),
			(external_dir_llamacpp .. "/cpu/lib/Debug/llava_static.lib"),  -- Begin VLM Addition/End VLM Addition
			(external_dir_llamacpp .. "/cpu/lib/Debug/mtmd_static.lib"),  -- Begin VLM Addition/End VLM Addition
			(external_dir_llamacpp .. "/cpu/lib/Debug/ggml-cpu.lib"),
			(external_dir_llamacpp .. "/cpu/lib/Debug/ggml-base.lib"),
		}

	filter {"system:windows", "configurations:not Debug"}
		links {
			(external_dir_llamacpp .. "/cpu/lib/NotDebug/common.lib"),
			(external_dir_llamacpp .. "/cpu/lib/NotDebug/ggml.lib"), 
			(external_dir_llamacpp .. "/cpu/lib/NotDebug/llama.lib"),
			(external_dir_llamacpp .. "/cpu/lib/NotDebug/llava_static.lib"),  -- Begin VLM Addition/End VLM Addition
			(external_dir_llamacpp .. "/cpu/lib/NotDebug/mtmd_static.lib"),   -- Begin VLM Addition/End VLM Addition
			(external_dir_llamacpp .. "/cpu/lib/NotDebug/ggml-cpu.lib"),
			(external_dir_llamacpp .. "/cpu/lib/NotDebug/ggml-base.lib"),
		}
	
	filter {"system:linux"}
		libdirs {
			external_dir_llamacpp .. "/cpu/lib64/",
		}
		linkoptions {"-fopenmp"}
		links {"llama", "common", "ggml", "llava_static", "mtmd_static"}  -- Begin VLM Addition -- added mtmd_static -- End VLM Addition

end

group ""
