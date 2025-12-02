group "plugins/gpt"

local external_dir_llamacpp = externaldir .. "llama.cpp"

project "nvigi.plugin.gpt.ggml.vk"	
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
		external_dir_llamacpp .. "/src",
		external_dir_llamacpp .. "/tools/mtmd/", -- Begin VLM Addition/End VLM Addition
		externaldir .. "vulkanSDK/include"
	}
	
	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
	
	-- some 3rd party libs (ggml) do not compile without it in production mode when security flags are set
	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"} 
	filter{}

	libdirs {
		externaldir .."vulkanSDK/Lib",
		external_dir_llamacpp .. "/vulkan/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
	}

	filter {"system:windows"}
		links {
			"common.lib",
			"ggml.lib", 
			"llama.lib",
			"mtmd_static.lib", -- Begin VLM Addition/End VLM Addition
			"ggml-cpu.lib",
			"ggml-base.lib",
			"ggml-vulkan.lib",
			"vulkan-1.lib"
		}

	filter {"system:linux"}
		libdirs {
			external_dir_llamacpp .. "/vulkan/lib64/"
		}
		linkoptions {"-fopenmp"}
		links {"llama", "common", "ggml", "mtmd_static", "ggml-cpu","ggml-vulkan","ggml-base","dl", "pthread", "rt", "vulkan"}  -- Begin VLM Addition -- added mtmd_static -- End VLM Addition

	filter {}	

project "nvigi.plugin.gpt.ggml.cuda"
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
		external_dir_llamacpp .. "/src",
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
			external_dir_llamacpp .. "/cuda/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
		}

		links {"cuda.lib", "cublas.lib"}


		links {
			"common.lib",
			"ggml.lib", 
			"llama.lib",
			"mtmd_static.lib",  -- Begin VLM Addition/End VLM Addition
			"ggml-cpu.lib",
			"ggml-cuda.lib",
			"ggml-base.lib"
		}

	filter {"system:linux"}
		libdirs {
			externaldir .."cuda/lib64",
			externaldir .."cuda/lib64/stubs",
			external_dir_llamacpp .. "/cuda/lib64/",
		}
		linkoptions {"-fopenmp", "-lcublas"}
		links {"llama", "common", "ggml", "mtmd_static", "ggml-cpu","ggml-base","ggml-cuda", "cuda", "cublas", "dl", "pthread", "rt"}  -- Begin VLM Addition -- added mtmd_static -- End VLM Addition

	filter {}

	add_cuda_dependencies()

if os.istarget("windows") then

project "nvigi.plugin.gpt.ggml.cpu"
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
		external_dir_llamacpp .. "/src",
		external_dir_llamacpp .. "/tools/mtmd/",                  -- Begin VLM Addition/End VLM Addition
	}

	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}

	filter {"system:windows"}
		libdirs {
			external_dir_llamacpp .. "/cpu/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
		}

		links {
			"common.lib",
			"ggml.lib", 
			"llama.lib",
			"mtmd_static.lib",  -- Begin VLM Addition/End VLM Addition
			"ggml-cpu.lib",
			"ggml-base.lib",
		}
	
	filter {"system:linux"}
		libdirs {
			external_dir_llamacpp .. "/cpu/lib64/",
		}
		linkoptions {"-fopenmp"}
		links {"llama", "common", "ggml", "mtmd_static"}  -- Begin VLM Addition -- added mtmd_static -- End VLM Addition

end

group ""
