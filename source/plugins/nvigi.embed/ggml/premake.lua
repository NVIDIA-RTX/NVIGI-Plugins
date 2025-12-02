group "plugins/embed"

local external_dir_llamacpp = externaldir .. "llama.cpp"
project "nvigi.plugin.embed.ggml.cuda"
	pluginBasicSetup("embed/ggml")
	
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
				coresdkdir .. "source/utils/nvigi.ai", ROOT .. "source/plugins/nvigi.embed/ggml/",
				external_dir_llamacpp .. "/common", 
				external_dir_llamacpp .. "/ggml/include", 
				external_dir_llamacpp .. "/include",
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
		links {"common", "llama", "ggml", "ggml-base","ggml-cpu","ggml-cuda","cuda", "cublas", "dl", "pthread", "rt"}

	filter {}

	add_cuda_dependencies()

project "nvigi.plugin.embed.ggml.cpu"
	pluginBasicSetup("embed/ggml")
	
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
		
	}
	includedirs {
		coresdkdir .. "source/utils/nvigi.ai", 
		ROOT .. "source/plugins/nvigi.embed/ggml/",
		external_dir_llamacpp .. "/common", 
		external_dir_llamacpp .. "/ggml/include", 
		external_dir_llamacpp .. "/include"
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
			"ggml-cpu.lib",
			"ggml-base.lib"
		}
	
	filter {"system:linux"}
		libdirs {
			external_dir_llamacpp .. "/cpu/lib64/",
		}
		linkoptions {"-fopenmp"}
		links {"common", "llama", "ggml", "ggml-base","ggml-cpu","dl", "pthread", "rt"}

group ""