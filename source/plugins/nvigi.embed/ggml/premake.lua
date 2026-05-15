group "plugins/embed"

local external_dir_llamacpp = externaldir .. "llama.cpp"

project "nvigi.plugin.embed.ggml.cuda"
	if premakeX64Targets() then
		filter { "platforms:x64" }
			kind "SharedLib"
		filter {}
	end
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

	files {
		"../*.h",
		"*.h",
		"*.cpp"
	}

	filter {"system:windows", "platforms:x64"}
		vectorextensions "AVX2"
		includedirs {
			externaldir .."cuda/include",
			coresdkdir .. "source/utils/nvigi.ai",
			ROOT .. "source/plugins/nvigi.embed/ggml/",
			external_dir_llamacpp .. "/common",
			external_dir_llamacpp .. "/ggml/include",
			external_dir_llamacpp .. "/include",
		}

		libdirs {
			externaldir .."cuda//lib/x64",
			external_dir_llamacpp .. "/cuda/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
		}
	filter {}


	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"} 
	filter{}

	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}

	links {"cuda.lib", "cublas.lib"}

	links {
		"common.lib",
		"cpp-httplib.lib",
		"ggml.lib",
		"llama.lib",
		"ggml-cpu.lib",
		"ggml-cuda.lib",
		"ggml-base.lib"
	}

	add_cuda_dependencies()

project "nvigi.plugin.embed.ggml.cpu"
	if premakeX64Targets() then
		filter { "platforms:x64" }
			kind "SharedLib"
		filter {}
	end

	pluginBasicSetup("embed/ggml")

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
		includedirs {
			coresdkdir .. "source/utils/nvigi.ai",
			ROOT .. "source/plugins/nvigi.embed/ggml/",
			external_dir_llamacpp .. "/common",
			external_dir_llamacpp .. "/ggml/include",
			external_dir_llamacpp .. "/include"
		}

		vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}

		libdirs {
			external_dir_llamacpp .. "/cpu/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
		}
	filter {}


	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"}
	filter{}

	links {
		"common.lib",
		"cpp-httplib.lib",
		"ggml.lib",
		"llama.lib",
		"ggml-cpu.lib",
		"ggml-base.lib"
	}

	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}

group ""
