group "plugins/gpt"

local external_dir_llamacpp = externaldir .. "llama.cpp"

project "nvigi.plugin.gpt.ggml.vk"
	if premakeX64Targets() then
		filter { "platforms:x64" }
			kind "SharedLib"
		filter {}
	end
		
	pluginBasicSetup("gpt/ggml")

	defines {
		"GGML_USE_VULKAN=1",
		"GGML_SCHED_MAX_COPIES=4",
		"K_QUANTS_PER_ITERATION=2",
    }

	files {
		"../*.h",
		"*.h",
		"*.cpp"
	}

	includedirs {
		coresdkdir .. "source/utils/nvigi.ai",
		ROOT .. "source/plugins/nvigi.gpt/ggml/",
	}

	-- Per-platform include/lib paths
	filter {"system:windows", "platforms:x64"}
		vectorextensions "AVX2"
		includedirs {
			external_dir_llamacpp .. "/common",
			external_dir_llamacpp .. "/ggml/include",
			external_dir_llamacpp .. "/include",
			external_dir_llamacpp .. "/src",
			external_dir_llamacpp .. "/tools/mtmd/", -- VLM
			external_dir_llamacpp .. "/tools/server/", -- Server API
			externaldir .. "vulkanSDK/include"
		}
		libdirs {
			externaldir .."vulkanSDK/Lib",
			externaldir.."llama.cpp/vulkan/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
		}
	filter{}


	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}

	-- some 3rd party libs (ggml) do not compile without it in production mode when security flags are set
	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"}
	filter{}

	links {
		"common.lib",
		"ggml.lib",
		"llama.lib",
		"llama-server-lib.lib",
		"ggml-cpu.lib",
		"ggml-base.lib",
		"ggml-vulkan.lib",
		"vulkan-1.lib"
	}

project "nvigi.plugin.gpt.ggml.cuda"
	if premakeX64Targets() then
		filter { "platforms:x64" }
			kind "SharedLib"
		filter {}
	end
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
			ROOT .. "source/plugins/nvigi.gpt/ggml/",
			external_dir_llamacpp .. "/common",
			external_dir_llamacpp .. "/ggml/include",
			external_dir_llamacpp .. "/include",
			external_dir_llamacpp .. "/src",
			external_dir_llamacpp .. "/tools/mtmd/", -- VLM
			external_dir_llamacpp .. "/tools/server/", -- Server API
		}

		libdirs {
			externaldir .."cuda//lib/x64",
			externaldir.."llama.cpp/cuda/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
		}
	filter {}


	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}

	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"}
	filter{}

	links {"cuda.lib", "cublas.lib"}
	links {
		"common.lib",
		"ggml.lib",
		"llama.lib",
		"llama-server-lib.lib",
		"ggml-cpu.lib",
		"ggml-cuda.lib",
		"ggml-base.lib"
	}

	add_cuda_dependencies()

if os.istarget("windows") then

project "nvigi.plugin.gpt.ggml.cpu"
	if premakeX64Targets() then
		filter { "platforms:x64" }
			kind "SharedLib"
		filter {}
	end

	pluginBasicSetup("gpt/ggml")

	defines {
		"K_QUANTS_PER_ITERATION=2",
		"GGML_USE_K_QUANTS"
    }

	files {
		"../*.h",
		"./*.h",
		"./*.cpp"
	}

	includedirs {
		coresdkdir .. "source/utils/nvigi.ai",
		ROOT .. "source/plugins/nvigi.gpt/ggml/",
	}

	filter {"system:windows", "platforms:x64"}
		vectorextensions "AVX2"
		includedirs {
			external_dir_llamacpp .. "/common",
			external_dir_llamacpp .. "/ggml/include",
			external_dir_llamacpp .. "/include",
			external_dir_llamacpp .. "/src",
			external_dir_llamacpp .. "/tools/server/", -- Server API
		}

		libdirs {
			externaldir.."llama.cpp/cpu/lib/".."%{iif(cfg.buildcfg:startswith(\"Debug\"), \"Debug\", \"NotDebug\")}",
		}
	filter {}


	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}

	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"}
	filter{}

	links {
		"common.lib",
		"ggml.lib",
		"llama.lib",
		"llama-server-lib.lib",
		"ggml-cpu.lib",
		"ggml-base.lib",
	}

end

group ""
