group "plugins/embed"

local llama_cpp_version = "a1fe56b"

local external_dir_llamacpp = externaldir .. "llamacpp/" .. llama_cpp_version
project "nvigi.plugin.embed.ggml.cuda"
	kind "SharedLib"
	targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	
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

	local suffixe_llamacpp_lib = "_" .. llama_cpp_version .. "_cuda.lib"

	-- some 3rd party libs (ggml) do not compile without it in production mode when security flags are set
	filter {"system:windows","configurations:Production"}
			defines {"_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"} 
	filter{}

	filter {"system:windows"}

		libdirs {externaldir .."cuda//lib/x64"}
		links {"cuda.lib", "cublas.lib"}


		filter {"system:windows", "configurations:Debug"}
			links {
				(external_dir_llamacpp .. "/lib/Debug/common"..suffixe_llamacpp_lib),
				(external_dir_llamacpp .. "/lib/Debug/ggml"..suffixe_llamacpp_lib), 
				(external_dir_llamacpp .. "/lib/Debug/llama"..suffixe_llamacpp_lib)
			}

		filter {"system:windows", "configurations:not Debug"}
			links {
				(external_dir_llamacpp .. "/lib/NotDebug/common"..suffixe_llamacpp_lib),
				(external_dir_llamacpp .. "/lib/NotDebug/ggml"..suffixe_llamacpp_lib), 
				(external_dir_llamacpp .. "/lib/NotDebug/llama"..suffixe_llamacpp_lib)
			}


	filter {"system:linux"}
		local suffixe_llamacpp_lib = "_" .. llama_cpp_version .. "_cuda"
		libdirs {
			externaldir .."cuda/lib64",
			externaldir .."cuda/lib64/stubs",
			external_dir_llamacpp .. "/lib64/",
		}
		linkoptions {"-fopenmp", "-lcublas"}
		links {"llama"..suffixe_llamacpp_lib, "common"..suffixe_llamacpp_lib, "ggml"..suffixe_llamacpp_lib,"cuda", "cublas", "dl", "pthread", "rt"}

	filter {}

	add_cuda_dependencies()

project "nvigi.plugin.embed.ggml.cpu"
	kind "SharedLib"
	targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	
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

	local suffixe_llamacpp_lib = "_" .. llama_cpp_version .. "_cpu.lib"
	filter {"system:windows", "configurations:Debug"}
		links {
			(external_dir_llamacpp .. "/lib/Debug/common"..suffixe_llamacpp_lib),
			(external_dir_llamacpp .. "/lib/Debug/ggml"..suffixe_llamacpp_lib), 
			(external_dir_llamacpp .. "/lib/Debug/llama"..suffixe_llamacpp_lib)
		}

	filter {"system:windows", "configurations:not Debug"}
		links {
			(external_dir_llamacpp .. "/lib/NotDebug/common"..suffixe_llamacpp_lib),
			(external_dir_llamacpp .. "/lib/NotDebug/ggml"..suffixe_llamacpp_lib), 
			(external_dir_llamacpp .. "/lib/NotDebug/llama"..suffixe_llamacpp_lib)
		}
	
	filter {"system:linux"}
		local suffixe_llamacpp_lib = "_" .. llama_cpp_version .. "_cpu"
		libdirs {
			external_dir_llamacpp .. "/lib64/",
		}
		linkoptions {"-fopenmp"}
		links {"llama"..suffixe_llamacpp_lib, "common"..suffixe_llamacpp_lib, "ggml"..suffixe_llamacpp_lib}

group ""