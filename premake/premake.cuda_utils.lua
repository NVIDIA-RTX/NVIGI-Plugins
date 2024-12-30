require("vstudio")

nvccPath = ROOT .. "external/cuda/bin/nvcc"

function make_nvcc_command(nvccPath, nvccHostCompilerFlags, nvccFlags)
    ext = iif(os.ishost("windows"), ".obj", ".o")
    local buildString =  "\""..nvccPath.."\" -std=c++17 "..nvccFlags.." -Xcompiler="..commaficate(nvccHostCompilerFlags).." -c "..' %{table.implode(cfg.defines, "-D", "", " ")} '.." %{get_include_string(cfg.includedirs)} \"%{file.abspath}\" -o \"%{cfg.objdir}/%{file.basename}"..ext.."\""
	if os.istarget("windows") then
		buildcommands { buildString }
	elseif os.istarget("linux") then
	    buildcommands { "{MKDIR} %{cfg.objdir} ", buildString }
	end
    buildmessage (buildString)
    buildoutputs { "%{cfg.objdir}/%{file.basename}"..ext }    
end

function add_cuda_dependencies_no_cudart()
	--Fermi	Kepler	Maxwell	Pascal	Volta	Turing	Ampere	Ada		Hopper
	--sm_20	sm_30	sm_50	sm_60	sm_70	sm_75	sm_80	sm_89	sm_90
	
	-- If we want to allow running this on newer architectures, generate PTX code for SM 89 (latest arch)
	local cudaTargets = "-arch=compute_70 -code=compute_70,sm_70 --generate-code=arch=compute_80,code=[sm_80] --generate-code=arch=compute_80,code=[compute_80]"
	-- supress warning "pointless comparison of unsigned integer with zero"
	local warningsOff = "-Xcudafe \"--diag_suppress=unsigned_compare_with_zero\" "
	local nvccFlags = "--threads 8 -Xptxas -O3,-v --expt-extended-lambda -m64 -Wno-deprecated-gpu-targets --expt-relaxed-constexpr -std=c++17 "..warningsOff ..cudaTargets

	if os.istarget("windows") then
        nvccWin = " -Xcompiler=\"/EHsc /Zc:__cplusplus /bigobj -Ob2 /wd4819 \" "
		filter { "files:**.cu", "configurations:Debug"}
			make_nvcc_command(nvccPath, "/MDd", nvccWin..nvccFlags)
		filter { "files:**.cu", "configurations:not Debug" }
			make_nvcc_command(nvccPath, "/MD", nvccWin..nvccFlags)
		filter {}
	elseif os.istarget("linux") then		
		filter { "files:**.cu", "system:linux", "configurations:Debug"}
			make_nvcc_command(nvccPath, "-fPIC -g", "-g "..nvccFlags)
		filter { "files:**.cu", "system:linux", "configurations:not Debug" }
			make_nvcc_command(nvccPath, "-fPIC", nvccFlags)
		filter {}
	end
end

function add_cuda_dependencies()
	add_cuda_dependencies_no_cudart()
    filter {"system:windows"}
		links {"cudart_static.lib"}
    filter {"system:linux"}
		links {"cudart_static", "dl", "pthread", "rt"}
	filter {}
end

