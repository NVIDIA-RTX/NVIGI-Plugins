	
	include ("premake.utils.lua")

	-- _ACTION is the argument you passed into premake5 when you ran it.
	local project_action = "UNDEFINED"
	if _ACTION ~= nill then project_action = _ACTION end
	

	-- Where the project files (vs project, solution, etc) go
	location( ROOT .. "_project/" .. project_action)
	configurations { "Debug", "Production", "Release" }
	

	newoption {
		trigger = "x64",
		description = "Enable building of amd64 (setup.bat passes this by default)"
	}

	-- True when this premake run should emit x64 project kinds/targets.
	-- Default ON so `premake5 vs2022 --file=premake.lua` matches legacy single-arch behavior.
	function premakeX64Targets()
		if _OPTIONS["x64"] ~= nil then
			return true
		end
		return true
	end

	if _OPTIONS["x64"] ~= nil then
		print("Enabling x64 build support")
		platforms { "x64" }
	end
	filter { "platforms:x64" }
		architecture "x64"
	filter {}

	language "c++"
	preferredtoolarchitecture "x86_64"
	targetprefix ""

	externaldir = (ROOT .."external/")
	artifactsdir = (ROOT .."_artifacts/")
	coresdkdir = (ROOT .."nvigi_core/")

	-- NVTX before CUDA includes so nvtx3/... resolves to the packman NVTX package (headers under c/include per upstream layout).
	includedirs 
	{ 
		externaldir .. "nvtx/c/include",
		ROOT .. "include",
		ROOT,
		coresdkdir,
		coresdkdir..'include',
		ROOT.."external/json/source"
	}
   	 
	if os.ishost("windows") then
		local winSDK = os.winSdkVersion() .. ".0"
		print("WinSDK", winSDK)
		systemversion(winSDK)

		local function getVSInstallPath()
			local vswhere = os.getenv("ProgramFiles(x86)") .. "/Microsoft Visual Studio/Installer/vswhere.exe"
			local command = string.format('"%s" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath', vswhere)
			return os.outputof(command)
		end
		
		local vsInstallPath = getVSInstallPath()

		local function getVCToolsVersion(vsInstallPath)
			local versionFile = vsInstallPath .. "/VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt"
			local f = io.open(versionFile, "r")
			if f then
				local version = f:read("*all"):match("(%d+%.%d+%.%d+)")
				f:close()
				return version
			end
			return nil
		end
		
		local toolsVersion = getVCToolsVersion(vsInstallPath)
		if toolsVersion then
			print("MS Toolchain Version: " .. toolsVersion)
		else
			print("Failed to detect MS Toolchain Version")
		end		
	end
	
	-- DO NOT REMOVE, required for security --
	filter {"system:windows","configurations:Production"}
			buildoptions {"/guard:ehcont","/guard:cf","/sdl"}		
			linkoptions {"/HIGHENTROPYVA"}			
	filter{}
	filter {"system:windows","configurations:Production", "platforms:x64"}
			linkoptions {"/CETCOMPAT"}			
	filter{}

    filter {"system:windows"}
		externalincludedirs { externaldir }
		externalwarnings "Off"
		--flags {"FatalWarnings"}
		defines { "NVIGI_SDK", "NVIGI_WINDOWS", "WIN32" , "WIN64" , "_CONSOLE", "NOMINMAX"}
		buildoptions {"/utf-8", "/Zc:__cplusplus"}
		defines {
			"_CRT_SECURE_NO_WARNINGS"
		}				
	-- when building any visual studio project
	filter {"system:windows", "action:vs*"}
		multiprocessorcompile "On"
		minimalrebuild "Off"
    filter {}

    filter {"system:windows"}
		cppdialect "C++latest"
	filter {} 
	
	filter "configurations:not Production"
		defines { "NVIGI_VALIDATE_MEMORY" }
	filter "configurations:Debug"
		defines { "DEBUG", "_DEBUG", "NVIGI_ENABLE_TIMING=1", "NVIGI_DEBUG", "NVIGI_VALIDATE_MEMORY" }
		symbols "Full"
	filter {} 
				
	filter "configurations:Release"
		defines { "NDEBUG", "NVIGI_ENABLE_TIMING=1", "NVIGI_RELEASE" }
		optimize "On"
		symbols "On"

	filter "configurations:Production"
		defines { "NDEBUG","NVIGI_ENABLE_TIMING=0","NVIGI_ENABLE_PROFILING=0","NVIGI_PRODUCTION" }
		optimize "On"
		symbols "On"
		linktimeoptimization "On"

	filter {} -- clear filter when you know you no longer need it!


	filter {"system:windows"}
		defines { 
			"NVIGI_DEF_MIN_OS_MAJOR=10",
			"NVIGI_DEF_MIN_OS_MINOR=0",
			"NVIGI_DEF_MIN_OS_BUILD=19041"
		}
    filter {"system:windows"}
		defines { 
			"NVIGI_CUDA_MIN_GPU_ARCH=0x00000140",
			"NVIGI_CUDA_MIN_DRIVER_MAJOR=555",
			"NVIGI_CUDA_MIN_DRIVER_MINOR=85",
			"NVIGI_CUDA_MIN_DRIVER_BUILD=0",
			"NVIGI_D3D12_MIN_DRIVER_MAJOR=580",
			"NVIGI_D3D12_MIN_DRIVER_MINOR=61",
			"NVIGI_D3D12_MIN_DRIVER_BUILD=0"
		}
	filter {}
