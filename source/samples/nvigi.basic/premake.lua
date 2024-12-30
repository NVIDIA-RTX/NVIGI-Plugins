if os.istarget("windows") then

group "samples"

project "nvigi.basic"
	kind "ConsoleApp"
	targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	
	dependson {"gitVersion"}

	--floatingpoint "Fast"

	files {
		"./**.h",
		"./**.cpp",
		"./**.rc",
		coresdkdir .. "source/utils/**.h",
	}

	includedirs {coresdkdir .. "source/core/nvigi.api", coresdkdir .. "source/utils/nvigi.ai"}
	-- allows the sample to compile using include directives that match a packaged SDK
	includedirs {
		ROOT .. "source/plugins/nvigi.asr",
		ROOT .. "source/plugins/nvigi.gpt"
	}
	filter {"system:windows"}
		vpaths { ["impl"] = {"./**.h","./**.cpp", }}
		vpaths { ["utils"] = {ROOT .. "source/utils/**.h",ROOT .. "source/utils/**.cpp", }}
		links { "dsound.lib", "winmm.lib" }
	filter {"system:linux"}
	filter {}

group ""

end
