if os.istarget("windows") then

group "samples"

project "nvigi.pipeline"
	if premakeX64Targets() then
		filter { "platforms:x64" }
			kind "ConsoleApp"
		filter {}
	end

	targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.platform}/%{cfg.buildcfg}")
	objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.platform}/%{cfg.buildcfg}")
	
	dependson {"gitVersion"}

	--floatingpoint "Fast"

	files {
		"./**.h",
		"./**.cpp",
		ROOT .. "source/utils/**.h",
	}

	includedirs {coresdkdir .. "source/core/nvigi.api", coresdkdir .. "source/utils/nvigi.ai"}
	-- allows the sample to compile using include directives that match a packaged SDK
	includedirs {
		ROOT .. "source/plugins/nvigi.aip",
		ROOT .. "source/plugins/nvigi.asr",
		ROOT .. "source/plugins/nvigi.gpt",
		ROOT .. "source/plugins/nvigi.tts"
	}
	filter {"system:windows"}
		vpaths { ["impl"] = {"./**.h","./**.cpp", }}
		vpaths { ["utils"] = {ROOT .. "source/utils/**.h",ROOT .. "source/utils/**.cpp", }}
		links { "dsound.lib", "winmm.lib" }
	filter {}

group ""

end
