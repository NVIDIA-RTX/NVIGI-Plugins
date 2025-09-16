if os.istarget("windows") then

group "samples"

project "nvigi.fcall"
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
		ROOT .. "source/plugins/nvigi.gpt"
	}

	-- Begin for libcurl
	libdirs {externaldir .."libcurl/lib/rt_dynamic/release"}
	filter {"system:windows", "configurations:Debug"}
		libdirs { externaldir .."/zlib/debug/lib"}
		links {"zlibd.lib"}
	filter {"system:windows", "configurations:not Debug"}
		libdirs { externaldir .."/zlib/lib"}
		links {"zlib.lib"}
	-- End for libcurl

	filter {"system:windows"}
		vpaths { ["impl"] = {"./**.h","./**.cpp", }}
		vpaths { ["utils"] = {ROOT .. "source/utils/**.h",ROOT .. "source/utils/**.cpp", }}
		-- Begin for libcurl
		links { "libcurl.lib",  "ws2_32.lib", "wldap32.lib","advapi32.lib", "crypt32.lib", "normaliz.lib"}
		-- End for libcurl
	filter {"system:linux"}
	filter {}

group ""

end
