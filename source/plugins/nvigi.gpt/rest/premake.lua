group "plugins/gpt"

project "nvigi.plugin.gpt.cloud.rest"
	kind "SharedLib"
	targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	
	pluginBasicSetup("gpt/rest")
	
	files {
		"../*.h",
		"./*.h",
		"./*.cpp"	
	}

	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
		

group ""