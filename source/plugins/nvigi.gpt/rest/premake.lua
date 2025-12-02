group "plugins/gpt"

project "nvigi.plugin.gpt.cloud.rest"
	pluginBasicSetup("gpt/rest")
	
	files {
		"../*.h",
		"./*.h",
		"./*.cpp"	
	}

	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
		

group ""