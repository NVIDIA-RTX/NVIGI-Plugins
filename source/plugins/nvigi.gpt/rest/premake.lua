group "plugins/gpt"

project "nvigi.plugin.gpt.cloud.rest"
	if premakeX64Targets() then
		filter { "platforms:x64" }
			kind "SharedLib"
		filter {}
	end
		
	pluginBasicSetup("gpt/rest")
	
	files {
		"../*.h",
		"./*.h",
		"./*.cpp"	
	}

	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
		

group ""