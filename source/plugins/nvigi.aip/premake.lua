if os.istarget("windows") then

group "plugins/pipelines"	
	project "nvigi.plugin.ai.pipeline"
		if premakeX64Targets() then
			filter { "platforms:x64" }
				kind "SharedLib"
			filter {}
		end
		
		pluginBasicSetup("aip")
	
		files { 
			"./*.h",
			"./*.cpp",
		}

		vpaths { ["impl"] = {"./*.h", "./*.cpp" }}				
group ""

end