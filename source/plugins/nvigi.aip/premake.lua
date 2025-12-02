if os.istarget("windows") then

group "plugins/pipelines"	
	project "nvigi.plugin.ai.pipeline"
		pluginBasicSetup("aip")
	
		files { 
			"./*.h",
			"./*.cpp",
		}

		vpaths { ["impl"] = {"./*.h", "./*.cpp" }}				
group ""

end