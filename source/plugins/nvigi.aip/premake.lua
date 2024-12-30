if os.istarget("windows") then

group "plugins/pipelines"	
	project "nvigi.plugin.ai.pipeline"
		kind "SharedLib"	
		targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
		objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}") 
		
		pluginBasicSetup("aip")
	
		files { 
			"./*.h",
			"./*.cpp",
		}

		vpaths { ["impl"] = {"./*.h", "./*.cpp" }}				
group ""

end