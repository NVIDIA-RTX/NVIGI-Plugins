function pluginBasicSetupInternal(name, sdk)
	kind "SharedLib"
	targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")

	files { 
		coresdkdir .. "source/core/nvigi.api/**.h",
		coresdkdir .. "source/core/nvigi.log/**.h",				
		coresdkdir .. "source/core/nvigi.types/**.h",		
		coresdkdir .. "source/core/nvigi.file/**.h",
		coresdkdir .. "source/core/nvigi.file/**.cpp",
		coresdkdir .. "source/core/nvigi.extra/**.h",		
		coresdkdir .. "source/core/nvigi.plugin/**.h",
		coresdkdir .. "source/core/nvigi.plugin/**.cpp",
		coresdkdir .. "source/utils/**.h",		
		ROOT .. "source/plugins/nvigi."..name.."/versions.h",
		ROOT .. "source/plugins/nvigi."..name.."/resource.h",
		ROOT .. "source/plugins/nvigi."..name.."/premake.lua",
	}
		
	includedirs 
	{ 
		coresdkdir .. "source/core/nvigi.api", 
		coresdkdir .. "source/utils/nvigi.ai"
	}

	dependson {"gitVersion"}

	filter {"system:windows"}	
		files {	ROOT .. "source/plugins/nvigi."..name.."/**.rc"}
		vpaths { ["api"] = {coresdkdir .. "source/core/nvigi.api/**.h"}}
		vpaths { ["types"] = {coresdkdir .. "source/core/nvigi.types/**.h"}}
		vpaths { ["log"] = {coresdkdir .. "source/core/nvigi.log/**.h",coresdkdir .. "source/core/nvigi.log/**.cpp"}}	
		vpaths { ["utils/ai"] = {coresdkdir .. "source/utils/nvigi.ai/*.h"}}	
		vpaths { ["utils/dsound"] = {coresdkdir .. "source/utils/nvigi.dsound/*.h"}}	
		vpaths { ["file"] = {coresdkdir .. "source/core/nvigi.file/**.h", coresdkdir .. "source/core/nvigi.file/**.cpp"}}	
		vpaths { ["extra"] = {coresdkdir .. "source/core/nvigi.extra/**.h"}}		
		vpaths { ["plugin"] = {coresdkdir .. "source/core/nvigi.plugin/**.h",coresdkdir .. "source/core/nvigi.plugin/**.cpp"}}	
		vpaths { ["version"] = {coresdkdir .. "source/plugins/nvigi."..name.."/resource.h",ROOT .. "source/plugins/nvigi."..name.."/versions.h",ROOT .. "source/plugins/nvigi."..name.."/**.rc"}}
		vpaths {["premake"] = {ROOT .. "source/plugins/nvigi."..name.."/premake.lua"}
	}
		-- NOTE: moved the warning section after the files setup because filter was messing up inclusion of *.rc
		
		-- disable warnings coming from external source code
		externalincludedirs { ROOT .. "source/plugins/nvigi."..name.."/external" }
		-- Apply settings to suppress warnings for external files
		filter { "files:external/**.cpp or files:**/external/**.cpp or files:external/**.c or files:**/external/**.c or files:**/external/**.cc or files:external/**.cc or files:**/external/**.cu or files:external/**.cu" }
			warnings "Off"
	filter {}
end

function pluginBasicSetupSDK(name)
	pluginBasicSetupInternal(name, True)
end

function pluginBasicSetup(name)
	pluginBasicSetupInternal(name, False)
end
