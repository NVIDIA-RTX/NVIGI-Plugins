if os.istarget("windows") then

group "plugins/gpt"

project "nvigi.plugin.gpt.onnxgenai.dml"
	kind "SharedLib"
	targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	objdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
	
	pluginBasicSetup("gpt/onnxgenai")
	
	defines {
    }

	vectorextensions "AVX2"
	
	files {
		"../*.h",
		"./*.h",
		"./*.cpp",
		"./external/include/*.h",
	}

	vpaths { ["impl"] = {"../*.h", "./*.h", "./*.cpp" }}
	vpaths { ["onnxgenai"] = {
		"./external/incude/*.h",
	}}
	links {"onnxruntime-genai.lib"}
	includedirs {
		externaldir .."Microsoft.ML.OnnxRuntimeGenAI.DirectML/build/native/include",
		".."
	}
	libdirs { externaldir .."Microsoft.ML.OnnxRuntimeGenAI.DirectML/runtimes/win-x64/native" }

group ""

end