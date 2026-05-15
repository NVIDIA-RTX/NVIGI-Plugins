group "samples/codeagentlua"

-- Build Lua as a static library to be used by nvigi.codeagentlua
project "liblua"
	if premakeX64Targets() then
		filter { "platforms:x64" }
			kind "StaticLib"
		filter {}
	end

    language "C"
    targetname "lua" -- produces lua.lib
    targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.platform}/%{cfg.buildcfg}")
    objdir    (ROOT .. "_artifacts/%{prj.name}/%{cfg.platform}/%{cfg.buildcfg}")

    characterset "MBCS"

    -- Lua sources (adjust externaldir if needed)
    includedirs { "../external/lua/src" }

    files {
        "../external/lua/src/lapi.c",
        "../external/lua/src/lcode.c",
        "../external/lua/src/lctype.c",
        "../external/lua/src/ldebug.c",
        "../external/lua/src/ldo.c",
        "../external/lua/src/ldump.c",
        "../external/lua/src/lfunc.c",
        "../external/lua/src/lgc.c",
        "../external/lua/src/llex.c",
        "../external/lua/src/lmem.c",
        "../external/lua/src/lobject.c",
        "../external/lua/src/lopcodes.c",
        "../external/lua/src/lparser.c",
        "../external/lua/src/lstate.c",
        "../external/lua/src/lstring.c",
        "../external/lua/src/ltable.c",
        "../external/lua/src/ltm.c",
        "../external/lua/src/lundump.c",
        "../external/lua/src/lvm.c",
        "../external/lua/src/lzio.c",

        "../external/lua/src/lauxlib.c",
        "../external/lua/src/lbaselib.c",
        "../external/lua/src/lcorolib.c",
        "../external/lua/src/ldblib.c",
        "../external/lua/src/liolib.c",
        "../external/lua/src/lmathlib.c",
        "../external/lua/src/loadlib.c",
        "../external/lua/src/loslib.c",
        "../external/lua/src/lstrlib.c",
        "../external/lua/src/ltablib.c",
        "../external/lua/src/lutf8lib.c",
        "../external/lua/src/linit.c"
    }

if os.istarget("windows") then

project "nvigi.codeagentlua"
	if premakeX64Targets() then
		filter { "platforms:x64" }
			kind "ConsoleApp"
		filter {}
	end

    targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.platform}/%{cfg.buildcfg}")
    objdir    (ROOT .. "_artifacts/%{prj.name}/%{cfg.platform}/%{cfg.buildcfg}")
    dependson { "gitVersion", "liblua" }

    files {
        "./**.h",
        "./**.cpp",
        "./**.rc",
        coresdkdir .. "source/utils/**.h",
        coresdkdir .. "source/utils/**.cpp",
    }

    -- Core and plugin headers
    includedirs {
        coresdkdir .. "source/core/nvigi.api",
        coresdkdir .. "source/utils/nvigi.ai",
        ROOT .. "source/plugins/nvigi.gpt",
        "../external/lua/src"        -- Lua headers
    }

    filter { "system:windows" }
        vpaths { ["impl"]  = { "./**.h", "./**.cpp" } }
        vpaths { ["utils"] = { ROOT .. "source/utils/**.h", ROOT .. "source/utils/**.cpp" } }

        -- Link Lua
        links {
            "liblua",
        }

    filter {}

group ""

end
