group "samples/codeagentlua"

-- Build Lua as a static library to be used by nvigi.codeagentlua
project "liblua"
    kind "StaticLib"
    language "C"
    targetname "lua" -- produces lua.lib
    targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
    objdir    (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")

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

    filter "system:not windows"
        links { "m" }
    filter {}


if os.istarget("windows") then

project "nvigi.codeagentlua"
    kind "ConsoleApp"
    targetdir (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
    objdir    (ROOT .. "_artifacts/%{prj.name}/%{cfg.buildcfg}_%{cfg.platform}")
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

    filter { "system:linux" }
        -- linux-specific flags if ever needed

    filter {}

group ""

end
