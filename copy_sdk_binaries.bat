echo off
set Cfg=%1

if "%Cfg%" NEQ "Release" (
    if "%Cfg%" NEQ "Debug" (
        if "%Cfg%" NEQ "Production" (
            echo Script requires a single argument: the build config to copy.  Must be Release, Debug or Production
            exit /b 1
        )
    )
)

cd %~dp0
if not exist "bin\x64\%Cfg%" mkdir bin\x64\%Cfg%

REM nvigi_core layout: platform/config e.g. out\x64\%Cfg% or bin\x64\%Cfg%
if exist "nvigi_core\out\x64\%Cfg%\" (
    copy /Y nvigi_core\out\x64\%Cfg%\*.dll bin\x64\%Cfg%\
) else (
    copy /Y nvigi_core\bin\x64\%Cfg%\*.dll bin\x64\%Cfg%\
)

for /D %%b in ("_artifacts\*") do (
    for %%c in ("%%~b\x64\%Cfg%\*.dll") do (
        echo COPYING %%c to bin\x64\%Cfg%\
        copy %%c bin\x64\%Cfg%\
    )
    for %%c in ("%%~b\x64\%Cfg%\*.exe") do (
        echo COPYING %%c to bin\x64\%Cfg%\
        copy %%c bin\x64\%Cfg%\
    )
)

REM Custom for nvigi.3d
for %%c in ("_artifacts\nvigi.3d\x64\%Cfg%\tts\*.*") do (
    copy %%c bin\x64\%Cfg%\
)

robocopy /s _artifacts\nvigi.3d\x64\%Cfg%\shaders bin\x64\%Cfg%\shaders

if not exist "source\plugins" goto :eof

echo Copying Plugin Headers
if not exist "include" mkdir include
pushd source\plugins
for /R %%f in (nvigi_*.h) do (
    echo COPYING %%f to include
        copy %%f ..\..\include\
    )
)
popd

:eof
