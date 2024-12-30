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
if not exist "bin\x64" mkdir bin\x64

copy /Y nvigi_core\bin\%Cfg%_x64\*.dll bin\x64\

for /D %%b in ("_artifacts\*") do (
    for %%c in ("%%~b\%Cfg%_x64\*.dll") do (
        echo COPYING %%c to bin\x64\
        copy %%c bin\x64\
    )
    for %%c in ("%%~b\%Cfg%_x64\*.exe") do (
        echo COPYING %%c to bin\x64\
        copy %%c bin\x64\
    )
)

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
