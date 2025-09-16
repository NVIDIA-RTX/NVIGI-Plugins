@echo off
setlocal enabledelayedexpansion

if exist "tools\cmake-windows-x86_64\bin\cmake.exe" (
    set CMAKE_LOCAL=tools\cmake-windows-x86_64\bin\cmake.exe 
) else (
    set CMAKE_LOCAL=cmake
)

mkdir _bin
echo Building and installing for all configurations...
echo.

set CONFIGS=Release Production Debug

for %%c in (%CONFIGS%) do (
    echo ========================================
    echo Building configuration: %%c
    echo ========================================
    
    %CMAKE_LOCAL% --build _build --config %%c --target donut_app donut_core donut_engine donut_render 
    if !errorlevel! neq 0 (
        echo ERROR: Build failed for configuration %%c
        exit /b !errorlevel!
    )
    
    echo.
    echo ========================================
    echo Installing configuration: %%c
    echo ========================================
    
    %CMAKE_LOCAL% --install _build --config %%c
    if !errorlevel! neq 0 (
        echo ERROR: Install failed for configuration %%c
        exit /b !errorlevel!
    )
    
    echo.
    echo Configuration %%c completed successfully.
    echo.
)

echo ========================================
echo All configurations built and installed successfully!
echo ========================================