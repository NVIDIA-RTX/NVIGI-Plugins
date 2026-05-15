set Cfg=%1
if "%Cfg%" NEQ "Release" (
    if "%Cfg%" NEQ "Debug" (
        if "%Cfg%" NEQ "Production" (
            echo Script requires a single argument: the build config to copy.  Must be Release, Debug or Production
            exit /b 1
        )
    )
)
copy external\zlib\debug\bin\zlibd1.dll bin\x64\%Cfg%\
copy external\zlib\bin\zlib1.dll bin\x64\%Cfg%\
copy external\cuda\extras\CUPTI\lib64\cupti64_2025.1.0.dll bin\x64\%Cfg%\
copy external\cig_scheduler_settings\bin\Release_x64\cig_scheduler_settings.dll bin\x64\%Cfg%\
copy external\cuda\bin\cublas64_12.dll bin\x64\%Cfg%\
copy external\cuda\bin\cublasLt64_12.dll bin\x64\%Cfg%\
copy external\cuda\bin\cudart64_12.dll bin\x64\%Cfg%\
copy external\cuda\bin\nvrtc64_120_0.dll bin\x64\%Cfg%\
copy external\cuda\bin\nvrtc-builtins64_128.dll bin\x64\%Cfg%\
mkdir bin\x64\%Cfg%
copy external\cuda\bin\nvrtc-builtins64_128.dll bin\x64\%Cfg%\nvrtc-builtins.alt64_128.dll
copy external\dxc-redist\bin\x64\dxcompiler.dll bin\x64\%Cfg%\
copy external\dxc-redist\bin\x64\dxil.dll bin\x64\%Cfg%\
copy external\dxc-redist\bin\x64\dxcompiler.dll bin\x64\%Cfg%\
copy external\dxc-redist\bin\x64\dxil.dll bin\x64\%Cfg%\
copy external\SimpleFarForTTS\x64\Release\RivaNormalizer.dll bin\x64\%Cfg%\
copy external\cuda\extras\CUPTI\lib64\cupti64_2025.1.0.dll bin\x64\%Cfg%\
copy external\cig_scheduler_settings\bin\Release_x64\cig_scheduler_settings.dll bin\x64\%Cfg%\
copy external\cuda\bin\cublas64_12.dll bin\x64\%Cfg%\
copy external\cuda\bin\cublasLt64_12.dll bin\x64\%Cfg%\
copy external\cuda\bin\cudart64_12.dll bin\x64\%Cfg%\
copy external\cuda\bin\nvrtc64_120_0.dll bin\x64\%Cfg%\
copy external\cuda\bin\nvrtc-builtins64_128.dll bin\x64\%Cfg%\
mkdir bin\x64\%Cfg%
copy external\cuda\bin\nvrtc-builtins64_128.dll bin\x64\%Cfg%\nvrtc-builtins.alt64_128.dll
copy external\cuda\bin\cublas64_12.dll bin\x64\%Cfg%\
copy external\cuda\bin\cublasLt64_12.dll bin\x64\%Cfg%\
copy external\cuda\bin\cudart64_12.dll bin\x64\%Cfg%\
copy external\cuda\bin\nvrtc64_120_0.dll bin\x64\%Cfg%\
copy external\cuda\bin\nvrtc-builtins64_128.dll bin\x64\%Cfg%\
mkdir bin\x64\%Cfg%
copy external\cuda\bin\nvrtc-builtins64_128.dll bin\x64\%Cfg%\nvrtc-builtins.alt64_128.dll
copy external\dxc-redist\bin\x64\dxcompiler.dll bin\x64\%Cfg%\
copy external\dxc-redist\bin\x64\dxil.dll bin\x64\%Cfg%\
copy external\dxc-redist\bin\x64\dxcompiler.dll bin\x64\%Cfg%\
copy external\dxc-redist\bin\x64\dxil.dll bin\x64\%Cfg%\
copy external\SimpleFarForTTS\x64\Release\RivaNormalizer.dll bin\x64\%Cfg%\
copy external\cuda\extras\CUPTI\lib64\cupti64_2025.1.0.dll bin\x64\%Cfg%\
copy external\cig_scheduler_settings\bin\Release_x64\cig_scheduler_settings.dll bin\x64\%Cfg%\
copy external\cuda\bin\cublas64_12.dll bin\x64\%Cfg%\
copy external\cuda\bin\cublasLt64_12.dll bin\x64\%Cfg%\
copy external\cuda\bin\cudart64_12.dll bin\x64\%Cfg%\
copy external\cuda\bin\nvrtc64_120_0.dll bin\x64\%Cfg%\
copy external\cuda\bin\nvrtc-builtins64_128.dll bin\x64\%Cfg%\
mkdir bin\x64\%Cfg%
copy external\cuda\bin\nvrtc-builtins64_128.dll bin\x64\%Cfg%\nvrtc-builtins.alt64_128.dll
mkdir bin\x64\%Cfg%\D3D12
copy external\agility-sdk-redist\build\native\bin\x64\D3D12Core.dll bin\x64\%Cfg%\D3D12
mkdir bin\x64\%Cfg%\D3D12
copy external\agility-sdk-redist\build\native\bin\x64\d3d12SDKLayers.dll bin\x64\%Cfg%\D3D12
mkdir bin\x64\%Cfg%\D3D12
copy external\agility-sdk-redist\build\native\bin\x64\D3D12Core.dll bin\x64\%Cfg%\D3D12
mkdir bin\x64\%Cfg%\D3D12
copy external\agility-sdk-redist\build\native\bin\x64\d3d12SDKLayers.dll bin\x64\%Cfg%\D3D12
mkdir bin\x64\%Cfg%\D3D12
copy external\agility-sdk-redist\build\native\bin\x64\D3D12Core.dll bin\x64\%Cfg%\D3D12
mkdir bin\x64\%Cfg%\D3D12
copy external\agility-sdk-redist\build\native\bin\x64\d3d12SDKLayers.dll bin\x64\%Cfg%\D3D12
mkdir bin\x64\%Cfg%\D3D12
copy external\agility-sdk-redist\build\native\bin\x64\D3D12Core.dll bin\x64\%Cfg%\D3D12
mkdir bin\x64\%Cfg%\D3D12
copy external\agility-sdk-redist\build\native\bin\x64\d3d12SDKLayers.dll bin\x64\%Cfg%\D3D12
