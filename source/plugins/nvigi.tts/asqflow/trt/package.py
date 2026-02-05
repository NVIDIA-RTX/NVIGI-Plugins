# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for TTS ASqFlow TRT plugin
# This file is imported by tools/packaging/package.py

def get_components(externals):
    """
    Returns component definitions for this plugin.
    Args:
        externals: dict of external package definitions from main package.py
    """
    win_plat = ['win-x64']
    
    # Define externals specific to this component
    cuptiDlls = {'win-x64':[
        'external/cuda/extras/CUPTI/lib64/cupti64_2025.1.0.dll',
        'external/cig_scheduler_settings/bin/Release_x64/cig_scheduler_settings.dll'],
    }
    
    cuda12dlls = {'win-x64':[
        'external/cuda/bin/cublas64_12.dll',
        'external/cuda/bin/cublasLt64_12.dll', 
        'external/cuda/bin/cudart64_12.dll'
        ],
                'linux-x64': [
        'external/cuda/lib64/libcublas.so*',
        'external/cuda/lib64/libcublasLt.so*']
    }
    
    trt_ext = [{"dep":"tensorrt", "path":"external/tensorrt",
               "items":[{"name":"tensorrt", "version":"TensorRT-10.10.0.31.Windows.win10.cuda-12.9_v2","platforms":"windows-x86_64"}]
              }]
    
    tts_exts = [{"dep":"SimpleFarForTTS", "path":"external/SimpleFarForTTS",
                 "items":[{"name":"SimpleFarForTTS", "version":"0.1.2-windows","platforms":"windows-x86_64"},
                          {"name":"SimpleFarForTTS", "version":"0.1.2-linux","platforms":"linux-x86_64"}
                         ]}
               ]
    
    ort_dml_exts = [
        {"dep":"Microsoft.AI.DirectML.1.15.4", "path":"external/Microsoft.AI.DirectML.1.15.4",
                    "items":[{"name":"Microsoft.AI.DirectML.1.15.4", "version":"1.15.4","platforms":"windows-x86_64"}]},
        {"dep":"microsoft.ml.onnxruntime.directml.1.20.1", "path":"external/microsoft.ml.onnxruntime.directml.1.20.1",
                    "items":[{"name":"microsoft.ml.onnxruntime.directml.1.20.1", "version":"1.20.1","platforms":"windows-x86_64"}]}
    ]
    
    asqflow_tts_dlls_trt = {'win-x64':[
        'external/microsoft.ml.onnxruntime.directml.1.20.1/runtimes/win-x64/native/onnxruntime.dll',
        'external/tensorrt/lib/nvinfer_10.dll',
        'external/cuda/bin/cudart64_12.dll',
        'external/SimpleFarForTTS/x64/Release/RivaNormalizer.dll'] + cuptiDlls['win-x64']
    }
    
    components = {
        'tts.asqflow-trt': {
            'platforms': win_plat,
            '3rdparty': asqflow_tts_dlls_trt,
            'docs': ["ProgrammingGuideTTSASqFlow.md"],
            'sharedlib': ['nvigi.plugin.tts.asqflow-trt'],
            'includes': ['source/plugins/nvigi.tts/nvigi_tts.h'],
            'models': ['nvigi.plugin.tts.asqflow-trt'],
            'public_models': ['{81320D1D-DF3C-4CFC-B9FA-4D3FF95FC35F}'],
            'sources': ['plugins/nvigi.tts/nvigi_tts.h', 'plugins/nvigi.tts/asqflow/trt'],
            'externals': ort_dml_exts + tts_exts + trt_ext + [externals['cuda_ext']],
            'premake': 'source/plugins/nvigi.tts/asqflow/trt/premake.lua',
            'data': ['nvigi.tts']
        }
    }
    
    return components

