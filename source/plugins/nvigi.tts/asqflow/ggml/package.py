# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for TTS ASqFlow GGML plugins
# This file is imported by tools/packaging/package.py

def get_components(externals):
    """
    Returns component definitions for this plugin.
    Args:
        externals: dict of external package definitions from main package.py
    """
    all_plat = ['win-x64']
    win_plat = ['win-x64']
    
    # Define externals specific to TTS ASqFlow GGML plugin family
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
    
    tts_exts = [{"dep":"SimpleFarForTTS", "path":"external/SimpleFarForTTS",
                 "items":[{"name":"SimpleFarForTTS", "version":"0.1.2-windows","platforms":"windows-x86_64"},
                          {"name":"SimpleFarForTTS", "version":"0.1.2-linux","platforms":"linux-x86_64"}
                         ]}
               ]
    
    asqflowcpp_ext = {"dep":"asqflowcpp", "path":"external/asqflow.cpp",
                      "items":[{"name":"asqflowcpp", "version":"ea3dba0_x64-windows","platforms":"windows-x86_64"},
                               {"name":"asqflowcpp", "version":"e47974e_x64-linux","platforms":"linux-x86_64"}
                              ]
                     }
    
    asqflowcpp_d3d12_ext = {"dep":"asqflowcpp-d3d12", "path":"external/asqflow.cpp-d3d12",
                            "items":[{"name":"asqflowcpp", "version":"ea3dba0_x64-windows_d3d12","platforms":"windows-x86_64"},
                                     {"name":"asqflowcpp", "version":"e47974e_x64-linux","platforms":"linux-x86_64"}
                                    ]
                           }
    
    asqflow_tts_dlls_ggml = {'win-x64':[
        'external/SimpleFarForTTS/x64/Release/RivaNormalizer.dll'] + cuptiDlls['win-x64'] + cuda12dlls['win-x64'],
        'linux-x64':[
        'external/SimpleFarForTTS/lib/Release/libRivaNormalizer.so.1.0.0'] + cuda12dlls['linux-x64']
    }
    
    components = {
        'tts.asqflow-ggml.cuda': {
            'platforms': all_plat,
            '3rdparty': asqflow_tts_dlls_ggml,
            'docs': ["ProgrammingGuideTTSASqFlow.md"],
            'sharedlib': ['nvigi.plugin.tts.asqflow-ggml.cuda'],
            'includes': ['source/plugins/nvigi.tts/nvigi_tts.h'],
            'models': ['nvigi.plugin.tts.asqflow-ggml'],
            'public_models': ['{16EEB8EA-55A8-4F40-BECE-CE995AF44101}', '{3D52FDC0-5B6D-48E1-B108-84D308818602}'],
            'sources': ['plugins/nvigi.tts/nvigi_tts.h', 'plugins/nvigi.tts/asqflow/ggml'],
            'externals': [externals['nlohmann_json_ext'], externals['cuda_ext'], asqflowcpp_ext] + tts_exts,
            'premake': 'source/plugins/nvigi.tts/asqflow/ggml/premake.lua',
            'data': ['nvigi.tts']
        },
        'tts.asqflow-ggml.vk': {
            'platforms': all_plat,
            'docs': ["ProgrammingGuideTTSASqFlow.md"],
            'sharedlib': ['nvigi.plugin.tts.asqflow-ggml.vk'],
            'includes': ['source/plugins/nvigi.tts/nvigi_tts.h'],
            'models': ['nvigi.plugin.tts.asqflow-ggml'],
            'public_models': ['{16EEB8EA-55A8-4F40-BECE-CE995AF44101}', '{3D52FDC0-5B6D-48E1-B108-84D308818602}'],
            'sources': ['plugins/nvigi.tts/nvigi_tts.h', 'plugins/nvigi.tts/asqflow/ggml'],
            'externals': [externals['nlohmann_json_ext'], externals['vulkan_ext'], asqflowcpp_ext] + tts_exts,
            'premake': 'source/plugins/nvigi.tts/asqflow/ggml/premake.lua',
            'data': ['nvigi.tts']
        },
        'tts.asqflow-ggml.d3d12': {
            'platforms': win_plat,
            '3rdparty_private': externals['dxc_redist_ext'],
            'docs': ["ProgrammingGuideTTSASqFlow.md"],
            'sharedlib': ['nvigi.plugin.tts.asqflow-ggml.d3d12'],
            'includes': ['source/plugins/nvigi.tts/nvigi_tts.h'],
            'models': ['nvigi.plugin.tts.asqflow-ggml'],
            'public_models': ['{16EEB8EA-55A8-4F40-BECE-CE995AF44101}', '{3D52FDC0-5B6D-48E1-B108-84D308818602}'],
            'sources': ['plugins/nvigi.tts/nvigi_tts.h', 'plugins/nvigi.tts/asqflow/ggml'],
            'externals': [externals['nlohmann_json_ext'], asqflowcpp_ext] + tts_exts,
            'externals_private': [externals['agility_sdk_redist_ext'], asqflowcpp_d3d12_ext, externals['dxc_redist_ext']],
            'premake': 'source/plugins/nvigi.tts/asqflow/ggml/premake.lua',
            'data': ['nvigi.tts']
        }
    }
    
    return components

