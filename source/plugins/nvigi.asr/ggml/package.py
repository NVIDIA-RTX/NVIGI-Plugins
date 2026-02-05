# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for ASR GGML plugins
# This file is imported by tools/packaging/package.py

def get_components(externals):
    """
    Returns component definitions for this plugin.
    Args:
        externals: dict of external package definitions from main package.py
    """
    all_plat = ['win-x64']
    win_plat = ['win-x64']
    
    d3d12dlls = externals['d3d12dlls']
    cuda12dlls = externals['cuda12dlls']
    agilityDlls = externals['agilityDlls']
    
    # Define externals specific to ASR plugin family
    whispercpp_ext = {"dep":"whispercpp", "path":"external/whisper.cpp",
                      "items":[{"name":"whispercpp", "version":"a329b6a_x64-windows","platforms":"windows-x86_64"},
                               {"name":"whispercpp", "version":"4600650_x64-linux","platforms":"linux-x86_64"}
                              ]
                     }
    
    whispercpp_d3d12_ext = {"dep":"whispercpp-d3d12", "path":"external/whisper.cpp-d3d12",
                            "items":[{"name":"whispercpp", "version":"03bee8e_x64-windows_d3d12","platforms":"windows-x86_64"},
                                     {"name":"whispercpp", "version":"4600650_x64-linux","platforms":"linux-x86_64"}
                                    ]
                           }
    
    components = {
        'asr.ggml.cpu': {
            'platforms': win_plat,
            'sharedlib': ['nvigi.plugin.asr.ggml.cpu'],
            'docs': ['ProgrammingGuideASRWhisper.md'],
            'includes': ['source/plugins/nvigi.asr/nvigi_asr_whisper.h'],
            'sources': ['plugins/nvigi.asr/nvigi_asr_whisper.h', 'plugins/nvigi.asr/ggml', 'shared'],
            'externals': [externals['nlohmann_json_ext'], whispercpp_ext],
            'premake': 'source/plugins/nvigi.asr/ggml/premake.lua',
            'models': ['nvigi.plugin.asr.ggml'],
            'data': ['nvigi.asr'],
            'public_models': ['{5CAD3A03-1272-4D43-9F3D-655417526170}']
        },
        'asr.ggml.d3d12': {
            'platforms': win_plat,
            '3rdparty': d3d12dlls,
            '3rdparty_private': [agilityDlls, externals['dxc_redist_ext']],
            'sharedlib': ['nvigi.plugin.asr.ggml.d3d12'],
            'docs': ['ProgrammingGuideASRWhisper.md'],
            'includes': ['source/plugins/nvigi.asr/nvigi_asr_whisper.h'],
            'sources': ['plugins/nvigi.asr/nvigi_asr_whisper.h', 'plugins/nvigi.asr/ggml', 'shared'],
            'externals': [externals['nlohmann_json_ext'], whispercpp_ext, externals['agility_sdk_ext']],
            'externals_private': [externals['agility_sdk_redist_ext'], whispercpp_d3d12_ext, externals['dxc_redist_ext']],
            'premake': 'source/plugins/nvigi.asr/ggml/premake.lua',
            'models': ['nvigi.plugin.asr.ggml'],
            'data': ['nvigi.asr'],
            'public_models': ['{5CAD3A03-1272-4D43-9F3D-655417526170}']
        },
        'asr.ggml.cuda': {
            'platforms': all_plat,
            '3rdparty': cuda12dlls,
            '3rdparty_private': agilityDlls,
            'sharedlib': ['nvigi.plugin.asr.ggml.cuda'],
            'docs': ['ProgrammingGuideASRWhisper.md'],
            'includes': ['source/plugins/nvigi.asr/nvigi_asr_whisper.h'],
            'sources': ['plugins/nvigi.asr/nvigi_asr_whisper.h', 'plugins/nvigi.asr/ggml', 'shared'],
            'externals': [externals['nlohmann_json_ext'], whispercpp_ext, externals['cuda_ext']],
            'premake': 'source/plugins/nvigi.asr/ggml/premake.lua',
            'models': ['nvigi.plugin.asr.ggml'],
            'data': ['nvigi.asr'],
            'public_models': ['{5CAD3A03-1272-4D43-9F3D-655417526170}']
        },
        'asr.ggml.vk': {
            'platforms': all_plat,
            'sharedlib': ['nvigi.plugin.asr.ggml.vk'],
            'docs': ['ProgrammingGuideASRWhisper.md'],
            'includes': ['source/plugins/nvigi.asr/nvigi_asr_whisper.h'],
            'sources': ['plugins/nvigi.asr/nvigi_asr_whisper.h', 'plugins/nvigi.asr/ggml', 'shared'],
            'externals': [externals['vulkan_ext'], externals['nlohmann_json_ext'], whispercpp_ext],
            'premake': 'source/plugins/nvigi.asr/ggml/premake.lua',
            'models': ['nvigi.plugin.asr.ggml'],
            'data': ['nvigi.asr'],
            'public_models': ['{5CAD3A03-1272-4D43-9F3D-655417526170}']
        }
    }
    
    return components

