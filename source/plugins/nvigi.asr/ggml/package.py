# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for ASR GGML plugins
# This file is imported by tools/packaging/package.py
# Component platforms default to win-x64; extend when adding targets (see ALL_PLATFORMS).

def get_components(externals):
    """
    Returns component definitions for this plugin.
    Args:
        externals: dict of external package definitions from main package.py (includes DEFAULT_COMPONENT_PLATFORMS, ALL_PLATFORMS).
    """
    supported_platforms = externals['ALL_PLATFORMS']

    d3d12dlls = externals['d3d12dlls']
    cudaDlls = externals['cudaDlls']
    agilityDlls = externals['agilityDlls']
    
    # Packman rows for this plugin family (x64).
    whispercpp_ext = [
        {"dep":"whispercpp", "target":"win-x64", "path":"external/whisper.cpp",
         "name":"whispercpp", "version":"98c55af_x64-windows", "host_platform":"windows-x86_64"},
    ]

    whispercpp_d3d12_ext = [
        {"dep":"whispercpp-d3d12", "target":"win-x64", "path":"external/whisper.cpp-d3d12",
         "name":"whispercpp", "version":"98c55af_x64-windows_d3d12", "host_platform":"windows-x86_64"},
    ]
    
    nvapi_ext = [{"dep":"nvapi", "target":"win-x64", "path":"external/nvapi",
                  "name":"nvapi", "version":"r575-public-windows-x86_64", "host_platform":"windows-x86_64"}]

    components = {
        'asr.ggml.cpu': {
            'platforms': supported_platforms,
            'sharedlib': ['nvigi.plugin.asr.ggml.cpu'],
            'docs': ['ProgrammingGuideASRWhisper.md'],
            'includes': ['source/plugins/nvigi.asr/nvigi_asr_whisper.h'],
            'sources': ['plugins/nvigi.asr/nvigi_asr_whisper.h', 'plugins/nvigi.asr/ggml', 'shared'],
            'externals': [externals['nlohmann_json_ext'], whispercpp_ext, externals['nvtx_ext']],
            'premake': 'source/plugins/nvigi.asr/ggml/premake.lua',
            'models': ['nvigi.plugin.asr.ggml'],
            'data': ['nvigi.asr'],
            'public_models': ['{5CAD3A03-1272-4D43-9F3D-655417526170}']
        },
        'asr.ggml.d3d12': {
            'platforms': supported_platforms,
            '3rdparty': d3d12dlls,
            '3rdparty_private': agilityDlls,
            'sharedlib': ['nvigi.plugin.asr.ggml.d3d12'],
            'docs': ['ProgrammingGuideASRWhisper.md'],
            'includes': ['source/plugins/nvigi.asr/nvigi_asr_whisper.h'],
            'sources': ['plugins/nvigi.asr/nvigi_asr_whisper.h', 'plugins/nvigi.asr/ggml', 'shared'],
            'externals': [externals['nlohmann_json_ext'], whispercpp_ext, externals['agility_sdk_ext'], externals['nvtx_ext']],
            'externals_private': [externals['agility_sdk_redist_ext'], whispercpp_d3d12_ext, externals['dxc_redist_ext']],
            'premake': 'source/plugins/nvigi.asr/ggml/premake.lua',
            'models': ['nvigi.plugin.asr.ggml'],
            'data': ['nvigi.asr'],
            'public_models': ['{5CAD3A03-1272-4D43-9F3D-655417526170}']
        },
        'asr.ggml.cuda': {
            'platforms': supported_platforms,
            '3rdparty': cudaDlls,
            '3rdparty_private': agilityDlls,
            'sharedlib': ['nvigi.plugin.asr.ggml.cuda'],
            'docs': ['ProgrammingGuideASRWhisper.md'],
            'includes': ['source/plugins/nvigi.asr/nvigi_asr_whisper.h'],
            'sources': ['plugins/nvigi.asr/nvigi_asr_whisper.h', 'plugins/nvigi.asr/ggml', 'shared'],
            'externals': [externals['nlohmann_json_ext'], whispercpp_ext, externals['cuda_ext'], externals['nvtx_ext']],
            'premake': 'source/plugins/nvigi.asr/ggml/premake.lua',
            'models': ['nvigi.plugin.asr.ggml'],
            'data': ['nvigi.asr'],
            'public_models': ['{5CAD3A03-1272-4D43-9F3D-655417526170}']
        },
        'asr.ggml.vk': {
            'platforms': supported_platforms,
            'sharedlib': ['nvigi.plugin.asr.ggml.vk'],
            'docs': ['ProgrammingGuideASRWhisper.md'],
            'includes': ['source/plugins/nvigi.asr/nvigi_asr_whisper.h'],
            'sources': ['plugins/nvigi.asr/nvigi_asr_whisper.h', 'plugins/nvigi.asr/ggml', 'shared'],
            'externals': [externals['vulkan_ext'], externals['nlohmann_json_ext'], whispercpp_ext, externals['nvtx_ext']],
            'premake': 'source/plugins/nvigi.asr/ggml/premake.lua',
            'models': ['nvigi.plugin.asr.ggml'],
            'data': ['nvigi.asr'],
            'public_models': ['{5CAD3A03-1272-4D43-9F3D-655417526170}']
        }
    }
    
    return components

