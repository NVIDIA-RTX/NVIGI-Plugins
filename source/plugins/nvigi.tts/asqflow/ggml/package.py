# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for TTS ASqFlow GGML plugins
# This file is imported by tools/packaging/package.py
# Component platforms default to win-x64; extend when adding targets (see ALL_PLATFORMS).

def get_components(externals):
    """
    Returns component definitions for this plugin.
    Args:
        externals: dict of external package definitions from main package.py (includes DEFAULT_COMPONENT_PLATFORMS, ALL_PLATFORMS).
    """
    supported_platforms = externals['DEFAULT_COMPONENT_PLATFORMS']

    # Define externals specific to TTS ASqFlow GGML plugin family
    cuptiDlls = externals['cuptiDlls']
    cudaDlls = externals['cudaDlls']

    # Packman rows for this plugin family (x64).
    tts_exts = [
        {"dep":"SimpleFarForTTS", "target":"win-x64", "path":"external/SimpleFarForTTS",
         "name":"SimpleFarForTTS", "version":"0.1.2-windows", "host_platform":"windows-x86_64"},
    ]

    asqflowcpp_ext = [
        {"dep":"asqflowcpp", "target":"win-x64", "path":"external/asqflow.cpp",
         "name":"asqflowcpp", "version":"f655bcc_x64-windows", "host_platform":"windows-x86_64"},
    ]

    asqflowcpp_d3d12_ext = [
        {"dep":"asqflowcpp-d3d12", "target":"win-x64", "path":"external/asqflow.cpp-d3d12",
         "name":"asqflowcpp", "version":"f655bcc_x64-windows_d3d12", "host_platform":"windows-x86_64"},
    ]
    
    asqflow_tts_dlls_ggml = {'win-x64':[
        'external/SimpleFarForTTS/x64/Release/RivaNormalizer.dll'] + cuptiDlls['win-x64'] + cudaDlls['win-x64'],

    }
    
    components = {
        'tts.asqflow-ggml.cuda': {
            'platforms': supported_platforms,
            '3rdparty': asqflow_tts_dlls_ggml,
            'docs': ["ProgrammingGuideTTSASqFlow.md"],
            'sharedlib': ['nvigi.plugin.tts.asqflow-ggml.cuda'],
            'includes': ['source/plugins/nvigi.tts/nvigi_tts.h'],
            'models': ['nvigi.plugin.tts.asqflow-ggml'],
            'public_models': ['{16EEB8EA-55A8-4F40-BECE-CE995AF44101}', '{3D52FDC0-5B6D-48E1-B108-84D308818602}'],
            'sources': ['plugins/nvigi.tts/nvigi_tts.h', 'plugins/nvigi.tts/asqflow/ggml'],
            'externals': [externals['nlohmann_json_ext'], externals['cuda_ext'], externals['cig_scheduler_settings_ext'], asqflowcpp_ext, externals['nvtx_ext']] + tts_exts,
            'premake': 'source/plugins/nvigi.tts/asqflow/ggml/premake.lua',
            'data': ['nvigi.tts']
        },
        'tts.asqflow-ggml.vk': {
            'platforms': externals['DEFAULT_COMPONENT_PLATFORMS'],
            'docs': ["ProgrammingGuideTTSASqFlow.md"],
            'sharedlib': ['nvigi.plugin.tts.asqflow-ggml.vk'],
            'includes': ['source/plugins/nvigi.tts/nvigi_tts.h'],
            'models': ['nvigi.plugin.tts.asqflow-ggml'],
            'public_models': ['{16EEB8EA-55A8-4F40-BECE-CE995AF44101}', '{3D52FDC0-5B6D-48E1-B108-84D308818602}'],
            'sources': ['plugins/nvigi.tts/nvigi_tts.h', 'plugins/nvigi.tts/asqflow/ggml'],
            'externals': [externals['nlohmann_json_ext'], externals['vulkan_ext'], asqflowcpp_ext, externals['nvtx_ext']] + tts_exts,
            'premake': 'source/plugins/nvigi.tts/asqflow/ggml/premake.lua',
            'data': ['nvigi.tts']
        },
        'tts.asqflow-ggml.d3d12': {
            'platforms': supported_platforms,
            'docs': ["ProgrammingGuideTTSASqFlow.md"],
            'sharedlib': ['nvigi.plugin.tts.asqflow-ggml.d3d12'],
            'includes': ['source/plugins/nvigi.tts/nvigi_tts.h'],
            'models': ['nvigi.plugin.tts.asqflow-ggml'],
            'public_models': ['{16EEB8EA-55A8-4F40-BECE-CE995AF44101}', '{3D52FDC0-5B6D-48E1-B108-84D308818602}'],
            'sources': ['plugins/nvigi.tts/nvigi_tts.h', 'plugins/nvigi.tts/asqflow/ggml'],
            'externals': [externals['nlohmann_json_ext'], asqflowcpp_ext, externals['nvtx_ext']] + tts_exts,
            'externals_private': [externals['agility_sdk_redist_ext'], asqflowcpp_d3d12_ext, externals['dxc_redist_ext']],
            'premake': 'source/plugins/nvigi.tts/asqflow/ggml/premake.lua',
            'data': ['nvigi.tts']
        }
    }
    
    return components

