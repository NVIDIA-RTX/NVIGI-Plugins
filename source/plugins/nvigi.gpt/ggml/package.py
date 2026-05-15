# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for GPT GGML plugins
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

    # Packman rows for this plugin family (x64).
    llamacpp_ext = [
        {"dep":"llamacpp", "target":"win-x64", "path":"external/llama.cpp",
         "name":"llamacpp", "version":"f5d4e5e_x64-windows", "host_platform":"windows-x86_64"},
    ]

    llamacpp_d3d12_ext = [
        {"dep":"llamacpp-d3d12", "target":"win-x64", "path":"external/llama.cpp-d3d12",
         "name":"llamacpp", "version":"f5d4e5e_x64-windows_d3d12", "host_platform":"windows-x86_64"},
    ]

    nvapi_ext = [
        {"dep":"nvapi", "target":"win-x64", "path":"external/nvapi",
         "name":"nvapi", "version":"r575-public-windows-x86_64", "host_platform":"windows-x86_64"},
    ]
    
    components = {
        'gpt.ggml.vk': {
            'platforms': supported_platforms,
            'sharedlib': ['nvigi.plugin.gpt.ggml.vk'],
            'docs': ['ProgrammingGuideGPT.md'],
            'includes': ['source/plugins/nvigi.gpt/nvigi_gpt.h'],
            'sources': ['plugins/nvigi.gpt/nvigi_gpt.h', 'plugins/nvigi.gpt/ggml', 'shared'],
            'premake': 'source/plugins/nvigi.gpt/ggml/premake.lua', 
            'externals': [externals['nlohmann_json_ext'], llamacpp_ext, externals['vulkan_ext'], externals['nvtx_ext']],
            'models': ['nvigi.plugin.gpt.ggml'],
            'public_models': ['{01F43B70-CE23-42CA-9606-74E80C5ED0B6}',
                              '{8E31808B-C182-4016-9ED8-64804FF5B40D}']
        },
        'gpt.ggml.d3d12': {
            'platforms': supported_platforms,
            '3rdparty': d3d12dlls,
            'sharedlib': ['nvigi.plugin.gpt.ggml.d3d12'],
            'docs': ['ProgrammingGuideGPT.md'],
            'includes': ['source/plugins/nvigi.gpt/nvigi_gpt.h'],
            'sources': ['plugins/nvigi.gpt/nvigi_gpt.h', 'plugins/nvigi.gpt/ggml', 'shared'],
            'premake': 'source/plugins/nvigi.gpt/ggml/premake.lua', 
            'externals': [externals['nlohmann_json_ext'], llamacpp_ext, externals['agility_sdk_ext'], externals['nvtx_ext']],
            'externals_private': [externals['agility_sdk_redist_ext'], llamacpp_d3d12_ext, nvapi_ext, externals['dxc_redist_ext']],
            'models': ['nvigi.plugin.gpt.ggml'],
            'public_models': ['{01F43B70-CE23-42CA-9606-74E80C5ED0B6}',
                              '{8E31808B-C182-4016-9ED8-64804FF5B40D}']
        },
        'gpt.ggml.cuda': {
            'platforms': supported_platforms,
            'sharedlib': ['nvigi.plugin.gpt.ggml.cuda'],
            'docs': ['ProgrammingGuideGPT.md'],
            'includes': ['source/plugins/nvigi.gpt/nvigi_gpt.h'],
            'sources': ['plugins/nvigi.gpt/nvigi_gpt.h', 'plugins/nvigi.gpt/ggml', 'shared'],
            'premake': 'source/plugins/nvigi.gpt/ggml/premake.lua', 
            'externals': [externals['nlohmann_json_ext'], llamacpp_ext, externals['cuda_ext'], externals['cig_scheduler_settings_ext'], externals['nvtx_ext']],
            'models': ['nvigi.plugin.gpt.ggml'],
            'public_models': ['{01F43B70-CE23-42CA-9606-74E80C5ED0B6}',
                              '{545F7EC2-4C29-499B-8FC8-61720DF3C626}',
                              '{8E31808B-C182-4016-9ED8-64804FF5B40D}',
                              '{0BAEDD5C-F2CA-49AA-9892-621C40030D12}',
                              '{0BAEDD5C-F2CA-49AA-9892-621C40030D13}']
        },
        'gpt.ggml.cpu': {
            'platforms': supported_platforms,
            'sharedlib': ['nvigi.plugin.gpt.ggml.cpu'],
            'docs': ['ProgrammingGuideGPT.md'],
            'includes': ['source/plugins/nvigi.gpt/nvigi_gpt.h'],
            'sources': ['plugins/nvigi.gpt/nvigi_gpt.h', 'plugins/nvigi.gpt/ggml', 'shared'],
            'premake': 'source/plugins/nvigi.gpt/ggml/premake.lua',
            'externals': [externals['nlohmann_json_ext'], llamacpp_ext, externals['nvtx_ext']],
            'models': ['nvigi.plugin.gpt.ggml'],
            'public_models': ['{01F43B70-CE23-42CA-9606-74E80C5ED0B6}',
                              '{8E31808B-C182-4016-9ED8-64804FF5B40D}']
        }
    }
    
    return components

