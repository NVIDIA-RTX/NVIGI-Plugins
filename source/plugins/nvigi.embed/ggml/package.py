# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for Embed GGML plugins
# This file is imported by tools/packaging/package.py
# Component platforms default to win-x64; extend when adding targets (see ALL_PLATFORMS).

def get_components(externals):
    """
    Returns component definitions for this plugin.
    Args:
        externals: dict of external package definitions from main package.py (includes DEFAULT_COMPONENT_PLATFORMS, ALL_PLATFORMS).
    """
    supported_platforms = externals['DEFAULT_COMPONENT_PLATFORMS']

    # Packman rows for this plugin family (x64).
    llamacpp_ext = [
        {"dep":"llamacpp", "target":"win-x64", "path":"external/llama.cpp",
         "name":"llamacpp", "version":"f5d4e5e_x64-windows", "host_platform":"windows-x86_64"},
    ]
    
    components = {
        'embed.ggml.cuda': {
            'platforms': supported_platforms,
            'sharedlib': ['nvigi.plugin.embed.ggml.cuda'],
            'docs': ['ProgrammingGuideEmbed.md'],
            'includes': ['source/plugins/nvigi.embed/nvigi_embed.h'],
            'sources': ['plugins/nvigi.embed/nvigi_embed.h', 'plugins/nvigi.embed/ggml', 'shared'],
            'premake': 'source/plugins/nvigi.embed/ggml/premake.lua',
            'externals': [externals['nlohmann_json_ext'], llamacpp_ext, externals['cuda_ext'], externals['nvtx_ext']],
            'models': ['nvigi.plugin.embed.ggml'],
            'public_models': ['{5D458A64-C62E-4A9C-9086-2ADBF6B241C7}']
        },
        'embed.ggml.cpu': {
            'platforms': supported_platforms,
            'sharedlib': ['nvigi.plugin.embed.ggml.cpu'],
            'docs': ['ProgrammingGuideEmbed.md'],
            'includes': ['source/plugins/nvigi.embed/nvigi_embed.h'],
            'sources': ['plugins/nvigi.embed/nvigi_embed.h', 'plugins/nvigi.embed/ggml', 'shared'],
            'premake': 'source/plugins/nvigi.embed/ggml/premake.lua',
            'externals': [externals['nlohmann_json_ext'], llamacpp_ext, externals['nvtx_ext']],
            'models': ['nvigi.plugin.embed.ggml'],
            'public_models': ['{5D458A64-C62E-4A9C-9086-2ADBF6B241C7}']
        }
    }
    
    return components

