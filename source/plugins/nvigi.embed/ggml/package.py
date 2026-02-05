# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for Embed GGML plugins
# This file is imported by tools/packaging/package.py

def get_components(externals):
    """
    Returns component definitions for this plugin.
    Args:
        externals: dict of external package definitions from main package.py
    """
    all_plat = ['win-x64']
    
    # Define externals specific to Embed plugin family
    llamacpp_ext = {"dep":"llamacpp", "path":"external/llama.cpp",
                    "items":[{"name":"llamacpp", "version":"40c45b4_x64-windows","platforms":"windows-x86_64"},
                             {"name":"llamacpp", "version":"e6b2933_x64-linux","platforms":"linux-x86_64"}
                            ]
                   }
    
    components = {
        'embed.ggml.cuda': {
            'platforms': all_plat,
            'sharedlib': ['nvigi.plugin.embed.ggml.cuda'],
            'docs': ['ProgrammingGuideEmbed.md'],
            'includes': ['source/plugins/nvigi.embed/nvigi_embed.h'],
            'sources': ['plugins/nvigi.embed/nvigi_embed.h', 'plugins/nvigi.embed/ggml', 'shared'],
            'premake': 'source/plugins/nvigi.embed/ggml/premake.lua',
            'externals': [externals['nlohmann_json_ext'], llamacpp_ext, externals['cuda_ext']],
            'models': ['nvigi.plugin.embed.ggml'],
            'public_models': ['{5D458A64-C62E-4A9C-9086-2ADBF6B241C7}']
        },
        'embed.ggml.cpu': {
            'platforms': all_plat,
            'sharedlib': ['nvigi.plugin.embed.ggml.cpu'],
            'docs': ['ProgrammingGuideEmbed.md'],
            'includes': ['source/plugins/nvigi.embed/nvigi_embed.h'],
            'sources': ['plugins/nvigi.embed/nvigi_embed.h', 'plugins/nvigi.embed/ggml', 'shared'],
            'premake': 'source/plugins/nvigi.embed/ggml/premake.lua',
            'externals': [externals['nlohmann_json_ext'], llamacpp_ext],
            'models': ['nvigi.plugin.embed.ggml'],
            'public_models': ['{5D458A64-C62E-4A9C-9086-2ADBF6B241C7}']
        }
    }
    
    return components

