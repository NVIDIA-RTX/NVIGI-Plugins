# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for AI Pipeline plugin
# This file is imported by tools/packaging/package.py
# Component platforms default to win-x64; extend when adding targets (see ALL_PLATFORMS).

def get_components(externals):
    """
    Returns component definitions for this plugin.
    Args:
        externals: dict of external package definitions from main package.py (includes DEFAULT_COMPONENT_PLATFORMS, ALL_PLATFORMS).
    """
    supported_platforms = externals['ALL_PLATFORMS']

    components = {
        'ai.pipeline': {
            'platforms': supported_platforms,
            'sharedlib': ['nvigi.plugin.ai.pipeline'],
            'includes': ['source/plugins/nvigi.aip/nvigi_aip.h'],
            'sources': ['plugins/nvigi.aip', 'shared'],
            'externals': [externals['nvtx_ext']],
            'premake': 'source/plugins/nvigi.aip/premake.lua'
        }
    }
    
    return components

