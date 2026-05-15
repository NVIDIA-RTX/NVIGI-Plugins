# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for nvigi.basic.tts.cxx sample
# This file is imported by tools/packaging/package.py
# Component platforms default to win-x64; extend when adding targets (see ALL_PLATFORMS).

def get_components(externals):
    """
    Returns component definitions for this sample.
    Args:
        externals: dict of external package definitions from main package.py (includes DEFAULT_COMPONENT_PLATFORMS, ALL_PLATFORMS).
    """
    supported_platforms = externals['ALL_PLATFORMS']

    components = {
        'basic.tts.cxx': {
            'platforms': supported_platforms,
            'exes': ['nvigi.basic.tts.cxx'],
            'sources': ['samples/nvigi.basic.cxx/tts',
                        'samples/shared/cxx_wrappers'],
            'premake': 'source/samples/nvigi.basic.cxx/tts/premake.lua',
            'externals': [externals['vulkan_ext']]
        }
    }
    
    return components

