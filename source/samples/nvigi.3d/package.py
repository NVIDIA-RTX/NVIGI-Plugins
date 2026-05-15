# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for nvigi.3d sample
# This file is imported by tools/packaging/package.py
# Component platforms default to win-x64; extend when adding targets (see ALL_PLATFORMS).

def get_components(externals):
    """
    Returns component definitions for this sample.
    Args:
        externals: dict of external package definitions from main package.py (includes DEFAULT_COMPONENT_PLATFORMS, ALL_PLATFORMS).
    """
    supported_platforms = externals['ALL_PLATFORMS']

    # Define externals specific to this component
    donut_prebuilt_ext = [
        {"dep":"donut_prebuilt", "target":"win-x64", "path":"external/donut",
         "name":"donut-prebuild-x86-x64", "version":"aa8412f-20250811_2","host_platform":"windows-x86_64"},
    ]
    
    components = {
        '3d': {
            'platforms': supported_platforms,
            'exes': ['nvigi.3d'],
            'bin_extras': [('shaders', 'shaders'), ('tts', '')],
            'sources': ['samples/nvigi.3d',
                        'samples/shared/cxx_wrappers'],
            'sources_skip': ['opt-local-donut/_bin', 'opt-local-donut/_build', 'opt-local-donut/_package', 'opt-local-donut/Donut'],
            'externals': [donut_prebuilt_ext, externals['nvtx_ext']],
            'premake': 'source/samples/nvigi.3d/premake.lua',
            'data': ['nvigi.3d'],
        }
    }
    
    return components

