# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for nvigi.fcall sample
# This file is imported by tools/packaging/package.py
# Component platforms default to win-x64; extend when adding targets (see ALL_PLATFORMS).

def get_components(externals):
    """
    Returns component definitions for this sample.
    Args:
        externals: dict of external package definitions from main package.py (includes DEFAULT_COMPONENT_PLATFORMS, ALL_PLATFORMS).
    """
    supported_platforms = externals['DEFAULT_COMPONENT_PLATFORMS']

    # Define externals specific to this component
    curl_ext = [
        {"dep":"libcurl", "target":"win-x64", "path":"external/libcurl",
        "name":"libcurl", "version":"7.80.0+nv1-windows-x86_64","host_platform":"windows-x86_64"},
    ]
    
    components = {
        'fcall': {
            'platforms': supported_platforms,
            'exes': ['nvigi.fcall'],
            'sources': ['samples/nvigi.fcall'],
            'externals': [externals['nlohmann_json_ext'], curl_ext, externals['zlib_ext']],
            'premake': 'source/samples/nvigi.fcall/premake.lua'
        }
    }
    
    return components

