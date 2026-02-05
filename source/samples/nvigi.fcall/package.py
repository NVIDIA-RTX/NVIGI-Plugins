# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for nvigi.fcall sample
# This file is imported by tools/packaging/package.py

def get_components(externals):
    """
    Returns component definitions for this sample.
    Args:
        externals: dict of external package definitions from main package.py
    """
    win_plat = ['win-x64']
    
    # Define externals specific to this component
    curl_ext = {"dep":"libcurl", "path":"external/libcurl",
                "items":[{"name":"libcurl", "version":"7.80.0+nv1-windows-x86_64","platforms":"windows-x86_64"},
                         {"name":"libcurl", "version":"8.1.2-3-linux-x86_64-static-release","platforms":"linux-x86_64"}
                        ]
               }
    
    components = {
        'fcall': {
            'platforms': win_plat,
            'exes': ['nvigi.fcall'],
            'sources': ['samples/nvigi.fcall'],
            'externals': [externals['nlohmann_json_ext'], curl_ext, externals['zlib_ext']],
            'premake': 'source/samples/nvigi.fcall/premake.lua'
        }
    }
    
    return components

