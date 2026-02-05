# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for nvigi.basic.asr.cxx sample
# This file is imported by tools/packaging/package.py

def get_components(externals):
    """
    Returns component definitions for this sample.
    Args:
        externals: dict of external package definitions from main package.py
    """
    win_plat = ['win-x64']
    
    components = {
        'basic.asr.cxx': {
            'platforms': win_plat,
            'exes': ['nvigi.basic.asr.cxx'],
            'sources': ['samples/nvigi.basic.cxx/asr',
                        'samples/shared/cxx_wrappers'],
            'premake': 'source/samples/nvigi.basic.cxx/asr/premake.lua',
            'externals': [externals['vulkan_ext']]
        }
    }
    
    return components

