# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for AI Pipeline plugin
# This file is imported by tools/packaging/package.py

def get_components(externals):
    """
    Returns component definitions for this plugin.
    Args:
        externals: dict of external package definitions from main package.py
    """
    win_plat = ['win-x64']
    
    components = {
        'ai.pipeline': {
            'platforms': win_plat,
            'sharedlib': ['nvigi.plugin.ai.pipeline'],
            'includes': ['source/plugins/nvigi.aip/nvigi_aip.h'],
            'sources': ['plugins/nvigi.aip', 'shared'],
            'premake': 'source/plugins/nvigi.aip/premake.lua'
        }
    }
    
    return components

