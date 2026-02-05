# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for nvigi.pipeline sample
# This file is imported by tools/packaging/package.py

def get_components(externals):
    """
    Returns component definitions for this sample.
    Args:
        externals: dict of external package definitions from main package.py
    """
    win_plat = ['win-x64']
    
    components = {
        'pipeline': {
            'platforms': win_plat,
            'exes': ['nvigi.pipeline'],
            'sources': ['samples/nvigi.pipeline'],
            'premake': 'source/samples/nvigi.pipeline/premake.lua',
            'data': ['nvigi.rag']
        }
    }
    
    return components

