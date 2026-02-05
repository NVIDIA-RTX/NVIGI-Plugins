# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for nvigi.basic sample
# This file is imported by tools/packaging/package.py

def get_components(externals):
    """
    Returns component definitions for this sample.
    Args:
        externals: dict of external package definitions from main package.py
    """
    win_plat = ['win-x64']
    
    components = {
        'basic': {
            'platforms': win_plat,
            'exes': ['nvigi.basic'],
            'sources': ['samples/nvigi.basic'],
            'premake': 'source/samples/nvigi.basic/premake.lua'
        }
    }
    
    return components

