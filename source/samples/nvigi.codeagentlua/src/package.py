# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for nvigi.codeagentlua sample
# This file is imported by tools/packaging/package.py

def get_components(externals):
    """
    Returns component definitions for this sample.
    Args:
        externals: dict of external package definitions from main package.py
    """
    win_plat = ['win-x64']
    
    components = {
        'codeagentlua': {
            'platforms': win_plat,
            'exes': ['nvigi.codeagentlua'],
            'sources': ['samples/nvigi.codeagentlua'],
            'premake': 'source/samples/nvigi.codeagentlua/src/premake.lua',
            'data': ['nvigi.codeagentlua'],
        }
    }
    
    return components

