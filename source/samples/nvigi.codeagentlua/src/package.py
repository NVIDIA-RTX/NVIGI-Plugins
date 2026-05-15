# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for nvigi.codeagentlua sample
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
        'codeagentlua': {
            'platforms': supported_platforms,
            'exes': ['nvigi.codeagentlua'],
            'sources': ['samples/nvigi.codeagentlua'],
            'premake': 'source/samples/nvigi.codeagentlua/src/premake.lua',
            'data': ['nvigi.codeagentlua'],
        }
    }
    
    return components

