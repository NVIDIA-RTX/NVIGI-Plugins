# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Component packaging configuration for GPT Cloud REST plugin
# This file is imported by tools/packaging/package.py
# Component platforms default to win-x64; extend when adding targets (see ALL_PLATFORMS).

def get_components(externals):
    """
    Returns component definitions for this plugin.
    Args:
        externals: dict of external package definitions from main package.py (includes DEFAULT_COMPONENT_PLATFORMS, ALL_PLATFORMS).
    """
    supported_platforms = externals['ALL_PLATFORMS']

    components = {
        'gpt.cloud.rest': {
            'platforms': supported_platforms,
            'sharedlib': ['nvigi.plugin.gpt.cloud.rest'],
            'docs': ['ProgrammingGuideGPT.md'],
            'includes': ['source/plugins/nvigi.gpt/nvigi_gpt.h'],
            'scripts': [],
            'sources': ['plugins/nvigi.gpt/nvigi_gpt.h', 'plugins/nvigi.gpt/rest', 'shared'],
            'premake': 'source/plugins/nvigi.gpt/rest/premake.lua',
            'models': ['nvigi.plugin.gpt.cloud'],
            'public_models': ['{01F43B70-CE23-42CA-9606-74E80C5ED0B6}',
                              '{8E31808B-C182-4016-9ED8-64804FF5B40D}',
                              '{E9102ACB-8CD8-4345-BCBF-CCF6DC758E58}']
        }
    }
    
    return components

