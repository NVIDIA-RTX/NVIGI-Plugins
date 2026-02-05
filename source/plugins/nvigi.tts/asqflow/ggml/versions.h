// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "source/shared/sharedVersions.h"

#if defined(GGML_USE_VULKAN) || defined(GGML_USE_D3D12)
#define VERSION_MAJOR 0
#define VERSION_MINOR 5
#define VERSION_PATCH 0
#else
#define VERSION_MAJOR SHARED_VERSION_MAJOR
#define VERSION_MINOR SHARED_VERSION_MINOR
#define VERSION_PATCH SHARED_VERSION_PATCH
#endif
