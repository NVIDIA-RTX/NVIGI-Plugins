// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#include <source/shared/coresdkVersions.h>
#define API_MAJOR CORESDK_API_VERSION_MAJOR
#define API_MINOR CORESDK_API_VERSION_MINOR
#define API_PATCH CORESDK_API_VERSION_PATCH

// We set the versions of all plugins in the project centrally
// But this is 100% independent of the CoreSDK version
// (The API Version of each plugin must be set to the CoreSDK API version against which it was built)
#define SHARED_VERSION_MAJOR 1
#define SHARED_VERSION_MINOR 2
#define SHARED_VERSION_PATCH 0
#if defined(NVIGI_PRODUCTION)
#define BUILD_CONFIG_INFO "PRODUCTION"
#elif defined(NVIGI_DEBUG)
#define BUILD_CONFIG_INFO "DEBUG"
#elif defined(NVIGI_RELEASE)
#define BUILD_CONFIG_INFO "RELEASE"
#else
#error "Unsupported build config"
#endif

#if defined(NVIGI_PRODUCTION)
#define DISTRIBUTION_INFO "PRODUCTION"
#else
#define DISTRIBUTION_INFO "NOT FOR PRODUCTION"
#endif

