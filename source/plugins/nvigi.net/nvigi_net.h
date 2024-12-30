// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <string>
#include <vector>

#include "source/core/nvigi.api/nvigi_struct.h"
#include "source/core/nvigi.types/types.h"

namespace nvigi
{
namespace plugin
{
namespace net
{
constexpr PluginID kId = { {0xb73ed870, 0x8091, 0x491e,{0xa4, 0x9b, 0x6e, 0x19, 0x8f, 0xe9, 0x0e, 0x2c}}, 0x544a60 };  // {B73ED870-8091-491E-A49B-6E198FE90E2C} [nvigi.plugin.net]
}
}

namespace net
{

// {8560A124-99B4-4ED8-89FE-4406EF08CB30}
struct alignas(8) Parameters {
    Parameters() {}; 
    NVIGI_UID(UID({ 0x8560a124, 0x99b4, 0x4ed8,{ 0x89, 0xfe, 0x44, 0x6, 0xef, 0x8, 0xcb, 0x30 } }), kStructVersion1);
    //! IMPORTANT: Using nvigi::types ABI stable implementations
    //! 
    types::string url{};
    types::vector<types::string> headers{};
    types::vector<uint8_t> data{};
};

NVIGI_VALIDATE_STRUCT(Parameters)

constexpr uint32_t kResultNetMissingAuthentication = 1 << 24 | plugin::net::kId.crc24;
constexpr uint32_t kResultNetFailedToInitializeCurl = 2 << 24 | plugin::net::kId.crc24;
constexpr uint32_t kResultNetCurlError = 3 << 24 | plugin::net::kId.crc24;
constexpr uint32_t kResultNetServerError = 4 << 24 | plugin::net::kId.crc24;
constexpr uint32_t kResultNetTimeout = 5 << 24 | plugin::net::kId.crc24;

// {E70C7C30-5E61-4F3A-B40F-A6F561EDB563}
struct alignas(8) INet {
    INet() {}; 
    NVIGI_UID(UID({ 0xe70c7c30, 0x5e61, 0x4f3a,{ 0xb4, 0xf, 0xa6, 0xf5, 0x61, 0xed, 0xb5, 0x63 } }), kStructVersion1);
    Result (*setVerboseMode)(bool flag);
    Result (*nvcfSetToken)(const char* token);
    Result (*nvcfGet)(const Parameters& params, types::string& response);
    Result (*nvcfPost)(const Parameters& params, types::string& response);
    Result (*nvcfUploadAsset)(const types::string& contentType, const types::string& description, const types::vector<uint8_t>& asset, types::string& assetId);
};

NVIGI_VALIDATE_STRUCT(INet)

}
}