// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <string>
#include <vector>

#include "source/core/nvigi.types/types.h"
#include "source/plugins/nvigi.net/nvigi_net.h"
#include "external/json/source/nlohmann/json.hpp"
using json = nlohmann::json;

namespace nvigi
{

namespace net
{

struct INetworkInternal
{
    virtual Result initialize() = 0;
    virtual Result shutdown() = 0;
    virtual Result setVerboseMode(bool flag) = 0;
    virtual Result nvcfSetToken(const char* token) = 0;
    virtual Result nvcfGet(const Parameters& params, json& response) = 0;
    virtual Result nvcfPost(const Parameters& params, json& response) = 0;
    virtual Result nvcfUploadAsset(const types::string& contentType, const types::string& description, const types::vector<uint8_t>& asset, types::string& assetId) = 0;
};

INetworkInternal* getInterface();
void destroyInterface();

}
}