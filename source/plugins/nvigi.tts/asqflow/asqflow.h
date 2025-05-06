// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#include "../../../../nvigi_core/source/core/nvigi.log/log.h"
#include <mutex>
#include <queue>
#include <string>

namespace nvigi
{
namespace asqflow
{

// Prompts buffer used when async mode is used
class PromptsBuffer
{
  public:
    PromptsBuffer()
    {
    }

    void clear()
    {
        std::scoped_lock lock(mtx_);
        while (!buffer_.empty())
            buffer_.pop();
    }

    void write(const std::string prompt)
    {
        std::scoped_lock lock(mtx_);
        buffer_.push(prompt);
    }

    void read_and_pop(std::string &prompt)
    {
        std::scoped_lock lock(mtx_);
        if (buffer_.empty())
        {
            NVIGI_LOG_ERROR("Buffer is empty");
            return;
        }
        prompt = buffer_.front();
        buffer_.pop();
        return;
    }

    bool empty()
    {
        std::scoped_lock lock(mtx_);
        return buffer_.empty();
    }

    // Get the amount of data currently stored in the buffer
    size_t getDataSize()
    {
        std::scoped_lock lock(mtx_);
        return buffer_.size();
    }

  private:
    std::queue<std::string> buffer_; // The buffer of prompts
    std::mutex mtx_;
};

} // namespace asqflow
} // namespace nvigi
