// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <deque>
#include <string>
#include <vector>

using SizeType32 = std::int32_t;
using TokenIdType = std::int32_t;
using VecTokens = std::vector<TokenIdType>;
// hardcoded newline tokens size
SizeType32 n_newlineTokens = 3;

struct ChatData {
    std::string query{};
    std::string response{};
    SizeType32 n_queryTokens{};
    SizeType32 n_responseTokens{};

    // Default constructor
    ChatData() = default;
    ~ChatData() = default;

    // Returns the total tokens in this chat (query + response + '\n')
    SizeType32 totalTokens() const {
        return n_queryTokens + n_responseTokens + n_newlineTokens;
    }

    // Returns concatenated chat string (for building the full prompt)
    std::string getChatString() const {
        return query + "\n" + response;
    }
};

class ChatHistory {
private:
    SizeType32 maxSize = 20;
    SizeType32 maxTokens = 1024;
    std::deque<ChatData> history;

public:
    // Constructor to set maximum tokens allowed
    ChatHistory(SizeType32 newMaxTokens) : maxTokens(newMaxTokens) {}

    // Default constructor
    ChatHistory() = default;
    ~ChatHistory() = default;

    void setMaxTokens(SizeType32 newMaxTokens) {
        maxTokens = newMaxTokens;
    }

    // Add a chat to the history, remove oldest if exceeding maxSize
    void addChat(const ChatData& chat) {
        if (history.size() == maxSize) {
            history.pop_front();
        }
        history.push_back(chat);
    }

    // Concatenate chats until token count exceeds maximum allowed
    std::string getHistoryPrompt(SizeType32 initialCount = 0) {
        std::string historyPrompt;
        SizeType32 tokenCount = initialCount;

        // Traverse chats from the latest to the earliest, appending until we hit the token limit
        for (auto it = history.rbegin(); it != history.rend(); ++it) {
            SizeType32 chatTokens = it->totalTokens();
            if (tokenCount + chatTokens > maxTokens) {
                break;
            }
            // Prepend the chat to the prompt and update token count
            historyPrompt = it->getChatString() + "\n" + historyPrompt;
            tokenCount += chatTokens + n_newlineTokens;
        }
        return historyPrompt;
    }

    // Get the size of the chat history buffer
    SizeType32 size() const {
        return history.size();
    }

    // Clear the chat history buffer
    void clear() {
        history.clear();
    }
};