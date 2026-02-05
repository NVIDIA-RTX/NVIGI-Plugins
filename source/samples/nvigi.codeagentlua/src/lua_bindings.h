// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "lua.hpp"
#include "game.h"

// Create a new Lua state with memory limiting (call INSTEAD of luaL_newstate)
lua_State* createLuaStateWithMemoryLimit(size_t limitBytes = 100 * 1024 * 1024);

// Initialize Lua with sandboxed libraries (call INSTEAD of luaL_openlibs)
void initLuaSandbox(lua_State* L);

// Initialize Lua state with game functions
void initLuaBindings(lua_State* L);

// Memory tracking utilities
size_t getLuaMemoryUsage();
size_t getLuaPeakMemoryUsage();
void cleanupLuaMemoryTracker();

// Call Lua AI update function if it exists
std::string callLuaAIFunc(lua_State* L,
                          Entity& player, 
                          Entity& ai, 
                          std::vector<Entity>& monsters,
                          std::vector<std::string>& maze,
                          std::vector<Entity>& items);
