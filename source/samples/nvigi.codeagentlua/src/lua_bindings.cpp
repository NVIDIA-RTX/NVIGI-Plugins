// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "lua_bindings.h"
#include <sstream>
#include <chrono>

// Metatable name for Entity userdata
static const char* ENTITY_META = "EntityMeta";

// Registry key for C++ maze pointer (avoids copying maze from Lua every call)
static const char* MAZE_REGISTRY_KEY = "cpp_maze_ptr";

//=============================================================================
// Lua Execution Timeout and Recursion Limit
//=============================================================================

// Configuration
static const int LUA_TIMEOUT_MS = 1000;              // Max execution time in milliseconds
static const int LUA_HOOK_INSTRUCTION_COUNT = 1000;  // Check timeout every N instructions
static const int LUA_MAX_CALL_DEPTH = 200;           // Max recursion depth

// Thread-local state (safe for multiple Lua states)
static thread_local std::chrono::steady_clock::time_point g_luaStartTime;
static thread_local bool g_luaTimedOut = false;
static thread_local bool g_luaStackOverflow = false;
static thread_local int g_luaCallDepth = 0;

// Debug hook for timeout and recursion checking
static void luaTimeoutHook(lua_State* L, lua_Debug* ar)
{
    // Handle function call/return for recursion tracking
    if (ar->event == LUA_HOOKCALL || ar->event == LUA_HOOKTAILCALL)
    {
        g_luaCallDepth++;
        if (g_luaCallDepth > LUA_MAX_CALL_DEPTH)
        {
            g_luaStackOverflow = true;
            luaL_error(L, "Stack overflow (recursion depth exceeded %d)", LUA_MAX_CALL_DEPTH);
        }
    }
    else if (ar->event == LUA_HOOKRET)
    {
        if (g_luaCallDepth > 0)
        {
            g_luaCallDepth--;
        }
    }
    
    // Check timeout on instruction count events
    if (ar->event == LUA_HOOKCOUNT)
    {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_luaStartTime).count();
        
        if (elapsed > LUA_TIMEOUT_MS)
        {
            g_luaTimedOut = true;
            luaL_error(L, "Lua execution timed out (exceeded %d ms)", LUA_TIMEOUT_MS);
        }
    }
}

// Start timeout and recursion tracking, install hook
static void startLuaTimeout(lua_State* L)
{
    g_luaStartTime = std::chrono::steady_clock::now();
    g_luaTimedOut = false;
    g_luaStackOverflow = false;
    g_luaCallDepth = 0;
    // Hook on: function calls, returns, and every N instructions
    lua_sethook(L, luaTimeoutHook, LUA_MASKCALL | LUA_MASKRET | LUA_MASKCOUNT, LUA_HOOK_INSTRUCTION_COUNT);
}

// Stop timeout tracking and remove hook
static void stopLuaTimeout(lua_State* L)
{
    lua_sethook(L, nullptr, 0, 0);
}

// Check if last execution timed out
static bool didLuaTimeout()
{
    return g_luaTimedOut;
}

// Check if last execution had stack overflow
static bool didLuaStackOverflow()
{
    return g_luaStackOverflow;
}

//=============================================================================
// Lua Memory Limiting
//=============================================================================

// Memory limit configuration
static const size_t LUA_MEMORY_LIMIT_BYTES = 100 * 1024 * 1024;  // 100 MB

// Memory tracking structure (passed as userdata to allocator)
struct LuaMemoryTracker
{
    size_t currentUsage;
    size_t peakUsage;
    size_t limit;
};

// Global tracker (for access outside allocator)
static LuaMemoryTracker* g_memoryTracker = nullptr;

// Custom allocator function for Lua
// See: https://www.lua.org/manual/5.4/manual.html#lua_Alloc
static void* luaLimitedAlloc(void* ud, void* ptr, size_t osize, size_t nsize)
{
    LuaMemoryTracker* tracker = static_cast<LuaMemoryTracker*>(ud);
    
    // Free operation (nsize == 0)
    if (nsize == 0)
    {
        if (ptr != nullptr)
        {
            tracker->currentUsage -= osize;
            free(ptr);
        }
        return nullptr;
    }
    
    // Calculate the change in memory usage
    size_t delta = nsize - (ptr ? osize : 0);
    size_t newUsage = tracker->currentUsage + delta;
    
    // Check if this would exceed the limit
    if (newUsage > tracker->limit)
    {
        // Refuse allocation - Lua will throw out of memory error
        return nullptr;
    }
    
    // Perform the allocation/reallocation
    void* newPtr = realloc(ptr, nsize);
    if (newPtr != nullptr)
    {
        tracker->currentUsage = newUsage;
        if (newUsage > tracker->peakUsage)
        {
            tracker->peakUsage = newUsage;
        }
    }
    
    return newPtr;
}

// Create a new Lua state with memory limiting
lua_State* createLuaStateWithMemoryLimit(size_t limitBytes)
{
    // Allocate tracker (lives for duration of Lua state)
    LuaMemoryTracker* tracker = new LuaMemoryTracker();
    tracker->currentUsage = 0;
    tracker->peakUsage = 0;
    tracker->limit = limitBytes;
    
    g_memoryTracker = tracker;
    
    // Use high-resolution time as seed for Lua's hash randomization (Lua 5.5+)
    // This helps prevent hash-flooding attacks by making hash behavior unpredictable
    auto seed = static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    
    lua_State* L = lua_newstate(luaLimitedAlloc, tracker, seed);
    return L;
}

// Get current Lua memory usage in bytes
size_t getLuaMemoryUsage()
{
    return g_memoryTracker ? g_memoryTracker->currentUsage : 0;
}

// Get peak Lua memory usage in bytes
size_t getLuaPeakMemoryUsage()
{
    return g_memoryTracker ? g_memoryTracker->peakUsage : 0;
}

// Cleanup memory tracker (call when closing Lua state)
void cleanupLuaMemoryTracker()
{
    if (g_memoryTracker)
    {
        delete g_memoryTracker;
        g_memoryTracker = nullptr;
    }
}

//=============================================================================
// Lua Sandboxing - Restrict dangerous functionality
//=============================================================================

// Remove a global function by setting it to nil
static void removeGlobal(lua_State* L, const char* name)
{
    lua_pushnil(L);
    lua_setglobal(L, name);
}

// Remove a function from a library table
static void removeFromLib(lua_State* L, const char* libName, const char* funcName)
{
    lua_getglobal(L, libName);
    if (lua_istable(L, -1))
    {
        lua_pushnil(L);
        lua_setfield(L, -2, funcName);
    }
    lua_pop(L, 1);
}

// Initialize Lua with only safe libraries (call INSTEAD of luaL_openlibs)
void initLuaSandbox(lua_State* L)
{
    // Load safe libraries
    luaL_requiref(L, "_G", luaopen_base, 1);
    lua_pop(L, 1);
    
    luaL_requiref(L, LUA_TABLIBNAME, luaopen_table, 1);
    lua_pop(L, 1);
    
    luaL_requiref(L, LUA_STRLIBNAME, luaopen_string, 1);
    lua_pop(L, 1);
    
    luaL_requiref(L, LUA_MATHLIBNAME, luaopen_math, 1);
    lua_pop(L, 1);
    
    luaL_requiref(L, LUA_UTF8LIBNAME, luaopen_utf8, 1);
    lua_pop(L, 1);
    
    // Load os library but then remove dangerous functions
    luaL_requiref(L, LUA_OSLIBNAME, luaopen_os, 1);
    lua_pop(L, 1);
    
    // NOT loading these dangerous libraries:
    // - io: File I/O
    // - debug: Can break sandbox
    // - package: Can load arbitrary modules/files
    // - coroutine: Not needed, could complicate timeout handling
    
    // Remove dangerous os functions (keep time, date, difftime, clock)
    removeFromLib(L, "os", "execute");    // Run shell commands
    removeFromLib(L, "os", "exit");       // Terminate program
    removeFromLib(L, "os", "remove");     // Delete files
    removeFromLib(L, "os", "rename");     // Rename/move files
    removeFromLib(L, "os", "setlocale");  // Modify locale
    removeFromLib(L, "os", "getenv");     // Read environment variables
    removeFromLib(L, "os", "tmpname");    // Create temp file names
    
    // Remove dangerous string functions
    removeFromLib(L, "string", "dump");   // Dump bytecode - not useful without load()
    
    // Remove dangerous base functions
    removeGlobal(L, "dofile");       // Execute Lua file
    removeGlobal(L, "loadfile");     // Load Lua file
    removeGlobal(L, "load");         // Load arbitrary code string with environment access
    removeGlobal(L, "loadstring");   // Lua 5.1 compat - load code string
    removeGlobal(L, "rawequal");     // Bypass metatables
    removeGlobal(L, "rawget");       // Bypass metatables
    removeGlobal(L, "rawset");       // Bypass metatables
    removeGlobal(L, "rawlen");       // Bypass metatables
    removeGlobal(L, "getmetatable"); // Could inspect/modify Entity metatable, potentially corrupting the entities tables. No valid agent use case, so removed
    removeGlobal(L, "setmetatable"); // Could inspect/modify Entity metatable, potentially corrupting the entities tables. No valid agent use case, so removed
    removeGlobal(L, "collectgarbage"); // GC control
    removeGlobal(L, "require");      // Module loading
    removeGlobal(L, "module");       // Module creation (Lua 5.1)
}

//=============================================================================
// Entity Custom Fields (Lua-side shadow storage)
//=============================================================================

// Unique key for the shadow table in Lua registry
static const char* ENTITY_SHADOW_TABLE_KEY = "EntityShadowTable";

// Ensure shadow table exists in registry, push it onto stack
static void pushShadowTable(lua_State* L)
{
    lua_getfield(L, LUA_REGISTRYINDEX, ENTITY_SHADOW_TABLE_KEY);
    if (lua_isnil(L, -1))
    {
        lua_pop(L, 1);
        // Create new shadow table with weak keys (entities can be GC'd)
        lua_newtable(L);
        
        // Make keys weak so entity entries are cleaned up when entity userdata is GC'd
        lua_newtable(L);  // metatable
        lua_pushstring(L, "k");
        lua_setfield(L, -2, "__mode");
        lua_setmetatable(L, -2);
        
        // Store in registry
        lua_pushvalue(L, -1);
        lua_setfield(L, LUA_REGISTRYINDEX, ENTITY_SHADOW_TABLE_KEY);
    }
}

// Get the custom fields table for an entity (creates if doesn't exist)
// Leaves the fields table on the stack
static void getEntityCustomFields(lua_State* L, Entity* entity)
{
    pushShadowTable(L);
    
    // Use entity pointer as light userdata key
    lua_pushlightuserdata(L, entity);
    lua_gettable(L, -2);
    
    if (lua_isnil(L, -1))
    {
        lua_pop(L, 1);
        // Create new table for this entity's custom fields
        lua_newtable(L);
        lua_pushlightuserdata(L, entity);
        lua_pushvalue(L, -2);  // copy the new table
        lua_settable(L, -4);   // shadow[entity] = newtable
    }
    
    // Remove shadow table, leaving only entity's custom fields table
    lua_remove(L, -2);
}

// Helper: Push a std::pair<int32_t, int32_t> as a Lua table with both array AND named access
static void pushPosition(lua_State* L, const std::pair<int32_t, int32_t>& pos)
{
    lua_createtable(L, 2, 2);  // 2 array slots, 2 hash slots
    
    // Array-style access (for compatibility): pos[1], pos[2]
    lua_pushinteger(L, pos.first);
    lua_rawseti(L, -2, 1);
    lua_pushinteger(L, pos.second);
    lua_rawseti(L, -2, 2);
    
    // Named access (more idiomatic): pos.row, pos.col
    lua_pushinteger(L, pos.first);
    lua_setfield(L, -2, "row");
    lua_pushinteger(L, pos.second);
    lua_setfield(L, -2, "col");
}

// Helper: Get a std::pair<int32_t, int32_t> from a Lua table at index
// Supports BOTH array-style {1, 2} AND named {row=1, col=2} or {x=1, y=2}
static std::pair<int32_t, int32_t> getPosition(lua_State* L, int index)
{
    if (!lua_istable(L, index))
    {
        luaL_error(L, "Expected table for position");
    }
    
    int32_t x, y;
    
    // First try array-style access: {row, col}
    lua_rawgeti(L, index, 1);
    if (!lua_isnil(L, -1))
    {
        x = (int32_t)luaL_checkinteger(L, -1);
        lua_pop(L, 1);
        lua_rawgeti(L, index, 2);
        y = (int32_t)luaL_checkinteger(L, -1);
        lua_pop(L, 1);
        return { x, y };
    }
    lua_pop(L, 1);
    
    // Fall back to named fields: try "row"/"col"
    lua_getfield(L, index, "row");
    if (!lua_isnil(L, -1))
    {
        x = (int32_t)luaL_checkinteger(L, -1);
        lua_pop(L, 1);
        lua_getfield(L, index, "col");
        y = (int32_t)luaL_checkinteger(L, -1);
        lua_pop(L, 1);
        return { x, y };
    }
    lua_pop(L, 1);
    
    // Try "x"/"y" as alternative
    lua_getfield(L, index, "x");
    if (!lua_isnil(L, -1))
    {
        x = (int32_t)luaL_checkinteger(L, -1);
        lua_pop(L, 1);
        lua_getfield(L, index, "y");
        y = (int32_t)luaL_checkinteger(L, -1);
        lua_pop(L, 1);
        return { x, y };
    }
    lua_pop(L, 1);
    
    luaL_error(L, "Position must be {row, col}, {row=N, col=N}, or {x=N, y=N}");
    return { 0, 0 };
}

// Helper: Push a vector of strings as a Lua table
static void pushStringVector(lua_State* L, const std::vector<std::string>& vec)
{
    lua_createtable(L, (int)vec.size(), 0);
    for (size_t i = 0; i < vec.size(); i++)
    {
        lua_pushstring(L, vec[i].c_str());
        lua_rawseti(L, -2, (int)(i + 1));
    }
}

// Forward declarations
static void pushEntity(lua_State* L, Entity* entity);
static Entity* checkEntity(lua_State* L, int index);
static void pushEntityVector(lua_State* L, const std::vector<Entity>& entities);

// Forward declarations for global Lua functions (used by entity_index)
static int lua_move_entity(lua_State* L);
static int lua_move_random(lua_State* L);
static int lua_has_item(lua_State* L);
static int lua_remove_item(lua_State* L);

// Forward declarations for entity methods with special behavior
static int entity_method_find_path_to(lua_State* L);
static int entity_method_distance_to(lua_State* L);

// Helper: Push an Entity as userdata with reference semantics
static void pushEntity(lua_State* L, Entity* entity)
{
    Entity** udata = (Entity**)lua_newuserdata(L, sizeof(Entity*));
    *udata = entity;
    luaL_getmetatable(L, ENTITY_META);
    lua_setmetatable(L, -2);
}

// Helper: Check and get Entity pointer from userdata
static Entity* checkEntity(lua_State* L, int index)
{
    void* ud = luaL_testudata(L, index, ENTITY_META);
    if (ud)
    {
        Entity** udata = (Entity**)ud;
        if (udata && *udata)
        {
            return *udata;
        }
    }
    luaL_error(L, "Expected Entity userdata");
    return nullptr;
}

// Helper: Push a vector of entities as a Lua table (with ipairs support).
// Careful - current code doesn't allow for modifying the vector (though we can modify entries in the vector)
// If we wanted to modify the vector, we would need to take care that the vector doesn't change out from under Lua 
// or vice versa, that Lua did not invalidate an iterator in the vector.
static void pushEntityVector(lua_State* L, const std::vector<Entity>& entities)
{
    lua_createtable(L, (int)entities.size(), 0);
    for (size_t i = 0; i < entities.size(); i++)
    {
        pushEntity(L, const_cast<Entity*>(&entities[i]));
        lua_rawseti(L, -2, (int)(i + 1));
    }
}

// Helper: Get maze pointer from registry (zero-copy, stored by callLuaAIFunc)
static const std::vector<std::string>& getMazeFromRegistry(lua_State* L)
{
    lua_getfield(L, LUA_REGISTRYINDEX, MAZE_REGISTRY_KEY);
    auto* mazePtr = static_cast<std::vector<std::string>*>(lua_touserdata(L, -1));
    lua_pop(L, 1);
    return *mazePtr;
}

// Helper: Build entities list from individual globals (player, ai, monsters)
// This is used for pathfinding to determine blocked positions
static std::vector<Entity> buildEntitiesFromGlobals(lua_State* L)
{
    std::vector<Entity> entities;
    
    // Get player
    lua_getglobal(L, "player");
    if (!lua_isnil(L, -1))
    {
        Entity* player = checkEntity(L, -1);
        entities.push_back(*player);
    }
    lua_pop(L, 1);
    
    // Get ai
    lua_getglobal(L, "ai");
    if (!lua_isnil(L, -1))
    {
        Entity* ai = checkEntity(L, -1);
        entities.push_back(*ai);
    }
    lua_pop(L, 1);
    
    // Get monsters
    lua_getglobal(L, "monsters");
    if (!lua_isnil(L, -1) && lua_istable(L, -1))
    {
        size_t len = lua_rawlen(L, -1);
        for (size_t i = 1; i <= len; i++)
        {
            lua_rawgeti(L, -1, (int)i);
            Entity* monster = checkEntity(L, -1);
            entities.push_back(*monster);
            lua_pop(L, 1);
        }
    }
    lua_pop(L, 1);
    
    return entities;
}

// Entity.__index metamethod - returns properties AND methods
static int entity_index(lua_State* L)
{
    Entity* e = checkEntity(L, 1);
    const char* key = luaL_checkstring(L, 2);

    // Entity methods - more idiomatic for Lua (ai:move("w") vs move_entity(ai, "w"))
    // and gives a higher chance of successful LLM code generation.

    // Check for methods first
    // For move, move_random, has_item, remove_item: push global functions directly
    // (stack layout is identical: entity as arg 1, other args follow)
    if (strcmp(key, "move") == 0)
    {
        lua_pushcfunction(L, lua_move_entity);
        return 1;
    }
    else if (strcmp(key, "move_random") == 0)
    {
        lua_pushcfunction(L, lua_move_random);
        return 1;
    }
    else if (strcmp(key, "has_item") == 0)
    {
        lua_pushcfunction(L, lua_has_item);
        return 1;
    }
    else if (strcmp(key, "remove_item") == 0)
    {
        lua_pushcfunction(L, lua_remove_item);
        return 1;
    }
    // find_path_to and distance_to have special behavior (accept entity OR position)
    else if (strcmp(key, "find_path_to") == 0)
    {
        lua_pushcfunction(L, entity_method_find_path_to);
        return 1;
    }
    else if (strcmp(key, "distance_to") == 0)
    {
        lua_pushcfunction(L, entity_method_distance_to);
        return 1;
    }
    // Then check for properties
    else if (strcmp(key, "name") == 0)
    {
        lua_pushstring(L, e->name.c_str());
    }
    else if (strcmp(key, "position") == 0)
    {
        pushPosition(L, e->position);
    }
    else if (strcmp(key, "symbol") == 0)
    {
        char buf[2] = { e->symbol, '\0' };
        lua_pushstring(L, buf);
    }
    else if (strcmp(key, "items") == 0)
    {
        pushEntityVector(L, e->items);
    }
    else if (strcmp(key, "health") == 0)
    {
        lua_pushinteger(L, e->health);
    }
    else if (strcmp(key, "weakness") == 0)
    {
        lua_pushstring(L, e->weakness.c_str());
    }
    else if (strcmp(key, "description") == 0)
    {
        lua_pushstring(L, e->description.c_str());
    }
    else
    {
        // Check shadow table for custom fields
        getEntityCustomFields(L, e);
        lua_pushstring(L, key);
        lua_gettable(L, -2);
        lua_remove(L, -2);  // Remove custom fields table, leave value
    }
    return 1;
}

// Entity.__newindex metamethod
// Silently ignores writes to built-in fields to prevent game state corruption.
// Only custom fields (for AI state tracking) are stored.
static int entity_newindex(lua_State* L)
{
    Entity* e = checkEntity(L, 1);
    const char* key = luaL_checkstring(L, 2);

    // Disallow modifications of entity state - use methods instead.
    if (strcmp(key, "name") == 0 || strcmp(key, "position") == 0 ||
        strcmp(key, "symbol") == 0 || strcmp(key, "items") == 0 ||
        strcmp(key, "health") == 0 || strcmp(key, "weakness") == 0 ||
        strcmp(key, "description") == 0)
    {
        return luaL_error(L, "Cannot modify '%s' - use methods instead", key);
    }

    // Allow store of custom fields in shadow table (for AI state tracking)
    getEntityCustomFields(L, e);
    lua_pushstring(L, key);
    lua_pushvalue(L, 3);  // Copy the value
    lua_settable(L, -3);
    lua_pop(L, 1);  // Pop custom fields table
    return 0;
}

// Lua binding: find_path_astar(start, dest)
// entities for obstacle avoidance are built from globals (player, ai, monsters)
static int lua_find_path_astar(lua_State* L)
{
    auto start = getPosition(L, 1);
    auto dest = getPosition(L, 2);
    const auto& maze = getMazeFromRegistry(L);
    auto entities = buildEntitiesFromGlobals(L);

    std::vector<char> path = findPathAStar(start, dest, maze, entities);

    lua_createtable(L, (int)path.size(), 0);
    for (size_t i = 0; i < path.size(); i++)
    {
        char buf[2] = { path[i], '\0' };
        lua_pushstring(L, buf);
        lua_rawseti(L, -2, (int)(i + 1));
    }
    return 1;
}

// Lua binding: move_entity(entity, direction)
static int lua_move_entity(lua_State* L)
{
    Entity* entity = checkEntity(L, 1);
    const char* dir = luaL_checkstring(L, 2);
    const auto& maze = getMazeFromRegistry(L);

    if (dir[0])
    {
        moveEntity(*entity, dir[0], maze);
    }
    return 0;
}

// Lua binding: manhattan(a, b)
static int lua_manhattan(lua_State* L)
{
    auto a = getPosition(L, 1);
    auto b = getPosition(L, 2);

    int32_t dist = manhattan(a, b);
    lua_pushinteger(L, dist);
    return 1;
}

// Lua binding: valid_move(position)
static int lua_valid_move(lua_State* L)
{
    auto pos = getPosition(L, 1);
    const auto& maze = getMazeFromRegistry(L);

    bool valid = validMove(pos, maze);
    lua_pushboolean(L, valid);
    return 1;
}

// Lua binding: move_random(entity)
static int lua_move_random(lua_State* L)
{
    Entity* entity = checkEntity(L, 1);
    const auto& maze = getMazeFromRegistry(L);

    moveRandom(*entity, maze);
    return 0;
}

// Lua binding: has_item(entity, item_name)
static int lua_has_item(lua_State* L)
{
    Entity* entity = checkEntity(L, 1);
    const char* itemName = luaL_checkstring(L, 2);

    bool has = hasItem(*entity, itemName);
    lua_pushboolean(L, has);
    return 1;
}

// Lua binding: remove_item(entity, item_name)
static int lua_remove_item(lua_State* L)
{
    Entity* entity = checkEntity(L, 1);
    const char* itemName = luaL_checkstring(L, 2);

    removeItem(*entity, itemName);
    return 0;
}

//=============================================================================
// Entity Methods with special behavior (accept entity OR position)
//=============================================================================

// Entity method: entity:find_path_to(dest_position_or_entity)
static int entity_method_find_path_to(lua_State* L)
{
    Entity* entity = checkEntity(L, 1);
    
    // dest can be a position table OR another entity
    std::pair<int32_t, int32_t> dest;
    if (luaL_testudata(L, 2, ENTITY_META))
    {
        Entity* destEntity = checkEntity(L, 2);
        dest = destEntity->position;
    }
    else
    {
        dest = getPosition(L, 2);
    }
    
    const auto& maze = getMazeFromRegistry(L);
    auto entities = buildEntitiesFromGlobals(L);
    
    std::vector<char> path = findPathAStar(entity->position, dest, maze, entities);
    
    lua_createtable(L, (int)path.size(), 0);
    for (size_t i = 0; i < path.size(); i++)
    {
        char buf[2] = { path[i], '\0' };
        lua_pushstring(L, buf);
        lua_rawseti(L, -2, (int)(i + 1));
    }
    return 1;
}

// Entity method: entity:distance_to(dest_position_or_entity)
static int entity_method_distance_to(lua_State* L)
{
    Entity* entity = checkEntity(L, 1);
    
    // dest can be a position table OR another entity
    std::pair<int32_t, int32_t> dest;
    if (luaL_testudata(L, 2, ENTITY_META))
    {
        Entity* destEntity = checkEntity(L, 2);
        dest = destEntity->position;
    }
    else
    {
        dest = getPosition(L, 2);
    }
    
    int32_t dist = manhattan(entity->position, dest);
    lua_pushinteger(L, dist);
    return 1;
}

// Initialize the Lua state with all game functions
void initLuaBindings(lua_State* L)
{
    // Create the Entity metatable
    luaL_newmetatable(L, ENTITY_META);

    lua_pushstring(L, "__index");
    lua_pushcfunction(L, entity_index);
    lua_settable(L, -3);

    lua_pushstring(L, "__newindex");
    lua_pushcfunction(L, entity_newindex);
    lua_settable(L, -3);

    lua_pop(L, 1); // Pop metatable

    // Register global functions
    // Note while the AI knows about far fewer global functions in the system prompt, this hedges
    // a bit allowing it to accidentally call a global function and still work.
    lua_register(L, "find_path_astar", lua_find_path_astar);
    lua_register(L, "move_entity", lua_move_entity);
    lua_register(L, "manhattan", lua_manhattan);
    lua_register(L, "valid_move", lua_valid_move);
    lua_register(L, "move_random", lua_move_random);
    lua_register(L, "has_item", lua_has_item);
    lua_register(L, "remove_item", lua_remove_item);

    // Set up initial values
    lua_pushstring(L, "");
    lua_setglobal(L, "ai_text");
}

// Call Lua AI update function if it exists
std::string callLuaAIFunc(lua_State* L,
    Entity& player,
    Entity& ai,
    std::vector<Entity>& monsters,
    std::vector<std::string>& maze,
    std::vector<Entity>& items)
{
    // Quick check if update_func exists before doing expensive setup
    lua_getglobal(L, "update_func");
    bool hasUpdateFunc = lua_isfunction(L, -1);
    lua_pop(L, 1);
    
    // Early exit if no AI function registered - skip all setup work
    if (!hasUpdateFunc)
    {
        return "";
    }
    
    // Store maze pointer in registry for efficient access (no copying)
    lua_pushlightuserdata(L, const_cast<std::vector<std::string>*>(&maze));
    lua_setfield(L, LUA_REGISTRYINDEX, MAZE_REGISTRY_KEY);

    // Set global variables with reference semantics (userdata pointers)
    pushEntity(L, &player);
    lua_setglobal(L, "player");

    pushEntity(L, &ai);
    lua_setglobal(L, "ai");

    pushEntityVector(L, monsters);
    lua_setglobal(L, "monsters");

    pushStringVector(L, maze);
    lua_setglobal(L, "maze");  // Still expose to Lua for scripts that want to inspect it

    pushEntityVector(L, items);
    lua_setglobal(L, "items");

    // Try to call update_func
    lua_getglobal(L, "update_func");
    if (lua_isfunction(L, -1))
    {
        pushEntity(L, &player);
        pushEntity(L, &ai);
        pushEntityVector(L, monsters);
        pushEntityVector(L, items);

        // Start timeout tracking
        startLuaTimeout(L);
        int callResult = lua_pcall(L, 4, 1, 0);
        stopLuaTimeout(L);

        if (callResult != LUA_OK)
        {
            std::string error;
            if (didLuaTimeout())
            {
                error = "AI function timed out (infinite loop?). Function has been removed.";
                // Remove the problematic function to prevent repeated timeouts
                lua_pushnil(L);
                lua_setglobal(L, "update_func");
            }
            else if (didLuaStackOverflow())
            {
                error = "AI function caused stack overflow (recursion too deep). Function has been removed.";
                // Remove the problematic function to prevent repeated stack overflows
                lua_pushnil(L);
                lua_setglobal(L, "update_func");
            }
            else if (callResult == LUA_ERRMEM)
            {
                error = "AI function exceeded memory limit. Function has been removed.";
                // Remove the problematic function to prevent repeated memory errors
                lua_pushnil(L);
                lua_setglobal(L, "update_func");
            }
            else
            {
                error = std::string("Function failed to execute.\n") + lua_tostring(L, -1);
            }
            lua_pop(L, 1);
            return error;
        }

        std::string result = luaL_optstring(L, -1, "");
        lua_pop(L, 1);

        // No sync-back needed - changes are by reference!
        return result;
    }
    lua_pop(L, 1);

    return "";
}
