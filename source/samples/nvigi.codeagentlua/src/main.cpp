// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "game.h"
#include "llm.h"
#include "render.h"
#include "lua_bindings.h"

#include <nvigi.h>

void loggingPrint(nvigi::LogType type, const char* msg)
{
#ifdef NVIGI_WINDOWS
    OutputDebugStringA(msg);
#endif
    std::cout << msg;
}

void loggingCallback(nvigi::LogType type, const char* msg)
{
#ifndef NVIGI_DEBUG
    if (type == nvigi::LogType::eError)
#endif
        loggingPrint(type, msg);
}

// Global AI text for communication
std::string ai_text;

// Max retry attempts for LLM code generation (covers both compile and runtime errors)
const int MAX_RETRIES = 1;

// Game data
Entity player = { "player", {1, 1}, 'P', {}, 3, "", "" };
Entity ai = { "ai", {1, 2}, 'A', {}, 3, "", "" };

std::vector<Entity> monsters = {
    {"gorgon", {9, 1}, 'G', {}, 1, "sword", ""},
    {"bat", {11, 10}, 'B', {}, 1, "arrow", ""},
    {"bat", {17, 17}, 'B', {}, 1, "arrow", ""},
    {"bat", {6, 13}, 'B', {}, 1, "arrow", ""},
};

std::vector<Entity> items = {
    {"sword", {20, 22}, '^', {}, 0, "", "a sharp bladed weapon"},
    {"bow", {1, 10}, '>', {}, 0, "", "needs arrows to shoot"},
    {"arrow", {7, 5}, '/', {}, 0, "", "disappears on use"},
    {"arrow", {9, 18}, '/', {}, 0, "", "disappears on use"},
    {"arrow", {17, 12}, '/', {}, 0, "", "disappears on use"},
};

std::vector<std::string> maze = {
    "########################",
    "#     #         #     ##",
    "# ### # ##### # # # #  #",
    "# #   #   #   #   # #  #",
    "#     #   ### ##  # ####",
    "# #       #     #      #",
    "### #     #     ###  ###",
    "#   #                 ##",
    "## ### ### #####   ### #",
    "#         #         #  #",
    "#   ###  ## ### # # #  #",
    "#   #         #   #    #",
    "### ### # #   #        #",
    "#         #       #    #",
    "#   ### ###   ## ## # ##",
    "#     #   #         #  #",
    "#   ##### ### #####   ##",
    "#         #           ##",
    "## ##### # ###   ### ###",
    "#     # #           #  #",
    "# ### #   #   ###  #   #",
    "#       #     #        #",
    "########################",
};

// ============================================================================
// DEBUG: Uncomment USE_DEBUG_LUA_CODE to bypass LLM and use hardcoded Lua below
// ============================================================================
//#define USE_DEBUG_LUA_CODE

#ifdef USE_DEBUG_LUA_CODE
std::string debugGetLuaCode()
{
    // Paste LLM-generated code here for debugging without calling the LLM.
    // This function is called instead of llmCreateAIFunc when USE_DEBUG_LUA_CODE is defined.
    return R"(
function update_func(player, ai, monsters, items)
    -- Example: AI fetches the bow and brings it back to player
    
    -- If AI has the bow, return to player
    if ai:has_item("bow") then
        if ai.position.row == player.position.row and ai.position.col == player.position.col then
            return "Here's the bow!"
        end
        local path = ai:find_path_to(player)
        if path and #path > 0 then
            ai:move(path[1])
            return "Returning with the bow..."
        end
        return "Can't find path back to you!"
    end
    
    -- Find bow on ground and go get it
    for _, item in ipairs(items) do
        if item.name == "bow" then
            local path = ai:find_path_to(item.position)
            if path and #path > 0 then
                ai:move(path[1])
                return "Heading to the bow..."
            end
            return "Can't reach the bow!"
        end
    end
    
    return "Bow not found on ground (maybe someone has it?) "
end
)";
}
#endif

// Main game update (one turn)
void update(Entity& player, Entity& ai, std::vector<Entity>& monsters, std::vector<Entity>& items,
    std::vector<std::string>& maze, char input, lua_State* L)
{
    // Handle Enter key - get prompt from user and create AI function
    if (input == '\r' || input == '\n')
    {
#ifdef USE_DEBUG_LUA_CODE
        // Debug mode: ignore user prompt, use hardcoded Lua code
        (void)getInputString(":", maze, player, ai, monsters, items);
        std::string code = debugGetLuaCode();
        
        if (luaL_dostring(L, code.c_str()) != LUA_OK)
        {
            const char* err = lua_tostring(L, -1);
            ai_text = std::string("Debug code error: ") + (err ? err : "unknown");
            lua_pop(L, 1);
        }
#else
        std::string prompt = getInputString(":", maze, player, ai, monsters, items);
        
        // Clear the prompt line
        int32_t row = static_cast<int32_t>(maze.size()) + 6;
        setCursor(0, row);
        std::cout << "                                                                    ";

        std::string lastError;
        bool success = false;

        std::string code;
        for (int attempt = 0; attempt <= MAX_RETRIES && !success; attempt++)
        {
            // Build prompt (with error context on retry)
            std::string fullPrompt;
            
            if (attempt == 0)
            {
                fullPrompt = "Now, write an update_func that satisfies the order \"" + prompt + "\"\n"
                    "Only give the function, nothing else.";
            }
            else
            {
                fullPrompt = "User request: \"" + prompt + "\"\n\n"
                    "Your previous code:\n" + code + "failed with this error:\n" + lastError + "\n\n"
                    "Generate corrected code. Only use methods documented in the API.\n"
                    "Only give the corrected function, nothing else.";
            }

            code = llmCreateAIFunc(fullPrompt);
            
            if (code.empty())
            {
                ai_text = "No AI function generated.";
                break;
            }

            // Try to compile
            if (luaL_dostring(L, code.c_str()) != LUA_OK)
            {
                const char* err = lua_tostring(L, -1);
                lastError = err ? err : "unknown compile error";
                lua_pop(L, 1);
                continue;  // Retry
            }

            // Try runtime (test the function once)
            std::string result = callLuaAIFunc(L, player, ai, monsters, maze, items);
            if (result.starts_with("Function failed"))
            {
                lastError = result;
                continue;  // Retry
            }

            // Success!
            success = true;
        }

        if (!success && !lastError.empty())
        {
            ai_text = "Unable to complete request: " + lastError;
            // Clear the broken function
            lua_pushnil(L);
            lua_setglobal(L, "update_func");
        }
#endif

        clearScreen();
        return;
    }

    // Update AI using Lua function (early-exits if no AI functions registered)
    ai_text = callLuaAIFunc(L, player, ai, monsters, maze, items);

    // Move player, AI, monsters; handle combat and item transfer
    moveEntity(player, input, maze);
    pickupItems(player, items);
    if (ai.health > 0)
    {
        moveEntity(ai, 0, maze);
        pickupItems(ai, items);
    }
    for (auto& m : monsters)
    {
        moveRandom(m, maze);
    }
    checkMonsterCollisions(player, ai, monsters);
    if (ai.health <= 0 && ai.position != std::make_pair(-1, -1))
    {
        // AI just died - drop its items around where it died
        dropItemsAroundPosition(ai.position, ai.items, items, maze);
        ai.items.clear();
        ai.position = { -1, -1 };
    }
    transferAIItemsIfSamePosition(player, ai);
}

int32_t main(int argc, char** argv)
{
    // Make sure the user understands what they are about to run.
    std::cout << "Note this sample uses an LLM to write LUA code to run the AI agent in the dungeon" << std::endl;
    std::cout << "<<Press ENTER to continue, q to quit>>" << std::endl;
    char ch;
    do
    {
        ch = _getch();
        if (ch == 'q') 
        {
            std::cout << "exiting...\n";
            exit(0);               // or return / set a flag, etc. [web:37]
        }
    } while (ch != '\r' && ch != '\n');

    lua_State* L = createLuaStateWithMemoryLimit();  // 100MB limit by default
    initLuaSandbox(L);  // Load only safe libraries (sandboxed)
    initLuaBindings(L); 

    // Initialize IGI
    if (argc != 3)
    {
        loggingPrint(nvigi::LogType::eError, "nvigi.codeagentlua.exe <path to models> <system prompt file>");
        return -1;
    }
    std::string modelDir = argv[1];
    std::string systemPromptPath = argv[2];
    if (llmInit(modelDir, systemPromptPath) != 0)
    {
        loggingPrint(nvigi::LogType::eError, "failed to initialize LLM");
        return -1;
    }

    srand((unsigned)time(0));

    // Hide console cursor for smoother appearance
    CONSOLE_CURSOR_INFO ccinfo;
    GetConsoleCursorInfo(hConsole, &ccinfo);
    ccinfo.bVisible = FALSE;
    SetConsoleCursorInfo(hConsole, &ccinfo);

    std::vector<Entity> localItems = items;
    std::vector<Entity> localMonsters = monsters;
    Entity localPlayer = player;
    Entity localAI = ai;

    clearScreen();

    DWORD prevTick = GetTickCount();
    while (true)
    {
        // Draw scene
        setCursor(0, 0);
        drawMaze(maze, localPlayer, localAI, localMonsters, localItems);

        // Handle input (non-blocking)
        char input = 0;
        if (_kbhit())
        {
            input = _getch();
            if (input == 'q') break;
            // Don't lowercase Enter key
            if (input != '\r' && input != '\n')
            {
                input = tolower(input);
            }
        }
        DWORD now = GetTickCount();

        // Update game every 1s or on player's input
        if (input || (now - prevTick) > 1000)
        {
            update(localPlayer, localAI, localMonsters, localItems, maze, input, L);
            prevTick = now;

            // Defeat message
            if (localPlayer.health <= 0)
            {
                setCursor(0, static_cast<int32_t>(maze.size()) + 5);
                std::cout << "XXX You have been defeated by the monsters! XXX";
                Sleep(2000);
                break;
            }
            // Victory message
            bool allMonstersDead = std::all_of(localMonsters.begin(), localMonsters.end(),
                [](const Entity& m) { return m.health == 0; });
            if (allMonstersDead)
            {
                setCursor(0, static_cast<int32_t>(maze.size()) + 5);
                std::cout << "!!! You have beaten the dungeon !!!";
                Sleep(2000);
                break;
            }
        }
        Sleep(50); // ~20FPS animation (but input is still responsive)
    }

    // Show cursor again
    ccinfo.bVisible = TRUE;
    SetConsoleCursorInfo(hConsole, &ccinfo);
    setCursor(0, static_cast<int32_t>(maze.size()) + 8);

    lua_close(L);
    cleanupLuaMemoryTracker();
    std::cout << "Done.\n";
    return llmShutdown();
}

