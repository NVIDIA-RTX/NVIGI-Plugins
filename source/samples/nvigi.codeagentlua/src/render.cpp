// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "render.h"

// Windows console handle
HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

// Helper to move the cursor
void setCursor(int32_t x, int32_t y)
{
    COORD coord = { (SHORT)x, (SHORT)y };
    SetConsoleCursorPosition(hConsole, coord);
}

// Clear the console screen
void clearScreen()
{
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    DWORD cellCount;
    DWORD count;
    COORD homeCoords = { 0, 0 };

    // Get the number of cells in the current buffer
    if (!GetConsoleScreenBufferInfo(hConsole, &csbi))
    {
        return;
    }
    cellCount = csbi.dwSize.X * csbi.dwSize.Y;

    // Fill the entire buffer with spaces
    if (!FillConsoleOutputCharacter(hConsole, (TCHAR)' ', cellCount, homeCoords, &count))
    {
        return;
    }

    // Fill the entire buffer with the current colors and attributes
    if (!FillConsoleOutputAttribute(hConsole, csbi.wAttributes, cellCount, homeCoords, &count))
    {
        return;
    }

    // Move the cursor home
    SetConsoleCursorPosition(hConsole, homeCoords);
}

// Flicker-free clear (no full clear, just redraw maze in place)
void drawMaze(const std::vector<std::string>& maze,
    const Entity& player,
    const Entity& ai,
    const std::vector<Entity>& monsters,
    const std::vector<Entity>& items)
{
    // Draw maze with entities and items
    for (int32_t r = 0; r < maze.size(); ++r)
    {
        setCursor(0, r);
        for (int32_t c = 0; c < maze[0].length(); ++c)
        {
            char out = maze[r][c];
            bool drawn = false;

            if (player.health > 0 && player.position == std::make_pair(r, c))
            {
                out = player.symbol;
                drawn = true;
            }
            if (!drawn && ai.health > 0 && ai.position == std::make_pair(r, c))
            {
                out = ai.symbol;
                drawn = true;
            }
            for (const auto& m : monsters)
            {
                if (m.health > 0 && m.position == std::make_pair(r, c))
                {
                    out = m.symbol;
                    drawn = true;
                    break;
                }
            }
            if (!drawn)
            {
                for (const auto& i : items)
                {
                    if (i.position == std::make_pair(r, c))
                    {
                        out = i.symbol;
                        drawn = true;
                        break;
                    }
                }
            }
            std::cout << out;
        }
        std::cout << std::endl;
    }
    int32_t row = static_cast<int32_t>(maze.size());
    setCursor(0, row++);
    std::cout << "move: wasd, quit: q, talk: enter";
    setCursor(0, row++);
    std::cout << "sword: ^, bow: >, arrow: /";
    setCursor(0, row++);
    std::cout << "Player - Health: " << std::string(player.health, '*')
        << " Pos: (" << player.position.first << ","
        << player.position.second << ") Items: ";
    for (const auto& it : player.items)
    {
        std::cout << it.name << " ";
    }
    // Clear out the rest of the stale display items
    std::cout << "                                                                    ";

    setCursor(0, row++);
    if (ai.health > 0)
    {
        std::cout << "AI - Health: " << std::string(ai.health, '*') << " Pos: (" << ai.position.first << "," << ai.position.second << ") Items: ";

        if (!ai.items.empty())
        {
            for (const auto& it : ai.items)
            {
                std::cout << it.name << " ";
            }
        }

        // Clear out the rest of the stale display items
        std::cout << "                                                                    ";

        if (!ai_text.empty())
        {
            setCursor(0, row++);
            std::cout << ai_text << "                                                     ";
        }
    }
    else
    {
        std::cout << "AI - Defeated                                                       ";
        if (!ai_text.empty())
        {
            setCursor(0, row++);
            std::cout << "                                                     ";
        }
    }
}

// Get input string from user with live maze display
std::string getInputString(const std::string& promptText,
    const std::vector<std::string>& maze,
    const Entity& player,
    const Entity& ai,
    const std::vector<Entity>& monsters,
    const std::vector<Entity>& items)
{
    std::string inputString;
    int32_t row = static_cast<int32_t>(maze.size()) + 6;
    
    // Initial draw
    setCursor(0, 0);
    drawMaze(maze, player, ai, monsters, items);
    setCursor(0, row);
    std::cout << promptText;

    while (true)
    {
        // Wait for next key (blocking would be fine, but use small poll for responsiveness)
        if (_kbhit())
        {
            char key = _getch();

            if (key == '\r' || key == '\n')
            {
                // Enter pressed - return the string
                break;
            }
            else if (key == '\b')
            {
                // Backspace
                if (!inputString.empty())
                {
                    inputString.pop_back();
                }
            }
            else if (key >= 32 && key <= 126)
            {
                // Printable ASCII character
                inputString += key;
            }
            // Ignore other special keys
            
            // Update only the prompt line (not the whole maze)
            setCursor(0, row);
            std::cout << promptText << inputString;
            // Clear rest of line
            std::cout << "                                                                    ";
            setCursor(static_cast<int32_t>(promptText.length() + inputString.length()), row);
        }

        Sleep(16); // Small delay to avoid busy-waiting
    }

    return inputString;
}

