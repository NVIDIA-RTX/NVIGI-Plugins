// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <windows.h>
#include <conio.h>
#include "game.h"

// Windows console handle
extern HANDLE hConsole;

// Global AI text for communication
extern std::string ai_text;

// Helper to move the cursor
void setCursor(int32_t x, int32_t y);

// Clear the console screen
void clearScreen();

// Draw the maze to the console
void drawMaze(const std::vector<std::string>& maze,
              const Entity& player,
              const Entity& ai,
              const std::vector<Entity>& monsters,
              const std::vector<Entity>& items);

// Get input string from user with live maze display
std::string getInputString(const std::string& promptText,
                          const std::vector<std::string>& maze,
                          const Entity& player,
                          const Entity& ai,
                          const std::vector<Entity>& monsters,
                          const std::vector<Entity>& items);

