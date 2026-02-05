// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <vector>
#include <string>
#include <ctime>
#include <inttypes.h>
#include <algorithm>
#include <queue>
#include <set>
#include <tuple>

// Forward declaration
struct Entity;

// Unified Entity structure for all game objects (players, monsters, items)
struct Entity
{
    std::string name;
    std::pair<int32_t, int32_t> position;
    char symbol;
    std::vector<Entity> items;  // Changed to store full Entity objects
    int32_t health;
    std::string weakness;      // Used for monsters
    std::string description;   // Used for items
};

// Helper for Manhattan distance
int32_t manhattan(std::pair<int32_t, int32_t> a, std::pair<int32_t, int32_t> b);

// A* pathfinding, returns sequence of direction chars
std::vector<char> findPathAStar(
    std::pair<int32_t, int32_t> start,
    std::pair<int32_t, int32_t> dest,
    const std::vector<std::string>& maze,
    const std::vector<Entity>& entities);

// Utility: check valid move
bool validMove(std::pair<int32_t, int32_t> pos,
    const std::vector<std::string>& maze);

// Move entity in a direction
void moveEntity(Entity& entity, char dir, const std::vector<std::string>& maze);

// Move entity randomly
void moveRandom(Entity& entity, const std::vector<std::string>& maze);

// Pickup items at entity's position
void pickupItems(Entity& entity, std::vector<Entity>& items);

// Check if entity has an item by name
bool hasItem(const Entity& entity, const std::string& itemName);

// Remove an item by name from entity's inventory
void removeItem(Entity& entity, const std::string& itemName);

// Get adjacent positions
std::vector<std::pair<int32_t, int32_t>> adjacent(std::pair<int32_t, int32_t> pos);

// Handle monster collisions and combat
void checkMonsterCollisions(Entity& player, Entity& ai, std::vector<Entity>& monsters);

// Transfer items from AI to player if same position
void transferAIItemsIfSamePosition(Entity& player, Entity& ai);

// Drop items around a position when entity dies
void dropItemsAroundPosition(std::pair<int32_t, int32_t> position,
    const std::vector<Entity>& itemsToDrop,
    std::vector<Entity>& items,
    const std::vector<std::string>& maze);

