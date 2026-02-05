// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "game.h"

// Helper for Manhattan distance
int32_t manhattan(std::pair<int32_t, int32_t> a, std::pair<int32_t, int32_t> b)
{
    return abs(a.first - b.first) + abs(a.second - b.second);
}

// A* pathfinding, returns sequence of direction chars
std::vector<char> findPathAStar(
    std::pair<int32_t, int32_t> start,
    std::pair<int32_t, int32_t> dest,
    const std::vector<std::string>& maze,
    const std::vector<Entity>& entities)
{
    std::vector<std::tuple<char, int32_t, int32_t>> moves = {
        {'w', -1, 0}, {'s',1,0}, {'a',0,-1}, {'d',0,1}
    };
    // Extract positions from entities that are alive
    std::set<std::pair<int32_t, int32_t>> entity_positions;
    for (const auto& e : entities)
    {
        if (e.health > 0)
        {
            entity_positions.insert(e.position);
        }
    }
    std::priority_queue<
        std::tuple<int32_t, int32_t, std::pair<int32_t, int32_t>, std::vector<char>>,
        std::vector<std::tuple<int32_t, int32_t, std::pair<int32_t, int32_t>, std::vector<char>>>,
        std::greater<>
    > open_set;
    std::set<std::pair<int32_t, int32_t>> visited;

    open_set.push({ manhattan(start, dest), 0, start, {} });
    while (!open_set.empty())
    {
        auto [prio, cost, curr, path] = open_set.top();
        open_set.pop();
        if (curr == dest) return path;
        if (visited.count(curr)) continue;
        visited.insert(curr);
        for (auto [label, dr, dc] : moves)
        {
            int32_t nr = curr.first + dr, nc = curr.second + dc;
            std::pair<int32_t, int32_t> next = { nr, nc };
            bool in_bounds = nr >= 0 && nr < maze.size() && nc >= 0 && nc < maze[0].length();
            bool not_wall = in_bounds && maze[nr][nc] == ' ';
            bool not_occupied = entity_positions.count(next) == 0 || next == dest;
            if (in_bounds && not_wall && not_occupied && !visited.count(next))
            {
                auto newpath = path;
                newpath.push_back(label);
                open_set.push({ cost + 1 + manhattan(next,dest), cost + 1, next, newpath });
            }
        }
    }
    return {}; // Not reachable
}

// Utility: check valid move
bool validMove(std::pair<int32_t, int32_t> pos,
    const std::vector<std::string>& maze)
{
    int32_t r = pos.first, c = pos.second;
    bool in_bounds = r >= 0 && r < maze.size() && c >= 0 && c < maze[0].length();
    bool not_wall = in_bounds && maze[r][c] == ' ';
    return in_bounds && not_wall;
}

void moveEntity(Entity& entity, char dir, const std::vector<std::string>& maze)
{
    int32_t r = entity.position.first, c = entity.position.second;
    std::pair<int32_t, int32_t> newpos = { r, c };
    if (dir == 'w') 
        newpos = { r - 1, c };
    else if (dir == 's') 
        newpos = { r + 1, c };
    else if (dir == 'a') 
        newpos = { r, c - 1 };
    else if (dir == 'd') 
        newpos = { r, c + 1 };

    if ((dir == 'w' || dir == 'a' || dir == 's' || dir == 'd') && validMove(newpos, maze))
        entity.position = newpos;
}

void moveRandom(Entity& entity, const std::vector<std::string>& maze)
{
    int32_t r = entity.position.first, c = entity.position.second;
    std::vector<std::pair<int32_t, int32_t>> opts = {
        {r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}
    };
    std::vector<std::pair<int32_t, int32_t>> valid;
    for (const auto& p : opts)
    {
        if (validMove(p, maze)) valid.push_back(p);
    }
    if (!valid.empty())
    {
        entity.position = valid[rand() % valid.size()];
    }
}

void pickupItems(Entity& entity, std::vector<Entity>& items)
{
    items.erase(std::remove_if(items.begin(), items.end(),
        [&](const Entity& item)
        {
            if (item.position == entity.position)
            {
                entity.items.push_back(item);  // Store full item object
                return true;
            }
            return false;
        }), items.end());
}

bool hasItem(const Entity& entity, const std::string& itemName)
{
    for (const auto& item : entity.items)
    {
        if (item.name == itemName)
        {
            return true;
        }
    }
    return false;
}

void removeItem(Entity& entity, const std::string& itemName)
{
    for (auto it = entity.items.begin(); it != entity.items.end(); ++it)
    {
        if (it->name == itemName)
        {
            entity.items.erase(it);
            return;
        }
    }
}

std::vector<std::pair<int32_t, int32_t>> adjacent(std::pair<int32_t, int32_t> pos)
{
    int32_t r = pos.first, c = pos.second;
    return { {r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1} };
}

// Detect collision and resolve combat for player/ai with monsters
void checkMonsterCollisions(Entity& player, Entity& ai, std::vector<Entity>& monsters)
{
    for (Entity* e : { &player, &ai })
    {
        if (e->health <= 0) continue;
        for (auto& m : monsters)
        {
            if (m.health <= 0) continue;
            bool adj = false;
            if (e->position == m.position)
                adj = true;
            for (const auto& p : adjacent(m.position))
                if (e->position == p)
                    adj = true;

            if (adj)
            {
                if (m.name == "bat")
                {
                    if (hasItem(*e, "bow") && hasItem(*e, "arrow"))
                    {
                        m.health -= 1;
                        removeItem(*e, "arrow");
                    }
                    else
                    {
                        e->health -= 1;
                    }
                }
                else if (hasItem(*e, m.weakness))
                {
                    m.health -= 1;
                }
                else
                {
                    e->health -= 1;
                }
                break; // one combat per turn
            }
        }
    }
    
    // Remove dead monsters from the vector (like items when picked up)
    monsters.erase(std::remove_if(monsters.begin(), monsters.end(),
        [](const Entity& m) { return m.health <= 0; }), monsters.end());
}

void transferAIItemsIfSamePosition(Entity& player, Entity& ai)
{
    if (player.position == ai.position && ai.health > 0 && !ai.items.empty())
    {
        player.items.insert(player.items.end(), ai.items.begin(), ai.items.end());
        ai.items.clear();
    }
}

void dropItemsAroundPosition(std::pair<int32_t, int32_t> position,
    const std::vector<Entity>& itemsToDrop,
    std::vector<Entity>& items,
    const std::vector<std::string>& maze)
{
    if (itemsToDrop.empty())
    {
        return;
    }

    // Find valid empty positions around the death location
    auto adjacentPositions = adjacent(position);
    std::vector<std::pair<int32_t, int32_t>> validPositions = { position };
    validPositions.insert(validPositions.end(), adjacentPositions.begin(), adjacentPositions.end());

    // Filter to only valid positions that aren't occupied by items
    validPositions.erase(std::remove_if(validPositions.begin(), validPositions.end(),
        [&](const std::pair<int32_t, int32_t>& pos)
        {
            if (!validMove(pos, maze))
            {
                return true;  // Remove invalid positions
            }
            // Check if position is occupied by an item
            for (const auto& item : items)
            {
                if (item.position == pos)
                {
                    return true;  // Remove occupied positions
                }
            }
            return false;  // Keep valid unoccupied positions
        }), validPositions.end());

    // Drop items on valid positions
    for (auto item : itemsToDrop)
    {
        if (validPositions.empty())
        {
            break;  // No more empty spaces
        }
        item.position = validPositions[0];
        validPositions.erase(validPositions.begin());
        items.push_back(item);
    }
}


