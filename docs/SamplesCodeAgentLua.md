# Code Agent Sample: Lua Dungeon Crawler

## 1. Sample Overview

### What Is This Sample?

This sample demonstrates a **code agent** — an LLM that generates executable code in response to natural language commands. Unlike tool-calling agents that select from predefined functions, code agents write arbitrary programs, enabling more flexible and creative behavior.

The sample is a text-based dungeon crawler where the player controls movement directly (WASD), but commands an AI companion using natural language. When the player types a command like "go get the sword," the LLM generates Lua code that implements that behavior. The AI companion then executes this code each game tick until the task is complete.

**NOTE**: This sample makes use of a CUDA-based backend, and therefore will not work on non-NVIDIA hardware.  Please reference other documentation here on how to switch backends.

### Required Models

The code agent sample requires the following models:

| Plugin | Model Name | GUID |
| ------ | ---------- | ---- |
| nvigi.plugin.gpt.ggml.* | Qwen3 8B Instruct | 545F7EC2-4C29-499B-8FC8-61720DF3C626 |

See the top-level documentation that shipped with your development pack for information on how to download these models.

> NOTE: Other code-capable models may work. Models with strong instruction-following and code generation abilities tend to perform best.

### How to Run

Because of the SDK layout, once the components are built they will be under the `_artifacts` tree; in order for the SDK to run an app like the sample, all of these DLLs and the executable must be in the same directory. We do this by copying the DLLs and EXE into the `bin\x64` directory, and running the app from within that directory, so all plugins are available.

**From Command Line:**

1. Open a command prompt in `<SDK_ROOT>`
2. Run:
```sh
bin\x64\nvigi.codeagentlua.exe data/nvigi.models data/nvigi.test/nvigi.codeagentlua/codeagent_prompt_lua.txt
```

**In Debugger:**

1. Edit project settings for `nvigi/samples/nvigi.codeagentlua`
2. Set "Command" to `<SDK_ROOT>\bin\x64\nvigi.codeagentlua.exe`
3. Set "Command Arguments" to `data/nvigi.models data/nvigi.test/nvigi.codeagentlua/codeagent_prompt_lua.txt`
4. Set "Working Directory" to `<SDK_ROOT>/bin/x64`
5. Build and run `copy_sdk_binaries.bat <cfg>` after each build

### Basic Gameplay

- **w/a/s/d** — Move the player (P) up/left/down/right
- **Enter** — Type a command for the AI companion (A)
- **q** — Quit the game

**Objective:** Navigate the dungeon, collect weapons, and defeat all monsters.

**Entities:**
- `P` — Player (you)
- `A` — AI, your companion
- `G` — Gorgon (defeated with sword)
- `B` — Bat (defeated with bow + arrow)
- `^` — Sword
- `>` — Bow
- `/` — Arrow

**Combat:** Walking adjacent to a monster triggers combat. You need the right weapon or you take damage.

### Example Commands

Commands that tend to work well:

- "Follow me"
- "Go get the sword"
- "Pick up the bow, then pick up an arrow, then kill a bat"
- "Explore the dungeon"
- "Stay here"
- "How many monsters are in the dungeon?"

The AI will generate Lua code that runs each game tick, moving one space at a time toward completing the goal.  

Depending on the LLM, it might require simpler commands. For instance, using GPT5 you can likely say "kill all of the bats" and it will reason out that it needs to get the bow, then the arrows then go kill all of the bats.  Using smaller local LLMs, that same command will likely send the AI to attack the bats unarmed.  Sometimes it is necessary for SLMs to even break compound tasks into multiple simple tasks.  At times, "Go get the sword, then kill the gorgon" might work, but it will likely work more often as "Go get the sword", wait for that task to complete, then follow it with "Go kill the gorgon".  Experiment with it and see what works.

Whenever you run a command, you can look at the file located in <WORKING_DIR>/ai_func_out.txt to see the function that was generated.

---

## 2. Code Agents vs Tool-Calling Agents

Modern LLM applications often need to interact with external systems — querying databases, calling APIs, or controlling game characters. There are two primary approaches: **tool-calling agents** and **code agents**.

### What Is a Tool-Calling Agent?

A tool-calling agent works by having the LLM select from a predefined set of functions. The application provides a schema describing available tools (name, parameters, descriptions), and the LLM outputs structured JSON indicating which tool to call and with what arguments.

```
User: "What's the weather in Austin?"

LLM Output: { "tool": "get_weather", "args": { "city": "Austin" } }

Application: Calls get_weather("Austin"), returns "72°F, sunny"

LLM: "The weather in Austin is 72°F and sunny."
```

**Advantages:**
- **Predictable** — Only predefined functions can be called
- **Easy to secure** — Each tool is explicitly implemented and validated
- **Structured output** — JSON parsing is well-understood
- **Lower risk** — The LLM cannot execute arbitrary logic

**Disadvantages:**
- **Limited flexibility** — Complex behaviors require many predefined tools
- **Combinatorial explosion** — As capabilities grow, tool count explodes
- **No novel solutions** — The LLM cannot invent behaviors you didn't anticipate
- **Slow** - Each function call is likely multiple inference calls to the LLM.
- **No logic** - We rely on the LLM for all logic, when code is superior at exact solutions.
- **Verbose prompts** - the json specifications make for large and difficult to read function declarations.

### What Is a Code Agent?

A code agent generates executable code rather than selecting from predefined functions. Given a natural language request, the LLM writes a program that implements the desired behavior. The application then executes that code in a sandboxed environment.

```
User: "Go get the sword, then come back to me."

LLM Output:
function update_func(player, ai, monsters, items)
    if not ai:has_item("sword") then
        for _, item in ipairs(items) do
            if item.name == "sword" then
                local path = ai:find_path_to(item.position)
                if path and #path > 0 then
                    ai:move(path[1])
                    return "Moving toward sword"
                end
            end
        end
    else
        local path = ai:find_path_to(player.position)
        if path and #path > 0 then
            ai:move(path[1])
            return "Returning to player"
        end
    end
    return "Done"
end

Application: Executes this Lua function each game tick
```

**Advantages:**
- **Highly flexible** — Can express any logic the language supports
- **Compositional** — Complex behaviors emerge from simple primitives
- **Novel solutions** — LLM can invent behaviors not explicitly designed
- **Fewer APIs** — Expose primitives, let the LLM compose them
- **More Readable Prompts** - Common scripting function declarations are the documentation.
- **Smaller prompts** - Being able to write code allows the system prompt to be more concise, and faster to load.
- **Faster** - one inference call can generate code that executes for the duration of the task.

**Disadvantages:**
- **Security risk** — Arbitrary code execution is inherently dangerous
- **Unpredictable** — Generated code may have bugs or unexpected behavior
- **Harder to validate** — Cannot easily verify correctness before execution
- **Requires sandboxing** — Must carefully restrict what the code can do

### When to Use Each

| Scenario | Recommended Approach |
|----------|---------------------|
| Limited, well-defined actions | Tool-calling |
| High-security requirements | Tool-calling |
| Simple request-response patterns | Tool-calling |
| Complex, multi-step behaviors | Code agent |
| Creative or exploratory tasks | Code agent |
| Game AI or robotics control | Code agent |
| When you can sandbox effectively | Code agent |

**Hybrid approaches** are also possible: use tool-calling for high-risk operations (payments, deletions) and code agents for low-risk, creative tasks (game AI, data analysis).

This sample demonstrates a code agent because controlling a game character requires flexible, multi-step logic that would be tedious to express as individual tool calls. The character needs to pathfind, make decisions based on game state, and adapt to changing circumstances — all of which are naturally expressed as code.

---

## 3. Security Considerations

Executing LLM-generated code is inherently risky. Before evaluating scripting languages, we must understand the threats we need to mitigate.

### The Risk of Code Agents

Unlike tool-calling agents where the application controls exactly what functions can be invoked, code agents execute arbitrary programs. The LLM might generate code that:

- Accidentally or intentionally accesses the filesystem, network, or system resources
- Consumes unbounded memory, crashing the application
- Enters infinite loops, hanging the application
- Exploits language features to escape the sandbox
- Corrupts application state in unexpected ways

These risks exist even when the LLM is "well-intentioned" — bugs in generated code can trigger any of these behaviors. A robust code agent must defend against both malicious and accidental misuse.

### Threat Model

We identified six categories of threats that a code agent sandbox must address:

#### Dangerous Function Access

**Threat:** Generated code calls functions that access the filesystem, execute shell commands, load arbitrary modules, or interact with the network.

**Examples (pseudocode):**
- `execute_shell("rm -rf /")` — Run shell commands
- `read_file("/etc/passwd")` — Read sensitive files
- `import("network_library")` — Load dangerous modules
- `eval(arbitrary_code_string)` — Execute arbitrary code

**Mitigation:** The sandbox must either not load dangerous libraries, or selectively remove dangerous functions after loading.

#### Memory Exhaustion

**Threat:** Generated code allocates unbounded memory, exhausting system resources and crashing the application.

**Examples (pseudocode):**
- `while true: list.append(large_string)` — Infinite allocation
- `s = ""; for i in 1..billion: s = s + "x"` — String concatenation bomb

**Mitigation:** The runtime must enforce a memory limit and fail gracefully when exceeded.

#### Stack Overflow

**Threat:** Generated code uses deep or infinite recursion, overflowing the call stack.

**Examples (pseudocode):**
- `function f(): f()` — Infinite recursion
- Mutually recursive functions that never terminate

**Mitigation:** The runtime must track call depth and abort execution when a threshold is exceeded.

#### Infinite Loops / Hangs

**Threat:** Generated code enters an infinite loop, hanging the application and preventing further user interaction.

**Examples (pseudocode):**
- `while true: pass` — Infinite loop with no I/O
- `for i in 1..infinity: pass` — Extremely long loop

**Mitigation:** The runtime must enforce a time limit or instruction count limit and abort long-running code.

#### Prototype/Metatable Manipulation

**Threat:** Generated code manipulates object prototypes or metatables to bypass sandbox restrictions or corrupt internal data structures. Many scripting languages allow customizing how objects behave through prototype chains, metatables, or similar mechanisms.

**Examples (pseudocode):**
- `set_prototype(entity, malicious_handler)` — Override entity behavior
- `get_prototype(entity).write_handler = null` — Disable write protection

**Mitigation:** Remove or restrict access to prototype/metatable manipulation functions.

#### Game State Corruption

**Threat:** Generated code directly modifies application state in ways that break game logic or cause crashes.

**Examples (pseudocode):**
- `player.health = -999` — Invalid health value
- `monster.position = null` — Break rendering/pathfinding
- `ai.items = "not a list"` — Type corruption

**Mitigation:** Expose application state through controlled interfaces that validate or reject invalid modifications.

### Summary

The six categories this sample has addressed are:

| Threat | Required Mitigation |
|--------|---------------------|
| Dangerous functions | Selective library loading, function removal |
| Memory exhaustion | Custom allocator with limits |
| Stack overflow | Call depth tracking |
| Infinite loops | Timeout / instruction counting |
| Metatable manipulation | Remove metatable functions |
| State corruption | Controlled state access, validation |

This is not exhaustive and can expand or contract depending on your use case and the language you choose to have the ai write in.

The next section evaluates scripting languages against these requirements.

--

## 4. Scripting Engine Evaluation

Choosing a scripting language for a code agent requires balancing several concerns: security (can we sandbox it?), embeddability (can we integrate it into C++?), LLM familiarity (will models generate correct code?), and runtime characteristics (performance, memory footprint).

### Candidates Considered

We evaluated three scripting languages:

#### Python

**Pros:**
- LLMs are extremely proficient at generating Python code
- Rich standard library
- Familiar to most developers
- Excellent documentation and community

**Cons:**
- Difficult to sandbox securely — the standard library has many escape hatches
- Larger than desired runtime footprint (~10-20MB+ for embedded Python)
- Larger than desired compile time footprint (~150MB+ for embedded Python)
- Complex C++ integration (reference counting, GIL, etc.)
- Hard to limit memory and CPU usage reliably

Python's sandboxing challenges are well-documented. Even "restricted" Python environments have historically been bypassed. For a code agent where untrusted LLM-generated code runs, this is a significant concern.

If you do consider python, pay special attention to these areas
- If you are still using the GIL (python <= 3.13), then threaded implementations become difficult.  It can become impossible to cleanly restart the python interpreter.
- Subinterpreters can be restarted, but at a cost of lost memory.
- Subprocesses can be restarted cleanly, handle memory loss properly, but can be difficult to debug.

#### ChaiScript

**Pros:**
- Designed specifically for C++ embedding
- Header-only library, easy to integrate
- Clean syntax, similar to JavaScript
- No external dependencies

**Cons:**
- LLMs have limited training data on ChaiScript — generated code often has syntax errors
- Smaller community and less documentation
- Sandboxing support is limited
- Less battle-tested than alternatives

ChaiScript's obscurity was a significant problem. Models frequently hallucinated non-existent functions or used incorrect syntax, requiring extensive prompt engineering and retry logic. 

If you do consider Chaiscript, pay special attention to these areas:
- ChaiScript also has limited ability to properly sandbox memory usage.  
- Special care must be paid when using Chaiscript in a threaded environment with how variable scoping works.

#### Lua

**Pros:**
- Designed for embedding from the start
- Tiny footprint (~200KB)
- Small compile time footprint (a few MB)
- Excellent sandboxing support — can selectively load libraries and remove functions
- Custom allocator support for memory limiting
- Debug hooks for timeout/recursion control
- LLMs generate reasonable Lua code (widely used in game modding, WoW, Roblox, etc.)
- Battle-tested in thousands of game engines

**Cons:**
- Syntax can be unfamiliar (1-indexed arrays, `~=` for not-equal, `:` vs `.`)
- Smaller standard library than Python
- LLMs occasionally make syntax errors (though fewer than ChaiScript)

### Why Lua Won

Lua was selected for this sample because it best balances our requirements:

| Requirement | Python | ChaiScript | Lua |
|-------------|--------|------------|-----|
| Sandboxing | ❌ Difficult | ⚠️ Limited | ✅ Excellent |
| Memory limiting | ❌ Hard | ❌ No | ✅ Custom allocator |
| Timeout control | ❌ Complex | ❌ No | ✅ Debug hooks |
| LLM code quality | ✅ Excellent | ❌ Poor | ⚠️ Good |
| Embedding ease | ⚠️ Complex | ✅ Easy | ✅ Easy |
| Runtime size | ❌ Large | ✅ Small | ✅ Tiny |

While Python would produce better LLM-generated code, it cannot be safely sandboxed. ChaiScript embeds easily but is difficult to effectively secure and LLMs struggle with its syntax. Lua provides the best combination: reasonable LLM output quality with excellent security controls.

The occasional Lua syntax errors from LLMs can be mitigated with retry logic (see Section 6), making it a practical choice for production code agents.

---

## 5. How Lua Addresses Each Threat

This section explains how the sample implements each security mitigation using Lua's features. The relevant code is in `lua_bindings.cpp`.

### Selective Library Loading

Unlike `luaL_openlibs()` which loads everything, we selectively load only safe libraries:

```cpp
// Load safe libraries
luaL_requiref(L, "_G", luaopen_base, 1);      // Basic functions
luaL_requiref(L, LUA_TABLIBNAME, luaopen_table, 1);   // Table manipulation
luaL_requiref(L, LUA_STRLIBNAME, luaopen_string, 1);  // String functions
luaL_requiref(L, LUA_MATHLIBNAME, luaopen_math, 1);   // Math functions
luaL_requiref(L, LUA_UTF8LIBNAME, luaopen_utf8, 1);   // UTF-8 support
luaL_requiref(L, LUA_OSLIBNAME, luaopen_os, 1);       // OS (then sanitized)

// NOT loading:
// - io: File I/O
// - debug: Can break sandbox
// - package: Can load arbitrary modules
// - coroutine: Could complicate timeout handling
```

By never loading dangerous libraries, those functions simply don't exist — there's nothing to exploit.

### Dangerous Function Removal

Even safe libraries contain dangerous functions. After loading, we remove them:

```cpp
// Remove dangerous os functions (keep time, date, difftime, clock)
removeFromLib(L, "os", "execute");    // Shell commands
removeFromLib(L, "os", "exit");       // Terminate program
removeFromLib(L, "os", "remove");     // Delete files
removeFromLib(L, "os", "rename");     // Move files
removeFromLib(L, "os", "getenv");     // Environment variables

// Remove dangerous base functions
removeGlobal(L, "dofile");            // Execute Lua file
removeGlobal(L, "loadfile");          // Load Lua file
removeGlobal(L, "load");              // Execute arbitrary strings
removeGlobal(L, "require");           // Module loading
```

The `removeFromLib` and `removeGlobal` helpers simply set these to `nil`, making them undefined.

### Custom Memory Allocator

Lua allows replacing its memory allocator via `lua_newstate()`. We provide a custom allocator that tracks usage and enforces a limit:

```cpp
static void* luaLimitedAlloc(void* ud, void* ptr, size_t osize, size_t nsize)
{
    LuaMemoryTracker* tracker = static_cast<LuaMemoryTracker*>(ud);
    
    // Calculate new usage
    size_t delta = nsize - (ptr ? osize : 0);
    size_t newUsage = tracker->currentUsage + delta;
    
    // Refuse allocation if it would exceed limit
    if (newUsage > tracker->limit)
    {
        return nullptr;  // Lua throws out-of-memory error
    }
    
    // Perform allocation and track usage
    void* newPtr = realloc(ptr, nsize);
    if (newPtr) tracker->currentUsage = newUsage;
    return newPtr;
}

// Create Lua state with 100MB limit
lua_State* L = lua_newstate(luaLimitedAlloc, tracker);
```

When the limit is reached, Lua receives a null pointer and throws a recoverable out-of-memory error.

### Debug Hooks for Timeout and Recursion

Lua's debug hooks let us intercept execution at key points. We use three hook types:

```cpp
static void luaTimeoutHook(lua_State* L, lua_Debug* ar)
{
    // Track call depth for recursion limit
    if (ar->event == LUA_HOOKCALL || ar->event == LUA_HOOKTAILCALL)
    {
        g_luaCallDepth++;
        if (g_luaCallDepth > LUA_MAX_CALL_DEPTH)  // 200
        {
            luaL_error(L, "Stack overflow (recursion depth exceeded)");
        }
    }
    else if (ar->event == LUA_HOOKRET)
    {
        g_luaCallDepth--;
    }
    
    // Check timeout every N instructions
    if (ar->event == LUA_HOOKCOUNT)
    {
        auto elapsed = now - g_luaStartTime;
        if (elapsed > LUA_TIMEOUT_MS)  // 1000ms
        {
            luaL_error(L, "Execution timed out");
        }
    }
}

// Install hook before calling Lua code
lua_sethook(L, luaTimeoutHook, LUA_MASKCALL | LUA_MASKRET | LUA_MASKCOUNT, 1000);
```

This catches both infinite loops (via timeout) and infinite recursion (via call depth tracking).

### Metatable Protection

Metatables control how Lua objects behave. If code could modify our Entity metatable, it could bypass protections. We remove the metatable functions entirely:

```cpp
removeGlobal(L, "getmetatable");  // Can't inspect metatables
removeGlobal(L, "setmetatable");  // Can't modify metatables
removeGlobal(L, "rawget");        // Can't bypass __index
removeGlobal(L, "rawset");        // Can't bypass __newindex
removeGlobal(L, "rawequal");      // Can't bypass __eq
removeGlobal(L, "rawlen");        // Can't bypass __len
```

Without these functions, code can only interact with entities through our controlled `__index` and `__newindex` metamethods.

### Entity Field Write Protection

Our `__newindex` metamethod controls what happens when code writes to entity fields. We block writes to built-in fields while allowing custom fields for AI state:

```cpp
static int entity_newindex(lua_State* L)
{
    Entity* e = checkEntity(L, 1);
    const char* key = luaL_checkstring(L, 2);

    // Block writes to built-in fields
    if (strcmp(key, "name") == 0 || strcmp(key, "position") == 0 ||
        strcmp(key, "health") == 0 || /* ... */)
    {
        return luaL_error(L, "Cannot modify '%s' - use methods instead", key);
    }

    // Allow custom fields (stored in shadow table)
    getEntityCustomFields(L, e);
    lua_pushstring(L, key);
    lua_pushvalue(L, 3);
    lua_settable(L, -3);
    return 0;
}
```

This prevents `monster.health = 0` (cheating) while allowing `ai.my_target = monster` (state tracking).

### Summary

| Threat | Lua Feature Used | Implementation |
|--------|------------------|----------------|
| Dangerous functions | Selective loading | `luaL_requiref` for safe libs only |
| Dangerous functions | Function removal | `removeGlobal`, `removeFromLib` |
| Memory exhaustion | Custom allocator | `lua_newstate` with tracking allocator |
| Stack overflow | Debug hooks | `LUA_MASKCALL` / `LUA_MASKRET` |
| Infinite loops | Debug hooks | `LUA_MASKCOUNT` with timeout |
| Metatable abuse | Function removal | Remove `get/setmetatable`, `raw*` |
| State corruption | Metamethods | `__newindex` blocks built-in fields |

---

## 6. Design Decisions for LLM Success

Security is necessary but not sufficient — the LLM must also generate *correct* code. This section covers design decisions that improve code generation success rates.

### API Design: Entity Methods vs Global Functions

We expose functionality as both entity methods and global functions, but the system prompt emphasizes methods:

```lua
-- Method syntax (preferred, documented in prompt)
ai:move("w")
ai:has_item("sword")
local path = ai:find_path_to(player.position)

-- Global function syntax (also works, but not documented)
move_entity(ai, "w")
has_item(ai, "sword")
local path = find_path_astar(ai.position, player.position, entities)
```

**Why methods work better:**

1. **More idiomatic** — LLMs trained on Lua (game mods, Roblox, etc.) see method syntax frequently
2. **Cleaner code** — `ai:move("w")` vs `move_entity(ai, "w")` is more readable
3. **Fewer parameters** — Methods automatically use the entity and global maze, reducing chances for error
4. **Better autocomplete patterns** — LLMs predict `ai:` then method name more reliably

We still register the global functions as a fallback — if the LLM accidentally generates `move_entity(ai, "w")`, it works. Defense in depth for correctness.

### Prompt Engineering

The system prompt (`codeagent_prompt_lua.txt`) is carefully structured to prevent common LLM errors:

**1. Explicit examples for ambiguous cases:**
```
NOTE: These methods take POSITIONS, not strings or names!
WRONG: ai:find_path_to("sword")      -- strings don't work!
WRONG: ai:distance_to(monster.weakness)  -- that's a string!
RIGHT: ai:find_path_to(item.position)    -- use .position
```

Without this, LLMs frequently hallucinate that `find_path_to("sword")` will find the sword by name.

**2. Clear item location rules:**
```
Items exist in ONE of two places (never both):
1) ON THE GROUND: Found in the global "items" table
2) IN AN INVENTORY: Found in entity.items (use entity:has_item() to check)
```

LLMs often confuse ground items with inventory items without explicit guidance.

**3. Function signature with parameter types:**
```lua
entity:find_path_to(position)
    --[[
    Parameters:
        position: A position table {row, col} - use entity.position for entities
    Returns:
        table: Array of direction strings, or empty {} if unreachable
    ]]
```

Documenting return types and parameter formats reduces hallucinations.

**4. Lua-specific reminders:**
```
- Lua tables are 1-indexed! Use: for i, item in ipairs(array) do ... end
- Method call: entity:method() (passes entity as first arg)
- Check empty table: if #path > 0 then ... end
```

LLMs trained primarily on Python/JavaScript often forget Lua's 1-indexing.

### Error Handling and Retry

LLM-generated code can fail on the first attempt. We implement a retry loop that feeds errors back to the model in an attempt to autocorrect.

```cpp
for (int attempt = 0; attempt <= MAX_RETRIES && !success; attempt++)
{
    if (attempt == 0)
    {
        fullPrompt = "Write an update_func that satisfies: \"" + prompt + "\"";
    }
    else
    {
        // Include the failed code and error message
        fullPrompt = "Your previous code:\n" + code + 
                     "\nfailed with this error:\n" + lastError +
                     "\n\nGenerate corrected code.";
    }

    code = llmCreateAIFunc(fullPrompt);
    
    // Try to compile
    if (luaL_dostring(L, code.c_str()) != LUA_OK)
    {
        lastError = lua_tostring(L, -1);
        continue;  // Retry with error context
    }

    // Try runtime
    std::string result = callLuaAIFunc(L, ...);
    if (result.starts_with("Function failed"))
    {
        lastError = result;
        continue;  // Retry with error context
    }

    success = true;
}
```

This catches both compile-time errors (syntax mistakes) and runtime errors (nil indexing, type errors), giving the LLM a chance to self-correct.

### Reference Semantics with Userdata

We pass C++ entities to Lua as userdata pointers, not copies:

```cpp
// Push entity as pointer (reference semantics)
static void pushEntity(lua_State* L, Entity* entity)
{
    Entity** udata = (Entity**)lua_newuserdata(L, sizeof(Entity*));
    *udata = entity;
    luaL_getmetatable(L, ENTITY_META);
    lua_setmetatable(L, -2);
}
```

**Why this matters:**

1. **Changes persist** — When Lua code calls `ai:move("w")`, the C++ entity's position actually changes. No sync-back needed.
2. **No stale data** — Reading `monster.health` always returns the current value
3. **Efficient** — No copying of entity data back and forth

The alternative (copying entities to Lua tables) would require syncing changes back to C++ after every Lua call, which is error-prone and inefficient.

### Shadow Tables for Custom Fields

The AI needs to store persistent state across function calls (e.g., "Tell me the number of unique positions I have occupied?"). We support custom fields on entities using a shadow table:

```cpp
// Shadow table: maps entity pointers to their custom fields
static void getEntityCustomFields(lua_State* L, Entity* entity)
{
    pushShadowTable(L);  // Global table in registry
    
    lua_pushlightuserdata(L, entity);  // Use pointer as key
    lua_gettable(L, -2);
    
    if (lua_isnil(L, -1))
    {
        // Create new table for this entity's custom fields
        lua_newtable(L);
        lua_pushlightuserdata(L, entity);
        lua_pushvalue(L, -2);
        lua_settable(L, -4);  // shadow[entity] = {}
    }
    lua_remove(L, -2);
}
```

This allows:
```lua
-- AI can store custom state for tracking
player.visited_positions = player.visited_positions or {}
local pos_key = player.position[1] .. "," .. player.position[2]
player.visited_positions[pos_key] = true

-- Count unique positions
local count = 0
for _ in pairs(player.visited_positions) do count = count + 1 end
return "You have visited " .. count .. " unique positions"

-- Built-in fields still work (read-only)
local pos = player.position  -- reads from C++
```

The shadow table uses weak keys, so when an entity is garbage collected, its custom fields are automatically cleaned up.

**Caveat:** While this feature works correctly, the current SLMs rarely generate code that uses it effectively, at least with Lua and our current SLM. Stateful tasks like "track unique positions visited" require the model to correctly initialize state, update it each call, and avoid resetting it with `local`. In practice, SLM-generated code often gets this wrong — especially with smaller models. Consider this an available capability rather than a reliable feature for LLM-generated code. Also consider future models or cloud models might be far superior at using such techniques.

### Summary

| Decision | Problem Solved | Implementation |
|----------|----------------|----------------|
| Method syntax | Cleaner code, better LLM predictions | `__index` returns bound methods |
| Explicit examples in prompt | Prevent common hallucinations | "WRONG/RIGHT" examples |
| Error feedback retry | Self-correction of mistakes | Loop with error context |
| Userdata pointers | Changes persist, no sync needed | `pushEntity` stores pointer |
| Shadow tables | Custom state without corrupting entities | Registry table with weak keys |

---

## 7. Alternative Architecture — Curated Function Libraries

While this sample demonstrates runtime code generation, some applications may require stricter control over what code can execute. An alternative approach combines LLM code generation during development with human curation for production.

### The Hybrid Approach

Instead of generating code at runtime, you can:

1. **Development Phase:** Have your team interact with the code agent during development, making the kinds of requests end users would make. The LLM generates functions as usual, but each generated function is cached and logged.

2. **Curation Phase:** Developers review the generated functions, validating and approving the ones that work correctly. Over time, you build a library of vetted, production-ready functions.

3. **Production Phase:** When an end user makes a request, use semantic matching (embeddings, similarity search) to find a pre-approved function that matches their intent. If a match exists, use it. If not, either:
   - **Strict mode:** Return an error ("I don't know how to do that yet")
   - **Fallback mode:** Generate code at runtime (with all the sandboxing protections)

### Benefits

- **Human oversight** — Every function that runs in production has been reviewed
- **Predictable behavior** — Users get tested, validated code paths
- **Reduced latency** — No LLM inference needed for common requests
- **Security confidence** — No runtime code generation in strict mode
- **Continuous improvement** — New requests in fallback mode become candidates for curation

### Implementation Considerations

**Function storage:** Store approved functions with metadata (original prompt, function code, semantic embedding of the request).

**Semantic matching:** Use an embedding model to convert user requests to vectors, then find the nearest approved function. Set a similarity threshold — below it, consider the request "unmatched."

**Parameterization:** Some functions may need light parameterization (e.g., "go get the sword" vs "go get the bow"). Consider whether exact matches are sufficient or if you need template-based functions.

**Fallback policy:** Decide whether unmatched requests should fail gracefully or trigger runtime generation. This is a security/flexibility tradeoff.

### When to Use This Approach

| Scenario | Recommendation |
|----------|----------------|
| High-security production environment | Strict mode (no runtime generation) |
| Internal tools with trusted users | Fallback mode acceptable |
| Games with predictable command patterns | Curated library works well |
| Open-ended creative applications | Runtime generation may be necessary |

This approach lets you capture the benefits of LLM code generation during development while maintaining tight control over what runs in production.

---

## 8. Conclusion

Code agents offer a powerful alternative to tool-calling agents for tasks that require flexible, multi-step logic. By having the LLM generate executable code rather than selecting from predefined functions, we can build AI companions that adapt to novel situations and compose behaviors in ways we didn't explicitly anticipate.

However, this power comes with responsibility. Executing LLM-generated code requires careful sandboxing to prevent:
- Dangerous function access (file I/O, shell execution)
- Resource exhaustion (memory, CPU, stack)
- Sandbox escapes (metatable manipulation)
- Application state corruption

Lua proved to be an excellent choice for this sample, offering:
- Selective library loading and function removal
- Custom memory allocators for hard limits
- Debug hooks for timeout and recursion control
- A syntax that LLMs can generate with reasonable accuracy

Beyond security, we found that API design and prompt engineering significantly impact LLM success rates. Method syntax, explicit WRONG/RIGHT examples, and error-feedback retry loops all contribute to more reliable code generation.

This sample demonstrates that code agents are practical today, but require thoughtful engineering. The techniques shown here — sandboxing, API design, prompt engineering, and graceful error handling — provide a foundation for building code agents in your own applications.

**Key Takeaways:**

1. **Code agents vs tool-calling** — Choose based on task complexity and security requirements
2. **Language choice matters** — Prioritize sandboxing capability, then LLM familiarity
3. **Defense in depth** — Multiple layers of protection for each threat category
4. **Design for LLM success** — API design and prompts are as important as the runtime
5. **Expect iteration** — Error feedback and retry loops improve success rates significantly

We encourage you to experiment with this sample, try different commands, and explore how the AI companion responds. The code is designed to be readable and modifiable — use it as a starting point for your own code agent implementations.

