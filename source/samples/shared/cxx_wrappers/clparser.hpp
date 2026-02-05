// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//

// Modern C++20/C++2x Command Line Parser

#pragma once

#include <algorithm>
#include <charconv>
#include <concepts>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <format>

namespace clparser {

// Concept for integral types (int, unsigned, size_t, etc.)
template<typename T>
concept Integral = std::integral<T>;

// Concept for floating point types (float, double)
template<typename T>
concept FloatingPoint = std::floating_point<T>;

// Concept for any numeric type
template<typename T>
concept Numeric = Integral<T> || FloatingPoint<T>;

// Concept for string-like types
template<typename T>
concept StringLike = std::convertible_to<T, std::string_view>;

// Command option configuration
struct CommandOption {
    std::string short_name;
    std::string long_name;
    std::string description;
    std::string default_value;
    bool required{false};

    // Current value (mutable state)
    std::string value;

    [[nodiscard]] constexpr bool has_value() const noexcept {
        return !value.empty();
    }

    [[nodiscard]] constexpr bool is_default() const noexcept {
        return value == default_value;
    }
};

// Parser result type
enum class ParseResult {
    Success,
    UnknownCommand,
    MissingValue,
    MissingRequired,
    InvalidFormat
};

// Exception type for parsing errors
class ParseError : public std::runtime_error {
public:
    explicit ParseError(std::string_view msg) : std::runtime_error(std::string(msg)) {}
    ParseError(ParseResult result, std::string_view details)
        : std::runtime_error(format_error(result, details)), result_(result) {}

    [[nodiscard]] ParseResult result() const noexcept { return result_; }

private:
    ParseResult result_{ParseResult::Success};

    static std::string format_error(ParseResult result, std::string_view details) {
        switch (result) {
            case ParseResult::UnknownCommand:
                return std::format("Unknown command: {}", details);
            case ParseResult::MissingValue:
                return std::format("Missing value for command: {}", details);
            case ParseResult::MissingRequired:
                return std::format("Missing required command: {}", details);
            case ParseResult::InvalidFormat:
                return std::format("Unexpected argument format: {}", details);
            default:
                return std::string(details);
        }
    }
};

// Main parser class
class CommandLineParser {
public:
    CommandLineParser() = default;
    ~CommandLineParser() = default;

    // Non-copyable, movable
    CommandLineParser(const CommandLineParser&) = delete;
    CommandLineParser& operator=(const CommandLineParser&) = delete;
    CommandLineParser(CommandLineParser&&) noexcept = default;
    CommandLineParser& operator=(CommandLineParser&&) noexcept = default;

    // Add a command option with builder-like pattern
    template<StringLike S1 = std::string_view, StringLike S2 = std::string_view,
             StringLike S3 = std::string_view, StringLike S4 = std::string_view>
    CommandLineParser& add_command(S1&& short_name, S2&& long_name,
                                   S3&& description, S4&& default_value = "",
                                   bool required = false) {
        std::string short_str(std::forward<S1>(short_name));
        std::string long_str(std::forward<S2>(long_name));
        std::string desc(std::forward<S3>(description));
        std::string def_val(std::forward<S4>(default_value));

        CommandOption option{
            .short_name = short_str,
            .long_name = long_str,
            .description = std::move(desc),
            .default_value = def_val,
            .required = required,
            .value = std::move(def_val)
        };

        if (!short_str.empty()) {
            commands_[short_str] = option;
        }
        commands_[long_str] = std::move(option);

        return *this;
    }

    // Parse command line arguments from span
    void parse(std::span<const char* const> args) {
        for (auto it = args.begin(); it != args.end(); ++it) {
            std::string_view arg(*it);

            if (!arg.starts_with('-')) {
                throw ParseError(ParseResult::InvalidFormat, arg);
            }

            // Extract key: handle both -x and --xxx formats
            std::string_view key = arg.starts_with("--") ? arg.substr(2) : arg.substr(1);
            
            auto cmd_it = commands_.find(std::string(key));
            if (cmd_it == commands_.end()) {
                throw ParseError(ParseResult::UnknownCommand, arg);
            }

            auto& cmd = cmd_it->second;
            
            // Check if next arg is a value
            if (std::next(it) != args.end()) {
                std::string_view next_arg(*std::next(it));
                if (!next_arg.starts_with('-')) {
                    cmd.value = *std::next(it);
                    ++it; // consume the value
                    
                    // Update alternate key (short/long) with same value
                    update_alternate_key(key, cmd);
                    continue;
                }
            }

            // No value provided - for boolean flags, set to "true" to indicate presence
            // For other types, use default value or error if no default
            if (cmd.default_value.empty()) {
                // Treat as boolean flag - presence means true
                cmd.value = "true";
                update_alternate_key(key, cmd);
            }
            // If default exists, the flag was set but we keep the existing value
        }

        validate_required_commands();
    }

    // Parse from argc/argv (traditional C-style)
    void parse(int argc, char* argv[]) {
        if (argc <= 1) return; // No arguments to parse
        std::vector<const char*> args_vec(argv + 1, argv + argc);
        parse(std::span{args_vec});
    }

    // Get command value with optional return
    [[nodiscard]] std::optional<std::string_view> get_optional(std::string_view name) const noexcept {
        if (auto it = commands_.find(std::string(name)); it != commands_.end()) {
            return std::string_view{it->second.value};
        }
        return std::nullopt;
    }

    // Get command value (throws if not found)
    [[nodiscard]] std::string_view get(std::string_view name) const {
        if (auto result = get_optional(name)) {
            return *result;
        }
        throw ParseError(ParseResult::UnknownCommand, name);
    }

    // Get command value with default fallback
    [[nodiscard]] std::string_view get_or(std::string_view name, std::string_view default_val) const noexcept {
        return get_optional(name).value_or(default_val);
    }

    // Check if a flag/parameter was present on the command line
    // For boolean flags: returns true if flag was present, false otherwise
    // Usage: if (parser.has("verbose")) { ... }
    [[nodiscard]] bool has(std::string_view name) const noexcept {
        if (auto it = commands_.find(std::string(name)); it != commands_.end()) {
            // For flags with empty defaults, "true" means it was present
            if (it->second.default_value.empty()) {
                return it->second.value == "true";
            }
            // For parameters with defaults, check if explicitly set
            return !it->second.is_default();
        }
        return false;
    }

    // Check if command was explicitly set (not default)
    [[nodiscard]] bool is_set(std::string_view name) const noexcept {
        if (auto it = commands_.find(std::string(name)); it != commands_.end()) {
            return !it->second.is_default();
        }
        return false;
    }

    // ========== Numeric Conversion Methods ==========

    // Convert string to numeric type (throws on error)
    template<Numeric T>
    [[nodiscard]] T get_as(std::string_view name) const {
        auto str_value = get(name);
        
        T result;
        auto [ptr, ec] = std::from_chars(str_value.data(), 
                                         str_value.data() + str_value.size(), 
                                         result);
        
        if (ec == std::errc()) {
            return result;
        } else if (ec == std::errc::invalid_argument) {
            throw ParseError(std::format("Invalid numeric value for '{}': '{}'", name, str_value));
        } else if (ec == std::errc::result_out_of_range) {
            throw ParseError(std::format("Numeric value out of range for '{}': '{}'", name, str_value));
        }
        
        throw ParseError(std::format("Failed to parse numeric value for '{}': '{}'", name, str_value));
    }

    // Convert string to numeric type with optional return (noexcept)
    template<Numeric T>
    [[nodiscard]] std::optional<T> get_as_optional(std::string_view name) const noexcept {
        auto str_opt = get_optional(name);
        if (!str_opt || str_opt->empty()) {
            return std::nullopt;
        }
        
        auto str_value = *str_opt;
        T result;
        auto [ptr, ec] = std::from_chars(str_value.data(), 
                                         str_value.data() + str_value.size(), 
                                         result);
        
        if (ec == std::errc()) {
            return result;
        }
        return std::nullopt;
    }

    // Convert string to numeric type with default fallback
    template<Numeric T>
    [[nodiscard]] T get_as_or(std::string_view name, T default_value) const noexcept {
        return get_as_optional<T>(name).value_or(default_value);
    }

    // Convenience methods for common types
    [[nodiscard]] int get_int(std::string_view name) const {
        return get_as<int>(name);
    }

    [[nodiscard]] std::optional<int> get_int_optional(std::string_view name) const noexcept {
        return get_as_optional<int>(name);
    }

    [[nodiscard]] int get_int_or(std::string_view name, int default_value) const noexcept {
        return get_as_or<int>(name, default_value);
    }

    [[nodiscard]] unsigned int get_uint(std::string_view name) const {
        return get_as<unsigned int>(name);
    }

    [[nodiscard]] std::optional<unsigned int> get_uint_optional(std::string_view name) const noexcept {
        return get_as_optional<unsigned int>(name);
    }

    [[nodiscard]] unsigned int get_uint_or(std::string_view name, unsigned int default_value) const noexcept {
        return get_as_or<unsigned int>(name, default_value);
    }

    [[nodiscard]] size_t get_size_t(std::string_view name) const {
        return get_as<size_t>(name);
    }

    [[nodiscard]] std::optional<size_t> get_size_t_optional(std::string_view name) const noexcept {
        return get_as_optional<size_t>(name);
    }

    [[nodiscard]] size_t get_size_t_or(std::string_view name, size_t default_value) const noexcept {
        return get_as_or<size_t>(name, default_value);
    }

    [[nodiscard]] long get_long(std::string_view name) const {
        return get_as<long>(name);
    }

    [[nodiscard]] std::optional<long> get_long_optional(std::string_view name) const noexcept {
        return get_as_optional<long>(name);
    }

    [[nodiscard]] long get_long_or(std::string_view name, long default_value) const noexcept {
        return get_as_or<long>(name, default_value);
    }

    [[nodiscard]] double get_double(std::string_view name) const {
        return get_as<double>(name);
    }

    [[nodiscard]] std::optional<double> get_double_optional(std::string_view name) const noexcept {
        return get_as_optional<double>(name);
    }

    [[nodiscard]] double get_double_or(std::string_view name, double default_value) const noexcept {
        return get_as_or<double>(name, default_value);
    }

    [[nodiscard]] float get_float(std::string_view name) const {
        return get_as<float>(name);
    }

    [[nodiscard]] std::optional<float> get_float_optional(std::string_view name) const noexcept {
        return get_as_optional<float>(name);
    }

    [[nodiscard]] float get_float_or(std::string_view name, float default_value) const noexcept {
        return get_as_or<float>(name, default_value);
    }

    // ========== File Reading Methods ==========

    // Read file from path specified by parameter and return contents as string_view (throws on error)
    [[nodiscard]] std::string_view get_file(std::string_view name) const {
        auto file_path = get(name);
        
        // Check cache first
        std::string cache_key = std::format("{}:{}", name, file_path);
        if (auto it = file_cache_.find(cache_key); it != file_cache_.end()) {
            return std::string_view{it->second};
        }
        
        // Check if file exists
        if (!std::filesystem::exists(file_path)) {
            throw ParseError(std::format("File not found for '{}': '{}'", name, file_path));
        }
        
        // Read file contents
        std::ifstream file(std::string(file_path), std::ios::binary | std::ios::ate);
        if (!file) {
            throw ParseError(std::format("Failed to open file for '{}': '{}'", name, file_path));
        }
        
        auto file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::string contents;
        contents.resize(static_cast<size_t>(file_size));
        
        if (!file.read(contents.data(), file_size)) {
            throw ParseError(std::format("Failed to read file for '{}': '{}'", name, file_path));
        }
        
        // Cache the contents and return view
        auto [iter, inserted] = file_cache_.emplace(std::move(cache_key), std::move(contents));
        return std::string_view{iter->second};
    }

    // Read file with optional return (returns nullopt if parameter not found or file doesn't exist)
    [[nodiscard]] std::optional<std::string_view> get_file_optional(std::string_view name) const noexcept {
        auto file_path_opt = get_optional(name);
        if (!file_path_opt || file_path_opt->empty()) {
            return std::nullopt;
        }
        
        auto file_path = *file_path_opt;
        
        // Check cache first
        std::string cache_key = std::format("{}:{}", name, file_path);
        if (auto it = file_cache_.find(cache_key); it != file_cache_.end()) {
            return std::string_view{it->second};
        }
        
        // Check if file exists
        if (!std::filesystem::exists(file_path)) {
            return std::nullopt;
        }
        
        // Read file contents
        std::ifstream file(std::string(file_path), std::ios::binary | std::ios::ate);
        if (!file) {
            return std::nullopt;
        }
        
        auto file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::string contents;
        contents.resize(static_cast<size_t>(file_size));
        
        if (!file.read(contents.data(), file_size)) {
            return std::nullopt;
        }
        
        // Cache the contents and return view
        auto [iter, inserted] = file_cache_.emplace(std::move(cache_key), std::move(contents));
        return std::string_view{iter->second};
    }

    // Read file with default fallback
    [[nodiscard]] std::string_view get_file_or(std::string_view name, std::string_view default_value) const noexcept {
        return get_file_optional(name).value_or(default_value);
    }

    // Print help message with modern formatting
    void print_help(std::string_view program_name) const {
        std::cout << std::format("Usage: {} [options]\n\n", program_name);

        // Collect unique commands (only print long name version)
        auto unique_commands = commands_ 
            | std::views::filter([](const auto& pair) {
                return pair.first == pair.second.long_name;
            })
            | std::views::values;

        for (const auto& cmd : unique_commands) {
            std::string option_str;
            if (!cmd.short_name.empty()) {
                option_str = std::format("  -{}, --{}", cmd.short_name, cmd.long_name);
            } else {
                option_str = std::format("  --{}", cmd.long_name);
            }

            // Pad to align descriptions
            constexpr size_t alignment_width = 30;
            if (option_str.length() < alignment_width) {
                option_str.append(alignment_width - option_str.length(), ' ');
            }

            std::string req_marker = cmd.required ? " [required]" : "";
            std::string default_info = !cmd.default_value.empty() 
                ? std::format(" (default: {})", cmd.default_value)
                : "";

            std::cout << std::format("{}{}{}{}\n", 
                option_str, cmd.description, default_info, req_marker);
        }
    }

    // Get all commands (for inspection/debugging)
    [[nodiscard]] auto get_all_commands() const noexcept -> const auto& {
        return commands_;
    }

    // Clear all commands
    void clear() noexcept {
        commands_.clear();
    }

    // Check if parser has any commands defined
    [[nodiscard]] bool empty() const noexcept {
        return commands_.empty();
    }

    // Count of unique commands (by long name)
    [[nodiscard]] size_t size() const noexcept {
        return std::ranges::count_if(commands_, [](const auto& pair) {
            return pair.first == pair.second.long_name;
        });
    }

private:
    std::unordered_map<std::string, CommandOption> commands_;
    mutable std::unordered_map<std::string, std::string> file_cache_;

    void update_alternate_key(std::string_view key, const CommandOption& cmd) {
        std::string_view alt_key = (cmd.long_name == key) ? cmd.short_name : cmd.long_name;
        if (!alt_key.empty()) {
            if (auto alt_it = commands_.find(std::string(alt_key)); alt_it != commands_.end()) {
                alt_it->second.value = cmd.value;
            }
        }
    }

    void validate_required_commands() const {
        for (const auto& [key, cmd] : commands_) {
            if (cmd.required && cmd.is_default() && key == cmd.long_name) {
                throw ParseError(ParseResult::MissingRequired, std::format("--{}", cmd.long_name));
            }
        }
    }
};

// Helper function to create and configure a parser
[[nodiscard]] inline CommandLineParser create_parser() {
    return CommandLineParser{};
}

} // namespace clparser

