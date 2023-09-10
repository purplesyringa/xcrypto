#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

enum class TokenKind {
    PUNCTUATION,
    KEYWORD,
    IDENTIFIER,
    LITERAL,
    PREPROCESSOR
};

struct Token {
    TokenKind kind;
    std::string text;

    Token(TokenKind kind, std::string text) : kind(kind), text(std::move(text)) {}

    int32_t hash() const {
        return std::hash<std::string>{}(text) * 5 + static_cast<size_t>(kind);
    }
    bool operator==(const Token& rhs) const {
        return hash() == rhs.hash();
        // return kind == rhs.kind && text == rhs.text;
    }
    bool operator!=(const Token& rhs) const {
        return !(*this == rhs);
    }
};


std::vector<Token> read_tokens() {
    std::vector<Token> tokens;
    std::string kind, text;
    while (std::cin >> kind) {
        std::getline(std::cin, text);
        text.erase(text.begin());
        TokenKind token_kind;
        if (kind == "punctuation") {
            token_kind = TokenKind::PUNCTUATION;
        } else if (kind == "keyword") {
            token_kind = TokenKind::KEYWORD;
        } else if (kind == "identifier") {
            token_kind = TokenKind::IDENTIFIER;
        } else if (kind == "literal") {
            token_kind = TokenKind::LITERAL;
        } else if (kind == "preprocessor") {
            token_kind = TokenKind::PREPROCESSOR;
        } else {
            std::cerr << "Unknown token kind " << kind << std::endl;
            std::exit(1);
        }
        tokens.emplace_back(token_kind, std::move(text));
    }
    return tokens;
}


bool may_concat_tokens(const Token& a, const Token& b) {
    if (a.kind != TokenKind::PUNCTUATION && b.kind != TokenKind::PUNCTUATION) {
        return false;
    }
    if (a.kind != TokenKind::PUNCTUATION || b.kind != TokenKind::PUNCTUATION) {
        return true;
    }
    std::array<std::string_view, 6> forbidden_strings{"++", "+++", "++++", "--", "---", "----"};
    return std::find(forbidden_strings.begin(), forbidden_strings.end(), a.text + b.text) == forbidden_strings.end();
}


template<typename It>
std::string stringify_tokens(It begin, It end) {
    const Token* last_token = nullptr;
    std::string s;
    for (auto it = begin; it != end; ++it) {
        auto& token = *it;
        if (token.kind == TokenKind::PREPROCESSOR) {
            if (last_token != nullptr) {
                s += '\n';
            }
            s += token.text + '\n';
            last_token = nullptr;
            continue;
        }
        if (last_token != nullptr && !may_concat_tokens(*last_token, token)) {
            s += ' ';
        }
        s += token.text;
        last_token = &token;
    }
    if (s.back() == '\n') {
        s.pop_back();
    }
    return s;
}


// Find the longest list of non-intersecting equal series of given length in a string
std::vector<size_t> find_non_intersecting_matches_by_length(const std::vector<Token>& tokens,
                                                            size_t length) {
    static constexpr int32_t POW = 312;
    static constexpr int32_t MOD1 = 1000000007;
    static constexpr int32_t MOD2 = 1000000009;

    int32_t pow_length1 = 1;
    int32_t pow_length2 = 1;
    for (size_t i = 0; i < length; i++) {
        pow_length1 = int64_t{pow_length1} * POW % MOD1;
        pow_length2 = int64_t{pow_length2} * POW % MOD2;
    }

    std::map<std::pair<int32_t, int32_t>, std::vector<size_t>> offsets_by_hash;
    int32_t hsh1 = 0;
    int32_t hsh2 = 0;
    for (size_t i = 0; i < length; i++) {
        hsh1 = (int64_t{hsh1} * POW + tokens[i].hash()) % MOD1;
        hsh2 = (int64_t{hsh2} * POW + tokens[i].hash()) % MOD2;
    }

    size_t nearest_preprocessor_token = 0;
    while (nearest_preprocessor_token < tokens.size() && tokens[nearest_preprocessor_token].kind != TokenKind::PREPROCESSOR) {
        nearest_preprocessor_token++;
    }

    for (size_t offset = 0; offset + length <= tokens.size(); offset++) {
        if (nearest_preprocessor_token >= offset + length) {
            offsets_by_hash[{hsh1, hsh2}].push_back(offset);
        }
        if (offset + length < tokens.size()) {
            if (nearest_preprocessor_token == offset) {
                nearest_preprocessor_token++;
                while (nearest_preprocessor_token < tokens.size() && tokens[nearest_preprocessor_token].kind != TokenKind::PREPROCESSOR) {
                    nearest_preprocessor_token++;
                }
            }
            hsh1 = (
                int64_t{hsh1} * POW
                + tokens[offset + length].hash()
                - int64_t{tokens[offset].hash()} * pow_length1
            ) % MOD1;
            hsh2 = (
                int64_t{hsh2} * POW
                + tokens[offset + length].hash()
                - int64_t{tokens[offset].hash()} * pow_length2
            ) % MOD2;
            if (hsh1 < 0) {
                hsh1 += MOD1;
            }
            if (hsh2 < 0) {
                hsh2 += MOD2;
            }
        }
    }

    size_t max_count = 1;
    std::vector<size_t> max_count_offsets{0};

    for (auto& [hsh, offsets]: offsets_by_hash) {
        if (offsets.back() - offsets[0] < length) {
            continue;
        }
        size_t count = 0;
        size_t last_offset = -length;
        for (size_t offset: offsets) {
            if (offset >= last_offset + length) {
                count++;
                last_offset = offset;
            }
        }
        if (count > max_count) {
            max_count = count;
            max_count_offsets = std::move(offsets);
        }
    }

    std::vector<size_t> non_intersecting_offsets;
    size_t last_offset = -length;
    for (size_t offset: max_count_offsets) {
        if (offset >= last_offset + length) {
            non_intersecting_offsets.push_back(offset);
            last_offset = offset;
        }
    }

    assert(non_intersecting_offsets.size() == max_count);
    return non_intersecting_offsets;
}


// Find a substring t of string s maximizing (len(t) - repl_len) * (matches(s, t) - 1), where the
// matches are non-intersecting. Returns offsets at which the substring is present and length.
std::pair<std::vector<size_t>, size_t> find_refren(const std::vector<Token>& tokens, size_t repl_len) {
    size_t length = 1;
    auto offsets = find_non_intersecting_matches_by_length(tokens, length);

    size_t best_win = 0;
    std::vector<size_t> best_offsets;
    size_t best_length = 0;

    while (length <= tokens.size() / 2 && offsets.size() > 1) {
        std::string s = stringify_tokens(tokens.begin() + offsets[0], tokens.begin() + offsets[0] + length);
        if (s.size() > repl_len) {
            size_t win = (s.size() - repl_len) * (offsets.size() - 1);
            if (win > best_win) {
                best_win = win;
                best_offsets = offsets;
                best_length = length;
            }
        }

        // Chances are length-1 will have the same len(offsets), which is... not terribly efficient.
        // Binary-search just how much we have to increase length to decrease len(offsets)

        size_t step = length;
        bool incremented = false;
        while (step >= 3) {
            auto offsets2 = find_non_intersecting_matches_by_length(tokens, length + step);
            if (offsets.size() == offsets2.size()) {
                offsets = offsets2;
                length += step;
                incremented = true;
            }
            step /= 2;
        }

        if (!incremented) {
            length++;
            offsets = find_non_intersecting_matches_by_length(tokens, length);
        }
    }

    return {best_offsets, best_length};
}


std::string FIRST_CHAR_ALPHABET = "abcdefghijklmnopqrstuvwxyz";
std::string NEXT_CHARS_ALPHABET = FIRST_CHAR_ALPHABET + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";


std::string get_id_by_index(size_t index) {
    size_t length = 1;
    size_t count_of_this_length = FIRST_CHAR_ALPHABET.size();
    while (index >= count_of_this_length) {
        index -= count_of_this_length;
        length++;
        count_of_this_length *= NEXT_CHARS_ALPHABET.size();
    }
    std::string s(length + 1, '\0');
    for (size_t i = 0; i < length - 1; i++) {
        s[length - i] = NEXT_CHARS_ALPHABET[index % NEXT_CHARS_ALPHABET.size()];
        index /= NEXT_CHARS_ALPHABET.size();
    }
    s[0] = '_';
    s[1] = FIRST_CHAR_ALPHABET[index];
    return s;
}


bool compress_once(
    std::vector<Token>& tokens,
    std::vector<std::string>& defines,
    size_t next_free_id_index
) {
    auto id = get_id_by_index(next_free_id_index);

    auto [offsets, length] = find_refren(tokens, id.size());
    if (length == 0) {
        return false;
    }
    auto s = stringify_tokens(tokens.begin() + offsets[0], tokens.begin() + offsets[0] + length);

    defines.push_back("#define " + id + " " + s);

    std::vector<Token> new_tokens;
    auto cur_end = tokens.begin();
    for (size_t offset: offsets) {
        new_tokens.insert(new_tokens.end(), cur_end, tokens.begin() + offset);
        new_tokens.emplace_back(TokenKind::IDENTIFIER, id);
        cur_end = tokens.begin() + offset + length;
    }
    new_tokens.insert(new_tokens.end(), cur_end, tokens.end());

    tokens = std::move(new_tokens);
    return true;
}


int main(int argc, char** argv) {
    bool compress = argc <= 1 || std::strcmp(argv[1], "-0") != 0;

    size_t next_free_id_index;
    std::cin >> next_free_id_index;
    next_free_id_index = 0;

    // Separate code from #include's so that #define's are placed under includes
    std::vector<Token> tokens;
    std::vector<std::string> prologue_directives;
    for (auto& token: read_tokens()) {
        if (token.kind == TokenKind::PREPROCESSOR && token.text.substr(0, 8) == "#include") {
            prologue_directives.push_back(std::move(token.text));
        } else {
            tokens.push_back(std::move(token));
        }
    }

    std::string best_s;
    while (true) {
        std::string s;
        for (auto& line: prologue_directives) {
            s += line;
            s += '\n';
        }
        s += stringify_tokens(tokens.begin(), tokens.end());

        if (best_s.empty() || s.size() < best_s.size()) {
            best_s = std::move(s);
        }
        if (!compress) {
            break;
        }
        if (!compress_once(tokens, prologue_directives, next_free_id_index++)) {
            break;
        }
    }

    std::cout << best_s << std::endl;

    return 0;
}
