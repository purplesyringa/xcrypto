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
  PREPROCESSOR,
  STREAM_SEPARATOR
};

struct Token {
  TokenKind kind;
  std::string text;
  int32_t hash;

  Token(TokenKind kind, std::string text)
      : kind(kind), text(std::move(text)),
        hash(std::hash<std::string>{}(this->text) * 6 +
             static_cast<size_t>(kind)) {}

  bool operator==(const Token &rhs) const { return hash == rhs.hash; }
  bool operator!=(const Token &rhs) const { return !(*this == rhs); }

  bool is_compressible() const {
    return kind != TokenKind::PREPROCESSOR &&
           kind != TokenKind::STREAM_SEPARATOR;
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

bool may_concat_tokens(const Token &a, const Token &b) {
  if (a.kind != TokenKind::PUNCTUATION && b.kind != TokenKind::PUNCTUATION) {
    return false;
  }
  if (a.kind != TokenKind::PUNCTUATION || b.kind != TokenKind::PUNCTUATION) {
    return true;
  }
  std::array<std::string_view, 6> forbidden_strings{"++", "+++", "++++",
                                                    "--", "---", "----"};
  return std::find(forbidden_strings.begin(), forbidden_strings.end(),
                   a.text + b.text) == forbidden_strings.end();
}

template <typename It> std::string stringify_tokens(It begin, It end) {
  const Token *last_token = nullptr;
  std::string s;
  for (auto it = begin; it != end; ++it) {
    auto &token = *it;
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

// Find a list of non-intersecting equal series of given length in a string
// maximizing f
template <typename F>
std::vector<size_t>
find_non_intersecting_matches_by_token_count(const std::vector<Token> &tokens,
                                             size_t token_count, F f) {
  static constexpr int32_t POW = 312;
  static constexpr int32_t MOD1 = 1000000007;
  static constexpr int32_t MOD2 = 1000000009;

  int32_t pow_token_count1 = 1;
  int32_t pow_token_count2 = 1;
  for (size_t i = 0; i < token_count; i++) {
    pow_token_count1 = int64_t{pow_token_count1} * POW % MOD1;
    pow_token_count2 = int64_t{pow_token_count2} * POW % MOD2;
  }

  std::map<std::pair<int32_t, int32_t>, std::vector<size_t>> offsets_by_hash;
  int32_t hsh1 = 0;
  int32_t hsh2 = 0;
  for (size_t i = 0; i < token_count; i++) {
    hsh1 = (int64_t{hsh1} * POW + tokens[i].hash) % MOD1;
    hsh2 = (int64_t{hsh2} * POW + tokens[i].hash) % MOD2;
  }

  size_t nearest_unsplittable_token = 0;
  while (nearest_unsplittable_token < tokens.size() &&
         tokens[nearest_unsplittable_token].is_compressible()) {
    nearest_unsplittable_token++;
  }

  for (size_t offset = 0; offset + token_count <= tokens.size(); offset++) {
    if (nearest_unsplittable_token >= offset + token_count) {
      offsets_by_hash[{hsh1, hsh2}].push_back(offset);
    }
    if (offset + token_count < tokens.size()) {
      if (nearest_unsplittable_token == offset) {
        nearest_unsplittable_token++;
        while (nearest_unsplittable_token < tokens.size() &&
               tokens[nearest_unsplittable_token].is_compressible()) {
          nearest_unsplittable_token++;
        }
      }
      hsh1 = (int64_t{hsh1} * POW + tokens[offset + token_count].hash -
              int64_t{tokens[offset].hash} * pow_token_count1) %
             MOD1;
      hsh2 = (int64_t{hsh2} * POW + tokens[offset + token_count].hash -
              int64_t{tokens[offset].hash} * pow_token_count2) %
             MOD2;
      if (hsh1 < 0) {
        hsh1 += MOD1;
      }
      if (hsh2 < 0) {
        hsh2 += MOD2;
      }
    }
  }

  size_t max_value = 0;
  std::vector<size_t> best_offsets;

  for (auto &[hsh, offsets] : offsets_by_hash) {
    if (offsets.back() - offsets[0] < token_count) {
      continue;
    }
    size_t count = 0;
    size_t last_offset = -token_count;
    for (size_t offset : offsets) {
      if (offset >= last_offset + token_count) {
        count++;
        last_offset = offset;
      }
    }
    size_t value = f(offsets[0], count);
    if (value > max_value) {
      max_value = value;
      best_offsets = std::move(offsets);
    }
  }

  std::vector<size_t> non_intersecting_offsets;
  size_t last_offset = -token_count;
  for (size_t offset : best_offsets) {
    if (offset >= last_offset + token_count) {
      non_intersecting_offsets.push_back(offset);
      last_offset = offset;
    }
  }
  return non_intersecting_offsets;
}

// Find a substring t of string s maximizing (len(t) - repl_len) * (matches(s,
// t) - 1), where the matches are non-intersecting. Returns offsets at which the
// substring is present and length.
std::pair<std::vector<size_t>, size_t>
find_refren(const std::vector<Token> &tokens, size_t repl_len) {
  size_t best_value = 0;
  size_t best_token_count = 0;
  std::vector<size_t> best_offsets;

  for (size_t token_count = 1; token_count <= tokens.size();) {
    auto f = [&](size_t offset, size_t count) {
      std::string s = stringify_tokens(tokens.begin() + offset,
                                       tokens.begin() + offset + token_count);
      if (s.size() < repl_len) {
        return size_t{0};
      }
      return (s.size() - repl_len) * (count - 1);
    };

    auto offsets =
        find_non_intersecting_matches_by_token_count(tokens, token_count, f);
    if (offsets.empty()) {
      break;
    }

    // Can we extend all these offsets?
    bool extended = false;
    while (offsets.back() + token_count + 1 < tokens.size() &&
           tokens[offsets[0] + token_count].is_compressible()) {
      bool good = true;
      for (size_t i = 1; i < offsets.size(); i++) {
        good = (good && offsets[i] - offsets[i - 1] >= token_count + 1 &&
                tokens[offsets[i] + token_count] ==
                    tokens[offsets[0] + token_count]);
      }
      if (good) {
        extended = true;
        token_count++;
      } else {
        break;
      }
    }

    if (!extended) {
      size_t value = f(offsets[0], offsets.size());
      if (value > best_value) {
        best_value = value;
        best_token_count = token_count;
        best_offsets = std::move(offsets);
      }
      token_count++;
    }
  }

  return {best_offsets, best_token_count};
}

std::string FIRST_CHAR_ALPHABET =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_$";
std::string ALPHABET = "0123456789" + FIRST_CHAR_ALPHABET;

void increment_id(std::string &id) {
  do {
    size_t i;
    for (i = id.size() - 1; i > 0; i--) {
      size_t j = ALPHABET.find(id[i]);
      j++;
      if (j == ALPHABET.size()) {
        id[i] = ALPHABET[0];
      } else {
        id[i] = ALPHABET[j];
        break;
      }
    }
    if (i == 0) {
      size_t j = FIRST_CHAR_ALPHABET.find(id[0]);
      j++;
      if (j == FIRST_CHAR_ALPHABET.size()) {
        id[0] = ALPHABET[0];
        id.insert(id.begin(), FIRST_CHAR_ALPHABET[0]);
      } else {
        id[i] = FIRST_CHAR_ALPHABET[j];
      }
    }
  } while (id.find('$') == std::string::npos);
}

bool compress_once(std::vector<Token> &tokens, const std::string &id,
                   std::pair<std::string, std::vector<Token>> &define) {
  auto [offsets, token_count] = find_refren(tokens, id.size());
  if (token_count == 0) {
    return false;
  }
  define = {
      id,
      {tokens.begin() + offsets[0], tokens.begin() + offsets[0] + token_count}};

  std::vector<Token> new_tokens;
  auto cur_end = tokens.begin();
  for (size_t offset : offsets) {
    new_tokens.insert(new_tokens.end(), cur_end, tokens.begin() + offset);
    new_tokens.emplace_back(TokenKind::IDENTIFIER, id);
    cur_end = tokens.begin() + offset + token_count;
  }
  new_tokens.insert(new_tokens.end(), cur_end, tokens.end());

  tokens = std::move(new_tokens);
  return true;
}

int main(int argc, char **argv) {
  bool compress = argc <= 1 || std::strcmp(argv[1], "-0") != 0;

  size_t next_free_id_index;
  std::cin >> next_free_id_index;

  std::string next_free_id = "$";

  // Separate code from #include's so that #define's are placed under includes
  std::vector<Token> body;
  std::vector<std::string> prologue_directives;
  for (auto &token : read_tokens()) {
    if (token.kind == TokenKind::PREPROCESSOR &&
        token.text.substr(0, 8) == "#include") {
      prologue_directives.push_back(std::move(token.text));
    } else {
      body.push_back(std::move(token));
    }
  }

  std::map<std::string, std::vector<Token>> defines;
  std::string best_s;

  size_t last_good_step = 0;
  size_t steps_total = 0;

  while (true) {
    std::string s;
    for (auto &line : prologue_directives) {
      s += line;
      s += '\n';
    }
    for (auto &[name, define_tokens] : defines) {
      s += "#define ";
      s += name;
      // When uncommented, this still works, but yields warnings
      // if (define_tokens[0].kind != TokenKind::PUNCTUATION ||
      // define_tokens[0].text == "(") {
      s += ' ';
      // }
      s += stringify_tokens(define_tokens.begin(), define_tokens.end());
      s += '\n';
    }
    s += stringify_tokens(body.begin(), body.end());

    if (best_s.empty() || s.size() < best_s.size()) {
      best_s = std::move(s);
      last_good_step = steps_total;
    }
    steps_total++;

    if (steps_total >= 10 && last_good_step * 2 < steps_total) {
      break;
    }

    if (!compress) {
      break;
    }

    std::vector<Token> tokens;
    for (auto &[name, define_tokens] : defines) {
      tokens.insert(tokens.end(),
                    std::make_move_iterator(define_tokens.begin()),
                    std::make_move_iterator(define_tokens.end()));
      tokens.emplace_back(TokenKind::STREAM_SEPARATOR, std::move(name));
    }
    tokens.insert(tokens.end(), std::make_move_iterator(body.begin()),
                  std::make_move_iterator(body.end()));

    std::pair<std::string, std::vector<Token>> define;
    if (!compress_once(tokens, next_free_id, define)) {
      break;
    }
    increment_id(next_free_id);

    body.clear();
    defines.clear();
    for (auto &token : tokens) {
      if (token.kind == TokenKind::STREAM_SEPARATOR) {
        defines[token.text] = std::move(body);
      } else {
        body.push_back(std::move(token));
      }
    }
    defines[define.first] = std::move(define.second);
  }

  std::cout << best_s << std::endl;

  return 0;
}
