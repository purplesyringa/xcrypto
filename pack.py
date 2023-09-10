import ast
import clang.cindex
from clang.cindex import Cursor, CursorKind, Token, TokenKind
from collections import defaultdict
from dataclasses import dataclass, field
import re
import subprocess
import sys
from typing import Literal, Optional


SOURCE = sys.argv[1]
COMPILER = sys.argv[2]


# Prevent clang from expanding intrinsics when compiling for gcc
def undef_intrinsics():
    if COMPILER == "clang":
        return ""

    proc = subprocess.run(
        [
            "clang++-16",
            "-E",
            "-dM",
            "-x", "c++",
            "/dev/null",
            "-include", "immintrin.h"
        ],
        capture_output=True,
        check=True
    )
    code = proc.stdout.decode()

    result_code = "#include <immintrin.h>\n\n"

    for line in code.splitlines():
        if line.count(" ") < 2:
            continue
        _, name, value = line.split(" ", 2)
        if name.startswith("_mm") and "(" in name:
            name, _, args = name[:-1].partition("(")
            args = args.split(",")
            ty = f"decltype({value})"
            if value[0] == "(":
                s = value[1:].partition(")")[0].lstrip("(")
                if s.lstrip("_").isalnum():
                    ty = s
            result_code += (
                f"#undef {name}\n"
                + "template<"
                + ", ".join(f"typename T_{arg}" for arg in args)
                + f"> auto {name}("
                + ", ".join(f"T_{arg} {arg}" for arg in args)
                + f") -> {ty} {{ __builtin_unreachable(); }}\n"
            )

    return result_code


pack_hpp = undef_intrinsics()


def preprocess_code(path: str):
    proc = subprocess.run(
        [
            "clang++-16",
            "-E",
            "-x", "c++",
            "-D", "EXPORT=__attribute__((annotate(\"pack_export\")))",
            "-include", "/dev/stdin",
            path
        ],
        input=pack_hpp.encode(),
        capture_output=True,
        check=True
    )
    code = proc.stdout.decode()

    file_name = ""
    for line in code.splitlines():
        if line.startswith("# "):
            new_file_name = line.split()[2].replace("\"", "")
            if file_name in (path, "/dev/stdin") and new_file_name.startswith("/"):
                yield "#include <" + new_file_name.rpartition("/")[-1] + ">"
            file_name = new_file_name
        elif file_name == path:
            yield line


def get_include_paths() -> list[str]:
    proc = subprocess.run(["clang++-16", "-E", "-x", "c++", "/dev/null", "-v"], capture_output=True, check=True)
    lines = proc.stderr.decode().splitlines()
    lines = lines[lines.index("#include <...> search starts here:") + 1:]
    lines = lines[:lines.index("End of search list.")]
    return [line.lstrip() for line in lines]


clang.cindex.conf.set_library_file("libclang-16.so.1")

code = "\n".join(preprocess_code(SOURCE))
args = [
    "-std=c++17",
    "-mavx2", "-mfma",
    "-include", "pack.hpp"
]
for path in get_include_paths():
    args += ["-I", path]

index = clang.cindex.Index.create()
translation_unit = index.parse(
    SOURCE,
    args=args,
    unsaved_files=[
        (SOURCE, code),
        ("./pack.hpp", pack_hpp)
    ]
)

for diagnostic in translation_unit.diagnostics:
    if diagnostic.category_number != 0:
        print(diagnostic, file=sys.stderr)
        raise SystemExit(1)


def get_text(node: Cursor) -> str:
    return code[node.extent.start.offset:node.extent.end.offset]


@dataclass
class Scope:
    name: str
    self_id: Optional[str] = None
    identifiers: dict[str, str] = field(default_factory=dict)
    auto_inheritable_identifiers: set[str] = field(default_factory=set)
    inherited_identifiers: set[str] = field(default_factory=set)
    preserved_identifiers: set[str] = field(default_factory=set)
    renames: dict[str, str] = field(default_factory=dict)
    parent: Optional["Scope"] = None
    children: list["Scope"] = field(default_factory=list)

    def child(self, **kwargs) -> "Scope":
        scope = Scope(**kwargs, parent=self)
        self.children.append(scope)
        return scope


scope_of_node = {}
scope_of_node_extent = {}
unscoped_identifier_uses = {}


def collect_identifiers(node: Cursor, scope: Scope):
    if node.location.file is not None and node.location.file.name != SOURCE:
        return

    if node.kind == CursorKind.NAMESPACE:
        usr = node.get_definition().get_usr()
        if usr in scope_of_node:
            scope = scope_of_node[usr]
        else:
            scope.identifiers[node.spelling] = "namespace"
            scope = scope.child(name=f"namespace {node.spelling}", self_id=node.spelling)
            scope_of_node[usr] = scope
    elif node.kind in (
        CursorKind.STRUCT_DECL,
        CursorKind.UNION_DECL,
        CursorKind.CLASS_DECL,
        CursorKind.CLASS_TEMPLATE
    ):
        usr = node.get_definition().get_usr()
        if usr in scope_of_node:
            scope = scope_of_node[usr]
        else:
            scope.identifiers[node.spelling] = "type"
            scope = scope.child(name=f"type {node.spelling}", self_id=node.spelling)
            scope_of_node[usr] = scope
            # Prevent a method from being named just like a constructor
            scope.inherited_identifiers.add(node.spelling)
    elif node.kind == CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION:
        assert False
    elif node.kind == CursorKind.FIELD_DECL:
        scope.identifiers[node.spelling] = f"field: {node.type.spelling}"
    elif node.kind == CursorKind.VAR_DECL:
        scope.identifiers[node.spelling] = f"var: {node.type.spelling}"
    elif node.kind == CursorKind.PARM_DECL and node.spelling != "":
        scope.identifiers[node.spelling] = f"param: {node.type.spelling}"
    elif node.kind == CursorKind.TEMPLATE_TYPE_PARAMETER and node.spelling != "":
        scope.identifiers[node.spelling] = "template type"
        scope.auto_inheritable_identifiers.add(node.spelling)
    elif node.kind == CursorKind.TEMPLATE_NON_TYPE_PARAMETER and node.spelling != "":
        scope.identifiers[node.spelling] = "template var: {node.type.spelling}"
        scope.auto_inheritable_identifiers.add(node.spelling)
    elif node.kind == CursorKind.TEMPLATE_TEMPLATE_PARAMETER and node.spelling != "":
        scope.identifiers[node.spelling] = "template template type"
        scope.auto_inheritable_identifiers.add(node.spelling)
    elif node.kind == CursorKind.UNEXPOSED_DECL:
        # Most likely a structured binding or its elements
        if not node.spelling.startswith("["):
            scope.identifiers[node.spelling] = "binding"
    elif node.kind in (
        CursorKind.FUNCTION_DECL,
        CursorKind.CXX_METHOD,
        CursorKind.CONSTRUCTOR,
        CursorKind.DESTRUCTOR,
        CursorKind.CONVERSION_FUNCTION,
        CursorKind.FUNCTION_TEMPLATE
    ):
        usr = node.get_definition().get_usr()
        if usr in scope_of_node:
            scope = scope_of_node[usr]
        else:
            scope = scope_of_node.get(node.semantic_parent.get_usr(), scope)
            if (
                node.kind not in (
                    CursorKind.CONSTRUCTOR,
                    CursorKind.DESTRUCTOR,
                    CursorKind.CONVERSION_FUNCTION
                )
                and not node.spelling.startswith("operator")
            ):
                scope.identifiers[node.spelling] = f"fn: {node.type.spelling}"
            scope = scope.child(name=f"fn {node.spelling}", self_id=node.spelling)
            scope_of_node[usr] = scope
    elif node.kind == CursorKind.LAMBDA_EXPR:
        scope = scope.child(name=f"lambda {node.type.spelling}")
        scope_of_node[node.type.get_declaration().get_usr()] = scope
    elif node.kind in (
        CursorKind.COMPOUND_STMT,
        CursorKind.IF_STMT,
        CursorKind.SWITCH_STMT,
        CursorKind.WHILE_STMT,
        CursorKind.DO_STMT,
        CursorKind.FOR_STMT
    ):
        kind = {
            CursorKind.COMPOUND_STMT: "{}",
            CursorKind.IF_STMT: "if",
            CursorKind.SWITCH_STMT: "switch",
            CursorKind.WHILE_STMT: "while",
            CursorKind.DO_STMT: "do",
            CursorKind.FOR_STMT: "for"
        }[node.kind]
        scope = scope.child(name=f"stmt {kind}")
    elif node.kind in (
        CursorKind.DECL_REF_EXPR,
        CursorKind.OVERLOADED_DECL_REF,
        CursorKind.TYPE_REF,
        CursorKind.TEMPLATE_REF,
        CursorKind.MEMBER_REF,
        CursorKind.VARIABLE_REF
    ):
        children = list(node.get_children())
        if children and len(children) == 1 and children[0].kind == CursorKind.NAMESPACE_REF:
            # Namespace qualifier
            namespace = children[0]
            scope1 = scope_of_node.get(namespace.get_definition().get_usr())
            assert node.spelling != ""
            unscoped_identifier_uses[node.extent.start.offset] = (node, scope1)
        elif not children:
            # Local identifier
            assert node.spelling != ""
            unscoped_identifier_uses[node.extent.start.offset] = (node, scope)
    elif node.kind == CursorKind.MEMBER_REF_EXPR:
        children = list(node.get_children())
        if not children:
            # Local identifier
            assert node.spelling != ""
            unscoped_identifier_uses[node.extent.start.offset] = (node, scope)
    elif node.kind == CursorKind.ANNOTATE_ATTR and node.spelling == "pack_export":
        if scope.self_id is None:
            raise ValueError(f"EXPORT is not applicable to {scope.name}")
        scope.parent.preserved_identifiers.add(scope.self_id)
    elif node.kind == CursorKind.TYPE_ALIAS_DECL:
        scope.identifiers[node.spelling] = "type"
        scope = scope.child(name=f"typedef {node.spelling}", self_id=node.spelling)
        scope_of_node[node.get_usr()] = scope

    scope_of_node_extent[node.extent.start.offset] = scope

    for child in node.get_children():
        collect_identifiers(child, scope)


root_scope = Scope(name="root")
collect_identifiers(translation_unit.cursor, root_scope)
assert "" not in scope_of_node


for node, scope in unscoped_identifier_uses.values():
    # Use T instead of class T for type names
    name = (node.get_definition() or node).spelling
    while scope is not None:
        if name in scope.identifiers or name in scope.inherited_identifiers:
            break
        scope.inherited_identifiers.add(name)
        scope = scope.parent


FIRST_CHAR_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
NEXT_CHARS_ALPHABET = FIRST_CHAR_ALPHABET + "0123456789"


def repeat_next_chars_alphabet(n: int, prefix: str):
    if n == 0:
        yield prefix
    else:
        for c in NEXT_CHARS_ALPHABET:
            yield from repeat_next_chars_alphabet(n - 1, prefix + c)


def list_new_identifiers():
    n = 0
    while True:
        for c in FIRST_CHAR_ALPHABET:
            yield from repeat_next_chars_alphabet(n, c)
        n += 1


max_used_new_identifiers = 0


def rename_in_scope(
    scope: Scope,
    inherited_renames: dict[str, str],
    auto_inheritable_identifiers: set[str]
):
    global max_used_new_identifiers

    inherited_renames = {
        id: inherited_renames[id]
        for id in auto_inheritable_identifiers | scope.inherited_identifiers
    }
    used_renamed_ids = set(inherited_renames.values())

    gen = list_new_identifiers()
    i = 0
    for id in sorted(scope.identifiers):
        if id in scope.preserved_identifiers:
            scope.renames[id] = id
            continue
        renamed_id = next(gen)
        i += 1
        while renamed_id in used_renamed_ids:
            renamed_id = next(gen)
            i += 1
        scope.renames[id] = renamed_id
    scope.renames |= inherited_renames
    max_used_new_identifiers = max(max_used_new_identifiers, i)

    for child in scope.children:
        rename_in_scope(
            child,
            scope.renames,
            auto_inheritable_identifiers | scope.auto_inheritable_identifiers
        )


rename_in_scope(root_scope, {id: id for id in root_scope.inherited_identifiers}, set())


def print_scope(scope: Scope, nesting: int = 0):
    print("--" * nesting, scope.name)
    if scope.inherited_identifiers:
        print("  " * nesting, "| inherit", ", ".join(sorted(scope.inherited_identifiers)))
    for id, role in sorted(scope.identifiers.items()):
        renamed_id = scope.renames[id]
        print("  " * nesting, f"| {id} -> {renamed_id}: {role}")
    for child in scope.children:
        print_scope(child, nesting + 1)


# print_scope(root_scope)
# raise SystemExit(0)


def map_identifier_token(token: Token) -> (str, TokenKind):
    text = token.spelling
    node = token.cursor

    if node.kind in (
        CursorKind.NAMESPACE,
        CursorKind.STRUCT_DECL,
        CursorKind.UNION_DECL,
        CursorKind.CLASS_DECL,
        CursorKind.CLASS_TEMPLATE,
        CursorKind.FUNCTION_DECL,
        CursorKind.CXX_METHOD,
        CursorKind.CONVERSION_FUNCTION,
        CursorKind.FUNCTION_TEMPLATE,
        CursorKind.TYPE_ALIAS_DECL
    ):
        scope = scope_of_node[node.get_definition().get_usr()]
        text = scope.parent.renames[node.spelling]
    elif node.kind == CursorKind.NAMESPACE_REF:
        scope = scope_of_node.get(node.get_definition().get_usr())
        if scope is not None:
            text = scope.parent.renames[node.spelling]
    elif node.kind in (CursorKind.CONSTRUCTOR, CursorKind.DESTRUCTOR):
        scope = scope_of_node[node.semantic_parent.get_usr()]
        # If we use node.spelling here, ~T is used instead of T
        text = scope.parent.renames[token.spelling]
    elif node.kind in (
        CursorKind.FIELD_DECL,
        CursorKind.VAR_DECL,
        CursorKind.PARM_DECL,
        CursorKind.TEMPLATE_TYPE_PARAMETER,
        CursorKind.TEMPLATE_NON_TYPE_PARAMETER,
        CursorKind.TEMPLATE_TEMPLATE_PARAMETER
    ):
        scope = scope_of_node_extent[node.extent.start.offset]
        # If we use node.spelling here instead, the token "T" in "T a" might yield a node of
        # type (say) PERM_DECL instead of TYPE_REF and thus have spelling "a", which is utterly
        # wrong, but we still have to work around it
        text = scope.renames[token.spelling]
    elif node.kind == CursorKind.UNEXPOSED_DECL and not node.spelling.startswith("["):
        scope = scope_of_node_extent[node.extent.start.offset]
        text = scope.renames[node.spelling]
    elif node.kind in (
        CursorKind.DECL_REF_EXPR,
        CursorKind.OVERLOADED_DECL_REF,
        CursorKind.TYPE_REF,
        CursorKind.TEMPLATE_REF,
        CursorKind.VARIABLE_REF,
        # May arise in an expansion of a non-type template parameter
        CursorKind.INTEGER_LITERAL,
        CursorKind.FLOATING_LITERAL,
        CursorKind.IMAGINARY_LITERAL,
        CursorKind.STRING_LITERAL,
        CursorKind.CHARACTER_LITERAL
    ):
        _, scope = unscoped_identifier_uses[node.extent.start.offset]
        if scope is not None:
            # XXX: If we use node.spelling instead of token.spelling here, calls to functors are
            # parsed as calls to operator()
            text = scope.renames[token.spelling]
    elif node.kind in (CursorKind.MEMBER_REF, CursorKind.MEMBER_REF_EXPR):
        definition = node.get_definition()
        if definition is not None:
            scope = scope_of_node.get(definition.semantic_parent.get_usr())
            if scope is not None:
                text = scope.renames[node.spelling]

    return text


def may_concat_tokens(a: Token, b: Token) -> bool:
    if a.kind != TokenKind.PUNCTUATION and b.kind != TokenKind.PUNCTUATION:
        return False
    if a.kind != TokenKind.PUNCTUATION or b.kind != TokenKind.PUNCTUATION:
        return True
    return a.spelling + b.spelling not in ("++", "+++", "++++", "--", "---", "----")


def strip_export_attribute(tokens: list[Token]) -> list[Token]:
    result = []
    skip = 0
    for token in tokens:
        node = token.cursor
        if node.kind == CursorKind.ANNOTATE_ATTR and token.spelling == "\"pack_export\"":
            del result[-5:]
            skip = 3
        elif skip > 0:
            skip -= 1
        else:
            result.append(token)
    return result


def concat_tokens(tokens: list[Token]) -> str:
    last_token = None
    s = ""
    for token in tokens:
        if last_token is not None and not may_concat_tokens(last_token, token):
            s += " "
        s += token.spelling
        last_token = token
    return s


last_preprocessor_directive_line = None
last_preprocessor_directive_tokens = []
new_tokens = []

for token in strip_export_attribute(list(translation_unit.cursor.get_tokens())):
    if token.kind == TokenKind.COMMENT:
        continue

    if last_preprocessor_directive_line == token.extent.start.line:
        # We're inside a preprocessor directive
        last_preprocessor_directive_tokens.append(token)
        continue
    elif last_preprocessor_directive_line is not None:
        # The preprocessor directive has just ended
        new_tokens.append(("preprocessor", concat_tokens(last_preprocessor_directive_tokens)))
        last_preprocessor_directive_line = None
        last_preprocessor_directive_tokens = []

    if token.kind == TokenKind.PUNCTUATION and token.spelling == "#":
        # Start of a preprocessor directive
        last_preprocessor_directive_line = token.extent.start.line
        last_preprocessor_directive_tokens = [token]
        continue

    if token.kind == TokenKind.PUNCTUATION:
        new_tokens.append(("punctuation", token.spelling))
    elif token.kind == TokenKind.KEYWORD:
        new_tokens.append(("keyword", token.spelling))
    elif token.kind == TokenKind.IDENTIFIER:
        new_tokens.append(("identifier", map_identifier_token(token)))
    elif token.kind == TokenKind.LITERAL:
        if (
            token.spelling[0] == "\""
            and new_tokens
            and new_tokens[-1][0] == "literal"
            and new_tokens[-1][1][0] == "\""
        ):
            new_tokens[-1] = ("literal", new_tokens[-1][1][:-1] + token.spelling[1:])
        else:
            new_tokens.append(("literal", token.spelling))
    else:
        assert False


def replace_asm_named_operand(code: str, name: str, i: int) -> str:
    return re.sub(r"%(\w*)\[" + re.escape(name) + r"\]", r"%\g<1>" + str(i), code)


def compress_asm_statements(tokens: list[(str, str)]) -> list[(str, str)]:
    state = "none"
    paren_nesting = 0
    seen_colon = False
    code = None
    code_i = None
    operand_i = None
    has_operand_without_comma = False

    result = []
    buffered_tokens = []

    for i, token in enumerate(tokens):
        if state == "none":
            if token == ("keyword", "asm"):
                state = "inside_asm"
                paren_nesting = 0
                seen_colon = False
                code = None
                code_i = None
                operand_i = 0
                has_operand_without_comma = False
                buffered_tokens = [token]
            else:
                result.append(token)
        elif state == "inside_asm":
            if token[0] == "literal" and paren_nesting == 1 and not seen_colon:
                code_i = len(buffered_tokens)
                code = token[1].replace(", ", ",")
            elif token == ("punctuation", ":"):
                seen_colon = True
                if has_operand_without_comma:
                    operand_i += 1
                has_operand_without_comma = False
            else:
                if seen_colon:
                    has_operand_without_comma = True
                if token == ("punctuation", "("):
                    paren_nesting += 1
                elif token == ("punctuation", ")"):
                    paren_nesting -= 1
                    if paren_nesting == 0:
                        buffered_tokens[code_i] = ("literal", code)
                        buffered_tokens.append(token)
                        result += buffered_tokens
                        state = "none"
                        continue
                elif token == ("punctuation", "[") and paren_nesting == 1:
                    state = "named_operand"
                    continue
                elif token == ("punctuation", ",") and paren_nesting == 1:
                    operand_i += 1
                    has_operand_without_comma = False
            buffered_tokens.append(token)
        elif state == "named_operand":
            assert token[0] == "identifier"
            code = replace_asm_named_operand(code, token[1], operand_i)
            state = "after_named_operand"
        elif state == "after_named_operand":
            assert token == ("punctuation", "]")
            state = "inside_asm"

    assert state == "none"
    return result


print(max_used_new_identifiers)
for kind, text in compress_asm_statements(new_tokens):
    print(kind, text)
