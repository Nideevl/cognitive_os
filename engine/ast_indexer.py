# engine/ast_indexer.py
"""Deterministic, local Tree-Sitter indexer. Zero LLM tokens.

Produces:
  * a compact skeleton (outline with line ranges) for level-2 workers
  * a structured member list (used by the reference / edit tools)
  * reference lookup: which fields / same-class methods a line range uses
  * a syntax check (used to reject or roll back a bad edit)

Range convention: a member's range STARTS at its first annotation (annotations
live inside the declaration node's `modifiers`), so a reader/editor that gets
the range always gets the annotations too. Leading javadoc/comments are NOT
included.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tree_sitter import Language, Parser
import tree_sitter_java

TYPE_KINDS = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "record_declaration": "record",
    "annotation_type_declaration": "@interface",
}
FIELD_TYPES = {"field_declaration", "constant_declaration"}
METHOD_TYPES = {
    "method_declaration": "method",
    "constructor_declaration": "constructor",
    "compact_constructor_declaration": "constructor",
    "annotation_type_element_declaration": "method",
}
BODY_TYPES = {"block", "constructor_body"}
COMMENTS = {"line_comment", "block_comment"}

ANN_LIMIT = 70      # max chars kept per annotation
FIELD_LIMIT = 110   # max chars kept per field declaration
SIG_LIMIT = 220     # max chars kept per method / class signature


@dataclass
class Member:
    kind: str                 # class|interface|enum|record|@interface|field|method|constructor|enum_constant|initializer
    name: str
    names: List[str]
    head: str                 # annotations + modifiers + signature (one line)
    start: int                # 1-based, first annotation line
    end: int
    depth: int
    has_body: bool = True

    @property
    def is_type(self) -> bool:
        return self.kind in TYPE_KINDS.values()


@dataclass
class FileIndex:
    path: Path
    source: bytes
    tree: object
    package: Optional[Tuple[str, int, int]]
    imports: Optional[Tuple[int, int, int]]   # (start, end, count)
    members: List[Member] = field(default_factory=list)


class ASTIndexer:
    def __init__(self):
        self.language = Language(tree_sitter_java.language())
        try:
            self.parser = Parser(self.language)
        except TypeError:
            self.parser = Parser()
            self.parser.set_language(self.language)

    # ------------------------------------------------------------ text helpers

    @staticmethod
    def _t(src: bytes, node) -> str:
        return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

    @staticmethod
    def _one(text: str, limit: Optional[int] = None) -> str:
        s = " ".join(text.split())
        if limit and len(s) > limit:
            s = s[: limit - 1] + "…"
        return s

    def _mods(self, node, src) -> Tuple[List[str], List[str]]:
        anns, kws = [], []
        for ch in node.children:
            if ch.type != "modifiers":
                continue
            for m in ch.children:
                if m.type in ("annotation", "marker_annotation"):
                    anns.append(self._one(self._t(src, m), ANN_LIMIT))
                elif m.type not in COMMENTS:
                    kws.append(self._t(src, m))
        return anns, kws

    def _rest_text(self, node, src, body, limit) -> str:
        """Source text of the declaration without modifiers and without the body."""
        first = None
        for ch in node.children:
            if ch.type == "modifiers" or ch.type in COMMENTS:
                continue
            first = ch
            break
        if first is None:
            return ""
        end = body.start_byte if body is not None else node.end_byte
        text = src[first.start_byte:end].decode("utf-8", errors="ignore").rstrip()
        if text.endswith(";"):
            text = text[:-1].rstrip()
        return self._one(text, limit)

    def _decl_names(self, node, src) -> List[str]:
        n = node.child_by_field_name("name")
        if n is not None:
            return [self._t(src, n)]
        names = []
        for ch in node.children:
            if ch.type == "variable_declarator":
                nm = ch.child_by_field_name("name")
                if nm is not None:
                    names.append(self._t(src, nm))
        return names

    # ------------------------------------------------------------------ parse

    def parse(self, path: Path) -> FileIndex:
        src = Path(path).read_bytes()
        tree = self.parser.parse(src)
        root = tree.root_node

        package = None
        imp_nodes = []
        members: List[Member] = []

        for child in root.children:
            if child.type == "package_declaration":
                package = (
                    " ".join(self._t(src, child).split()),
                    child.start_point[0] + 1,
                    child.end_point[0] + 1,
                )
            elif child.type == "import_declaration":
                imp_nodes.append(child)
            elif child.type in TYPE_KINDS:
                self._walk_type(child, src, 0, members)

        imports = None
        if imp_nodes:
            imports = (
                imp_nodes[0].start_point[0] + 1,
                imp_nodes[-1].end_point[0] + 1,
                len(imp_nodes),
            )
        return FileIndex(Path(path), src, tree, package, imports, members)

    def _walk_type(self, node, src, depth, out):
        kind = TYPE_KINDS[node.type]
        anns, kws = self._mods(node, src)
        body = node.child_by_field_name("body")
        rest = self._rest_text(node, src, body, SIG_LIMIT)
        names = self._decl_names(node, src)
        head = " ".join(anns + kws + [rest])
        out.append(Member(kind, names[0] if names else "?", names, head,
                          node.start_point[0] + 1, node.end_point[0] + 1, depth))
        if body is not None:
            self._walk_body(body, src, depth + 1, out)

    def _walk_body(self, body, src, depth, out):
        for m in body.children:
            t = m.type
            s, e = m.start_point[0] + 1, m.end_point[0] + 1
            if t in TYPE_KINDS:
                self._walk_type(m, src, depth, out)
            elif t in FIELD_TYPES:
                anns, kws = self._mods(m, src)
                rest = self._rest_text(m, src, None, FIELD_LIMIT)
                names = self._decl_names(m, src)
                out.append(Member("field", names[0] if names else "?", names,
                                  " ".join(anns + kws + [rest]) + ";", s, e, depth))
            elif t in METHOD_TYPES:
                anns, kws = self._mods(m, src)
                blk = next((c for c in m.children if c.type in BODY_TYPES), None)
                rest = self._rest_text(m, src, blk, SIG_LIMIT)
                names = self._decl_names(m, src)
                out.append(Member(METHOD_TYPES[t], names[0] if names else "?", names,
                                  " ".join(anns + kws + [rest]), s, e, depth,
                                  has_body=blk is not None))
            elif t == "enum_constant":
                anns, _ = self._mods(m, src)
                names = self._decl_names(m, src)
                text = self._one(self._t(src, m), FIELD_LIMIT)
                out.append(Member("enum_constant", names[0] if names else text, names, text, s, e, depth))
            elif t == "enum_body_declarations":
                self._walk_body(m, src, depth, out)
            elif t in ("static_initializer", "block"):
                label = "static {...}" if t == "static_initializer" else "{...} (initializer)"
                out.append(Member("initializer", "<init>", [], label, s, e, depth))

    # --------------------------------------------------------------- skeleton

    def generate_skeleton(self, file_path: Path) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not file_path.read_bytes().strip():
            return "[EMPTY FILE]"

        idx = self.parse(file_path)
        out: List[str] = []

        err = self._first_error(idx.tree)
        if err:
            out.append(f"# WARNING: file has syntax errors ({err}); outline may be incomplete")
        if idx.package:
            out.append(f"{idx.package[0]} {idx.package[1]}-{idx.package[2]}")
        if idx.imports:
            s, e, n = idx.imports
            out.append(f"import {{...}} {s}-{e} ({n} imports)")

        stack: List[int] = []
        for m in idx.members:
            while stack and stack[-1] >= m.depth:
                out.append("    " * stack.pop() + "}")
            ind = "    " * m.depth
            if m.is_type:
                out.append(f"{ind}{m.head} {{ {m.start}-{m.end}")
                stack.append(m.depth)
            elif m.kind in ("method", "constructor"):
                body = " {...}" if m.has_body else ";"
                out.append(f"{ind}{m.head}{body} {m.start}-{m.end}")
            else:
                out.append(f"{ind}{m.head} {m.start}-{m.end}")
        while stack:
            out.append("    " * stack.pop() + "}")
        return "\n".join(out)

    # ------------------------------------------------------------- references

    def references(self, path: Path, start: int, end: int, idx: Optional[FileIndex] = None):
        """Fields / same-class methods used inside lines [start, end] whose declarations
        lie OUTSIDE that range. Returns (fields, methods) as lists of Member."""
        idx = idx or self.parse(path)
        src = idx.source

        fields: Dict[str, List[Member]] = {}
        methods: Dict[str, List[Member]] = {}
        for m in idx.members:
            if m.kind in ("field", "enum_constant"):
                for n in m.names:
                    fields.setdefault(n, []).append(m)
            elif m.kind in ("method", "constructor"):
                methods.setdefault(m.name, []).append(m)

        def same(a, b) -> bool:
            return b is not None and (a.start_byte, a.end_byte) == (b.start_byte, b.end_byte)

        local_names = set()
        idents: List[str] = []
        calls: List[str] = []

        stack = [idx.tree.root_node]
        while stack:
            n = stack.pop()
            if n.end_point[0] + 1 < start or n.start_point[0] + 1 > end:
                continue
            t = n.type
            if t in ("formal_parameter", "catch_formal_parameter", "enhanced_for_statement"):
                nm = n.child_by_field_name("name")
                if nm is not None:
                    local_names.add(self._t(src, nm))
            elif t == "spread_parameter":
                for c in n.children:
                    if c.type == "variable_declarator":
                        nm = c.child_by_field_name("name")
                        if nm is not None:
                            local_names.add(self._t(src, nm))
            elif t == "variable_declarator":
                if n.parent is not None and n.parent.type not in FIELD_TYPES:
                    nm = n.child_by_field_name("name")
                    if nm is not None:
                        local_names.add(self._t(src, nm))
            elif t == "inferred_parameters":
                for c in n.children:
                    if c.type == "identifier":
                        local_names.add(self._t(src, c))
            elif t == "lambda_expression":
                p = n.child_by_field_name("parameters")
                if p is not None and p.type == "identifier":
                    local_names.add(self._t(src, p))
            elif t == "identifier":
                p = n.parent
                name = self._t(src, n)
                if p is None:
                    pass
                elif p.type == "method_invocation":
                    if same(n, p.child_by_field_name("name")):
                        obj = p.child_by_field_name("object")
                        if obj is None or obj.type == "this":
                            calls.append(name)
                    else:
                        idents.append(name)
                elif p.type == "field_access":
                    if same(n, p.child_by_field_name("field")):
                        obj = p.child_by_field_name("object")
                        if obj is not None and obj.type == "this":
                            idents.append(name)
                    else:
                        idents.append(name)
                elif p.type == "variable_declarator" and same(n, p.child_by_field_name("name")):
                    pass
                elif p.type in ("formal_parameter", "catch_formal_parameter", "spread_parameter") \
                        and same(n, p.child_by_field_name("name")):
                    pass
                else:
                    idents.append(name)
            stack.extend(n.children)

        def outside(m: Member) -> bool:
            return not (start <= m.start and m.end <= end)

        used_fields: Dict[Tuple[int, int], Member] = {}
        for nm in dict.fromkeys(idents):
            if nm in local_names:
                continue
            for m in fields.get(nm, []):
                if outside(m):
                    used_fields[(m.start, m.end)] = m

        used_methods: Dict[Tuple[int, int], Member] = {}
        for nm in dict.fromkeys(calls):
            for m in methods.get(nm, []):
                if outside(m):
                    used_methods[(m.start, m.end)] = m

        return (
            sorted(used_fields.values(), key=lambda m: m.start),
            sorted(used_methods.values(), key=lambda m: m.start),
        )

    # ----------------------------------------------------------------- syntax

    def _first_error(self, tree) -> Optional[str]:
        root = tree.root_node
        if not root.has_error:
            return None
        stack = [root]
        while stack:
            n = stack.pop()
            if not n.has_error and not n.is_missing and n.type != "ERROR":
                continue
            if n.type == "ERROR":
                return f"line {n.start_point[0] + 1}: unexpected code"
            if n.is_missing:
                return f"line {n.start_point[0] + 1}: missing '{n.type}'"
            stack.extend(reversed(n.children))
        return f"line {root.start_point[0] + 1}: syntax error"

    def syntax_error(self, source: bytes) -> Optional[str]:
        """None if the Java source parses cleanly, else 'line N: ...'."""
        return self._first_error(self.parser.parse(source))