# engine/ast_indexer.py
from pathlib import Path
from typing import Dict, Optional
from tree_sitter import Language, Parser
import tree_sitter_java


class ASTIndexer:
    """Deterministic, local Tree-Sitter parser that extracts compact AST skeletons

    with folded imports, absorbed method annotations, and explicit line coordinates.
    Zero LLM tokens used.
    """

    def __init__(self):
        self.language = Language(tree_sitter_java.language())
        try:
            self.parser = Parser(self.language)
        except TypeError:
            self.parser = Parser()
            self.parser.set_language(self.language)

    def generate_skeleton(self, file_path: Path) -> str:
        if not file_path.exists():
            return f"Error: File '{file_path}' does not exist."

        raw_bytes = file_path.read_bytes()
        tree = self.parser.parse(raw_bytes)
        root = tree.root_node

        output_lines = []
        import_nodes = []

        for child in root.children:
            if child.type == "import_declaration":
                import_nodes.append(child)

        import_start = import_nodes[0].start_point[0] + 1 if import_nodes else None
        import_end = import_nodes[-1].end_point[0] + 1 if import_nodes else None
        imports_handled = False

        for child in root.children:
            # 1. Package declaration
            if child.type == "package_declaration":
                start_l = child.start_point[0] + 1
                end_l = child.end_point[0] + 1
                pkg_text = raw_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore").strip()
                output_lines.append(f"{pkg_text} {start_l}-{end_l}")

            # 2. Folded imports block
            elif child.type == "import_declaration":
                if not imports_handled:
                    output_lines.append(f"import {{...}}  {import_start}-{import_end}")
                    imports_handled = True

            # 3. Class or Interface definition
            elif child.type in ("class_declaration", "interface_declaration"):
                self._render_compact_class(child, raw_bytes, output_lines)

        return "\n".join(output_lines)

    def _render_compact_class(self, class_node, raw_bytes: bytes, out: list):
        start_l = class_node.start_point[0] + 1
        end_l = class_node.end_point[0] + 1

        class_annotations = []
        sig_tokens = []
        body_node = None

        for child in class_node.children:
            if child.type in ("annotation", "marker_annotation"):
                class_annotations.append(raw_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore").strip())
            elif child.type == "class_body":
                body_node = child
            elif child.type not in ("block_comment", "line_comment"):
                sig_tokens.append(raw_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore").strip())

        for ann in class_annotations:
            out.append(ann)
        class_head = " ".join([t for t in sig_tokens if t])
        out.append(f"{class_head} {{ {start_l}-{end_l}")

        if not body_node:
            out.append("}")
            return

        for member in body_node.children:
            m_start = member.start_point[0] + 1
            m_end = member.end_point[0] + 1

            # Fields & Constants
            if member.type == "field_declaration":
                field_text = raw_bytes[member.start_byte:member.end_byte].decode("utf-8", errors="ignore").strip()
                out.append(f"    {field_text} {m_start}-{m_end}")

            # Constructors & Methods (annotations folded into line bounds)
            elif member.type in ("method_declaration", "constructor_declaration"):
                sig = self._extract_clean_signature(member, raw_bytes)
                out.append(f"    {sig} {{...}} {m_start}-{m_end}")

            # Nested classes
            elif member.type in ("class_declaration", "interface_declaration"):
                self._render_compact_class(member, raw_bytes, out)

        out.append("}")

    def _extract_clean_signature(self, node, raw_bytes: bytes) -> str:
        tokens = []
        for child in node.children:
            if child.type in ("block", "constructor_body", "annotation", "marker_annotation"):
                continue
            if child.type not in ("block_comment", "line_comment"):
                tokens.append(raw_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore").strip())
        return " ".join([t for t in tokens if t])