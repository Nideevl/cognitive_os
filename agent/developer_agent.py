import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from agent.key_manager import GroqKeyRotator
from engine.ast_indexer import ASTIndexer

# ------------------------------------------------------------------ limits
MAX_TOOL_CHARS = 6000        # cap on grep / list results fed to a navigator
MAX_SKELETON_CHARS = 30000   # cap per file skeleton
MAX_CONTEXT_CHARS = 16000    # navigator context budget (~4.5k tokens, under the 7k/request model limit)
MAX_REQUEST_TOKENS = 6000    # _chat shortens old results before sending anything bigger than this
MAX_FIND_FILES = 40
MAX_READER_CHARS = 14000     # max code shown to one leaf
MAX_OUT_TOKENS = 1000        # some keys allow only 1000 output tokens/min
MAX_WORKER_TURNS = 10
MAX_ROUNDS = 3
MAX_WORKERS_PER_ROUND = 8
MAX_GREP_HITS = 60
MAX_FILES_PER_WORKER = 1      # a level-2 navigator gets exactly one file

TOOL_NAMES = ("skeleton", "read", "refs", "grep", "list_files",
              "edit", "replace_preview", "replace_all")
ALIASES = {"get_file_skeleton": "skeleton", "grep_files": "grep"}
LINE_TOOL_RE = re.compile(r"^\s*(?:[-*]\s*)?TOOL:\s*(\w+)\((.*)\)\s*$", re.MULTILINE)
WRITE_RE = re.compile(
    r"WRITE_FILE:\s*(\S+)[ \t]*\r?\n```[\w+-]*\r?\n(.*?)\r?\n```",
    re.DOTALL,
)
FENCE_RE = re.compile(r"```[\w+-]*[ \t]*\r?\n(.*?)```", re.DOTALL)
MECHANICAL_RE = re.compile(
    r"replace_(?:all|preview)|\brename\b|repo-?wide|every occurrence|all occurrences|everywhere",
    re.IGNORECASE,
)
TEXT_EXTS = {".java", ".xml", ".yml", ".yaml", ".properties", ".json", ".gradle", ".md", ".sql"}
SKIP_DIRS = {"target", "build", ".git", ".idea", "node_modules", ".cognitive_backup"}

TOOLS_HELP = (
    "Valid tools (one per line, single-line arguments, ' | ' separates parts): "
    "TOOL: skeleton(file) | TOOL: read(file | ranges | question) | TOOL: refs(file | ranges) | "
    "TOOL: grep(pattern) | TOOL: list_files(dir) | TOOL: edit(file | range | instruction)"
)


def looks_like_tool_markup(msg: str) -> bool:
    return any(s in msg for s in ("<tool_call>", "<function", "TOOL:"))


def parse_tool_calls(msg: str, limit: int = 4):
    """Return a list of (name, arg). Line format: TOOL: name(arg). Falls back to
    the model's native markup for the simple single-argument tools."""
    calls = []
    for m in LINE_TOOL_RE.finditer(msg):
        name = ALIASES.get(m.group(1), m.group(1))
        if name in TOOL_NAMES:
            calls.append((name, m.group(2).strip()))
    if calls:
        return calls[:limit]

    if looks_like_tool_markup(msg):
        # native markup or a missing ')' : accept  read(a | b | c  /  edit(a | b | c)
        for m in re.finditer(r"\b(read|refs|edit|replace_preview|replace_all)\(([^\n<]*)", msg):
            arg = m.group(2).strip()
            if arg.endswith(")"):
                arg = arg[:-1].strip()
            if arg:
                calls.append((m.group(1), arg))
        if calls:
            return calls[:limit]

        seen = set()
        for m in re.finditer(r"\b(skeleton|get_file_skeleton|list_files|grep|grep_files)\b", msg):
            name = ALIASES.get(m.group(1), m.group(1))
            tail = re.sub(r"</?parameter[^>]*>", " ", msg[m.end():])
            if name == "grep":
                g = re.match(r"[^\w\\/.^(\[]*([^<\n]+?)\s*(?:</|\)|\n|$)", tail)
            else:
                g = re.match(r"[^\w./\\-]*([\w./\\-]+)", tail)
            arg = g.group(1).strip().strip("'\"") if g else ""
            if arg and (name, arg) not in seen:
                seen.add((name, arg))
                calls.append((name, arg))
    return calls[:limit]


@dataclass
class ChildReport:
    worker_id: str
    task: str
    findings: str
    status: str


@dataclass
class WorkingFrame:
    global_intent: str = ""
    plan: str = ""
    findings: str = ""
    next_tasks: str = ""
    synthesis: str = ""
    child_reports: List[ChildReport] = None
    awareness_state: str = "PLANNING"

    def __post_init__(self):
        if self.child_reports is None:
            self.child_reports = []

    def render(self) -> str:
        reports_str = ""
        if self.child_reports:
            reports_str = "\nCHILD REPORTS:\n"
            for r in self.child_reports:
                reports_str += f"  [{r.worker_id}] {r.status}\n{r.findings}\n\n"

        synthesis_str = ""
        if self.synthesis:
            synthesis_str = f"\nLATEST SYNTHESIS:\n{self.synthesis}\n"

        return (
            "=== ORCHESTRATOR STATE ===\n"
            f"GOAL: {self.global_intent}\n\n"
            f"PLAN:\n{self.plan}\n\n"
            f"FINDINGS:\n{self.findings}\n\n"
            f"NEXT TASKS:\n{self.next_tasks}\n"
            f"{reports_str}"
            f"{synthesis_str}"
            f"STATE: {self.awareness_state}\n"
            "=========================="
        )

    def to_dict(self) -> dict:
        return {
            "global_intent": self.global_intent,
            "plan": self.plan,
            "findings": self.findings,
            "next_tasks": self.next_tasks,
            "synthesis": self.synthesis,
            "child_reports": [
                {
                    "worker_id": r.worker_id,
                    "task": r.task,
                    "findings": r.findings,
                    "status": r.status,
                }
                for r in self.child_reports
            ],
            "awareness_state": self.awareness_state,
            "timestamp": datetime.now().isoformat(),
        }


class DeveloperAgent:
    """
    Level 1  Orchestrator : plans, delegates goals, synthesizes.
    Level 2  Navigator    : sees the SKELETON of its files (outline + line ranges), plans,
                            delegates exact line ranges to level 3, never reads code itself.
    Level 3  Reader/Writer: single LLM call, code preloaded, NO tools. Cannot spawn anything.
    """

    def __init__(
        self,
        project_root: str,
        key_rotator: GroqKeyRotator,
        model: str = "qwen/qwen3.8-27b",
        session_file: str = None,
    ):
        self.root = Path(project_root).resolve()
        self.rotator = key_rotator
        self.model = model
        self.indexer = ASTIndexer()
        self.token_log: List[dict] = []
        self._unscheduled: List[str] = []
        self.frame = WorkingFrame()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.file_index: Dict[str, List[Path]] = {}
        self.target_services: set = set()
        self.session_file = session_file or "cognitive_os_session.json"
        self.backup_root = self._new_backup_root()
        self._backed_up: set = set()
        self._build_index()

    # ------------------------------------------------------------------ infra

    def _new_backup_root(self) -> Path:
        return self.root / ".cognitive_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")

    def _reset(self):
        """Every question starts from zero: no frame, no reports, no synthesis, no target services."""
        self.frame = WorkingFrame()
        self.target_services = set()
        self.token_log = []
        self._unscheduled = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.backup_root = self._new_backup_root()
        self._backed_up = set()
        self._build_index()   # re-scan disk so files changed since the last question are seen

    def _skip(self, p: Path) -> bool:
        try:
            parts = p.relative_to(self.root).parts
        except Exception:
            parts = p.parts
        return any(part in SKIP_DIRS or part == "test" for part in parts)

    def _build_index(self):
        self.file_index.clear()
        for p in self.root.rglob("*.java"):
            if not self._skip(p):
                self.file_index.setdefault(p.name.lower(), []).append(p)

    def list_services(self) -> List[str]:
        services = []
        for p in self.root.iterdir():
            if p.is_dir() and not p.name.startswith((".", "target", "build")):
                if (p / "pom.xml").exists() or (p / "src").exists():
                    services.append(p.name)
        return sorted(services)

    def _service_of(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).parts[0]
        except Exception:
            return ""

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)).replace("\\", "/")
        except Exception:
            return str(path)

    @staticmethod
    def _fit(messages) -> None:
        """Keep one request under MAX_REQUEST_TOKENS: shorten the oldest tool results first (estimate: 3.2 chars/token)."""
        limit = int(MAX_REQUEST_TOKENS * 3.2)
        size = lambda: sum(len(m["content"]) for m in messages)
        for m in messages[2:]:
            if size() <= limit:
                return
            if m["role"] == "user" and len(m["content"]) > 500:
                m["content"] = m["content"][:500] + "\n...[shortened to fit the request limit; re-run the tool if needed]"
        if len(messages) > 2 and size() > limit:   # last resort, navigator turns only: cut the fresh task/skeleton block
            over = size() - limit
            m = messages[1]
            m["content"] = m["content"][: max(2000, len(m["content"]) - over)] + "\n...[skeleton cut to fit the request limit]"

    def _chat(self, messages, max_tokens: int = 2000) -> str:
        self._fit(messages)
        def call(client):
            res = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
                tools=None,
            )
            if res.usage:
                self.total_prompt_tokens += res.usage.prompt_tokens
                self.total_completion_tokens += res.usage.completion_tokens
            return res

        res = self.rotator.execute_with_failover(call)
        text = res.choices[0].message.content or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _snap(self) -> Tuple[int, int]:
        return self.total_prompt_tokens, self.total_completion_tokens

    def _log_tokens(self, label: str, before: Tuple[int, int]):
        p = self.total_prompt_tokens - before[0]
        c = self.total_completion_tokens - before[1]
        self.token_log.append({"label": label, "prompt": p, "completion": c})
        print(f"[Tokens] {label}: {p + c} (prompt {p} / completion {c})")

    def _token_summary(self):
        groups: Dict[str, int] = {}
        for e in self.token_log:
            lab = e["label"]
            key = "workers" if lab.startswith("W") else lab.split()[0]
            groups[key] = groups.get(key, 0) + e["prompt"] + e["completion"]
        print("[Tokens] by phase: " + " | ".join(f"{k}: {v}" for k, v in groups.items()))

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _chat_json(self, messages, max_tokens: int = MAX_OUT_TOKENS):
        """Returns (data, text). Retries once with a compact-output instruction if JSON is cut off."""
        text = self._chat(messages, max_tokens=max_tokens)
        data = self._extract_json(text)
        if data is not None:
            return data, text
        retry = [dict(m) for m in messages]
        retry[-1]["content"] += (
            "\n\nYour previous reply was cut off or not valid JSON. Reply again with ONLY compact JSON: "
            "at most 4 workers, each task under 25 words, files as short paths, no prose."
        )
        text = self._chat(retry, max_tokens=max_tokens)
        return self._extract_json(text), text

    @staticmethod
    def _cap(text: str, limit: int = MAX_TOOL_CHARS) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n...[truncated, {len(text)} chars total]"

    # --------------------------------------------------------------- file utils

    @staticmethod
    def _read_lines(path: Path, strict: bool = False) -> Tuple[List[str], str]:
        """(lines, newline). Lines have no line endings. Last element is '' if the file
        ends with a newline, so len(lines) may be one more than the real line count."""
        raw = path.read_bytes()
        text = raw.decode("utf-8") if strict else raw.decode("utf-8", errors="ignore")
        nl = "\r\n" if "\r\n" in text else "\n"
        return text.replace("\r\n", "\n").split("\n"), nl

    @staticmethod
    def _n(lines: List[str]) -> int:
        return len(lines) - 1 if lines and lines[-1] == "" else len(lines)

    @staticmethod
    def _parse_ranges(spec: str, n: int) -> List[Tuple[int, int]]:
        spec = spec.strip().strip("'\"")
        if n <= 0:
            return []
        if spec.lower() in ("all", "*", "whole"):
            return [(1, n)]
        rs = []
        for m in re.finditer(r"(\d+)\s*-\s*(\d+)|(\d+)", spec):
            if m.group(3):
                a = b = int(m.group(3))
            else:
                a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            a, b = max(1, a), min(n, b)
            if a <= b:
                rs.append((a, b))
        rs.sort()
        merged: List[Tuple[int, int]] = []
        for a, b in rs:
            if merged and a <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        return merged

    def _backup(self, path: Path):
        """Copy the original once per question into .cognitive_backup/<timestamp>/."""
        if path in self._backed_up or not path.exists():
            return
        dest = self.backup_root / path.relative_to(self.root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        self._backed_up.add(path)

    def _change_report(self) -> str:
        """Unified diff of every file changed this question (backup vs disk). Zero tokens, no trust in worker claims."""
        import difflib
        out = []
        for p in sorted(self._backed_up):
            bak = self.backup_root / p.relative_to(self.root)
            try:
                a = bak.read_text(encoding="utf-8", errors="ignore").splitlines()
                b = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            d = list(difflib.unified_diff(a, b, f"a/{self._rel(p)}", f"b/{self._rel(p)}", lineterm="", n=1))
            if d:
                out.append("\n".join(d))
        return self._cap("\n\n".join(out) or "(no files changed)", 8000)

    def _text_files(self):
        for p in self.root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in TEXT_EXTS:
                continue
            try:
                parts = p.relative_to(self.root).parts
            except Exception:
                continue
            if any(part in SKIP_DIRS for part in parts):
                continue
            yield p

    # ------------------------------------------------------------------ tools

    def _resolve(self, name: str, fuzzy: bool = True) -> Tuple[Optional[Path], Optional[str]]:
        name = name.strip().strip("'\"").replace("\\", "/").lstrip("/")
        if not name:
            return None, "Error: empty file name."

        # paths may be repo-relative or service-relative
        bases = [self.root] + [self.root / s for s in sorted(self.target_services)]
        for b in bases:
            if (b / name).is_file():
                return b / name, None

        matches = self.file_index.get(Path(name).name.lower(), [])
        if "/" in name:
            narrowed = [m for m in matches if self._rel(m).lower().endswith(name.lower())]
            # models often guess wrong folder names; fall back to filename (reads only)
            matches = narrowed if (narrowed or not fuzzy) else matches

        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            preferred = [m for m in matches if self._service_of(m) in self.target_services]
            if len(preferred) == 1:
                return preferred[0], None
            opts = ", ".join(self._rel(m) for m in matches)
            return None, f"Ambiguous '{name}'. Use one of: {opts}"
        return None, f"Error: File '{name}' not found in repo."

    def _skeleton_text(self, path: Path) -> str:
        try:
            return self._cap(self.indexer.generate_skeleton(path), MAX_SKELETON_CHARS)
        except Exception as e:
            return f"Error building skeleton: {e}"

    def _skeleton_block(self, files: List[Path]) -> str:
        if not files:
            return ""
        out = ["\n\nCURRENT SKELETONS (always up to date; 'a-b' are line ranges):"]
        for p in files:
            try:
                n = self._n(self._read_lines(p)[0])
            except Exception:
                n = "?"
            out.append(f"\n[{self._rel(p)}] ({n} lines)\n{self._skeleton_text(p)}")
        return "\n".join(out)

    def list_files(self, directory: str = "") -> str:
        directory = directory.strip().strip("'\"").replace("\\", "/").lstrip("/")
        bases = [self.root / directory] if directory else [self.root]
        if directory:
            bases += [self.root / s / directory for s in sorted(self.target_services)]
        base = next((b for b in bases if b.is_dir()), None)
        if base is None:
            return (f"Error: directory '{directory}' not found. "
                    f"Services: {', '.join(self.list_services())}")
        files = [self._rel(p) for p in sorted(base.rglob("*.java")) if not self._skip(p)]
        return self._cap("\n".join(files)) if files else "(no java files)"

    def grep_files(self, pattern: str) -> str:
        pattern = pattern.strip().strip("'\"")
        if not pattern:
            return "Error: empty pattern."
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))

        roots = [self.root / s for s in sorted(self.target_services)] if self.target_services else [self.root]

        hits = []
        for base in roots:
            for p in base.rglob("*.java"):
                if self._skip(p):
                    continue
                try:
                    for i, line in enumerate(
                        p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
                    ):
                        if rx.search(line):
                            hits.append(f"{self._rel(p)}:{i}: {line.strip()[:160]}")
                            if len(hits) >= MAX_GREP_HITS:
                                hits.append("...[hit limit reached]")
                                return "\n".join(hits)
                except Exception:
                    continue
        return "\n".join(hits) if hits else "(no matches)"

    # ------------------------------------------- level-3 leaves (single LLM call)

    def _class_context(self, path: Path, ranges: List[Tuple[int, int]]) -> str:
        """Declarations (from elsewhere in the file) that the ranges refer to. Deterministic."""
        if path.suffix != ".java":
            return ""
        try:
            idx = self.indexer.parse(path)
        except Exception:
            return ""
        fields, methods = {}, {}
        for a, b in ranges:
            f, m = self.indexer.references(path, a, b, idx)
            for x in f:
                fields[(x.start, x.end)] = x
            for x in m:
                methods[(x.start, x.end)] = x

        def inside(x):
            return any(a <= x.start and x.end <= b for a, b in ranges)

        lines = []
        for x in sorted(fields.values(), key=lambda x: x.start):
            if not inside(x):
                lines.append(f"  field  {x.start}-{x.end}: {x.head}")
        for x in sorted(methods.values(), key=lambda x: x.start):
            if not inside(x):
                lines.append(f"  method {x.start}-{x.end}: {x.head}")
        return "\n".join(lines[:40])

    @staticmethod
    def _numbered(lines: List[str], ranges: List[Tuple[int, int]], rel: str) -> str:
        blocks = []
        for a, b in ranges:
            body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(a, b + 1))
            blocks.append(f"--- {rel} lines {a}-{b} ---\n{body}")
        return "\n\n".join(blocks)

    def tool_read(self, arg: str) -> str:
        """Level-3 READER: preloaded ranges, one LLM call, no tools."""
        parts = [p.strip().strip("'\"") for p in arg.split("|", 2)]
        if len(parts) < 3 or not parts[2]:
            return "ERROR: use read(file | ranges | question)   e.g. read(A.java | 10-20, 40-45 | what does foo validate?)"
        path, err = self._resolve(parts[0])
        if err:
            return err
        try:
            lines, _ = self._read_lines(path)
        except Exception as e:
            return f"Error reading file: {e}"
        n = self._n(lines)
        ranges = self._parse_ranges(parts[1], n)
        if not ranges:
            return f"ERROR: bad ranges '{parts[1]}' (file has {n} lines). Use e.g. 10-20, 40-45 or all."

        rel = self._rel(path)
        code = self._cap(self._numbered(lines, ranges, rel), MAX_READER_CHARS)
        ctx = self._class_context(path, ranges)

        system_prompt = (
            "You are a code reader. You see ONLY the line ranges below, taken from one file. "
            "You cannot see the rest of the file and you have no tools.\n"
            "Answer the QUESTION strictly from the code shown. Never guess about code you cannot see.\n"
            "Reply in exactly this format and nothing else:\n"
            "ANSWER: <direct, concise answer>\n"
            "EVIDENCE: <up to 8 lines, each 'LINE: code' copied from the shown code>\n"
            "LEADS: <methods/types/classes the code uses that are defined elsewhere, as 'name (line N)'; or NONE>\n"
            "COVERAGE: FOUND IN RANGE | NOT IN RANGE (say what is missing)"
        )
        user = f"FILE: {rel}\nQUESTION: {parts[2]}\n\nCODE (line-number prefixes are not part of the code):\n{code}"
        if ctx:
            user += ("\n\nCLASS CONTEXT (declarations elsewhere in the file that the code refers to; "
                     f"NOT part of your ranges):\n{ctx}")
        answer = self._chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
            max_tokens=MAX_OUT_TOKENS,
        )
        if not answer or looks_like_tool_markup(answer):
            answer = "ANSWER: (reader produced no usable answer)\nCOVERAGE: UNKNOWN"
        spans = ", ".join(f"{a}-{b}" for a, b in ranges)
        return f"[reader {rel} lines {spans}]\n{answer}"

    def tool_refs(self, arg: str) -> str:
        """Deterministic usage map for line ranges (no LLM)."""
        parts = [p.strip().strip("'\"") for p in arg.split("|", 1)]
        if len(parts) < 2:
            return "ERROR: use refs(file | ranges)"
        path, err = self._resolve(parts[0])
        if err:
            return err
        if path.suffix != ".java":
            return "ERROR: refs works on .java files only."
        n = self._n(self._read_lines(path)[0])
        ranges = self._parse_ranges(parts[1], n)
        if not ranges:
            return f"ERROR: bad ranges '{parts[1]}' (file has {n} lines)."
        ctx = self._class_context(path, ranges)
        spans = ", ".join(f"{a}-{b}" for a, b in ranges)
        return f"[refs {self._rel(path)} lines {spans}]\n" + (ctx or "(no references to other members of this file)")

    def tool_edit(self, arg: str) -> str:
        """Level-3 WRITER: replaces exactly one line range. Harness applies + validates + reports."""
        parts = [p.strip().strip("'\"") for p in arg.split("|", 2)]
        if len(parts) < 3 or not parts[2]:
            return "ERROR: use edit(file | range | instruction)"
        path, err = self._resolve(parts[0], fuzzy=False)
        if err:
            return err
        try:
            lines, nl = self._read_lines(path, strict=True)
        except Exception as e:
            return f"ERROR: cannot edit (file is not valid UTF-8 or unreadable): {e}"
        n = self._n(lines)
        ranges = self._parse_ranges(parts[1], n)
        if len(ranges) != 1:
            return "ERROR: edit needs exactly ONE contiguous range, e.g. edit(A.java | 18-24 | instruction)"
        a, b = ranges[0]
        rel = self._rel(path)
        is_java = path.suffix == ".java"

        old_text_bytes = path.read_bytes()
        old_err = self.indexer.syntax_error(old_text_bytes) if is_java else None
        old_skel = self.indexer.generate_skeleton(path).splitlines() if is_java else []

        code = self._cap(self._numbered(lines, [(a, b)], rel), MAX_READER_CHARS)
        ctx = self._class_context(path, [(a, b)])
        skel = self._cap(self.indexer.generate_skeleton(path), 12000) if is_java else ""

        system_prompt = (
            f"You are a code editor. Replace EXACTLY lines {a}-{b} of the file with new text.\n"
            "Follow the INSTRUCTION. Keep everything in the range that the instruction does not change. "
            "Match the file's indentation and style. Include annotations that belong to the range.\n"
            "Output ONLY one fenced code block containing the full replacement text for those lines "
            "(no line-number prefixes, no explanation). To delete the lines, output an empty block."
        )
        user = f"FILE: {rel}\nINSTRUCTION: {parts[2]}\n\nLINES TO REPLACE:\n{code}"
        if ctx:
            user += f"\n\nCLASS CONTEXT (elsewhere in the file, do not output these):\n{ctx}"
        if skel:
            user += f"\n\nFILE OUTLINE (for reference, do not output):\n{skel}"

        msg = self._chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
            max_tokens=MAX_OUT_TOKENS,
        )
        fm = FENCE_RE.search(msg)
        if not fm:
            return ("ERROR: writer returned no complete code block (output may have been cut off). "
                    "Nothing was changed. Retry on a smaller range or with a simpler instruction.")
        body = fm.group(1)
        body_lines = body.replace("\r\n", "\n").split("\n")
        if body_lines and body_lines[-1] == "":
            body_lines.pop()
        # strip accidental 'N: ' line-number prefixes copied from the prompt
        if body_lines and all(re.match(r"^\d+: ?", l) for l in body_lines if l.strip()):
            body_lines = [re.sub(r"^\d+: ?", "", l) for l in body_lines]

        new_lines = lines[: a - 1] + body_lines + lines[b:]
        new_text = nl.join(new_lines)
        new_bytes = new_text.encode("utf-8")

        if is_java:
            new_err = self.indexer.syntax_error(new_bytes)
            if new_err and not old_err:
                return (f"REJECTED: the edit would introduce a syntax error ({new_err}). "
                        "Nothing was changed. Fix the instruction or the range and retry.")

        self._backup(path)
        path.write_bytes(new_bytes)
        self._build_index()

        delta = len(body_lines) - (b - a + 1)
        report = [
            f"EDITED {rel} lines {a}-{b} -> {len(body_lines)} line(s) (delta {delta:+d}). "
            + ("Syntax OK." if is_java and not (self.indexer.syntax_error(new_bytes)) else
               ("WARNING: file still has syntax errors." if is_java else "")),
        ]
        if is_java:
            new_skel = self.indexer.generate_skeleton(path).splitlines()
            strip = lambda s: re.sub(r"\s+\d+-\d+$", "", s)
            old_set, new_set = {strip(s) for s in old_skel}, {strip(s) for s in new_skel}
            added = [s for s in new_skel if strip(s) not in old_set]
            removed = [s.strip() for s in old_skel if strip(s) not in new_set]
            if added:
                report.append("NEW/CHANGED outline entries:\n" + "\n".join(f"  {s.strip()}" for s in added[:15]))
            if removed:
                report.append("REMOVED outline entries:\n" + "\n".join(f"  {s}" for s in removed[:10]))
            if delta:
                report.append(f"Everything after line {b} moved by {delta:+d}. The CURRENT SKELETONS block already reflects this.")
        new_n = self._n(new_lines)
        s2, e2 = a, min(new_n, a + max(len(body_lines), 1) - 1)
        if body_lines:
            preview = "\n".join(f"{i}: {new_lines[i - 1]}" for i in range(s2, min(e2, s2 + 60) + 1))
            report.append(f"READBACK:\n{preview}")
        return "\n".join(report)

    # ------------------------------------------------------- whole-file writes

    def write_file(self, file_name: str, content: str) -> str:
        """Create a new file or fully overwrite one (used for new/empty files)."""
        path, err = self._resolve(file_name, fuzzy=False)
        if err:
            if err.startswith("Ambiguous"):
                return err
            rel = file_name.strip().replace("\\", "/").lstrip("/")
            if "/" not in rel:
                return f"ERROR: '{file_name}' does not exist. Give a repo-relative path to create it."
            first = rel.split("/", 1)[0]
            if first not in self.list_services() and len(self.target_services) == 1:
                rel = f"{next(iter(self.target_services))}/{rel}"
            path = (self.root / rel).resolve()
        else:
            path = path.resolve()

        if self.root not in path.parents:
            return "ERROR: path outside repo."

        try:
            nl = "\n"
            if path.exists():
                self._backup(path)
                if b"\r\n" in path.read_bytes():
                    nl = "\r\n"
            data = content.replace("\r\n", "\n").replace("\n", nl)
            if path.suffix == ".java":
                serr = self.indexer.syntax_error(data.encode("utf-8"))
                if serr:
                    return f"REJECTED: content has a syntax error ({serr}). Nothing was written."
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data.encode("utf-8"))
        except Exception as e:
            return f"ERROR writing file: {e}"

        self._build_index()

        warn = ""
        posix = path.as_posix()
        pkg = re.search(r"^\s*package\s+([\w.]+)\s*;", content, re.MULTILINE)
        if pkg and "src/main/java/" in posix:
            expected = posix.split("src/main/java/", 1)[1].rsplit("/", 1)[0].replace("/", ".")
            if pkg.group(1) != expected:
                warn = f"\nWARNING: package '{pkg.group(1)}' does not match folder '{expected}'"

        return f"WROTE {self._rel(path)} ({len(content.splitlines())} lines){warn}"

    # ------------------------------------------------- deterministic repo tools

    @staticmethod
    def _split_pair(arg: str):
        for sep in ("=>", "->", "|"):
            if sep in arg:
                a, b = arg.split(sep, 1)
                return a.strip().strip("'\""), b.strip().strip("'\"")
        return None, None

    def replace_in_repo(self, arg: str, apply: bool) -> str:
        """Deterministic repo-wide literal replace. apply=False is a dry run."""
        old, new = self._split_pair(arg)
        if not old or new is None or old == new:
            return "ERROR: use  old => new"
        if len(old) < 3:
            return "ERROR: 'old' must be at least 3 characters."

        changed, total, by_ext = [], 0, {}
        for p in self._text_files():
            try:
                raw = p.read_bytes()
                text = raw.decode("utf-8")
            except Exception:
                continue
            n = text.count(old)
            if not n:
                continue
            total += n
            changed.append(self._rel(p))
            by_ext[p.suffix.lower()] = by_ext.get(p.suffix.lower(), 0) + n
            if apply:
                self._backup(p)
                p.write_bytes(text.replace(old, new).encode("utf-8"))

        ext_s = ", ".join(f"{k}:{v}" for k, v in sorted(by_ext.items())) or "none"
        sample = "\n".join(changed[:20]) + (f"\n...+{len(changed) - 20} more" if len(changed) > 20 else "")
        head = "APPLIED" if apply else "PREVIEW (nothing written)"
        out = f"{head}: '{old}' -> '{new}': {total} occurrence(s) in {len(changed)} file(s) [by type: {ext_s}]\n{sample}"

        if apply:
            self._build_index()
            resid = []
            for p in self._text_files():
                try:
                    if old.lower() in p.read_text(encoding="utf-8", errors="ignore").lower():
                        resid.append(self._rel(p))
                except Exception:
                    pass
            out += f"\nBackups: {self._rel(self.backup_root)}"
            out += f"\nREMAINING case-insensitive matches of '{old}': {len(resid)} file(s) {resid[:10]}"
            out += f"\nREPO OVERVIEW AFTER:\n{self._repo_overview()}"
        return out

    # --------------------------------------------------------------- topology

    def _repo_overview(self) -> str:
        lines = []
        fmt = lambda d: ", ".join(f"{k}({v})" for k, v in sorted(d.items()))
        for svc in self.list_services():
            dirs, pkgs, n = {}, {}, 0
            for p in (self.root / svc).rglob("*.java"):
                if self._skip(p):
                    continue
                n += 1
                posix = p.as_posix()
                if "src/main/java/" in posix:
                    d = ".".join(posix.split("src/main/java/", 1)[1].split("/")[:2])
                    dirs[d] = dirs.get(d, 0) + 1
                try:
                    head = p.read_text(encoding="utf-8", errors="ignore")[:600]
                except Exception:
                    continue
                m = re.search(r"^\s*package\s+([\w.]+)\s*;", head, re.MULTILINE)
                if m:
                    k = ".".join(m.group(1).split(".")[:2])
                    pkgs[k] = pkgs.get(k, 0) + 1
            lines.append(f"- {svc}: {n} java files | folder roots: {fmt(dirs)} | package roots: {fmt(pkgs)}")
        return "\n".join(lines)

    @staticmethod
    def _trim_history(messages) -> bool:
        """Drop oldest grep/list results until the context fits. Reader/edit results are never dropped."""
        total = lambda: sum(len(m["content"]) for m in messages)
        trimmed = False
        for m in messages[2:]:
            if total() <= MAX_CONTEXT_CHARS:
                break
            if m["role"] == "user" and m["content"].startswith("TOOL RESULT:") and len(m["content"]) > 200:
                m["content"] = "TOOL RESULT: [omitted to save tokens; re-run the tool if needed]"
                trimmed = True
        return trimmed

    def _extract_topology_facts(self, user_question: str) -> str:
        target_services = set()
        q = user_question.lower()
        words = re.findall(r"[\w\.-]+", q)

        for w in words:
            clean = w if w.endswith(".java") else f"{w}.java"
            for path in self.file_index.get(clean, []):
                svc = self._service_of(path)
                if svc:
                    target_services.add(svc)

        for srv in self.list_services():
            if srv.lower() in q or srv.replace("-", "").lower() in q:
                target_services.add(srv)

        self.target_services = target_services
        all_services = self.list_services()
        facts = [f"ALL SERVICES: {', '.join(all_services)}"]

        if not target_services:
            facts.append("TARGET SERVICES: (not identified)")
            facts.append("REPO OVERVIEW (per service: java files | folder roots | declared package roots):")
            facts.append(self._repo_overview())
            return "\n".join(facts)

        facts.append(f"TARGET SERVICES: {', '.join(sorted(target_services))}")

        for service in sorted(target_services):
            service_path = self.root / service
            lines = []
            for java_file in service_path.rglob("*.java"):
                if self._skip(java_file):
                    continue
                try:
                    rel = str(java_file.relative_to(service_path)).replace("\\", "/")
                    content = java_file.read_text(encoding="utf-8", errors="ignore")
                    if not content.strip():
                        lines.append(f"{java_file.name} (EMPTY FILE) at {rel}")
                        continue

                    cm = re.search(r"(class|interface|enum|record)\s+(\w+)", content)
                    if not cm:
                        continue
                    info = f"{cm.group(2)} ({cm.group(1)}) at {rel}"
                    ext = re.search(r"extends\s+([\w<>, ]+?)\s*(?:implements|\{)", content)
                    imp = re.search(r"implements\s+([^{]+)", content)
                    if ext:
                        info += f" extends {ext.group(1).strip()}"
                    if imp:
                        info += f" implements {imp.group(1).strip()}"
                    lines.append(info)
                except Exception:
                    continue

            if lines:
                facts.append(f"\n{service}:")
                facts.extend(f"  - {l}" for l in lines)

        return "\n".join(facts)

    # ---------------------------------------------------- level 2: navigators

    def _dispatch(self, name: str, arg: str, can_write: bool, known: List[Path]):
        """Returns (result_text, mutated_files)."""
        if name == "skeleton":
            path, err = self._resolve(arg)
            if err:
                return err, False
            if path.suffix != ".java":
                return (f"{self._rel(path)} is not a Java file. Use "
                        f"read({self._rel(path)} | all | question)."), False
            if path not in known:
                known.append(path)
            return f"{self._rel(path)} is now in CURRENT SKELETONS (see the task message).", False
        if name == "read":
            return self.tool_read(arg), False
        if name == "refs":
            return self.tool_refs(arg), False
        if name == "grep":
            return self.grep_files(arg), False
        if name == "list_files":
            return self.list_files(arg), False
        if name == "replace_preview":
            return self.replace_in_repo(arg, apply=False), False
        if name in ("edit", "replace_all"):
            if not can_write:
                return "ERROR: you do not have write permission.", False
            res = self.tool_edit(arg) if name == "edit" else self.replace_in_repo(arg, apply=True)
            return res, not res.startswith(("ERROR", "REJECTED"))
        return f"Unknown tool: {name}. {TOOLS_HELP}", False

    def spawn_worker(
        self,
        worker_id: str,
        task: str,
        target_files: str = "",
        can_write: bool = False,
        context: str = "",
    ) -> ChildReport:
        print(f"\n[Stack Push -> {worker_id}]{' (write)' if can_write else ''}")
        print(f"  Task: {task}")

        write_doc = ""
        if can_write:
            write_doc = """
- TOOL: edit(file | range | instruction)   (a level-3 editor replaces EXACTLY that one contiguous range; use the range from the CURRENT skeleton; the harness rejects edits that break the syntax)
- TOOL: replace_preview(old => new)   (repo-wide literal replace, DRY RUN: counts per file type, writes nothing)
- TOOL: replace_all(old => new)   (repo-wide literal replace in java/xml/yml/properties/json files; backs up originals; run replace_preview first and check the counts)
- To create a NEW or EMPTY file, reply with EXACTLY this and nothing else:
WRITE_FILE: <repo-relative path>
```java
<full file content>
```
Editing rules: to insert a new member, edit the range of a neighbouring member and tell the editor to keep it and add the new member after it. To add an import, edit the import range. After each edit the skeleton updates automatically; verify important edits with read(...)."""

        system_prompt = f"""You are Worker {worker_id}, a level-2 NAVIGATOR inside a code repository. You plan; you do not read code yourself.

You are shown the SKELETON of your files: an outline with the line range of every member (annotations included). Imports are folded into one range. Lombok/Spring annotations can generate members that are not in the source, so never conclude "no constructor/getter" from the skeleton alone.
To learn what code does, delegate to a level-3 reader: read(file | ranges | question). The reader sees ONLY those lines, answers your question, and is then discarded. Ask one precise question per read; give the smallest ranges that contain the answer; include the ranges of fields/constants the code uses (or call refs first).
To inspect imports, read the import range with a specific question (e.g. which package does X come from?).
Facts visible in the skeleton (member names, signatures, annotations, class/interface/enum) you may state directly. Behaviour and imports need a reader.
Say "not in this file" only after the skeleton has covered the whole file.
If the trail leaves your files (something defined elsewhere), do NOT chase it: report it as FOLLOW-UP.

TOOLS (up to 4 per message, one per line, single-line arguments, ' | ' separates parts):
- TOOL: read(file | ranges | question)   (ranges like 10-20, 40-45 or 'all')
- TOOL: refs(file | ranges)   (deterministic: fields/methods of this file that those lines use, with their ranges)
- TOOL: skeleton(file)   (add another .java file's skeleton to your view)
- TOOL: grep(pattern)   (regex or literal; returns path:line: text)
- TOOL: list_files(directory){write_doc}

RULES:
- Do only the assigned goal. Delegate the minimum reading needed.
- Base every claim on the skeleton or on reader answers; if something is unknown, say so.
- Finish with a report and NO tool call, in this format:
FINDINGS: <concise answer to your goal / what you changed>
EVIDENCE: <'file:line: code' lines that prove it>
FOLLOW-UP: <one per line 'file | member | why', or NONE>
- Tool calls must be written exactly as TOOL: name(arg). Do not use XML/<tool_call> tags."""

        base_user = f"ASSIGNED FILES/SCOPE: {target_files or '(not specified)'}\n"
        if context:
            base_user += f"\nCONTEXT FROM ORCHESTRATOR:\n{context}\n"
        base_user += f"\nGOAL: {task}"

        known: List[Path] = []
        for f in re.split(r"[,\n]", target_files or ""):
            f = f.strip()
            if not f:
                continue
            p, err = self._resolve(f)
            if p is not None and p.suffix == ".java" and p not in known:
                known.append(p)
        known = known[:4]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": base_user},
        ]

        actions: List[str] = []
        final = ""
        finished = False
        partial = False
        seen = set()
        parse_fails = 0

        for _ in range(MAX_WORKER_TURNS):
            messages[1]["content"] = base_user + self._skeleton_block(known)   # always fresh
            if self._trim_history(messages):
                seen.clear()
            msg = self._chat(messages, max_tokens=MAX_OUT_TOKENS)
            messages.append({"role": "assistant", "content": msg})
            final = msg

            w = WRITE_RE.search(msg)
            if w:
                if can_write:
                    result = self.write_file(w.group(1), w.group(2))
                    actions.append(result.split("\n")[0])
                    if result.startswith("WROTE"):
                        seen.clear()
                        p, _err = self._resolve(w.group(1), fuzzy=False)
                        if p is not None and p.suffix == ".java" and p not in known:
                            known.append(p)
                else:
                    result = "ERROR: you do not have write permission."
                messages.append({"role": "user", "content": f"RESULTS:\n{result}"})
                continue

            calls = parse_tool_calls(msg)
            if calls:
                parse_fails = 0
                outs, keep = [], False
                for name, arg in calls:
                    key = (name, arg.lower())
                    if key in seen:
                        outs.append(f"[{name}({arg})] already returned above; do not repeat it.")
                        continue
                    seen.add(key)
                    result, mutated = self._dispatch(name, arg, can_write, known)
                    if name in ("read", "refs", "edit", "replace_all"):
                        keep = True
                    if mutated:
                        seen.clear()
                        actions.append(result.split("\n")[0][:160])
                    outs.append(f"[{name}({arg})]\n{result}")
                prefix = "RESULTS:\n" if keep else "TOOL RESULT:\n"
                messages.append({"role": "user", "content": prefix + "\n\n".join(outs)})
                continue

            if looks_like_tool_markup(msg):
                parse_fails += 1
                if parse_fails >= 3:
                    break   # stop burning tokens; go to the forced wrap-up report
                ex = self._rel(known[0]).rsplit("/", 1)[-1] if known else "File.java"
                messages.append({"role": "user", "content":
                    "TOOL ERROR: could not parse that call. Write ONE line per call, starting with 'TOOL:' "
                    f"and ending with ')', no XML tags. Example: TOOL: read({ex} | 10-20 | what does this code throw?)"})
                continue

            finished = True
            break

        if not finished:
            # budget exhausted: force a report from what was already learned
            messages.append({"role": "user", "content":
                "Tool budget exhausted. Write your final report NOW from what you already learned, "
                "in the FINDINGS / EVIDENCE / FOLLOW-UP format. NO tool calls."})
            msg = self._chat(messages, max_tokens=MAX_OUT_TOKENS)
            if msg and not looks_like_tool_markup(msg):
                final = msg + "\n\n[partial: turn budget hit]"
                partial = True
            else:
                final += "\n\n[worker hit turn limit before finishing]"

        status = "COMPLETED" if finished else ("PARTIAL" if partial else "INCOMPLETE")
        if actions:
            final += "\n\nACTIONS: " + "; ".join(actions)

        report = ChildReport(worker_id, task, final, status)
        self.frame.child_reports.append(report)
        print(f"[Stack Pop <- {worker_id}]")
        print(f"  Findings: {final[:300]}...\n")
        return report

    # -------------------------------------------------- level 1: orchestration

    def _valid_target(self, f: str, allow_new: bool) -> bool:
        """A worker target must be a real .java file (write workers may also name a new file in an existing folder)."""
        p, _ = self._resolve(f)
        if p is not None:
            return p.suffix == ".java"
        if allow_new and f.endswith(".java") and "/" in f:
            norm = f.replace("\\", "/").lstrip("/")
            bases = [self.root] + [self.root / sv for sv in sorted(self.target_services)]
            return any((base / norm).parent.is_dir() for base in bases)
        return False

    def _clean_workers(self, raw) -> List[dict]:
        workers = []
        if not isinstance(raw, list):
            return workers
        for w in raw:
            if not isinstance(w, dict) or not w.get("task"):
                continue
            files = w.get("files", "")
            if isinstance(files, list):
                files = ", ".join(str(f) for f in files)
            write = bool(w.get("write", False))
            flist = [f.strip() for f in re.split(r"[,\n]", str(files)) if f.strip()]
            task_text = str(w["task"])
            valid = [f for f in flist if self._valid_target(f, write)]
            dropped = [f for f in flist if f not in valid]
            if dropped and not MECHANICAL_RE.search(task_text):
                task_text += f"\n(Scope: {', '.join(dropped)}. Locate the .java files with grep/list_files.)"
            if len(valid) > MAX_FILES_PER_WORKER:
                for i in range(0, len(valid), MAX_FILES_PER_WORKER):
                    chunk = valid[i:i + MAX_FILES_PER_WORKER]
                    workers.append({
                        "task": f"{task_text}\n(Focus ONLY on: {', '.join(chunk)}.)",
                        "files": ", ".join(chunk),
                        "write": write,
                        "find": "",
                    })
            else:
                workers.append({"task": task_text, "files": ", ".join(valid), "write": write,
                                "find": str(w.get("find", "") or "").strip()})
        return workers

    def _files_matching(self, pattern: str) -> List[str]:
        """Code only, no LLM: repo-relative .java files that contain a line matching the regex."""
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))
        found: List[str] = []
        for p in sorted(self.root.rglob("*.java")):
            if self._skip(p):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if any(rx.search(line) for line in text.splitlines()):
                found.append(self._rel(p))
                if len(found) >= MAX_FIND_FILES:
                    break
        return found

    def _expand_workers(self, workers: List[dict]) -> List[dict]:
        """Workers without a file but with a `find` regex get one worker per matching file (code-only search)."""
        self._unscheduled = []
        out: List[dict] = []
        for w in workers:
            if w["files"].strip() or MECHANICAL_RE.search(w["task"]):
                out.append(w)
                continue
            if not w.get("find"):
                out.append(w)   # nothing to search for: the worker locates its own files with grep/list_files
                continue
            found = self._files_matching(w["find"])
            low = w["task"].lower()
            # the first service named in the goal is the target; later mentions are usually the pattern to copy
            named = sorted(
                (low.index(sv.lower()), sv) for sv in self.list_services() if sv.lower() in low
            )
            if named:
                found = [f for f in found if f.split("/")[0] == named[0][1]]
            print(f"[Find] /{w['find']}/ -> {len(found)} file(s): {w['task'][:60]!r}")
            if not found:
                out.append(w)   # nothing found: the worker keeps its goal and uses grep itself
                continue
            for f in found:
                out.append({"task": f"{w['task']}\n(Focus ONLY on: {f}.)", "files": f, "write": w["write"], "find": ""})
        for w in out[MAX_WORKERS_PER_ROUND:]:
            self._unscheduled.append(w["files"] or w["task"][:80])
        return out[:MAX_WORKERS_PER_ROUND]

    def _set_next_tasks(self, workers: List[dict]):
        self.frame.next_tasks = "\n".join(
            f"{i}. {'[WRITE] ' if w['write'] else ''}{w['task']} ({w['files']})"
            for i, w in enumerate(workers, 1)
        )

    def create_intelligent_plan(self, user_question: str) -> List[dict]:
        self.frame.global_intent = user_question
        self.frame.awareness_state = "PLANNING"

        topology = self._extract_topology_facts(user_question)

        system_prompt = """You are the Orchestrator (level 1) of a multi-agent developer system working on a code repository.

Given repository facts and a user goal, produce a plan and delegate work to workers.

Each worker is a NAVIGATOR: it is shown the outline (skeleton with line ranges) of its assigned files and delegates the reading/editing of exact line ranges to tiny single-shot readers/editors. So give a worker a GOAL (what to find out or change) and exactly one file (or none: leave "files" empty and set "find" to a regex; the harness greps the repo with it and gives each matching file its own worker). Do NOT tell workers to read whole files or to "list all methods"; state the question.
Workers can also grep and list files. Grant "write": true only to workers whose goal is to create or modify files, and only if the goal actually requires changes.
If the goal only asks for analysis or explanation, use read-only workers. If the goal needs changes to files that are named or easy to locate, give WRITE workers the whole job (they read what they need through readers, then edit). Use a separate read-only round only when the edit depends on information from files the writer would not see.
Do not assume anything about files you have not been shown; have workers verify.

HARD LIMIT: each worker gets exactly ONE file. If a task spans more files, leave "files" empty, name the service in the goal and set "find" to a regex matching lines in the files to inspect or change (e.g. "findById|orElseThrow"); the harness greps and gives each matching file its own worker.
Use file paths exactly as shown in the facts (relative to the service folder). Never guess folder names.

If the facts/overview already answer the goal, return "workers": [] and put the answer in "answer".
Otherwise use the FEWEST workers possible. For work that spans many files use ONE worker entry with "files" empty and a "find" regex. Never list every service.
Keep the JSON compact: at most 4 workers, each task under 25 words, files as short paths.
For repo-wide mechanical changes (renaming a package/word, replacing a string everywhere), use ONE write worker that runs replace_preview, then replace_all. Do NOT enumerate files or services.

Respond with ONLY a JSON object:
{
  "plan": "2-3 sentence strategy",
  "findings": "files/components from the facts that look relevant and why",
  "answer": "",
  "workers": [
    {"task": "goal for the worker", "files": "one file name, or empty", "find": "regex or empty", "write": false}
  ]
}"""

        user_message = f"{topology}\n\nUSER GOAL: {user_question}\n\nProduce the JSON plan."

        data, text = self._chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=MAX_OUT_TOKENS,
        )

        if data:
            self.frame.plan = str(data.get("plan", ""))
            self.frame.findings = str(data.get("findings", ""))
            workers = self._clean_workers(data.get("workers"))
        else:
            self.frame.plan = "(planner returned no valid JSON)"
            workers = []

        answer = str(data.get("answer", "")).strip() if data else ""
        if answer and not workers:
            self.frame.synthesis = answer
            self.frame.awareness_state = "ANSWERED"
            return []

        if not workers:
            workers = [
                {
                    "task": f"Investigate the repository to address this goal and report concrete findings: {user_question}",
                    "files": "",
                    "write": False,
                }
            ]

        self.frame.next_tasks = "\n".join(
            f"{i}. {'[WRITE] ' if w['write'] else ''}{w['task']} ({w['files']})"
            for i, w in enumerate(workers, 1)
        )
        self.frame.awareness_state = "READY_FOR_EXECUTION"

        total = self.total_prompt_tokens + self.total_completion_tokens
        print(
            f"[Telemetry] Prompt: {self.total_prompt_tokens} | "
            f"Completion: {self.total_completion_tokens} | Total: {total}\n"
        )
        print(self.frame.render() + "\n")
        return workers

    def synthesize(self, round_reports: List[ChildReport]) -> dict:
        system_prompt = """You are the Orchestrator reviewing worker reports.

Decide whether the user's goal is fully satisfied.

Base your judgement only on the reports. Do not assume work happened that the reports do not show.
If the goal requires file changes and no worker reports a successful edit/write, it is NOT done yet.
If an edit happened, check the report's readback/warnings; if something is wrong, schedule a fix.
If more information is needed, or edits/fixes/verification remain, schedule workers (same rules: a goal + one file, "write": true only for file modification).
If a report has FOLLOW-UP entries that the goal needs, schedule one worker per follow-up file with a precise goal.
Pass worker goals all the specifics they need (exact names, signatures, package names) because they do not see other reports.

HARD LIMIT: each worker gets exactly ONE file. If a task spans more files, leave "files" empty, name the service in the goal and set "find" to a regex matching lines in the files to inspect or change (e.g. "findById|orElseThrow"); the harness greps and gives each matching file its own worker.
Use file paths exactly as shown in the reports (relative to the service folder). Never guess folder names.
The "files" field holds .java file paths only, never a service or folder name.
Do NOT tell workers to read whole files or to "list all methods"; state the question.
For work that spans many files use ONE worker entry with "files" empty and a "find" regex. Never list every service.
Keep the JSON compact: at most 4 workers, each task under 25 words, files as short paths.

If a worker is INCOMPLETE or PARTIAL, do NOT repeat its task. Split it into smaller goals of one file each.

Respond with ONLY a JSON object:
{
  "summary": "concise factual state: what was found, what was done, what remains",
  "status": "DONE" or "CONTINUE",
  "workers": [ {"task": "...", "files": "...", "find": "", "write": false} ]
}
Use an empty workers list when status is DONE."""

        reports_text = "\n\n".join(
            f"[{r.worker_id}] ({r.status}) GOAL: {r.task}\nREPORT:\n{r.findings}"
            for r in round_reports
        )
        prior = f"PRIOR SYNTHESIS:\n{self.frame.synthesis}\n\n" if self.frame.synthesis else ""
        user_message = (
            f"USER GOAL: {self.frame.global_intent}\n\nPLAN: {self.frame.plan}\n\n"
            f"{prior}LATEST WORKER REPORTS:\n{reports_text}\n\nProduce the JSON decision."
        )

        data, text = self._chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        )
        if not data:
            self.frame.synthesis = "Synthesis failed (no valid JSON). See worker reports above."
            return {"status": "DONE", "workers": []}

        self.frame.synthesis = str(data.get("summary", text))
        status = str(data.get("status", "DONE")).upper()
        return {"status": status, "workers": self._clean_workers(data.get("workers"))}

    def save_session(self):
        session_data = {
            "frame": self.frame.to_dict(),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "token_log": self.token_log,
            "session_file": self.session_file,
        }
        Path(self.session_file).write_text(json.dumps(session_data, indent=2), encoding="utf-8")
        print(f"\n[Session saved to: {self.session_file}]")

    def load_session(self, path: str = None) -> bool:
        """Restore a saved session (manual use only; never called automatically)."""
        p = Path(path or self.session_file)
        if not p.exists():
            return False
        data = json.loads(p.read_text(encoding="utf-8"))
        f = data.get("frame", {})
        self.frame = WorkingFrame(
            global_intent=f.get("global_intent", ""),
            plan=f.get("plan", ""),
            findings=f.get("findings", ""),
            next_tasks=f.get("next_tasks", ""),
            synthesis=f.get("synthesis", ""),
            child_reports=[ChildReport(**r) for r in f.get("child_reports", [])],
            awareness_state=f.get("awareness_state", "PLANNING"),
        )
        self.total_prompt_tokens = data.get("total_prompt_tokens", 0)
        self.total_completion_tokens = data.get("total_completion_tokens", 0)
        return True

    def execute_plan(self, user_question: str) -> str:
        self._reset()
        t0 = self._snap()
        workers = self.create_intelligent_plan(user_question)
        self._log_tokens("planning", t0)
        if not workers:
            print(self.frame.render())
            self.save_session()
            return self.frame.synthesis

        workers = self._expand_workers(workers)
        self._set_next_tasks(workers)

        for rnd in range(1, MAX_ROUNDS + 1):
            self.frame.awareness_state = f"EXECUTING_ROUND_{rnd}"
            print(f"[EXECUTION PHASE - Round {rnd}, {len(workers)} worker(s)]")

            round_reports = []
            for i, w in enumerate(workers, 1):   # sequential: edits to one file never run in parallel
                wid = f"W{rnd}.{i}"
                t = self._snap()
                report = self.spawn_worker(
                    worker_id=wid,
                    task=w["task"],
                    target_files=w["files"],
                    can_write=w["write"],
                    context=self.frame.synthesis if rnd > 1 else "",
                )
                self._log_tokens(wid, t)
                round_reports.append(report)

            if self._unscheduled:
                note = ChildReport(
                    "DISCOVERY",
                    "files found but not scheduled (round worker limit)",
                    "FINDINGS: more matching files exist than workers allowed in one round.\n"
                    "EVIDENCE: NONE\nFOLLOW-UP:\n" + "\n".join(f"{f} | - | not processed yet" for f in self._unscheduled),
                    "PARTIAL",
                )
                round_reports.append(note)
                self.frame.child_reports.append(note)
                self._unscheduled = []

            self.frame.awareness_state = "SYNTHESIZING"
            print("\n[SYNTHESIS PHASE]\n")
            t = self._snap()
            decision = self.synthesize(round_reports)
            self._log_tokens(f"synthesis r{rnd}", t)
            if any(r.status == "COMPLETED" and looks_like_tool_markup(r.findings) for r in round_reports):
                print("[WARN] a worker's final report is raw tool markup; parsing failed")

            if decision["status"] != "CONTINUE" or not decision["workers"]:
                break
            if rnd == MAX_ROUNDS:
                self.frame.synthesis += "\n\n[round limit reached; work may remain]"
                break
            workers = self._expand_workers(decision["workers"])
            self._set_next_tasks(workers)

        if self._backed_up:
            self.frame.synthesis += "\n\nACTUAL CHANGES ON DISK (diff vs originals):\n" + self._change_report()

        self.frame.awareness_state = "SESSION_SAVED"
        print(self.frame.render())
        self._token_summary()
        total = self.total_prompt_tokens + self.total_completion_tokens
        print(
            f"[Telemetry] Prompt: {self.total_prompt_tokens} | "
            f"Completion: {self.total_completion_tokens} | Total: {total}"
        )
        self.save_session()

        return self.frame.synthesis or "Execution complete. Session saved for continuation."

    def investigate(self, user_question: str) -> str:
        return self.execute_plan(user_question)