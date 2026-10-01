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
MAX_STATE_CHARS = 1800        # code-built state handed to the next round
MAX_KNOWN_CHARS = 5000        # KNOWN block inside one navigator
MAX_FACT_CHARS = 520          # one compact reader fact
QUOTE_MAX_LINES = 12          # raw lines a navigator may ask for explicitly

TOOL_NAMES = ("skeleton", "read", "refs", "grep", "list_files",
              "edit", "replace_preview", "replace_all", "quote")
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
    "TOOL: quote(file | range) | TOOL: grep(pattern) | TOOL: list_files(dir) | TOOL: edit(file | range | instruction)"
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
        for m in re.finditer(r"\b(read|refs|quote|edit|replace_preview|replace_all)\(([^\n<]*)", msg):
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
        self._state_lines: List[str] = []
        self._applied: List[str] = []
        self._held: Dict[str, str] = {}
        self._worker_file: Dict[str, str] = {}
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
        self._state_lines = []
        self._applied = []
        self._held = {}
        self._worker_file = {}
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
            "You are a code reader. You see ONLY the line ranges below (one file) and have no tools. "
            "Answer the QUESTION strictly from the shown code; never guess about unseen code.\n"
            "Reply in exactly these 4 lines:\n"
            "ANSWER: <max 25 words>\n"
            "AT: <line numbers that prove it, e.g. 55,57-58, or NONE>\n"
            "LEADS: <things the code uses that are defined elsewhere, as 'name (line N)', or NONE>\n"
            "COV: FOUND | NOT_IN_RANGE <max 10 words on what is missing>"
        )
        user = f"FILE: {rel}\nQUESTION: {parts[2]}\n\nCODE (line-number prefixes are not part of the code):\n{code}"
        if ctx:
            user += (f"\n\nCLASS CONTEXT (declarations elsewhere in the file that the code refers to; "
                     f"NOT part of your ranges):\n{ctx}")
        answer = self._chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
            max_tokens=MAX_OUT_TOKENS,
        )
        if not answer or looks_like_tool_markup(answer):
            answer = "ANSWER: (reader produced no usable answer)\nCOV: UNKNOWN"
        fields = {m.group(1): m.group(2).strip()
                  for m in re.finditer(r"^(ANSWER|AT|LEADS|COV):\s*(.*)$", answer, re.MULTILINE)}
        if "ANSWER" not in fields:
            fields = {"ANSWER": self._flat(answer, 200)}
        at = self._valid_at(fields.get("AT", "NONE"), ranges)
        spans = ",".join(f"{a}-{b}" for a, b in ranges)
        line = (f"§READ {rel} L{spans} | ANSWER: {fields['ANSWER']} | AT: {at} | "
                f"LEADS: {fields.get('LEADS', 'NONE')} | COV: {fields.get('COV', 'UNKNOWN')}")
        return self._flat(line, MAX_FACT_CHARS)

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

    def tool_quote(self, arg: str) -> str:
        """Deterministic raw-lines view for a navigator that explicitly needs a few lines (max QUOTE_MAX_LINES)."""
        parts = [p.strip().strip("'\"") for p in arg.split("|", 1)]
        if len(parts) < 2 or not parts[1]:
            return "ERROR: use quote(file | range)   e.g. quote(A.java | 55-58)"
        path, err = self._resolve(parts[0])
        if err:
            return err
        try:
            lines, _ = self._read_lines(path)
        except Exception as e:
            return f"Error reading file: {e}"
        n = self._n(lines)
        ranges = self._parse_ranges(parts[1], n)
        if len(ranges) != 1:
            return "ERROR: quote needs exactly one range, e.g. quote(A.java | 55-58)"
        a, b = ranges[0]
        if b - a + 1 > QUOTE_MAX_LINES:
            b = a + QUOTE_MAX_LINES - 1
        body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(a, b + 1))
        return f"[quote {self._rel(path)} {a}-{b}]\n{body}"

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
        if name == "quote":
            return self.tool_quote(arg), False
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

    # ---------------------------------------------------------- compact protocol

    @staticmethod
    def _flat(text: str, cap: int) -> str:
        t = re.sub(r"\s+", " ", text).strip()
        return t if len(t) <= cap else t[: cap - 3] + "..."

    @staticmethod
    def _valid_at(at_text: str, ranges) -> str:
        """Keep only claimed line numbers that lie inside the ranges the reader was shown."""
        keep, claimed = [], False
        for tok in re.split(r"[,\s]+", at_text.strip()):
            m = re.fullmatch(r"(\d+)(?:-(\d+))?", tok)
            if not m:
                continue
            claimed = True
            lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
            if any(a <= lo and hi <= b for a, b in ranges):
                keep.append(tok)
        if keep:
            return ",".join(keep)
        return "INVALID" if claimed else "NONE"

    @staticmethod
    def _parse_report(text: str) -> dict:
        lines = [l.strip() for l in text.splitlines() if l.strip().startswith("§")]
        status, verdicts = None, []
        for l in lines:
            m = re.match(r"§STATUS\s+(DONE|HELD|FAIL)\b", l)
            if m:
                status = m.group(1)
            m = re.match(r"§A\s*(\d+)\s+([YN?])", l)
            if m:
                verdicts.append((int(m.group(1)), m.group(2)))
        return {"status": status, "lines": lines, "verdicts": verdicts}

    def _records(self, r) -> List[str]:
        """Compact record lines of one report (what the next round and the orchestrator get)."""
        rep = self._parse_report(r.findings)
        head = f"§R {r.worker_id} {r.status} {self._worker_file.get(r.worker_id, '') or '-'}"
        body = [l for l in rep["lines"] if not l.startswith("§STATUS")]
        if not rep["lines"]:
            body = ["§FACT (unstructured) " + self._flat(r.findings, 300)]
        acts = re.search(r"ACTIONS:\s*(.+)", r.findings)
        if acts:
            body.append("§ACTIONS " + self._flat(acts.group(1), 300))
        return [head] + body

    @staticmethod
    def _flat_lines(lines: List[str], cap: int) -> str:
        text = "\n".join(lines)
        return text if len(text) <= cap else text[: cap - 3] + "..."

    def _state_block(self) -> str:
        lines = list(self._state_lines)
        while len("\n".join(lines)) > MAX_STATE_CHARS and len(lines) > 1:
            lines.pop(0)
        return "\n".join(lines)

    def _final_block(self) -> str:
        out = []
        if self._applied:
            out.append(f"§APPLIED {len(self._applied)}: " + "; ".join(self._applied))
        for f, q in self._held.items():
            out.append(f"§HELD {f}: {q}")
        return "\n".join(out)

    def spawn_worker(
        self,
        worker_id: str,
        task: str,
        target_files: str = "",
        can_write: bool = False,
        context: str = "",
        expect: str = "",
    ) -> ChildReport:
        print(f"\n[Stack Push -> {worker_id}]{' (write)' if can_write else ''}")
        print(f"  Task: {task}")
        if expect:
            print(f"  Expect: {expect}")
        self._worker_file[worker_id] = (target_files or "").strip()

        write_doc = ""
        if can_write:
            write_doc = """
TOOL: edit(file | range | instruction)  (a level-3 editor replaces EXACTLY that one contiguous range; use ranges from the CURRENT skeleton; syntax-breaking edits are rejected)
TOOL: replace_preview(old => new)  (repo-wide literal replace, dry run)
TOOL: replace_all(old => new)  (repo-wide literal replace in java/xml/yml/properties/json; backs up originals; preview first)
New or empty file: reply with EXACTLY this and nothing else:
WRITE_FILE: <repo-relative path>
```java
<full file content>
```
To add a member, edit a neighbouring member's range and tell the editor to keep it and add the new one after it. To add an import, edit the import range. The skeleton updates after each edit.
WRITE RULE: edit only if every EXPECT item is Y and the edit achieves the GOAL as far as your file shows. If any is N or ?, make NO edit and report §STATUS HELD with a §Q."""

        system_prompt = f"""You are {worker_id}, a level-2 NAVIGATOR. You see the SKELETON (member line ranges) of your file. You plan; you never read code yourself.
Skeleton facts (names, signatures, annotations) you may state. Behaviour and imports need a reader. Lombok/Spring may generate members that are not in the source. Say "not in this file" only after the skeleton covered the whole file.
KNOWN lists what your earlier tool calls established; never ask the same thing twice.
TOOLS (max 4 per message, one per line, ' | ' separates parts, written exactly as TOOL: name(arg), no XML):
TOOL: read(file | ranges | question)  (a level-3 reader sees ONLY those lines, answers, is discarded; one precise question, smallest ranges)
TOOL: refs(file | ranges)  (fields/methods those lines use)
TOOL: quote(file | range)  (raw lines, max {QUOTE_MAX_LINES}; only when a reader answer is not enough)
TOOL: grep(pattern)
TOOL: list_files(directory)
TOOL: skeleton(file){write_doc}
If the answer needs another file, do not chase it: report a §Q. Do only the goal.
FINAL REPLY (no tool call), records only, one per line:
§STATUS DONE|HELD|FAIL
§A <n> Y|N|? <line numbers or max 12 words>   (one per EXPECT item)
§FACT <max 25 words> @<file>:L<lines>   (confirmed facts, max 8)
§Q <question> -> <file>   (for every N or ? that another file can settle)
§SCOPE <files you actually examined>   (a "not found" claim holds only inside this scope)
§FOLLOW <file> | <member> | <why>"""

        base_user = f"FILE/SCOPE: {target_files or '(not specified)'}\n"
        if context:
            base_user += f"\nSTATE FROM EARLIER ROUNDS:\n{context}\n"
        base_user += f"\nGOAL: {task}"
        if expect:
            base_user += f"\nEXPECT (verify each, answer with §A): {expect}"

        known: List[Path] = []
        for f in re.split(r"[,\n]", target_files or ""):
            f = f.strip()
            if not f:
                continue
            p, err = self._resolve(f)
            if p is not None and p.suffix == ".java" and p not in known:
                known.append(p)
        known = known[:4]

        known_facts: List[str] = []
        last = ""
        actions: List[str] = []
        final = ""
        finished = False
        partial = False
        edits_made = False
        seen = set()
        parse_fails = 0
        turn_tokens: List[int] = []

        def user_block(extra: str = "") -> str:
            facts = list(known_facts)
            while len("\n".join(facts)) > MAX_KNOWN_CHARS and len(facts) > 1:
                idx = next((i for i, f in enumerate(facts) if not f.startswith("§EDIT")), 0)
                facts.pop(idx)
            u = base_user + self._skeleton_block(known)
            if facts:
                u += "\n\nKNOWN (already established):\n" + "\n".join(facts)
            if extra:
                u += "\n\n" + extra
            return u

        for _ in range(MAX_WORKER_TURNS):
            # every turn is a fresh call rebuilt from state: no transcript is carried over
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_block(last)},
            ]
            before = self._snap()
            msg = self._chat(messages, max_tokens=MAX_OUT_TOKENS)
            after = self._snap()
            turn_tokens.append((after[0] - before[0]) + (after[1] - before[1]))
            final = msg
            last = ""

            w = WRITE_RE.search(msg)
            if w:
                if can_write:
                    result = self.write_file(w.group(1), w.group(2))
                    actions.append(result.split("\n")[0])
                    if result.startswith("WROTE"):
                        edits_made = True
                        seen.clear()
                        p, _err = self._resolve(w.group(1), fuzzy=False)
                        if p is not None and p.suffix == ".java" and p not in known:
                            known.append(p)
                else:
                    result = "ERROR: you do not have write permission."
                known_facts.append("§WRITE " + self._flat(result, 200))
                continue

            calls = parse_tool_calls(msg)
            if calls:
                parse_fails = 0
                raw_outs = []
                for name, arg in calls:
                    key = (name, arg.lower())
                    if key in seen:
                        raw_outs.append(f"[{name}] already in KNOWN or shown above; do not repeat it.")
                        continue
                    seen.add(key)
                    result, mutated = self._dispatch(name, arg, can_write, known)
                    if mutated:
                        seen.clear()
                        edits_made = True
                        actions.append(result.split("\n")[0][:160])
                    if name == "read":
                        known_facts.append(result if result.startswith("§READ") else "§READ " + self._flat(result, 400))
                    elif name == "refs":
                        known_facts.append(f"§REFS {arg}: " + self._flat(result, 600))
                    elif name in ("edit", "replace_all"):
                        known_facts.append("§EDIT " + self._flat(result, 450))
                    elif name == "skeleton":
                        known_facts.append("§SKEL " + self._flat(result, 120))
                    else:
                        raw_outs.append(f"[{name}({arg})]\n{result}")
                if raw_outs:
                    last = "RESULT OF YOUR LAST TOOL CALL (shown once):\n" + "\n\n".join(raw_outs)
                continue

            if looks_like_tool_markup(msg):
                parse_fails += 1
                if parse_fails >= 3:
                    break   # stop burning tokens; go to the forced wrap-up report
                ex = self._rel(known[0]).rsplit("/", 1)[-1] if known else "File.java"
                last = ("TOOL ERROR: could not parse that call. Write ONE line per call, starting with 'TOOL:' "
                        f"and ending with ')', no XML tags. Example: TOOL: read({ex} | 10-20 | what does this code throw?)")
                continue

            finished = True
            break

        if not finished:
            wrap = "Tool budget exhausted. Write your final records NOW from KNOWN. NO tool calls."
            msg = self._chat(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_block(wrap)}],
                max_tokens=MAX_OUT_TOKENS,
            )
            if msg and not looks_like_tool_markup(msg):
                final = msg
                partial = True
            else:
                final += "\n\n[worker hit turn limit before finishing]"

        rep = self._parse_report(final)
        body = "\n".join(rep["lines"]) if rep["lines"] else final
        warns = []
        if can_write and edits_made and (rep["status"] == "HELD" or any(v != "Y" for _n, v in rep["verdicts"])):
            warns.append("§WARN edit was made although an EXPECT item was not Y")
        if can_write and rep["status"] == "DONE" and not edits_made and not MECHANICAL_RE.search(task):
            warns.append("§WARN status DONE but no edit was applied")
        if partial:
            warns.append("§WARN partial: turn budget hit")
        if warns:
            body += "\n" + "\n".join(warns)

        if rep["status"] == "HELD":
            status = "HELD"
        elif rep["status"] == "FAIL":
            status = "INCOMPLETE"
        elif finished:
            status = "COMPLETED"
        elif partial:
            status = "PARTIAL"
        else:
            status = "INCOMPLETE"

        if actions:
            body += "\n\nACTIONS: " + "; ".join(actions)
            self._applied.extend(self._flat(a, 160) for a in actions if a.startswith(("EDITED", "WROTE", "APPLIED")))

        fkey = self._worker_file.get(worker_id) or worker_id
        if status == "HELD":
            qs = [l for l in rep["lines"] if l.startswith("§Q")]
            self._held[fkey] = " ".join(qs) if qs else "(no question given)"
        elif status == "COMPLETED":
            self._held.pop(fkey, None)

        print(f"[Turns] {worker_id}: " + " ".join(str(t) for t in turn_tokens))
        self.token_log.append({"label": f"{worker_id} turns", "prompt": 0, "completion": 0, "turns": turn_tokens})

        report = ChildReport(worker_id, task, body, status)
        self.frame.child_reports.append(report)
        print(f"[Stack Pop <- {worker_id}]")
        print(f"  Findings: {body[:300]}...\n")
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
            expect = w.get("expect", "")
            if isinstance(expect, list):
                expect = " | ".join(str(x) for x in expect)
            expect = str(expect or "").strip()[:600]
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
                        "expect": expect,
                    })
            else:
                workers.append({"task": task_text, "files": ", ".join(valid), "write": write,
                                "find": str(w.get("find", "") or "").strip(), "expect": expect})
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
                out.append({"task": f"{w['task']}\n(Focus ONLY on: {f}.)", "files": f, "write": w["write"],
                            "find": "", "expect": w.get("expect", "")})
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

        system_prompt = """You are the Orchestrator (level 1) of a multi-agent developer system working on a code repository. Given repository facts and a user goal, produce a plan and delegate.

Workers are NAVIGATORS. Each sees only the outline (member line ranges) of ONE file and delegates reading/editing of exact line ranges to tiny readers/editors. A worker cannot see other files or other workers' reports.
Give each worker: "task" = the GOAL including the intent (what must be true afterwards); "files" = exactly one file; "expect" = what you assume about that file, as numbered claims (A1, A2, ...). The worker verifies every claim and edits only if all hold; if one is wrong, or depends on another file, it makes NO edit and reports back and you re-plan.
Work that spans many files: ONE worker entry with "files" empty and "find" = a regex matching lines in the files to inspect or change (e.g. "findById|orElseThrow"); name the service in the task. The harness greps and gives each matching file its own worker. "files" never holds a service or folder name. Never list every service.
Repo-wide mechanical changes (rename a package/word, replace a string everywhere): ONE write worker that runs replace_preview then replace_all; no files, no expect.
"write": true only for workers that must modify files; analysis-only goals use read-only workers. If an edit depends on facts in other files (what a method throws, a class's package or constructor), first schedule one-file read-only workers for those files, then the write workers.
Do not assume anything about files you have not been shown. Use the FEWEST workers: at most 4, each task under 25 words, file paths as shown in the facts (relative to the service folder). If the facts already answer the goal, return "workers": [] and put the answer in "answer".

Respond with ONLY a JSON object:
{
  "plan": "2-3 sentence strategy",
  "findings": "relevant files/components from the facts and why",
  "answer": "",
  "workers": [
    {"task": "goal + intent", "files": "one file or empty", "find": "regex or empty", "expect": ["A1 ...", "A2 ..."], "write": false}
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
                    "find": "",
                    "expect": "",
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

Reports are compact records: §STATUS DONE|HELD|FAIL; §A <n> Y|N|? (verdict on your EXPECT item n); §FACT; §Q <question> -> <file>; §SCOPE (a "not found" claim holds only inside that scope); §FOLLOW; §ACTIONS (edits the harness applied); §WARN.
Judge only from the reports; do not assume work happened that they do not show. If the goal needs file changes and no report shows an applied edit, it is not done.
HELD means the worker made NO edit. For each §Q schedule a one-file read-only worker on the file it names with a precise question; in a later round re-issue the write worker with corrected "expect". A §WARN about an edit made despite an unmet EXPECT: schedule a worker to check that file.
Workers do not see other reports: put every specific they need (names, signatures, package names, message convention) in the task or expect.
Limits: ONE file per worker; for work spanning many files leave "files" empty and set "find" (regex); never a service or folder name in "files"; at most 4 workers; each task under 25 words; paths exactly as in the reports. Never repeat a PARTIAL/INCOMPLETE task: split it into one-file goals.

When status is DONE, "summary" is the final answer for the user: confirmed results, what was changed, anything HELD with its question. Compact lines, no filler.

Respond with ONLY a JSON object:
{
  "summary": "state: found / done / remaining",
  "status": "DONE" or "CONTINUE",
  "workers": [ {"task": "...", "files": "...", "find": "", "expect": [], "write": false} ]
}
Use an empty workers list when status is DONE."""

        reports_text = "\n\n".join(
            f"[{r.worker_id}] ({r.status}) GOAL: {r.task}\n" + self._flat_lines(self._records(r), 1500)
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
                    context=self._state_block() if rnd > 1 else "",
                    expect=str(w.get("expect", "") or ""),
                )
                self._log_tokens(wid, t)
                round_reports.append(report)

            if self._unscheduled:
                note = ChildReport(
                    "DISCOVERY",
                    "files found but not scheduled (round worker limit)",
                    "\n".join(f"§FOLLOW {f} | - | matched the search but exceeded the round limit"
                              for f in self._unscheduled),
                    "PARTIAL",
                )
                round_reports.append(note)
                self.frame.child_reports.append(note)
                self._unscheduled = []

            for r in round_reports:
                self._state_lines.extend(self._records(r))

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

        final_block = self._final_block()
        if final_block:
            self.frame.synthesis += "\n\n" + final_block
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