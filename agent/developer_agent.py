import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from agent.key_manager import GroqKeyRotator
from engine.ast_indexer import ASTIndexer

MAX_TOOL_CHARS = 15000       # cap on any single tool result fed to a worker
MAX_WORKER_TURNS = 12   
MAX_ROUNDS = 3
MAX_WORKERS_PER_ROUND = 8
MAX_GREP_HITS = 60
MAX_FILES_PER_WORKER = 2

TOOL_RE = re.compile(r"TOOL:\s*(\w+)\(([^)]*)\)")
WRITE_RE = re.compile(
    r"WRITE_FILE:\s*(\S+)[ \t]*\r?\n```[\w+-]*\r?\n(.*?)\r?\n```",
    re.DOTALL,
)
TOOL_NAMES = ("read_file", "get_file_skeleton", "list_files", "grep_files")


def parse_tool_calls(msg: str, limit: int = 4):
    """Return a list of (name, arg) from any tool-call format."""
    calls = []
    for m in re.finditer(r"TOOL:\s*(\w+)\(([^)]*)\)", msg):
        if m.group(1) in TOOL_NAMES:
            calls.append((m.group(1), m.group(2).strip().strip("'\"")))
    if calls:
        return calls[:limit]

    if looks_like_tool_markup(msg):
        seen = set()
        for m in re.finditer(r"\b(" + "|".join(TOOL_NAMES) + r")\b", msg):
            name = m.group(1)
            tail = msg[m.end():]
            if name == "grep_files":
                g = re.match(r"[^\w\\/.^(\[]*([^<\n]+?)\s*(?:</|\)|\n|$)", tail)
            else:
                g = re.match(r"[^\w./\\-]*([\w./\\-]+)", tail)
            arg = g.group(1).strip().strip("'\"") if g else ""
            if arg and (name, arg) not in seen:
                seen.add((name, arg))
                calls.append((name, arg))
    return calls[:limit]

def looks_like_tool_markup(msg: str) -> bool:
    return any(s in msg for s in ("<tool_call>", "<function", "TOOL:"))


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
        self.frame = WorkingFrame()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.file_index: Dict[str, List[Path]] = {}
        self.target_services: set = set()
        self.session_file = session_file or "cognitive_os_session.json"
        self._build_index()

    # ------------------------------------------------------------------ infra

    def _build_index(self):
        self.file_index.clear()
        for p in self.root.rglob("*.java"):
            if "test" not in p.parts and "target" not in p.parts:
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

    def _chat(self, messages, max_tokens: int = 2000) -> str:
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

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    @staticmethod
    def _cap(text: str) -> str:
        if len(text) <= MAX_TOOL_CHARS:
            return text
        return text[:MAX_TOOL_CHARS] + f"\n...[truncated, {len(text)} chars total]"

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
    
    def read_file(self, file_name: str) -> str:
        path, err = self._resolve(file_name)
        if err:
            return err
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                return f"[EMPTY FILE] {self._rel(path)} is empty (0 bytes)"
            return self._cap(f"[{self._rel(path)}]\n{content}")
        except Exception as e:
            return f"Error reading file: {e}"

    def get_file_skeleton(self, file_name: str) -> str:
        path, err = self._resolve(file_name)
        if err:
            return err
        try:
            return self._cap(self.indexer.generate_skeleton(path))
        except Exception as e:
            return f"Error: {e}"

    def list_files(self, directory: str = "") -> str:
        directory = directory.strip().strip("'\"").replace("\\", "/").lstrip("/")
        bases = [self.root / directory] if directory else [self.root]
        if directory:
            bases += [self.root / s / directory for s in sorted(self.target_services)]
        base = next((b for b in bases if b.is_dir()), None)
        if base is None:
            return (f"Error: directory '{directory}' not found. "
                    f"Services: {', '.join(self.list_services())}")
        files = []
        for p in sorted(base.rglob("*.java")):
            if "test" in p.parts or "target" in p.parts:
                continue
            files.append(self._rel(p))
        return self._cap("\n".join(files)) if files else "(no java files)"

    def grep_files(self, pattern: str) -> str:
        pattern = pattern.strip().strip("'\"")
        if not pattern:
            return "Error: empty pattern."
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))

        if self.target_services:
            roots = [self.root / s for s in sorted(self.target_services)]
        else:
            roots = [self.root]

        hits = []
        for base in roots:
            for p in base.rglob("*.java"):
                if "test" in p.parts or "target" in p.parts:
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

    def write_file(self, file_name: str, content: str) -> str:
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
            if path.exists():
                old = path.read_text(encoding="utf-8", errors="ignore")
                if old.strip():
                    path.with_suffix(path.suffix + ".bak").write_text(old, encoding="utf-8")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
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

        return (
            f"WROTE {self._rel(path)} ({len(content.splitlines())} lines){warn}\n"
            f"READBACK:\n{self.read_file(self._rel(path))}"
        )

    # --------------------------------------------------------------- topology

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
            facts.append(
                "TARGET SERVICES: (not identified; workers should use list_files / grep_files "
                "to locate relevant code)"
            )
            return "\n".join(facts)

        facts.append(f"TARGET SERVICES: {', '.join(sorted(target_services))}")

        for service in sorted(target_services):
            service_path = self.root / service
            lines = []
            for java_file in service_path.rglob("*.java"):
                if "test" in java_file.parts or "target" in java_file.parts:
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

    # ---------------------------------------------------------------- workers

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
- To create or overwrite a file, reply with EXACTLY this format and nothing else:
WRITE_FILE: <path or filename>
```<language>
<full file content>
```
Before writing, read neighbouring/related files so your package, imports, naming and style
match the codebase. Write complete files only, never fragments."""

        system_prompt = f"""You are Worker {worker_id}, a focused sub-agent working inside a code repository.

TOOLS (you may call up to 4 per message, one per line):
- TOOL: read_file(filename_or_relative_path)
- TOOL: get_file_skeleton(filename_or_relative_path)
- TOOL: list_files(directory)   (directory relative to repo root, blank for all)
- TOOL: grep_files(pattern)     (regex or literal; returns path:line: text){write_doc}

RULES:
- Do only the assigned task. Read only what you need.
- Base every claim on file contents you actually read; if something is missing or unclear, say so.
- Finish with a complete, concrete report and NO tool call in that final message.
- Tool calls must be written exactly as `TOOL: name(arg)`. Do not use XML/<tool_call> tags.
- Your final report must include the actual code/signatures you found, not a statement that you will read."""

        user_message = f"ASSIGNED FILES/SCOPE: {target_files or '(not specified)'}\n"
        if context:
            user_message += f"\nCONTEXT FROM ORCHESTRATOR:\n{context}\n"
        user_message += f"\nTASK: {task}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        actions: List[str] = []
        final = ""
        finished = False
        partial = False
        seen = set()

        for _ in range(MAX_WORKER_TURNS):
            msg = self._chat(messages, max_tokens=3000)
            messages.append({"role": "assistant", "content": msg})
            final = msg

            w = WRITE_RE.search(msg)
            if w:
                if can_write:
                    result = self.write_file(w.group(1), w.group(2))
                    actions.append(result.split("\n")[0])
                else:
                    result = "ERROR: you do not have write permission."
                messages.append({"role": "user", "content": f"TOOL RESULT:\n{result}"})
                continue

            calls = parse_tool_calls(msg)
            if calls:
                outs = []
                for name, arg in calls:
                    key = (name, arg.lower())
                    if key in seen:
                        outs.append(f"[{name}({arg})] already returned above; do not repeat it.")
                        continue
                    seen.add(key)
                    if name == "read_file":
                        result = self.read_file(arg)
                    elif name == "get_file_skeleton":
                        result = self.get_file_skeleton(arg)
                    elif name == "list_files":
                        result = self.list_files(arg)
                    else:
                        result = self.grep_files(arg)
                    outs.append(f"[{name}({arg})]\n{result}")
                messages.append({"role": "user", "content": "TOOL RESULT:\n" + "\n\n".join(outs)})
                continue

            if looks_like_tool_markup(msg):
                messages.append({"role": "user", "content":
                    "TOOL ERROR: could not parse that call. Use one line per call: "
                    "TOOL: read_file(path/to/File.java)"})
                continue

            finished = True
            break

        if not finished:
            # budget exhausted: force a report from what was already read
            messages.append({"role": "user", "content":
                "Tool budget exhausted. Write your final report NOW from what you already read. "
                "Include actual code/signatures. NO tool calls."})
            msg = self._chat(messages, max_tokens=3000)
            if msg and not looks_like_tool_markup(msg):
                final = msg + "\n\n[partial: turn budget hit]"
                partial = True
            else:
                final += "\n\n[worker hit turn limit before finishing]"

        status = "COMPLETED" if finished else ("PARTIAL" if partial else "INCOMPLETE")
        report = ChildReport(worker_id, task, final, status)   
        if actions:
            final += "\n\nACTIONS: " + "; ".join(actions)

        report = ChildReport(worker_id, task, final, "COMPLETED" if finished else "INCOMPLETE")
        self.frame.child_reports.append(report)
        print(f"[Stack Pop <- {worker_id}]")
        print(f"  Findings: {final[:300]}...\n")
        return report

    # ----------------------------------------------------------- orchestration
    @staticmethod
    def _clean_workers(raw) -> List[dict]:
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
            if not write and len(flist) > MAX_FILES_PER_WORKER:
                for i in range(0, len(flist), MAX_FILES_PER_WORKER):
                    chunk = flist[i:i + MAX_FILES_PER_WORKER]
                    workers.append({
                        "task": f"{w['task']}\n(Focus ONLY on: {', '.join(chunk)}. "
                                f"Report the actual contents/signatures you found.)",
                        "files": ", ".join(chunk),
                        "write": False,
                    })
            else:
                workers.append({"task": str(w["task"]), "files": str(files), "write": write})
        return workers[:MAX_WORKERS_PER_ROUND]

    def create_intelligent_plan(self, user_question: str) -> List[dict]:
        self.frame.global_intent = user_question
        self.frame.awareness_state = "PLANNING"

        topology = self._extract_topology_facts(user_question)

        system_prompt = """You are the Orchestrator of a multi-agent developer system working on a code repository.

Given repository facts and a user goal, produce a plan and delegate work to child workers.

Each worker has a fresh context and tools to read files, view skeletons, list files, grep, and (only if permitted) write files.
Design workers so each has ONE narrow, self-contained task and reads only a few files.
Grant "write": true only to workers whose task is to create or modify files, and only if the goal actually requires changes.
If the goal only asks for analysis or explanation, use read-only workers.
If the goal needs changes, first use read-only workers to gather what is needed; later rounds will handle edits.
Do not assume anything about files you have not been shown; have workers verify.

HARD LIMIT: each worker gets at most 2 files. If a task needs more files, split it into several workers.
Use file paths exactly as shown in the facts (relative to the service folder). Never guess folder names.

Respond with ONLY a JSON object:
{
"plan": "2-3 sentence strategy",
"findings": "files/components from the facts that look relevant and why",
"workers": [
{"task": "specific instruction", "files": "comma-separated file names or scope", "write": false}
]
}"""

        user_message = f"{topology}\n\nUSER GOAL: {user_question}\n\nProduce the JSON plan."

        text = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1200,
        )
        data = self._extract_json(text)

        if data:
            self.frame.plan = str(data.get("plan", ""))
            self.frame.findings = str(data.get("findings", ""))
            workers = self._clean_workers(data.get("workers"))
        else:
            self.frame.plan = text
            workers = []

        if not workers:
            workers = [
                {
                    "task": (
                        "Investigate the repository to address this goal and report "
                        f"concrete findings: {user_question}"
                    ),
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
        system_prompt = """You are the Orchestrator reviewing child worker reports.

Decide whether the user's goal is fully satisfied.

Base your judgement only on the reports. Do not assume work happened that the reports do not show.
If the goal requires file changes and no worker has successfully written them, they are NOT done yet.
If a write happened, check the report's readback/warnings; if something is wrong, schedule a fix.
If more information is needed, or edits/fixes/verification remain, schedule workers (same rules: narrow tasks, "write": true only for file modification).
Pass worker tasks all the specifics they need (exact signatures, paths, package names) because they do not see other reports.

HARD LIMIT: each worker gets at most 2 files. If a task needs more files, split it into several workers.
Use file paths exactly as shown in the facts (relative to the service folder). Never guess folder names.

If a worker is INCOMPLETE or PARTIAL, do NOT repeat its task. Split it into smaller tasks of at most 2 files each, using the exact paths from the reports.
Respond with ONLY a JSON object:
{
"summary": "concise factual state: what was found, what was done, what remains",
"status": "DONE" or "CONTINUE",
"workers": [ {"task": "...", "files": "...", "write": false} ]
}
Use an empty workers list when status is DONE."""

        reports_text = "\n\n".join(
            f"[{r.worker_id}] ({r.status}) TASK: {r.task}\nREPORT:\n{r.findings}"
            for r in round_reports
        )
        prior = f"PRIOR SYNTHESIS:\n{self.frame.synthesis}\n\n" if self.frame.synthesis else ""
        user_message = (
            f"USER GOAL: {self.frame.global_intent}\n\nPLAN: {self.frame.plan}\n\n"
            f"{prior}LATEST WORKER REPORTS:\n{reports_text}\n\nProduce the JSON decision."
        )

        text = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=2500,
        )
        data = self._extract_json(text)
        if not data:
            self.frame.synthesis = text
            return {"status": "DONE", "workers": []}

        self.frame.synthesis = str(data.get("summary", text))
        status = str(data.get("status", "DONE")).upper()
        return {"status": status, "workers": self._clean_workers(data.get("workers"))}

    def save_session(self):
        session_data = {
            "frame": self.frame.to_dict(),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "session_file": self.session_file,
        }
        Path(self.session_file).write_text(json.dumps(session_data, indent=2), encoding="utf-8")
        print(f"\n[Session saved to: {self.session_file}]")

    def load_session(self, path: str = None) -> bool:
        """Restore a saved session so a new run can continue from it."""
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
        workers = self.create_intelligent_plan(user_question)

        for rnd in range(1, MAX_ROUNDS + 1):
            self.frame.awareness_state = f"EXECUTING_ROUND_{rnd}"
            print(f"[EXECUTION PHASE - Round {rnd}, {len(workers)} worker(s)]")

            round_reports = []
            for i, w in enumerate(workers, 1):
                report = self.spawn_worker(
                    worker_id=f"W{rnd}.{i}",
                    task=w["task"],
                    target_files=w["files"],
                    can_write=w["write"],
                    context=self.frame.synthesis if rnd > 1 else "",
                )
                round_reports.append(report)

            self.frame.awareness_state = "SYNTHESIZING"
            print("\n[SYNTHESIS PHASE]\n")
            decision = self.synthesize(round_reports)
            if any(r.status == "COMPLETED" and looks_like_tool_markup(r.findings) for r in round_reports):
                print("[WARN] a worker's final report is raw tool markup; parsing failed")

            if decision["status"] != "CONTINUE" or not decision["workers"]:
                break
            if rnd == MAX_ROUNDS:
                self.frame.synthesis += "\n\n[round limit reached; work may remain]"
                break
            workers = decision["workers"]
            self.frame.next_tasks = "\n".join(
                f"{i}. {'[WRITE] ' if w['write'] else ''}{w['task']}"
                for i, w in enumerate(workers, 1)
            )

        self.frame.awareness_state = "SESSION_SAVED"
        print(self.frame.render())
        total = self.total_prompt_tokens + self.total_completion_tokens
        print(
            f"[Telemetry] Prompt: {self.total_prompt_tokens} | "
            f"Completion: {self.total_completion_tokens} | Total: {total}"
        )
        self.save_session()

        return self.frame.synthesis or "Execution complete. Session saved for continuation."

    def investigate(self, user_question: str) -> str:
        return self.execute_plan(user_question)