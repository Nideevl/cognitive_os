import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from agent.key_manager import GroqKeyRotator
from engine.ast_indexer import ASTIndexer
from protocols.messages import FrameSpawnRequest, FrameReturnPacket


# =====================================================================
# WORKING FRAME (Cognitive State Dashboard)
# =====================================================================
@dataclass
class SubGoal:
    description: str
    status: str = "PENDING"        # PENDING | ACTIVE | DONE | INVALIDATED
    file_name: Optional[str] = None


@dataclass
class RevisionEntry:
    file_name: str
    snapshot_hash: str
    reason: str


@dataclass
class WorkingFrame:
    global_intent: str = ""
    working_hypothesis: str = ""
    active_sub_goal: Optional[str] = None
    sub_goal_stack: List[SubGoal] = field(default_factory=list)
    revision_ledger: List[RevisionEntry] = field(default_factory=list)
    awareness_state: str = "ORIENT"   # ORIENT | PINPOINT | MUTATE | PROPAGATE | BACKTRACK
    last_action: str = "(none yet)"

    def render(self) -> str:
        live_lines = []
        for sg in self.sub_goal_stack:
            if sg.status in ("PENDING", "ACTIVE"):
                marker = "[ACTIVE]" if sg.status == "ACTIVE" else "[PENDING]"
                file_suffix = f"  ({sg.file_name})" if sg.file_name else ""
                live_lines.append(f"   - {marker} {sg.description}{file_suffix}")
        live_str = "\n".join(live_lines) if live_lines else "   (empty)"

        ledger_lines = [
            f"   - {e.file_name} @ {e.snapshot_hash[:8]} — {e.reason}"
            for e in self.revision_ledger[-5:]
        ]
        ledger_str = "\n".join(ledger_lines) if ledger_lines else "   (no mutations yet)"

        return (
            "=== WORKING FRAME ===\n"
            f"INTENT      : {self.global_intent or '(unset)'}\n"
            f"ACTIVE GOAL : {self.active_sub_goal or '(none)'}\n"
            f"HYPOTHESIS  : {self.working_hypothesis or '(unset)'}\n"
            "LIVE STACK  :\n"
            f"{live_str}\n"
            "LEDGER (last 5 mutations):\n"
            f"{ledger_str}\n"
            f"STATE       : {self.awareness_state}\n"
            f"LAST ACTION : {self.last_action}\n"
            "====================="
        )


# =====================================================================
# DEVELOPER AGENT (Runtime Host & Stack Machine Orchestrator)
# =====================================================================
class DeveloperAgent:
    def __init__(
        self,
        project_root: str,
        key_rotator: GroqKeyRotator,
        model: str = "qwen/qwen3.8-27b"
    ):
        self.root = Path(project_root)
        self.rotator = key_rotator
        self.model = model

        # Local AST Indexer
        self.indexer = ASTIndexer()

        # Cognitive state
        self.frame = WorkingFrame()

        # Token telemetry
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Snapshot storage: { file_name: [ {content, hash, reason}, ... ] }
        self.file_snapshots: Dict[str, List[Dict[str, Any]]] = {}

        # Bare filename index: {"payment.java": Path(...)}
        self.file_index: Dict[str, Path] = {}
        self._build_index()

        # Inspection loop cache
        self.inspected_cache = set()

    # =================================================================
    # TOPOLOGY & INDEXING
    # =================================================================
    def _build_index(self):
        self.file_index.clear()
        for p in self.root.rglob("*.java"):
            if "test" not in p.parts and "target" not in p.parts:
                self.file_index[p.name.lower()] = p

    def list_services(self) -> List[str]:
        services = []
        for p in self.root.iterdir():
            if p.is_dir() and not p.name.startswith((".", "target", "build")):
                if (p / "pom.xml").exists() or (p / "src").exists():
                    services.append(p.name)
        return sorted(services)

    def list_service_files(self, service_name: str) -> List[str]:
        srv_path = self.root / service_name
        if not srv_path.exists():
            return [f"Service '{service_name}' not found."]
        files = []
        for f in srv_path.rglob("*.java"):
            if "test" not in f.parts and "target" not in f.parts:
                files.append(f.name)
        return sorted(files)

    def _resolve(self, name_or_path: str) -> Optional[Path]:
        clean = Path(name_or_path.strip().replace("\\", "/")).name.lower()
        return self.file_index.get(clean)

    # =================================================================
    # SCOPE RESOLVER
    # =================================================================
    def resolve_scope_content(self, target_scope: str) -> str:
        # 1. Check for line slices: e.g. "Payment.java lines 10-30"
        match_lines = re.search(
            r"([a-zA-Z0-9_\-\.]+\.java)\s+lines?\s+(\d+)\s*-\s*(\d+)",
            target_scope,
            re.IGNORECASE
        )
        if match_lines:
            file_name = match_lines.group(1)
            start_l = int(match_lines.group(2))
            end_l = int(match_lines.group(3))

            target_path = self._resolve(file_name)
            if not target_path:
                return f"Error: File '{file_name}' not found in repository index."

            raw_text = target_path.read_text(encoding="utf-8", errors="ignore")
            lines = raw_text.splitlines(keepends=True)
            if not lines or not raw_text.strip():
                return (f"Notice: File '{file_name}' exists at "
                        f"'{target_path.relative_to(self.root)}' but is completely EMPTY "
                        f"(0 lines). If you need to populate it, use `write_file`.")

            start = max(1, start_l)
            end = min(len(lines), end_l)
            return "".join([f"{i}: {l}" for i, l in enumerate(lines[start - 1:end], start=start)])

        # 2. Check for any Java file mentioned anywhere in the string
        match_java = re.search(r"([a-zA-Z0-9_\-\.]+\.java)", target_scope, re.IGNORECASE)
        if match_java:
            file_name = match_java.group(1)
            target_path = self._resolve(file_name)
            if not target_path:
                return f"Error: File '{file_name}' not found in repository index."

            raw_text = target_path.read_text(encoding="utf-8", errors="ignore")
            if not raw_text.strip():
                return (f"Notice: File '{file_name}' exists at "
                        f"'{target_path.relative_to(self.root)}' but is completely EMPTY "
                        f"(0 lines). If you need to populate it, use `write_file`.")

            return self.indexer.generate_skeleton(target_path)

        # 3. If target is a service name or generic scope
        for srv in self.list_services():
            if srv in target_scope:
                files = self.list_service_files(srv)
                return f"Service: {srv}\nFiles available: {', '.join(files)}"

        return (f"Scope '{target_scope}' did not match any file or service. "
                f"Available services: {', '.join(self.list_services())}")

    # =================================================================
    # FIX #1 — PINPOINT: SURGICAL LINES (empty-file notice)
    # =================================================================
    def read_surgical_lines(self, file_name: str, start_line: int, end_line: int) -> str:
        self.frame.awareness_state = "PINPOINT"
        target = self._resolve(file_name)
        if not target:
            return f"Error: File '{file_name}' not found in repository index."
        try:
            raw_text = target.read_text(encoding="utf-8", errors="ignore")
            lines = raw_text.splitlines(keepends=True)

            # Explicit notice for empty files so the model stops escalating
            # end_line hoping to find code "further down."
            if not lines or not raw_text.strip():
                try:
                    rel_path = target.relative_to(self.root)
                except ValueError:
                    rel_path = target.name
                return (
                    f"Notice: File '{file_name}' exists at '{rel_path}' "
                    f"but is completely EMPTY (0 lines). "
                    f"To populate it, inspect a sibling file for structure, "
                    f"then use `write_file`."
                )

            start = max(1, int(start_line))
            end = min(len(lines), int(end_line))
            return "".join([f"{i}: {l}" for i, l in enumerate(lines[start - 1:end], start=start)])
        except Exception as e:
            return f"Error reading lines: {e}"

    # =================================================================
    # FIX #2a — LOCATE FILE (immediate path lookup)
    # =================================================================
    def locate_file(self, file_name: str) -> str:
        """Immediately returns the exact owning service and relative path of any file."""
        target = self._resolve(file_name)
        if not target:
            return f"File '{file_name}' not found anywhere in repository index."
        try:
            return f"Found '{file_name}' at: {target.relative_to(self.root)}"
        except Exception:
            return f"Found '{file_name}' at: {target}"

    # =================================================================
    # STATE 3 — MUTATE: TRANSACTIONAL PATCHING
    # =================================================================
    def apply_code_patch(
        self,
        file_name: str,
        target_snippet: str,
        replacement_snippet: str,
        reason: str = "manual patch"
    ) -> str:
        self.frame.awareness_state = "MUTATE"
        target = self._resolve(file_name)
        if not target:
            return f"Error: File '{file_name}' not found."

        raw_content = target.read_text(encoding="utf-8")

        # Hard guard: patching an empty file is meaningless
        if not raw_content.strip():
            self.frame.last_action = f"PATCH BLOCKED on {file_name} (file is empty)"
            return (f"Error: File '{file_name}' is EMPTY. `apply_code_patch` requires an existing "
                    f"target snippet. Use `write_file` to populate the file instead.")

        has_crlf = "\r\n" in raw_content

        norm_content = raw_content.replace("\r\n", "\n")
        norm_target = target_snippet.replace("\r\n", "\n").strip("\r\n")
        norm_replacement = replacement_snippet.replace("\r\n", "\n").strip("\r\n")

        # Snapshot raw original code
        if file_name not in self.file_snapshots:
            self.file_snapshots[file_name] = []
        snapshot_hash = hashlib.sha1(raw_content.encode("utf-8")).hexdigest()
        self.file_snapshots[file_name].append({
            "content": raw_content,
            "hash": snapshot_hash,
            "reason": reason,
        })

        if norm_target not in norm_content:
            self.file_snapshots[file_name].pop()
            self.frame.last_action = f"PATCH FAILED on {file_name} (target not found)"
            return (f"Error: Target snippet not found in {file_name}. "
                    f"No snapshot retained. Inspect with surgical lines first.")

        updated_norm = norm_content.replace(norm_target, norm_replacement, 1)
        final_content = updated_norm.replace("\n", "\r\n") if has_crlf else updated_norm
        target.write_text(final_content, encoding="utf-8")

        self.frame.revision_ledger.append(
            RevisionEntry(file_name=file_name, snapshot_hash=snapshot_hash, reason=reason)
        )
        self.frame.last_action = f"PATCHED {file_name} @ {snapshot_hash[:8]} ({reason})"
        return (f"✅ Patched {file_name}. Snapshot {snapshot_hash[:8]} saved. "
                f"Next: verify dependent files (State 4: PROPAGATE).")

    # =================================================================
    # FIX #2b — WRITE (full-file write for empty/new files)
    # =================================================================
    def write_file(
        self,
        file_name: str,
        content: str,
        reason: str = "initial file creation"
    ) -> str:
        """Write complete content to a file — for empty files or brand-new files.

        Used when `apply_code_patch` cannot work because there is no
        target snippet to anchor to (e.g. the file is 0 bytes).
        """
        self.frame.awareness_state = "MUTATE"

        # Resolve to an existing path OR place alongside repo root
        target = self._resolve(file_name)
        if not target:
            target = self.root / file_name

        # Snapshot existing content (empty string if the file doesn't yet exist)
        existing_raw = target.read_text(encoding="utf-8") if target.exists() else ""
        snapshot_hash = hashlib.sha1(existing_raw.encode("utf-8")).hexdigest()

        if file_name not in self.file_snapshots:
            self.file_snapshots[file_name] = []
        self.file_snapshots[file_name].append({
            "content": existing_raw,
            "hash": snapshot_hash,
            "reason": reason,
        })

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        # Rebuild index so newly created files become resolvable
        self._build_index()

        self.frame.revision_ledger.append(
            RevisionEntry(file_name=file_name, snapshot_hash=snapshot_hash, reason=reason)
        )
        line_count = len(content.splitlines())
        self.frame.last_action = f"WROTE {file_name} ({line_count} lines, {reason})"
        return f"✅ Successfully wrote {line_count} lines to {file_name}."

    # =================================================================
    # STATE 5 — BACKTRACK: PHYSICAL ROLLBACK
    # =================================================================
    def rollback_file(self, file_name: str) -> str:
        self.frame.awareness_state = "BACKTRACK"
        stack = self.file_snapshots.get(file_name)
        if not stack:
            self.frame.last_action = f"ROLLBACK FAILED on {file_name} (no snapshot)"
            return f"Error: No previous snapshots found for {file_name}."

        previous = stack.pop()
        target = self._resolve(file_name)
        if not target:
            self.frame.last_action = f"ROLLBACK FAILED on {file_name} (file missing)"
            return f"Error: File '{file_name}' not found during rollback."

        target.write_text(previous["content"], encoding="utf-8")

        for sg in self.frame.sub_goal_stack:
            if sg.file_name == file_name and sg.status == "ACTIVE":
                sg.status = "INVALIDATED"

        self.frame.last_action = f"ROLLED BACK {file_name} → {previous['hash'][:8]}"
        return (f"↩️ Rollback successful: {file_name} restored to "
                f"snapshot {previous['hash'][:8]} (reason was: {previous['reason']}).")

    # =================================================================
    # SUB-GOAL STACK CONTROL PRIMITIVES
    # =================================================================
    def push_sub_goal(self, description: str, file_name: Optional[str] = None) -> str:
        for sg in self.frame.sub_goal_stack:
            if sg.status == "ACTIVE":
                sg.status = "PENDING"
        new_goal = SubGoal(description=description, status="ACTIVE", file_name=file_name)
        self.frame.sub_goal_stack.append(new_goal)
        self.frame.active_sub_goal = description
        self.frame.last_action = f"PUSHED sub-goal: {description}"
        return f"📌 Pushed sub-goal: {description} (file: {file_name or 'n/a'})"

    def declare_goal_complete(self) -> str:
        if not self.frame.sub_goal_stack:
            return "No active sub-goals to complete."
        for sg in reversed(self.frame.sub_goal_stack):
            if sg.status == "ACTIVE":
                sg.status = "DONE"
                break
        for sg in reversed(self.frame.sub_goal_stack):
            if sg.status == "PENDING":
                sg.status = "ACTIVE"
                self.frame.active_sub_goal = sg.description
                self.frame.last_action = f"GOAL DONE. Now ACTIVE: {sg.description}"
                return f"✅ Sub-goal complete. Now ACTIVE: {sg.description}"
        self.frame.active_sub_goal = None
        self.frame.last_action = "GOAL DONE. Stack empty."
        return "✅ Sub-goal complete. No further pending sub-goals."

    def update_hypothesis(self, hypothesis: str) -> str:
        self.frame.working_hypothesis = hypothesis
        self.frame.last_action = "HYPOTHESIS updated"
        return f"🧠 Hypothesis updated: {hypothesis}"

    # =================================================================
    # LLM CALL WRAPPER (With Key Rotation & Telemetry)
    # =================================================================
    def _call_model(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]], max_tokens: int = 800):
        def call(client):
            kwargs = dict(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens
            )
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            res = client.chat.completions.create(**kwargs)
            if res.usage:
                self.total_prompt_tokens += res.usage.prompt_tokens
                self.total_completion_tokens += res.usage.completion_tokens
            return res

        return self.rotator.execute_with_failover(call)

    # =================================================================
    # RECURSIVE FRAME DISPATCHER (Stack Push / Pop)
    # =================================================================
    def spawn_frame(self, request: FrameSpawnRequest) -> FrameReturnPacket:
        # Calculate current recursion depth from worker_id (e.g. "W1.1" -> 2)
        current_depth = len(request.child_id.split("."))

        print(f"\n[Stack Push -> {request.child_id}] (Depth: {current_depth}) Scope: {request.target_scope}")
        print(f"  Goal: {request.sub_goal}")

        scope_content = self.resolve_scope_content(request.target_scope)

        # ===== CRITICAL SEPARATION OF CONCERNS =====
        # Root worker (W1) gets ONLY orchestration + light inspection tools
        if current_depth == 1:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "locate_file",
                        "description": "Find the path of a file in the repo.",
                        "parameters": {
                            "type": "object",
                            "properties": {"file_name": {"type": "string"}},
                            "required": ["file_name"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_surgical_lines",
                        "description": "Read max 30 lines from a file to check if empty or understand structure.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_name": {"type": "string"},
                                "start_line": {"type": "integer"},
                                "end_line": {"type": "integer"}
                            },
                            "required": ["file_name", "start_line", "end_line"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "list_service_files",
                        "description": "List all Java files in a service.",
                        "parameters": {
                            "type": "object",
                            "properties": {"service_name": {"type": "string"}},
                            "required": ["service_name"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "update_hypothesis",
                        "description": "Update working hypothesis.",
                        "parameters": {
                            "type": "object",
                            "properties": {"hypothesis": {"type": "string"}},
                            "required": ["hypothesis"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_working_frame",
                        "description": "Get current working frame state.",
                        "parameters": {"type": "object", "properties": {}}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "spawn_child_worker",
                        "description": "Spawn a child worker to investigate a specific task. Use this for all detailed code inspection or file writes.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target_scope": {"type": "string"},
                                "sub_goal": {"type": "string"},
                                "expected_deliverable": {"type": "string"}
                            },
                            "required": ["target_scope", "sub_goal", "expected_deliverable"]
                        }
                    }
                }
            ]
        else:
            # Child workers (depth >= 2) get the mutation tools
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "locate_file",
                        "description": "Find the path of a file in the repo.",
                        "parameters": {
                            "type": "object",
                            "properties": {"file_name": {"type": "string"}},
                            "required": ["file_name"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_surgical_lines",
                        "description": "Read specific lines from a file.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_name": {"type": "string"},
                                "start_line": {"type": "integer"},
                                "end_line": {"type": "integer"}
                            },
                            "required": ["file_name", "start_line", "end_line"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "list_service_files",
                        "description": "List all Java files in a service.",
                        "parameters": {
                            "type": "object",
                            "properties": {"service_name": {"type": "string"}},
                            "required": ["service_name"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write complete content to a file.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_name": {"type": "string"},
                                "content": {"type": "string"},
                                "reason": {"type": "string"}
                            },
                            "required": ["file_name", "content"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "apply_code_patch",
                        "description": "Patch a specific snippet in a file.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_name": {"type": "string"},
                                "target_snippet": {"type": "string"},
                                "replacement_snippet": {"type": "string"},
                                "reason": {"type": "string"}
                            },
                            "required": ["file_name", "target_snippet", "replacement_snippet"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "rollback_file",
                        "description": "Rollback a file to previous snapshot.",
                        "parameters": {
                            "type": "object",
                            "properties": {"file_name": {"type": "string"}},
                            "required": ["file_name"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "update_hypothesis",
                        "description": "Update working hypothesis.",
                        "parameters": {
                            "type": "object",
                            "properties": {"hypothesis": {"type": "string"}},
                            "required": ["hypothesis"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_working_frame",
                        "description": "Get current working frame state.",
                        "parameters": {"type": "object", "properties": {}}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "spawn_child_worker",
                        "description": "Spawn another child to investigate a sub-component.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target_scope": {"type": "string"},
                                "sub_goal": {"type": "string"},
                                "expected_deliverable": {"type": "string"}
                            },
                            "required": ["target_scope", "sub_goal", "expected_deliverable"]
                        }
                    }
                }
            ]

        system_prompt = f"""You are Cognitive Worker Frame [{request.child_id}].
GOAL: {request.sub_goal}
EXPECTED DELIVERABLE: {request.expected_deliverable}

WHO YOU ARE (by depth):
CONSTRAINT FOR W1 (ORCHESTRATOR):
When you spawn a child, follow this EXACTLY:
1. Pick ONE specific task (not "read and analyze", pick "extract method list" OR "count lines" OR "verify caller")
2. Name the EXACT file or scope
3. End sub_goal with: "Do NOT read other files."
4. Make expected_deliverable ONE sentence, max. Example: "Report one number: how many lines?"
5. Wait for child to report back
6. Based on report, decide next spawn

EXAMPLES:
GOOD: sub_goal: "Count lines in ModelService.java. Report just the number."
BAD:  sub_goal: "Analyze ModelService.java and related files"

GOOD: sub_goal: "List all public methods in ModelManager. Format: name(params) return_type."
BAD:  sub_goal: "Understand what ModelService should contain"
- Depth 1 (W1): ORCHESTRATOR. You do NOT write code. You inspect topology and delegate all detailed work.
- Depth 2+ (W1.1, W1.2, ...): SPECIALIST. You execute focused tasks. You CAN write code when spawned for that purpose.

YOUR JOB (W1 only):
1. Understand the request
2. Locate files mentioned
3. Decide if the target file is empty → spawn a child to populate it
4. Spawn separate children to verify each caller/implementor
5. Collect their reports
6. Synthesize final report to parent (or dev if you're W1)

DO NOT (W1 only):
- Do NOT read 50 lines of code yourself
- Do NOT write files yourself
- Do NOT patch code yourself
SPAWN A CHILD for these tasks.

YOUR JOB (W1.X child):
1. Read assigned file thoroughly
2. Extract the specific facts asked for (method signatures, imports, callers, etc.)
3. Report findings clearly back to parent
4. If told to write/patch, do it and report success

CRITICAL: Each worker spawned is responsible for exactly ONE task. Never spawn a single child to do 3 different inspections."""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"SCOPE:\n{scope_content}\n\nBegin work."
            }
        ]

        child_reports = []
        mutations = []

        MAX_TURNS = 15
        for turn in range(MAX_TURNS):
            try:
                res = self._call_model(messages, tools, max_tokens=2000)
            except Exception as e:
                err_str = str(e)
                if "tool_use_failed" in err_str and "<tool_call>" in err_str:
                    match_fn = re.search(r"<function=([a-zA-Z0-9_]+)>", err_str)
                    if match_fn:
                        fn_name = match_fn.group(1)
                        raw_params = re.findall(
                            r"<parameter=([a-zA-Z0-9_]+)>\s*([\s\S]*?)\s*</parameter>", err_str
                        )
                        recovered_args = {}
                        for p_name, p_val in raw_params:
                            val = p_val.strip()
                            recovered_args[p_name] = int(val) if val.isdigit() else val

                        print(f"  [{request.child_id}] [Recovered XML Call] -> {fn_name}({recovered_args})")
                        output = self._dispatch_tool(fn_name, recovered_args, request.child_id, child_reports, mutations)
                        messages.append({
                            "role": "user",
                            "content": f"Tool returned:\n{output}\nContinue."
                        })
                        continue

                print(f"  [{request.child_id}] Warning: {e}. Forcing completion...")
                break

            msg = res.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                print(f"[Stack Pop <- {request.child_id}] Done\n")
                return FrameReturnPacket(
                    child_id=request.child_id,
                    parent_id=request.parent_id,
                    status="SUCCESS",
                    scalar_deduction=msg.content or "Completed.",
                    mutations_applied=mutations,
                    child_reports=child_reports
                )

            for call in msg.tool_calls:
                fn_name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                except Exception:
                    args = {}

                # Deduplication guard
                call_hash = f"{fn_name}:{json.dumps(args, sort_keys=True)}"
                if call_hash in self.inspected_cache:
                    print(f"  [{request.child_id}] [DUPLICATE BLOCKED] -> {fn_name}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": f"Already called {fn_name} with these params. Move forward."
                    })
                    continue

                self.inspected_cache.add(call_hash)
                print(f"  [{request.child_id}] -> {fn_name}({args})")
                tool_output = self._dispatch_tool(fn_name, args, request.child_id, child_reports, mutations)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(tool_output)[:3000]
                })

        return FrameReturnPacket(
            child_id=request.child_id,
            parent_id=request.parent_id,
            status="FAILED",
            scalar_deduction="Turn limit reached.",
            mutations_applied=mutations,
            child_reports=child_reports
        )

    def _dispatch_tool(self, fn_name: str, args: dict, current_worker_id: str, child_reports: list, mutations: list) -> str:
        if fn_name == "locate_file":
            return self.locate_file(file_name=args.get("file_name", ""))
        elif fn_name == "read_surgical_lines":
            return self.read_surgical_lines(
                file_name=args.get("file_name", ""),
                start_line=args.get("start_line", 1),
                end_line=args.get("end_line", 30)
            )
        elif fn_name == "list_service_files":
            return json.dumps(self.list_service_files(args.get("service_name", "")))
        elif fn_name == "spawn_child_worker":
            child_req = FrameSpawnRequest(
                child_id=f"{current_worker_id}.{len(child_reports) + 1}",
                parent_id=current_worker_id,
                target_scope=args.get("target_scope", ""),
                sub_goal=args.get("sub_goal", ""),
                expected_deliverable=args.get("expected_deliverable", "")
            )
            child_res = self.spawn_frame(child_req)
            child_reports.append(child_res.model_dump())
            mutations.extend(child_res.mutations_applied)
            return f"Child [{child_res.child_id}] finished. Result: {child_res.scalar_deduction}"
        elif fn_name == "write_file":
            res = self.write_file(
                file_name=args.get("file_name", ""),
                content=args.get("content", ""),
                reason=args.get("reason", "")
            )
            mutations.append(args.get("file_name", ""))
            return res
        elif fn_name == "apply_code_patch":
            res = self.apply_code_patch(
                file_name=args.get("file_name", ""),
                target_snippet=args.get("target_snippet", ""),
                replacement_snippet=args.get("replacement_snippet", ""),
                reason=args.get("reason", "")
            )
            mutations.append(args.get("file_name", ""))
            return res
        elif fn_name == "rollback_file":
            return self.rollback_file(file_name=args.get("file_name", ""))
        elif fn_name == "update_hypothesis":
            return self.update_hypothesis(hypothesis=args.get("hypothesis", ""))
        elif fn_name == "get_working_frame":
            return self.frame.render()
        return f"Error: Tool '{fn_name}' not recognized."

    def investigate(self, user_question: str) -> str:
        self.frame = WorkingFrame()
        self.frame.global_intent = user_question
        self.frame.awareness_state = "ORIENT"
        self.frame.last_action = "(Investigation booted)"

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.inspected_cache = set()

        services = self.list_services()
        root_scope = (
            f"Available Services: {', '.join(services)}\n"
            f"Project Root: {self.root.name}"
        )

        root_req = FrameSpawnRequest(
            child_id="W1",
            parent_id="ROOT",
            target_scope=root_scope,
            sub_goal=user_question,
            expected_deliverable="Clear report of findings and actions taken."
        )

        root_res = self.spawn_frame(root_req)

        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        print(f"\n[Telemetry] Prompt: {self.total_prompt_tokens} | Completion: {self.total_completion_tokens} | Total: {total_tokens}")
        print("\n[Final Working Frame]\n" + self.frame.render())

        return root_res.scalar_deduction