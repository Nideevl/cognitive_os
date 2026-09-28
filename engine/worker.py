# engine/worker.py
import json
from typing import Dict, Any, Callable, List, Optional
from protocols.messages import FrameSpawnRequest, FrameReturnPacket


class CognitiveWorker:
    """An isolated cognitive execution frame.
    Operates as root or as a child frame.
    Strictly follows: Digest Scope First -> Formulate Hypothesis -> Spawn Child if needed -> Synthesize.
    """

    def __init__(
        self,
        worker_id: str,
        goal: str,
        scope_content: str,
        expected_deliverable: str,
        call_model_fn: Callable[[List[Dict[str, str]], List[Dict[str, Any]], int], Any],
        spawn_child_fn: Callable[[FrameSpawnRequest], FrameReturnPacket],
        mutate_code_fn: Optional[Callable[[str, str, str, str], str]] = None,
        max_turns: int = 8
    ):
        self.worker_id = worker_id
        self.goal = goal
        self.scope_content = scope_content
        self.expected_deliverable = expected_deliverable
        self.call_model = call_model_fn
        self.spawn_child = spawn_child_fn
        self.mutate_code = mutate_code_fn
        self.max_turns = max_turns

        self.child_reports: List[Dict[str, Any]] = []
        self.mutations_applied: List[str] = []

    def execute(self) -> FrameReturnPacket:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "spawn_child_worker",
                    "description": "Spawn an isolated sub-worker to investigate a dependency, method slice, or related file. You will PAUSE until it finishes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_scope": {
                                "type": "string",
                                "description": "Filename or lines (e.g. 'PaymentService.java lines 82-124' or 'FraudClient.java')"
                            },
                            "sub_goal": {
                                "type": "string",
                                "description": "Exact question or task for the child"
                            },
                            "expected_deliverable": {
                                "type": "string",
                                "description": "Expected scalar deduction or factual finding"
                            }
                        },
                        "required": ["target_scope", "sub_goal", "expected_deliverable"]
                    }
                }
            }
        ]

        if self.mutate_code:
            tools.append({
                "type": "function",
                "function": {
                    "name": "apply_code_patch",
                    "description": "Surgically patch a target snippet in a file. Must have inspected lines first.",
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
            })

        system_prompt = f"""You are Cognitive Worker Frame [{self.worker_id}].
GOAL: {self.goal}
EXPECTED DELIVERABLE: {self.expected_deliverable}

OPERATING DISCIPLINE:
1. READ ENTIRE SCOPE FIRST: The code slice or skeleton is provided in your first prompt. Ingest every line before concluding.
2. DO NOT GUESS: If you encounter an injected bean, external contract, or folded method {{...}} needed to satisfy your goal, call `spawn_child_worker`. You will pause until it returns.
3. INVARIANT: A parent frame NEVER finishes until all children have reported back and their findings are integrated.
4. When finished, emit your clear factual synthesis satisfying the EXPECTED DELIVERABLE.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"ASSIGNED SCOPE:\n```\n{self.scope_content}\n```\n\nRead the full scope above and begin your deduction."
            }
        ]

        for turn in range(self.max_turns):
            res = self.call_model(messages, tools, 800)
            msg = res.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                # Factual synthesis reached
                return FrameReturnPacket(
                    child_id=self.worker_id,
                    parent_id="",
                    status="SUCCESS",
                    scalar_deduction=msg.content or "Completed without text.",
                    mutations_applied=self.mutations_applied,
                    child_reports=self.child_reports
                )

            for call in msg.tool_calls:
                fn_name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                except Exception:
                    args = {}

                print(f"  [{self.worker_id}] -> {fn_name}({args})")

                if fn_name == "spawn_child_worker":
                    req = FrameSpawnRequest(
                        child_id=f"{self.worker_id}.{len(self.child_reports) + 1}",
                        parent_id=self.worker_id,
                        target_scope=args.get("target_scope", ""),
                        sub_goal=args.get("sub_goal", ""),
                        expected_deliverable=args.get("expected_deliverable", "")
                    )
                    # Pauses this frame; runs child to completion
                    child_res = self.spawn_child(req)
                    self.child_reports.append(child_res.model_dump())
                    self.mutations_applied.extend(child_res.mutations_applied)
                    tool_result = f"Child Worker [{child_res.child_id}] Finished.\nResult: {child_res.scalar_deduction}"

                elif fn_name == "apply_code_patch" and self.mutate_code:
                    patch_res = self.mutate_code(
                        args.get("file_name", ""),
                        args.get("target_snippet", ""),
                        args.get("replacement_snippet", ""),
                        args.get("reason", "mutation")
                    )
                    self.mutations_applied.append(f"{args.get('file_name')} (turn {turn})")
                    tool_result = patch_res
                else:
                    tool_result = f"Error: Tool '{fn_name}' not available in this frame."

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(tool_result)[:3000]
                })

        return FrameReturnPacket(
            child_id=self.worker_id,
            parent_id="",
            status="FAILED",
            scalar_deduction="Turn limit reached before final answer was reached.",
            mutations_applied=self.mutations_applied,
            child_reports=self.child_reports
        )