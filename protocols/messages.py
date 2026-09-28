# protocols/messages.py
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FrameSpawnRequest(BaseModel):
    """Dispatched by a parent frame to spawn an isolated child."""
    child_id: str
    parent_id: str
    target_scope: str          # e.g., "PaymentService.java lines 82-124" or "Payment.java"
    sub_goal: str              # What the child must investigate or do
    expected_deliverable: str  # The scalar question/deliverable to return


class FrameReturnPacket(BaseModel):
    """Returned by a child frame when it terminates."""
    child_id: str
    parent_id: str
    status: str                # SUCCESS | FAILED
    scalar_deduction: str      # The concrete answer or finding
    mutations_applied: List[str] = Field(default_factory=list) # e.g. ["Payment.java@hash"]
    child_reports: List[Dict[str, Any]] = Field(default_factory=list) # Trace of deeper sub-children