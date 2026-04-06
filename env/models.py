from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from enum import Enum

class ActionType(str, Enum):
    CLASSIFY = "classify"
    REPLY = "reply"
    ESCALATE = "escalate"
    REFUND = "refund"
    ASK_INFO = "ask_info"

class Action(BaseModel):
    action_type: ActionType = Field(..., description="The type of action to perform")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the action (e.g. {'category': 'tech'}, {'message': '...'})")

class Observation(BaseModel):
    ticket_id: str
    user_inquiry: str
    history: List[str]
    is_terminated: bool
    available_actions: List[str]

class Reward(BaseModel):
    value: float = Field(description="Reward value for the current step")
    reason: str = Field(description="Explanation of the reward")

class EnvState(BaseModel):
    current_task_id: str
    step_count: int
    terminated: bool
    ticket_history: List[str]
    total_reward: float
