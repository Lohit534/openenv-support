"""
Grader for OpenEnv Customer Support Ticket Resolution tasks.
All final scores are strictly in the OPEN interval (0.0, 1.0) — never 0.0 or 1.0.
This is required by Phase 2 deep validation.
"""

SCORE_MIN = 0.01   # Minimum score — strictly greater than 0.0
SCORE_MAX = 0.95   # Maximum score — strictly less than 1.0

def clamp(value: float) -> float:
    """Strictly clamp to open interval (0, 1): never 0.0 or 1.0."""
    return max(SCORE_MIN, min(SCORE_MAX, value))


def check_partial_match(expected_args, actual_args):
    """Check if expected args string values are substrings of actual string values."""
    for k, v in expected_args.items():
        if k not in actual_args:
            return False
        if isinstance(v, str) and isinstance(actual_args[k], str):
            if v.lower() not in actual_args[k].lower():
                return False
        else:
            if v != actual_args[k]:
                return False
    return True


class TaskGrader:
    def __init__(self, expected_steps):
        self.expected_steps = expected_steps
        self.current_step = 0
        self.max_steps = len(expected_steps)
        # Start at SCORE_MIN so score is never exactly 0.0
        self.score = SCORE_MIN

    def grade_step(self, action_type: str, action_args: dict):
        if self.current_step >= self.max_steps:
            # Penalize for redundant actions after task completion
            self.score = clamp(self.score - 0.15)
            return -0.15, "Too many steps: task already achieved all goals.", True

        expected = self.expected_steps[self.current_step]

        if expected["action_type"] != action_type:
            # Wrong action type — modest penalty, don't advance step
            self.score = clamp(self.score - 0.08)
            return -0.08, f"Incorrect action type. Expected progress on step {self.current_step + 1}.", False

        if not check_partial_match(expected["args"], action_args):
            # Right action type but wrong arguments — partial penalty
            self.score = clamp(self.score - 0.04)
            return -0.04, f"Incorrect arguments for '{action_type}'. Missing required keywords.", False

        # Correct step — scale reward so max reachable is SCORE_MAX
        self.current_step += 1
        step_reward = (SCORE_MAX - SCORE_MIN) / self.max_steps
        self.score = clamp(self.score + step_reward)

        is_done = (self.current_step == self.max_steps)
        return round(step_reward, 4), "Correct step completed successfully.", is_done

    def get_final_score(self) -> float:
        """Return final episode score, guaranteed strictly in (0.0, 1.0)."""
        return clamp(self.score)
