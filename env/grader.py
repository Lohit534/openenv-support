def check_partial_match(expected_args, actual_args):
    """Checks if expected args string values are substrings of actual string values, and exact match otherwise."""
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
        self.score = 0.0

    def grade_step(self, action_type: str, action_args: dict):
        if self.current_step >= self.max_steps:
            # Penalize for taking redundant actions when task should be complete
            self.score = max(0.0, self.score - 0.2)
            return -0.2, "Too many steps, task already achieved goals.", True
            
        expected = self.expected_steps[self.current_step]
        
        if expected["action_type"] != action_type:
            # Incorrect action type -> modest penalty, don't advance step
            self.score = max(0.0, self.score - 0.1)
            return -0.1, f"Incorrect action type. Expected progress on step {self.current_step+1}.", False
            
        if not check_partial_match(expected["args"], action_args):
            # Incorrect arguments -> partial penalty
            self.score = max(0.0, self.score - 0.05)
            return -0.05, f"Incorrect arguments for {action_type}. Missing required keywords.", False
            
        # Correct step!
        self.current_step += 1
        step_reward = 1.0 / self.max_steps
        self.score = min(1.0, self.score + step_reward)
        
        is_done = self.current_step == self.max_steps
        return step_reward, "Correct step completed successfully.", is_done

    def get_final_score(self):
        return self.score
