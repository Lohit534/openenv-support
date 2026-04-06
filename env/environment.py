from .models import Observation, Action, Reward, EnvState
from .tasks import TASKS
from .grader import TaskGrader
import random
from fastapi import FastAPI, HTTPException

class SupportEnvironment:
    def __init__(self):
        self.tasks = TASKS
        self.current_task = None
        self.state_data = None
        self.grader = None

    def reset(self, task_idx=None) -> Observation:
        if task_idx is not None and task_idx < len(self.tasks):
            self.current_task = self.tasks[task_idx]
        else:
            self.current_task = random.choice(self.tasks)
            
        self.grader = TaskGrader(self.current_task["expected_steps"])
        
        self.state_data = EnvState(
            current_task_id=self.current_task["task_id"],
            step_count=0,
            terminated=False,
            ticket_history=[],
            total_reward=0.0
        )
        
        return self._get_obs()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        if self.state_data.terminated:
            return self._get_obs(), 0.0, True, {"error": "Environment is already terminated."}
            
        self.state_data.step_count += 1
        action_str = f"Agent Action: {action.action_type.value} args={action.args}"
        self.state_data.ticket_history.append(action_str)
        
        reward_val, reason, is_done = self.grader.grade_step(action.action_type.value, action.args)
        
        # Inject simulator dynamic response if we correctly asked for info
        if "ask_info" == action.action_type.value and reward_val > 0:
             self.state_data.ticket_history.append("User: No, restarting the router didn't help. The red light is blinking.")
             
        self.state_data.terminated = is_done or self.state_data.step_count >= 10
        self.state_data.total_reward = self.grader.get_final_score()
        
        obs = self._get_obs()
        return obs, reward_val, self.state_data.terminated, {"reason": reason, "total_score": self.state_data.total_reward}

    def _get_obs(self) -> Observation:
        return Observation(
            ticket_id=self.current_task["task_id"],
            user_inquiry=self.current_task["inquiry"],
            history=self.state_data.ticket_history,
            is_terminated=self.state_data.terminated,
            available_actions=["classify", "reply", "escalate", "refund", "ask_info"]
        )
        
    def state(self) -> EnvState:
        return self.state_data

# ==========================================
# OpenEnv Validator FastAPI Wrapper
# ==========================================

app = FastAPI(title="Support Ticket Environment", description="OpenEnv HTTP Wrapper")
global_env = SupportEnvironment()

@app.post("/reset", response_model=Observation)
def reset_env(task_idx: int = None):
    return global_env.reset(task_idx)

@app.post("/step")
def step_env(action: Action):
    if global_env.state_data is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    
    obs, reward, done, info = global_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }

@app.get("/state", response_model=EnvState)
def get_state():
    if global_env.state_data is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return global_env.state()
