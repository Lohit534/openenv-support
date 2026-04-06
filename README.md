---
title: OpenEnv Support
emoji: 🎫
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# OpenEnv Support Ticket Resolution


A real-world OpenEnv environment simulating a Customer Support Ticket Resolution System. Built with Pydantic for the Meta + HuggingFace agent evaluation requirements.

## Real-world Use Case
Simulates handling user inquiries. The agent must systematically process the ticket, classify the category, request additional info, reply, and escalate to higher tiers, mirroring enterprise customer support operations.

## Architecture
- `openenv-support/env/models.py`: Rigid Pydantic typing for Observation, Action, Reward, EnvState.
- `openenv-support/env/environment.py`: Implements stateless transition logic with the correct signature `step(action)`, `reset()`, `state()`.
- `openenv-support/env/tasks.py`: Contains defined tasks evaluated from easy to hard.
- `openenv-support/env/grader.py`: Calculates float-based partial & stepwise evaluation scores bounded in [0.0, 1.0].
- `openenv-support/inference.py`: Baseline inference script using standard OpenAI Client compatible endpoints (designed for HuggingFace `api-inference`).

## Action/Observation Space
* **Observation**: Dict containing `ticket_id`, `user_inquiry`, `history` of states, `is_terminated`, and `available_actions`.
* **Action**: Dict mapped to `action_type` (Enum) and `args` (Dict) payloads.

## Tasks Explanation
1. **Easy Task**: Single-step classification of a general query.
2. **Medium Task**: Dual-step classifying then replying to a user's billing query.
3. **Hard Task**: Multi-step technical escalation involving conditional info requests based on simulated user dialog.

## Reward Design
Designed without sparse restrictions. The `grader.py` handles fractional returns evaluated by step: `-0.1` or `-0.05` for out of distribution or missing required fields, and `+n step` fractional scaling based on total correct actions.

## Setup Instructions

### Local Run
```bash
pip install -r requirements.txt
export HF_TOKEN="your_hf_token_here"
python inference.py
```
*(If `HF_TOKEN` is unset or API limits apply, a mock inference loop is triggered producing valid determinist baseline bounds)*

### Docker
Ensure no heavy inference runs locally by utilizing remote endpoint fetching.
```bash
docker build -t openenvs-test .
docker run -e HF_TOKEN="your_key" openenvs-test
```
