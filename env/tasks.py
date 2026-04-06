TASKS = [
    {
        "task_id": "task_easy_1",
        "difficulty": "easy",
        "inquiry": "How do I reset my password? I forgot it.",
        "expected_steps": [
            {"action_type": "classify", "args": {"category": "general"}}
        ]
    },
    {
        "task_id": "task_medium_1",
        "difficulty": "medium",
        "inquiry": "I noticed an extra charge of $5 on my bill this month. Please explain.",
        "expected_steps": [
            {"action_type": "classify", "args": {"category": "billing"}},
            {"action_type": "reply", "args": {"message": "charge"}}
        ]
    },
    {
        "task_id": "task_hard_1",
        "difficulty": "hard",
        "inquiry": "My router is blinking red and I have no internet.",
        "expected_steps": [
            {"action_type": "classify", "args": {"category": "technical"}},
            {"action_type": "ask_info", "args": {"message": "restarting"}},
            {"action_type": "escalate", "args": {"department": "tier2"}}
        ]
    }
]
