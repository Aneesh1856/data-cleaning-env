"""
OpenEnv Inference Script — Baseline Agent for Data Cleaning Environment
"""
import os
import sys
import json
import requests

def main():
    env_base_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")

    tasks = ["easy", "medium", "hard"]
    task_instructions = {
        "easy": [
            {"operation": "fix_dates", "column": "join_date"},
            {"operation": "done"}
        ],
        "medium": [
            {"operation": "fix_dates", "column": "join_date"},
            {"operation": "drop_duplicates"},
            {"operation": "normalize_category", "column": "category"},
            {"operation": "done"}
        ],
        "hard": [
            {"operation": "fix_dates", "column": "join_date"},
            {"operation": "drop_duplicates"},
            {"operation": "normalize_category", "column": "category"},
            {"operation": "impute_nulls", "column": "salary", "params": {"strategy": "mean"}},
            {"operation": "done"}
        ]
    }

    for task_id in tasks:
        # [START] block
        print(f"[START] task={task_id}", flush=True)

        try:
            resp = requests.post(f"{env_base_url}/reset", json={"task_id": task_id})
            resp.raise_for_status()
            obs = resp.json()
        except Exception as e:
            print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
            continue

        max_steps = {"easy": 20, "medium": 35, "hard": 60}[task_id]
        step_count = 0
        final_score = 0.0
        done = False

        scripted = task_instructions[task_id].copy()

        while step_count < max_steps and not done:
            if scripted:
                parsed_action = scripted.pop(0)
            else:
                parsed_action = {"operation": "done"}

            try:
                step_resp = requests.post(f"{env_base_url}/step", json=parsed_action)
                step_resp.raise_for_status()
                step_data = step_resp.json()
            except Exception as e:
                break

            obs = step_data.get("observation", {})
            reward_data = step_data.get("reward", {})
            done = step_data.get("done", False)
            reward_value = reward_data.get("value", 0.0)

            step_count += 1

            # [STEP] block
            print(f"[STEP] task={task_id} step={step_count} reward={reward_value}", flush=True)

            if done:
                breakdown = reward_data.get("breakdown", {})
                grader_score = step_data.get("info", {}).get("grader_score", None)
                if grader_score is not None:
                    final_score = grader_score
                elif "grader" in breakdown:
                    final_score = breakdown["grader"]
                elif "bonus" in breakdown:
                    final_score = min(breakdown["bonus"] / 0.3, 1.0)
                else:
                    final_score = max(reward_value, 0.0)

        # [END] block
        print(f"[END] task={task_id} score={final_score} steps={step_count}", flush=True)

if __name__ == "__main__":
    main()
