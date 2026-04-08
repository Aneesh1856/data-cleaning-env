"""
OpenEnv Inference Script — Baseline Agent for Data Cleaning Environment
This script connects to the running server and executes cleaning actions.
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

    results = []

    for task_id in tasks:
        print(f"\n========== Evaluating Task: {task_id} ==========")

        try:
            resp = requests.post(f"{env_base_url}/reset", json={"task_id": task_id})
            resp.raise_for_status()
            obs = resp.json()
        except Exception as e:
            print(f"Failed to reset environment for task {task_id}: {e}")
            results.append({"task": task_id, "steps": 0, "score": 0.0})
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
                print(f"  /step call failed: {e}")
                break

            obs = step_data.get("observation", {})
            reward_data = step_data.get("reward", {})
            done = step_data.get("done", False)
            reward_value = reward_data.get("value", 0.0)

            print(f"Step {step_count+1:<2} | Action: {parsed_action.get('operation'):<22} | Reward: {reward_value:.4f}")

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

            step_count += 1

        print(f"\n-> Final Grader Score for '{task_id}': {final_score:.2f}")
        results.append({"task": task_id, "steps": step_count, "score": final_score})

    print("\n" + "="*40)
    print(" SUMMARY TABLE")
    print("="*40)
    print(f"{'Task':<10} | {'Steps Taken':<15} | {'Final Score'}")
    print("-" * 40)

    all_passed = True
    for r in results:
        print(f"{r['task']:<10} | {r['steps']:<15} | {r['score']:.2f}")
        if r["score"] <= 0.0:
            all_passed = False

    print()
    if all_passed:
        print("PASS: All three tasks scored > 0.0. Environment functional.")
        sys.exit(0)
    else:
        print("FAIL: One or more tasks scored 0.0.")
        sys.exit(1)

if __name__ == "__main__":
    main()
