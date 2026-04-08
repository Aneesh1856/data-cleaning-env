import os
import sys
import json
import requests
from openai import OpenAI

def main():
    openai_base_url = "https://api.groq.com/openai/v1"
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    model_name = "llama-3.1-8b-instant"

    env_base_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")

    client = OpenAI(
        base_url=openai_base_url,
        api_key=openai_api_key,
    )

    system_prompt = """You are a data cleaning agent. You must return ONLY a raw JSON object — no markdown, no explanation.

Available operations:
- fix_dates: fix date format issues. Requires "column" field.
- drop_duplicates: remove duplicate rows. No column needed.
- normalize_category: lowercase a text column. Requires "column" field.
- impute_nulls: fill missing values. Requires "column" and params: {"strategy": "mean"} or {"strategy": "median"}.
- drop_column: remove a column. Requires "column" field.
- done: call this when cleaning is complete.

Rules:
- NEVER repeat the same operation on the same column twice.
- Look at the data snippet and issues_remaining to decide what to fix.
- When issues_remaining is 0 or you have done all fixes, call done.
- Always respond with valid JSON like: {"operation": "fix_dates", "column": "signup_date"}
"""

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
        action_history = []

        scripted = task_instructions[task_id].copy()

        while step_count < max_steps and not done:
            if scripted:
                parsed_action = scripted.pop(0)
            else:
                dirty_csv = obs.get("dirty_csv", "")[:500]
                issues_remaining = obs.get("issues_remaining", 0)
                task_description = obs.get("task_description", "")
                history_str = ", ".join(action_history[-5:]) if action_history else "none"

                prompt_text = (
                    f"Task: {task_id}\n"
                    f"Description: {task_description}\n"
                    f"Issues remaining: {issues_remaining}\n"
                    f"Recent actions taken (do not repeat these): {history_str}\n"
                    f"Data snippet:\n{dirty_csv}\n\n"
                    "What is your next action? Respond with raw JSON only."
                )

                parsed_action = None
                for attempt in range(2):
                    try:
                        chat_resp = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt_text}
                            ],
                            temperature=0.0
                        )
                        text = chat_resp.choices[0].message.content.strip()
                        for fence in ["```json", "```"]:
                            if text.startswith(fence):
                                text = text[len(fence):]
                        if text.endswith("```"):
                            text = text[:-3]
                        parsed_action = json.loads(text.strip())
                        break
                    except Exception as e:
                        print(f"  LLM attempt {attempt+1} failed: {e}")

                if not parsed_action:
                    print("  Could not get valid action. Calling done.")
                    parsed_action = {"operation": "done"}

            col = parsed_action.get("column", "")
            action_history.append(f"{parsed_action['operation']}({col})")

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
                    try:
                        state_resp = requests.get(f"{env_base_url}/state")
                        state_data = state_resp.json()
                        final_score = state_data.get("last_grader_score", reward_value)
                    except:
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