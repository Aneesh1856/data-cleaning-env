"""
OpenEnv Inference Script — LLM Agent for Data Cleaning Environment
Uses API_BASE_URL and API_KEY environment variables provided by the checker.
"""
import os
import sys
import json
import requests
from openai import OpenAI

def main():
    env_base_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")

    # Use the checker-provided LLM proxy
    api_base_url = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    api_key = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", ""))

    client = OpenAI(
        base_url=api_base_url,
        api_key=api_key,
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
- Always respond with valid JSON like: {"operation": "fix_dates", "column": "join_date"}
"""

    tasks = ["easy", "medium", "hard"]

    for task_id in tasks:
        print(f"[START] task={task_id}", flush=True)

        try:
            resp = requests.post(f"{env_base_url}/reset", json={"task_id": task_id})
            resp.raise_for_status()
            obs = resp.json()
        except Exception as e:
            print(f"[END] task={task_id} score=0.001 steps=0", flush=True)
            continue

        max_steps = {"easy": 20, "medium": 35, "hard": 60}[task_id]
        step_count = 0
        final_score = 0.0
        done = False
        action_history = []

        while step_count < max_steps and not done:
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
                        model="gpt-4o-mini",
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
                    pass

            if not parsed_action:
                parsed_action = {"operation": "done"}

            col = parsed_action.get("column", "")
            action_history.append(f"{parsed_action['operation']}({col})")

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

        # Clamp score to strict (0, 1) open interval
        final_score = max(0.001, min(0.999, final_score))
        print(f"[END] task={task_id} score={final_score} steps={step_count}", flush=True)

if __name__ == "__main__":
    main()
