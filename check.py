import sys
import traceback
import os

def run_checks():
    errors = []
    
    # 1. & 2. Init and reset
    try:
        from env.environment import DataCleaningEnv
        from models import Action
        env = DataCleaningEnv('easy')
        obs = env.reset()
        if not hasattr(obs, 'dirty_csv') or not hasattr(obs, 'schema'):
            errors.append("Observation missing fields")
    except Exception as e:
        errors.append("1/2 Failed: " + str(e))
        traceback.print_exc()

    # 3. Step returns 4-tuple
    try:
        res = env.step(Action(operation='drop_duplicates'))
        if len(res) != 4:
            errors.append(f"Step returned {len(res)} tuple instead of 4")
        if not hasattr(res[0], 'dirty_csv'):
            errors.append("Step tuple 0 is not Observation")
        if not hasattr(res[1], 'value'):
            errors.append("Step tuple 1 is not Reward")
    except Exception as e:
        errors.append("3 Failed: " + str(e))
        traceback.print_exc()

    # 4. State returns plain dict
    try:
        state = env.state()
        if not isinstance(state, dict):
            errors.append("State is not a dict")
    except Exception as e:
        errors.append("4 Failed: " + str(e))

    # 5. Graders run without error
    try:
        from env.graders.graders import run_grader
        score = run_grader("easy", "env/data/task_easy_dirty.csv", "env/data/task_easy_gt.csv")
        if not isinstance(score, float) or not (0.0 <= score <= 1.0):
            errors.append("Grader score out of bounds or not float")
    except Exception as e:
        errors.append("5 Failed (Grader): " + str(e))

    if errors:
        print("ERRORS FOUND:")
        for er in errors:
            print(er)
    else:
        print("ALL PYTHON CHECKS PASSED")

if __name__ == "__main__":
    run_checks()
