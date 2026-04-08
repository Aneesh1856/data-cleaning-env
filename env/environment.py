import os
import io
import sys
import pandas as pd
import numpy as np

# Ensure root directory is in python path to allow imports
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import Observation, Action, Reward
from env.graders.graders import run_grader

class DataCleaningEnv:
    def __init__(self, task_id: str):
        self.task_id = task_id.lower()
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        
        self.dirty_path = os.path.join(self.data_dir, f"task_{self.task_id}_dirty.csv")
        self.gt_path = os.path.join(self.data_dir, f"task_{self.task_id}_gt.csv")
        
        self.original_df = pd.read_csv(self.dirty_path)
        self.current_df = self.original_df.copy()
        
        self.step_count = 0
        if self.task_id == "easy":
            self.max_steps = 20
        elif self.task_id == "medium":
            self.max_steps = 35
        else:
            self.max_steps = 60

    def _df_to_csv_string(self, df: pd.DataFrame) -> str:
        sio = io.StringIO()
        df.to_csv(sio, index=False)
        return sio.getvalue()
        
    def _get_issues_remaining(self, df: pd.DataFrame) -> int:
        null_count = df.isnull().sum().sum()
        dupe_count = df.duplicated().sum()
        
        bad_formats = 0
        for col in df.columns:
            if "date" in col.lower() and df[col].dtype == object:
                dates = df[col].dropna().astype(str)
                bad_formats += (~dates.str.match(r"^\d{4}-\d{2}-\d{2}$")).sum()
            elif "category" in col.lower() and df[col].dtype == object:
                cats = df[col].dropna().astype(str)
                bad_formats += (cats != cats.str.lower()).sum()
                
        return int(null_count + dupe_count + bad_formats)

    def reset(self) -> Observation:
        self.current_df = self.original_df.copy()
        self.step_count = 0
        
        schema = {col: str(self.current_df[col].dtype) for col in self.current_df.columns}
        desc = f"Clean the data for task {self.task_id}"
        
        return Observation(
            dirty_csv=self._df_to_csv_string(self.current_df),
            schema=schema,
            task_description=desc,
            step_count=self.step_count,
            issues_remaining=self._get_issues_remaining(self.current_df)
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        done = False
        info = {}
        reward_val = 0.0
        breakdown = {"step_reward": 0.0, "bonus": 0.0}
        
        if self.step_count >= self.max_steps:
            done = True
            
        old_issues = self._get_issues_remaining(self.current_df)
        
        try:
            if action.operation == "done" or done:
                done = True
                
                # Run grader
                temp_out = os.path.join(self.data_dir, f"temp_out_{self.task_id}.csv")
                self.current_df.to_csv(temp_out, index=False)
                
                final_score = run_grader(self.task_id, temp_out, self.gt_path)
                
                if os.path.exists(temp_out):
                    os.remove(temp_out)
                    
                bonus = final_score * 0.3
                reward_val += bonus
                breakdown["bonus"] = bonus
                
            else:
                df_before = self.current_df.copy()
                
                if action.operation == "fix_dates":
                    if action.column and action.column in self.current_df.columns:
                        fmt = action.params.get("format", None) if action.params else None
                        self.current_df[action.column] = pd.to_datetime(
                            self.current_df[action.column], format=fmt, errors='coerce'
                        ).dt.strftime('%Y-%m-%d')
                        
                elif action.operation == "drop_duplicates":
                    subset = action.params.get("subset", None) if action.params else None
                    if subset and isinstance(subset, str):
                        subset = [subset]
                    self.current_df = self.current_df.drop_duplicates(subset=subset).reset_index(drop=True)
                    
                elif action.operation == "normalize_category":
                    if action.column and action.column in self.current_df.columns:
                        self.current_df[action.column] = self.current_df[action.column].astype(str).str.lower().str.strip()
                        
                elif action.operation == "impute_nulls":
                    if action.column and action.column in self.current_df.columns:
                        strategy = action.params.get("strategy", "mean") if action.params else "mean"
                        col_data = pd.to_numeric(self.current_df[action.column], errors='coerce')
                        
                        if strategy == "mean":
                            val = col_data.mean()
                        elif strategy == "median":
                            val = col_data.median()
                        else:
                            val = strategy
                        self.current_df[action.column] = col_data.fillna(val)
                        
                elif action.operation == "rename_column":
                    if action.params and action.column:
                        new_name = action.params.get("new_name", None)
                        if new_name:
                            self.current_df = self.current_df.rename(columns={action.column: new_name})
                            
                elif action.operation == "drop_column":
                    if action.column and action.column in self.current_df.columns:
                        self.current_df = self.current_df.drop(columns=[action.column])
                
                # Intermediate reward calculation
                new_issues = self._get_issues_remaining(self.current_df)
                if df_before.equals(self.current_df):
                    step_rew = -0.02
                elif new_issues < old_issues:
                    step_rew = 0.05
                else:
                    step_rew = -0.02
                    
                reward_val += step_rew
                breakdown["step_reward"] = step_rew
                
        except Exception as e:
            step_rew = -0.05
            reward_val += step_rew
            breakdown["step_reward"] = step_rew
            info["error"] = str(e)
            
        self.step_count += 1
        
        if self.step_count > self.max_steps:
             reward_val -= 0.05
             breakdown["penalty_overstep"] = -0.05
             done = True
             
        schema = {col: str(self.current_df[col].dtype) for col in self.current_df.columns}
        desc = f"Clean the data for task {self.task_id}"
        next_obs = Observation(
            dirty_csv=self._df_to_csv_string(self.current_df),
            schema=schema,
            task_description=desc,
            step_count=self.step_count,
            issues_remaining=self._get_issues_remaining(self.current_df)
        )
        
        reward_obj = Reward(value=float(np.clip(reward_val, -1.0, 1.0)), breakdown=breakdown)
        
        return next_obs, reward_obj, done, info

    def state(self) -> dict:
        return {
            "current_df": self._df_to_csv_string(self.current_df),
            "step_count": self.step_count,
            "task_id": self.task_id
        }
