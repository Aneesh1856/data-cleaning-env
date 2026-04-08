from typing import Literal, Optional

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """
    Represents the state of the data cleaning environment at a specific step.
    
    Attributes:
        dirty_csv: The raw CSV content as a string.
        schema: Expected column names mapped to expected dtype strings 
            (e.g., "datetime", "str", "float").
        task_description: A natural language description of the data cleaning task.
        step_count: The current step number in the episode.
        issues_remaining: The count of remaining detectable issues (nulls + dupes + bad formats).
    """
    dirty_csv: str
    schema: dict
    task_description: str
    step_count: int
    issues_remaining: int


class Action(BaseModel):
    """
    Represents an action taken by the reinforcement learning agent to clean the data.
    
    Attributes:
        operation: The type of data cleaning operation to perform.
        column: The name of the column to apply the operation to. Optional.
        params: Additional parameters for the operation. Optional.
            Examples: {"strategy": "mean"} for impute_nulls, 
            {"format": "%d/%m/%Y"} for fix_dates.
    """
    operation: Literal[
        "fix_dates", 
        "drop_duplicates", 
        "normalize_category", 
        "impute_nulls", 
        "rename_column", 
        "drop_column", 
        "done"
    ]
    column: Optional[str] = None
    params: Optional[dict] = None


class Reward(BaseModel):
    """
    Represents the reward received by the agent after taking an action.
    
    Attributes:
        value: The overall reward value, between 0.0 and 1.0.
        breakdown: Sub-scores categorized by the type of fix applied.
            Example: {"nulls": 0.3, "dedup": 0.2, "formats": 0.4}
    """
    value: float = Field(ge=-1.0, le=1.0)
    breakdown: dict
