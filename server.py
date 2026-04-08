import os
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import UploadFile, File
import pandas as pd
import numpy as np
import io
import json
import os
import asyncio
from openai import OpenAI
from pydantic import BaseModel

from models import Action, Observation
from env.environment import DataCleaningEnv

app = FastAPI(title="Data Cleaning OpenEnv API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>OpenEnv Data Cleaning Environment is Running! (index.html missing)</h1>"

@app.post("/upload")
async def upload_and_clean_csv(file: UploadFile = File(...)):
    contents = await file.read()
    
    try:
        df_original = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        return {"error": f"Invalid CSV file: {e}"}

    # Detect date columns and category columns automatically
    date_cols = [c for c in df_original.columns if "date" in c.lower() or "time" in c.lower()]
    cat_cols = [c for c in df_original.columns if df_original[c].dtype == object and c not in date_cols and "name" not in c.lower() and "id" not in c.lower()]
    num_null_cols = [c for c in df_original.columns if df_original[c].isnull().any() and df_original[c].dtype in ["float64", "int64", object]]

    # Build a deterministic cleaning plan from analysis — no LLM needed for this part
    plan = []
    for col in date_cols:
        plan.append({"operation": "fix_dates", "column": col})
    plan.append({"operation": "drop_duplicates"})
    for col in cat_cols:
        plan.append({"operation": "normalize_category", "column": col})
    for col in num_null_cols:
        plan.append({"operation": "impute_nulls", "column": col, "params": {"strategy": "mean"}})

    # Run LLM to ALSO suggest additional steps beyond our plan
    groq_api_key = os.getenv("OPENAI_API_KEY", "")

    system_prompt = """You are a data cleaning expert. Analyze the CSV snippet and return a JSON array of cleaning operations.
Each item must be a JSON object with "operation" and optionally "column" and "params".
Available operations: fix_dates (needs column), drop_duplicates, normalize_category (needs column), impute_nulls (needs column + params.strategy=mean/median), done.
Return ONLY a raw JSON array. Example: [{"operation":"fix_dates","column":"join_date"},{"operation":"drop_duplicates"},{"operation":"done"}]"""

    def call_llm(df_csv_snippet):
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key
        )
        prompt = f"Here is the dirty CSV (first 10 rows):\n{df_csv_snippet}\n\nReturn a JSON array of cleaning steps."
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400
        )
        return resp.choices[0].message.content.strip()

    df = df_original.copy()
    action_history = []

    try:
        snippet = df.head(10).to_csv(index=False)
        llm_text = await asyncio.to_thread(call_llm, snippet)
        
        # Parse LLM array
        for fence in ["```json", "```"]:
            if llm_text.startswith(fence): llm_text = llm_text[len(fence):]
        if llm_text.endswith("```"): llm_text = llm_text[:-3]
        llm_plan = json.loads(llm_text.strip())
        if isinstance(llm_plan, list):
            plan = llm_plan  # LLM overrides with its full plan
        print(f"LLM Plan: {plan}")
    except Exception as e:
        print(f"LLM fallback to rule-based plan: {e}")
        # plan stays as the rule-based fallback built above

    # Execute cleaning plan
    for step in plan:
        op = step.get("operation")
        col = step.get("column")
        if op == "done":
            break
        elif op == "fix_dates" and col and col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
            action_history.append(f"fix_dates({col})")
        elif op == "drop_duplicates":
            df = df.drop_duplicates().reset_index(drop=True)
            action_history.append("drop_duplicates")
        elif op == "normalize_category" and col and col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            action_history.append(f"normalize_category({col})")
        elif op == "impute_nulls" and col and col in df.columns:
            strat = step.get("params", {}).get("strategy", "mean")
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            val = numeric_col.mean() if strat == "mean" else numeric_col.median()
            df[col] = numeric_col.fillna(val)
            action_history.append(f"impute_nulls({col})")
        elif op == "drop_column" and col and col in df.columns:
            df = df.drop(columns=[col])
            action_history.append(f"drop_column({col})")

    print(f"Cleaning complete. Steps applied: {action_history}")

    sio = io.StringIO()
    df.to_csv(sio, index=False)
    
    return StreamingResponse(
        iter([sio.getvalue()]), 
        media_type="text/csv", 
        headers={"Content-Disposition": f"attachment; filename=cleaned_{file.filename}"}
    )

# Global env instance
env = None

class ResetRequest(BaseModel):
    task_id: str

@app.post("/reset", response_model=Observation)
def reset_env(req: ResetRequest):
    global env
    if req.task_id not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="Invalid task_id. Must be 'easy', 'medium', or 'hard'.")
    
    env = DataCleaningEnv(task_id=req.task_id)
    obs = env.reset()
    return obs

@app.post("/step")
def step_env(action: Action):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return env.state()

@app.get("/tasks")
def get_tasks():
    yaml_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    if not os.path.exists(yaml_path):
        raise HTTPException(status_code=500, detail="openenv.yaml not found.")
    
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
        
    return {
        "tasks": config.get("tasks", []),
        "action_space": config.get("action_space", [])
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/generate_sample")
def generate_sample_csv():
    import uuid
    import random
    from datetime import datetime, timedelta
    
    n_rows_base = 50
    ids = np.arange(1, n_rows_base + 1)
    
    first_names = ["John", "Jane", "Alice", "Bob", "Charlie", "David", "Eve", "Frank"]
    last_names = ["Smith", "Doe", "Johnson", "Brown", "Davis", "Miller", "Wilson", "Moore"]
    names = [f"{random.choice(first_names)} {random.choice(last_names)}_{i}" for i in ids]
    
    ages = np.random.randint(22, 60, size=n_rows_base).astype(float)
    salaries = np.random.randint(40000, 150000, size=n_rows_base).astype(float)
    categories = np.random.choice(["electronics", "clothing", "home", "toys", "books"], size=n_rows_base)
    
    base_date = datetime(2015, 1, 1)
    dates = [base_date + timedelta(days=random.randint(0, 3000)) for _ in ids]
    
    df = pd.DataFrame({
        "id": ids,
        "full_name": names,
        "age": ages,
        "salary": salaries,
        "category": categories,
        "join_date": dates
    })
    
    # Chaos
    dirty_dates = []
    for d in dates:
        dirty_dates.append(d.strftime(random.choice(["%m/%d/%Y", "%b %d %Y", "%d-%m-%Y"])))
    df["join_date"] = dirty_dates
    
    dirty_cats = []
    for c in df["category"]:
        c_mod = c.upper() if random.random() > 0.5 else c.title()
        if random.random() > 0.5: c_mod = " " + c_mod + " "
        dirty_cats.append(c_mod)
    df["category"] = dirty_cats
    
    null_indices = np.random.choice(n_rows_base, size=10, replace=False)
    df.loc[null_indices, "salary"] = np.nan
    
    outlier_indices = np.random.choice(list(set(range(n_rows_base)) - set(null_indices)), size=5, replace=False)
    for idx in outlier_indices:
        df.loc[idx, "age"] = random.choice([-5.0, 999.0])
        
    duplicate_indices = np.random.choice(n_rows_base, size=8, replace=True)
    df_duplicates = df.iloc[duplicate_indices].copy()
    
    df_dirty = pd.concat([df, df_duplicates]).sample(frac=1).reset_index(drop=True)
    
    sio = io.StringIO()
    df_dirty.to_csv(sio, index=False)
    
    return StreamingResponse(
        iter([sio.getvalue()]), 
        media_type="text/csv", 
        headers={"Content-Disposition": f"attachment; filename=judge_dummy_data_{uuid.uuid4().hex[:6]}.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860)
