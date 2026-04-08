import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def create_directory():
    os.makedirs("env/data", exist_ok=True)

def generate_unified_datasets():
    np.random.seed(42)
    random.seed(42)
    
    n_rows_base = 400
    ids = np.arange(1, n_rows_base + 1)
    
    first_names = ["John", "Jane", "Alice", "Bob", "Charlie", "David", "Eve", "Frank"]
    last_names = ["Smith", "Doe", "Johnson", "Brown", "Davis", "Miller", "Wilson", "Moore"]
    names = [f"{random.choice(first_names)} {random.choice(last_names)}_{i}" for i in ids]
    
    ages = np.random.randint(22, 60, size=n_rows_base).astype(float)
    salaries = np.random.randint(40000, 150000, size=n_rows_base).astype(float)
    categories = np.random.choice(["electronics", "clothing", "home", "toys", "books"], size=n_rows_base)
    
    base_date = datetime(2015, 1, 1)
    dates = [base_date + timedelta(days=random.randint(0, 3000)) for _ in ids]
    clean_join_dates = [d.strftime("%Y-%m-%d") for d in dates]
    
    df_clean = pd.DataFrame({
        "id": ids,
        "full_name": names,
        "age": ages,
        "salary": salaries,
        "category": categories,
        "join_date": clean_join_dates
    })
    
    df_dirty_base = df_clean.copy()
    
    # 1. Malformed Dates
    dirty_dates = []
    format_choices = ["%m/%d/%Y", "%b %d %Y", "%d-%m-%Y"]
    for d in dates:
        dirty_dates.append(d.strftime(random.choice(format_choices)))
    df_dirty_base["join_date"] = dirty_dates
    
    # 2. Inconsistent Categories
    dirty_cats = []
    for c in df_dirty_base["category"]:
        c_mod = c.upper() if random.random() > 0.5 else c.title()
        if random.random() > 0.5:
            c_mod = " " + c_mod + " "
        dirty_cats.append(c_mod)
    df_dirty_base["category"] = dirty_cats
    
    # 3. Nulls
    null_indices = np.random.choice(n_rows_base, size=60, replace=False)
    df_dirty_base.loc[null_indices, "salary"] = np.nan
    
    # 4. Outliers (Schema violation)
    outlier_indices = np.random.choice(list(set(range(n_rows_base)) - set(null_indices)), size=40, replace=False)
    for idx in outlier_indices:
        df_dirty_base.loc[idx, "age"] = random.choice([-5.0, 999.0])
        
    # 5. Duplicates
    duplicate_indices = np.random.choice(n_rows_base, size=50, replace=True)
    df_duplicates = df_dirty_base.iloc[duplicate_indices].copy()
    
    df_dirty = pd.concat([df_dirty_base, df_duplicates]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Identical Dirty Targets For All Difficulties (Agent receives unified state)
    for t in ["easy", "medium", "hard"]:
        df_dirty.to_csv(f"env/data/task_{t}_dirty.csv", index=False)
    
    # Easy GT: ONLY Fix dates (Precision checks this isolated)
    df_easy_gt = df_dirty.copy()
    id_to_clean_date = dict(zip(df_clean["id"], df_clean["join_date"]))
    df_easy_gt["join_date"] = df_easy_gt["id"].map(id_to_clean_date)
    df_easy_gt.to_csv("env/data/task_easy_gt.csv", index=False)
    
    # Medium GT: Easy + Dedup + Normalize
    df_medium_gt = df_easy_gt.copy()
    df_medium_gt = df_medium_gt.drop_duplicates(subset=["id"], keep="first")
    df_medium_gt["category"] = df_medium_gt["category"].str.lower().str.strip()
    df_medium_gt.to_csv("env/data/task_medium_gt.csv", index=False)
    
    # Hard GT: Medium + Impute + Schema
    df_hard_gt = df_medium_gt.copy()
    true_mean = df_clean["salary"].mean()
    df_hard_gt["salary"] = df_hard_gt["salary"].fillna(true_mean)
    
    # Schema map: NaNs for invalid bound thresholds
    df_hard_gt.loc[(df_hard_gt["age"] < 18) | (df_hard_gt["age"] > 80), "age"] = np.nan
    df_hard_gt.to_csv("env/data/task_hard_gt.csv", index=False)

if __name__ == "__main__":
    create_directory()
    generate_unified_datasets()
    print("Successfully generated unified Data Cleaning matrices mapping precision/recall layers.")
