import os
import pandas as pd
import numpy as np

def robust_match(val1, val2):
    if str(val1).strip().lower() == str(val2).strip().lower():
        return True
    try:
        return float(val1) == float(val2)
    except:
        return False

def run_grader(task_id: str, output_csv: str, ground_truth_csv: str) -> float:
    try:
        dirty_csv = f"env/data/task_{task_id}_dirty.csv"
        if not os.path.exists(output_csv) or not os.path.exists(ground_truth_csv) or not os.path.exists(dirty_csv):
            return 0.001

        df_out = pd.read_csv(output_csv)
        df_gt = pd.read_csv(ground_truth_csv)
        df_dirty = pd.read_csv(dirty_csv)

        if "id" not in df_out.columns or "id" not in df_gt.columns or "id" not in df_dirty.columns:
            return 0.001
        
        dirty_ids = set(df_dirty["id"])
        gt_ids = set(df_gt["id"])
        out_ids = set(df_out["id"])
        
        expected_drops = dirty_ids - gt_ids
        actual_drops = dirty_ids - out_ids
        
        tp = len(expected_drops & actual_drops)
        fp = len(actual_drops - expected_drops)
        fn = len(expected_drops - actual_drops)
        
        df_out = df_out.drop_duplicates(subset=["id"], keep="first")
        df_gt = df_gt.drop_duplicates(subset=["id"], keep="first")
        df_dirty = df_dirty.drop_duplicates(subset=["id"], keep="first")
        
        out_idx = df_out.set_index("id").astype(str).replace("nan", np.nan).fillna("NAN_NULL")
        gt_idx = df_gt.set_index("id").astype(str).replace("nan", np.nan).fillna("NAN_NULL")
        dirty_idx = df_dirty.set_index("id").astype(str).replace("nan", np.nan).fillna("NAN_NULL")
        
        common_ids = gt_ids & out_ids
        cols = [c for c in gt_idx.columns if c in dirty_idx.columns and c in out_idx.columns]
        
        for cid in common_ids:
            for col in cols:
                v_dirty = dirty_idx.at[cid, col]
                v_gt = gt_idx.at[cid, col]
                v_out = out_idx.at[cid, col]
                
                is_changed_expected = not robust_match(v_gt, v_dirty)
                is_changed_actual = not robust_match(v_out, v_dirty)
                is_correct = robust_match(v_out, v_gt)
                
                if is_changed_expected:
                    if is_correct:
                        tp += 1
                    else:
                        fn += 1
                        if is_changed_actual:
                            fp += 1
                else:
                    if is_changed_actual:
                        fp += 1
                        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.001
            
        f1 = 2 * (precision * recall) / (precision + recall)
        # Clamp to strict (0, 1) open interval — checker rejects 0.0 and 1.0
        return float(np.clip(f1, 0.001, 0.999))

    except Exception as e:
        print(f"Universal Precision/Recall Grader error: {e}")
        return 0.001

