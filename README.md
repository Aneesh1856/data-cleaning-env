---
title: OpenEnv Data Cleaning Environment
emoji: 🧹
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - data-cleaning
  - reinforcement-learning
  - benchmark
pinned: false
---

# OpenEnv: data-cleaning-env

## Environment Description
This environment simulates programmatic data cleaning challenges representative of standard ELT (Extract, Load, Transform) engineering pipelines. Agent models evaluate dirty, real-world data distributions to normalize formats, impute absent data, discard malformed rows, and standardize columns via sequential discrete Pandas operations. It matters for agent evaluation because it provides a highly objective, tabular reasoning benchmark testing code sequence generation, schema inference, deterministic tool usage, and iterative visual anomaly detection.

## Observation Space
At each step, the environment returns a strictly typed JSON observation representing the current state of the dataset and episode.

| Field | Type | Description |
|---|---|---|
| `dirty_csv` | `str` | Truncated CSV string representation of the current active dataset state. |
| `schema` | `dict` | Column mappings linking column names to expected dtype strings. |
| `task_description` | `str` | NL prompt explicitly outlining the targeted cleaning objectives. |
| `step_count` | `int` | The numerical index of the current step in the episode. |
| `issues_remaining` | `int` | Programmatic estimation of remaining detectable malformations (nulls, duplicates, bad shapes). |

## Action Space
An agent triggers pipeline transformations via atomic JSON structures containing the `operation` and associated configurations.

| Operation | Required Params | Behavior |
|---|---|---|
| `fix_dates` | `column`, `format` (optional) | Parses arbitrary dates and normalizes them uniformly to ISO `YYYY-MM-DD` format. |
| `drop_duplicates` | `subset` (optional) | Drops duplicate rows spanning entire rows or specified subsets. |
| `normalize_category` | `column` | Lowercases and strips trailing/leading whitespaces on categorical strings. |
| `impute_nulls` | `column`, `strategy` (optional) | Imputes NaN values targeting specific central tendencies (`mean`, `median`, or absolute constant). |
| `rename_column` | `column`, `new_name` | Maps the active column vector to a targeted new name string. |
| `drop_column` | `column` | Drops target column entirely. |
| `done` | None | Terminates trajectory early and securely invokes final scoring protocols. |

## Tasks
The benchmark spans three incremental complexities assessing zero-shot data recovery limits.

| Task ID | Difficulty | Description | Max Steps |
|---|---|---|---|
| `easy` | easy | Fix generic date formats and drop basic null rows. | 20 |
| `medium` | medium | Remove exact duplicates, normalize diverse string casings, and impute revenue numeric variants. | 35 |
| `hard` | hard | Impute encrypted nulls, fix mixed-locale dates, resolve fuzzy duplicates, fix age outliers. | 60 |

**What makes `hard` genuinely hard for frontier models:**
The `hard` task features unstructured noise injection breaking standard assumptions. Nulls are encoded as hidden string artifacts (`"N/A"`, `"-"`) bypassing basic primitive `.isnull()` detection heuristics. Mixed US/EU date locales explicitly necessitate semantic logic branch processing. Finally, duplicates feature uncoordinated casing permutations that require prerequisite intermediary transformations (e.g., lowercasing variables) before discrete deduplication commands trigger correctly, expanding the reasoning horizon strictly required by the LLM solver.

## Reward Function
The environment generates composite reward scalars combining intermediate dense reward shaping alongside final accuracy sparse precision payouts.
- **Dense Rewards:** 
  - `+0.05`: Action effectively decreases measurable geometric issues.
  - `-0.02`: Agent triggers a sequence resulting in an unmutated dataframe (wasted step metric).
  - `-0.05`: Action explicitly triggers internal Pandas exceptions or exceeds the `max_steps` boundary constraint.
- **Sparse Bonus:** At step terminal (via `done` action invocation or hitting max iteration boundaries), the agent invokes ground-truth grader protocols. Unscaled terminal accuracies mapping `[0.0, 1.0]` are appended as an immediate terminal bonus scaled rigidly up to limits `[+0.0, +0.3]`.

## Grader Design
Final execution yields are audited automatically using deterministic Pandas comparison operations mapped cleanly against corresponding internal Ground-Truth tables compiled per task. 
- Fully deterministic indexing cross-validates absolute sets via exact `.merge()` joins targeting canonical internal ID indices. 
- It assesses granular mathematical tolerances: validating precise numeric distribution proximities (e.g., matching within explicit 5%-10% median envelopes), absolute fraction drop percentages on row anomalies, and strict regex parity matching ISO layouts mapping natively. 
- Explicitly circumvents stochastic LLM-as-a-judge methodologies to ensure un-hallucinated, stable, and functionally reproducible baseline measurements globally.

## Setup Instructions
The environment framework binds natively targeting Python 3.11 over an isolated FastAPI container module.

1. Build container:
```bash
docker build -t data-cleaning-env .
```

2. Run specific OpenEnv internal Server bindings:
```bash
docker run -p 7860:7860 data-cleaning-env
```

3. Launch generic Baseline Inference LLM logic:
```bash
python baseline_agent.py
```

## Baseline Scores
*Note: Reference table variables post executing standard native baseline evaluations.*

| Task | Steps Taken | Final Score |
|---|---|---|
| `easy` | 2 | 0.34 |
| `medium` | 4 | 0.34 |
| `hard` | 5 | 0.29 |

## HuggingFace Deployment
`tags: ["openenv"]`
