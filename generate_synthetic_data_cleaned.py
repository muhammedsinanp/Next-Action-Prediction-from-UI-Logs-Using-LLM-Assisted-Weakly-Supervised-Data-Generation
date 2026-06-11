import os
import re
import ast
import json
import time
import zipfile
import random
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

BASE_DIR = Path(".")

# Put few-shot JSON files or ZIP here
INPUT_DIR = BASE_DIR / "uploaded_fewshot_json"
INPUT_DIR.mkdir(exist_ok=True)

# Real next-action file
REAL_CSV_CANDIDATES = [
    Path("next_action_pairs_with_split.csv"),
    Path("real_next_action_pairs.csv"),
    Path("out_ui_logs/splits_by_task/next_action_pairs_with_split.csv"),
    Path("out_ui_logs/next_action_pairs_with_split.csv"),
    Path("pmcm/llm_synthetic_out/out_next_action_final/next_action_pairs_with_split.csv"),
]

OUT_DIR = BASE_DIR / "llm_synthetic_out_better"
OUT_DIR.mkdir(exist_ok=True)

PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

REPORTS_DIR = OUT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

FEWSHOT_PAIRS_CSV = OUT_DIR / "fewshot_pairs_from_json.csv"
LLM_SYNTHETIC_RAW_CSV = OUT_DIR / "llm_synthetic_raw_better.csv"
LLM_SYNTHETIC_FILTERED_CSV = OUT_DIR / "llm_synthetic_filtered_better.csv"
COMBINED_REAL_PLUS_SYNTH_CSV = OUT_DIR / "combined_real_plus_llm_synthetic_better.csv"

# Generation settings
SYNTHETIC_TARGET_TOTAL = 1500
MAX_RETRIES = 2

# Quality filters
MIN_TRANSITION_SCORE = 0.45
MAX_SYNTH_PER_CLASS_RATIO = 0.70

print("Input folder:", INPUT_DIR.resolve())
print("Output folder:", OUT_DIR.resolve())
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

KNOWN_ACTIONS = {
    "PROCESS_START",
    "PROCESS_END",

    "STEP_MENTION_SUGGESTION",
    "STEP_METION",
    "STEP_MENTION",
    "STEP_ENTITIES",
    "STEP_RELATION_SUGGESTION",
    "STEP_RELATIONS",

    "TOKEN_SELECTED",
    "TOKEN_DESELECTED",
    "TOKEN_SELECTED_MULTI",

    "MENTION_CREATED",
    "MENTION_DELETED",
    "MENTION_SELECTED",
    "MENTION_DESELECTED",
    "MENTION_SUGGESTION_ACCEPTED",
    "MENTION_SUGGESTION_REJECTED",
    "MENTION_SUGGESTION_BOUNDS_UPDATED",
    "MENTION_TYPE_UPDATED",
    "MENTION_BOUNDS_UPDATED",

    "RELATION_CREATED",
    "RELATION_DELETED",
    "RELATION_SELECTED",
    "RELATION_DESELECTED",
    "RELATION_MARKED",
    "RELATION_UNMARKED",
    "RELATION_SUGGESTION_ACCEPTED",
    "RELATION_SUGGESTION_REJECTED",
    "RELATION_SUGGESTION_MARKED",
    "RELATION_TYPE_UPDATED",

    "ENTITY_GROUPED",
    "ENTITY_REMOVED",

    "UNDO_ACTION",
}

print("Put your JSON files or ZIP file inside:")
print(INPUT_DIR.resolve())

print("\nFiles currently inside:")
for p in INPUT_DIR.iterdir():
    print(p)

json_files = list(INPUT_DIR.rglob("*.json"))

print("JSON files found:", len(json_files))

for p in json_files[:30]:
    print(p)

def safe_load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


records = []

for path in json_files:
    try:
        obj = safe_load_json(path)
        obj["_source_file"] = str(path)
        records.append(obj)
    except Exception as e:
        print("Failed:", path, e)

print("Loaded JSON records:", len(records))

def extract_actions_from_record(rec):
    logs = rec.get("logs", [])
    actions = []

    for event in logs:
        action = event.get("action")
        if action:
            action = str(action).strip()
            if action in KNOWN_ACTIONS:
                actions.append(action)

    return actions


sessions = []

for rec in records:
    actions = extract_actions_from_record(rec)

    if len(actions) < 2:
        continue

    sessions.append({
        "file_id": Path(rec.get("_source_file", "")).stem,
        "user_id": str(rec.get("userId", "unknown")),
        "task_id": str(rec.get("taskId", "unknown")),
        "document_text": rec.get("document", {}).get("text", ""),
        "actions": actions,
        "n_actions": len(actions),
    })

sessions_df = pd.DataFrame(sessions)

print("Few-shot sessions:", len(sessions_df))
display(sessions_df.head())

def make_next_action_pairs_from_sessions(sessions_df, max_prefix_len=None):
    rows = []

    for _, row in sessions_df.iterrows():
        actions = row["actions"]

        for i in range(1, len(actions)):
            prefix = actions[:i]
            next_action = actions[i]

            if max_prefix_len is not None:
                prefix = prefix[-max_prefix_len:]

            rows.append({
                "file_id": row["file_id"],
                "user_id": row["user_id"],
                "task_id": row["task_id"],
                "pair_index": i - 1,
                "prefix_len": len(prefix),
                "prefix": json.dumps(prefix),
                "next_action": next_action,
                "split": "train",
                "source": "fewshot_real_json",
            })

    return pd.DataFrame(rows)


fewshot_pairs = make_next_action_pairs_from_sessions(sessions_df)

print("Few-shot next-action pairs:", len(fewshot_pairs))
display(fewshot_pairs.head())

fewshot_pairs.to_csv(FEWSHOT_PAIRS_CSV, index=False)
print("Saved:", FEWSHOT_PAIRS_CSV)

def parse_prefix(x):
    if isinstance(x, list):
        return [str(a).strip() for a in x if str(a).strip() in KNOWN_ACTIONS]

    if pd.isna(x):
        return []

    s = str(x).strip()

    if not s:
        return []

    # JSON list
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(a).strip() for a in obj if str(a).strip() in KNOWN_ACTIONS]
    except Exception:
        pass

    # Python list string
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(a).strip() for a in obj if str(a).strip() in KNOWN_ACTIONS]
    except Exception:
        pass

    # Comma-separated
    if "," in s:
        parts = [
            a.strip().strip("'").strip('"')
            for a in s.split(",")
            if a.strip()
        ]
        return [p for p in parts if p in KNOWN_ACTIONS]

    # Space-separated action sequence
    parts = s.split()
    valid = [p for p in parts if p in KNOWN_ACTIONS]

    if valid:
        return valid

    if s in KNOWN_ACTIONS:
        return [s]

    return []


def clean_next_action(x):
    if pd.isna(x):
        return None

    s = str(x).strip()

    if s in KNOWN_ACTIONS:
        return s

    # If accidentally full sequence, keep last valid action
    parts = s.split()
    valid = [p for p in parts if p in KNOWN_ACTIONS]

    if valid:
        return valid[-1]

    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            valid = [str(a).strip() for a in obj if str(a).strip() in KNOWN_ACTIONS]
            return valid[-1] if valid else None
    except Exception:
        pass

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            valid = [str(a).strip() for a in obj if str(a).strip() in KNOWN_ACTIONS]
            return valid[-1] if valid else None
    except Exception:
        pass

    return None

def find_real_csv():
    for p in REAL_CSV_CANDIDATES:
        if p.exists():
            return p

    found = list(Path(".").rglob("*next_action_pairs_with_split*.csv"))
    if found:
        return found[0]

    found = list(Path(".").rglob("*.csv"))
    if found:
        print("CSV files found:")
        for f in found:
            print(f)

    return None


real_path = find_real_csv()

if real_path is None:
    print("No real CSV found. Using few-shot pairs as real data.")
    real_df = fewshot_pairs.copy()
else:
    print("Loading real CSV:", real_path)
    real_df = pd.read_csv(real_path)

print("Real dataframe shape:", real_df.shape)
display(real_df.head())

required_cols = ["task_id", "prefix", "next_action"]

missing = [c for c in required_cols if c not in real_df.columns]
if missing:
    raise ValueError(f"Missing required columns in real_df: {missing}")

real_df["task_id"] = real_df["task_id"].astype(str)

if "split" not in real_df.columns:
    real_df["split"] = "train"

real_df["split"] = real_df["split"].astype(str).str.lower()

real_df["next_action_original"] = real_df["next_action"]
real_df["next_action"] = real_df["next_action"].apply(clean_next_action)

real_df["prefix_list"] = real_df["prefix"].apply(parse_prefix)

before = len(real_df)

real_df = real_df[real_df["next_action"].notna()].copy()
real_df = real_df[real_df["prefix_list"].apply(len) > 0].copy()
real_df = real_df[real_df["next_action"].isin(KNOWN_ACTIONS)].copy()
real_df = real_df[real_df["prefix_list"].apply(lambda p: all(a in KNOWN_ACTIONS for a in p))].copy()

after = len(real_df)

print("Removed invalid real rows:", before - after)
print("Remaining real rows:", after)

display(real_df.head())

print("\nSplit distribution:")
display(real_df["split"].value_counts())

real_train = real_df[real_df["split"].eq("train")].copy()

if len(real_train) == 0:
    print("No train split found. Using all real data as train.")
    real_train = real_df.copy()
    real_train["split"] = "train"

print("Real train rows:", len(real_train))

all_actions = []

for p in real_train["prefix_list"]:
    all_actions.extend(p)

all_actions.extend(real_train["next_action"].tolist())

action_vocab = sorted(set(all_actions))
action_vocab = [a for a in action_vocab if a in KNOWN_ACTIONS]

action_counts = Counter(real_train["next_action"].astype(str))

print("Action vocab size:", len(action_vocab))
print(action_vocab)

dist_df = pd.DataFrame(action_counts.items(), columns=["next_action", "count"])
dist_df = dist_df.sort_values("count", ascending=False)

display(dist_df)

plt.figure(figsize=(14, 6))
plt.bar(dist_df["next_action"], dist_df["count"])
plt.xticks(rotation=90)
plt.title("Real Training Next-Action Distribution")
plt.xlabel("Next Action")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "real_training_next_action_distribution.png", dpi=200)
plt.show()

valid_bigrams = set()
valid_last_to_next = defaultdict(Counter)
valid_task_action_counts = defaultdict(Counter)

for _, row in real_train.iterrows():
    prefix = row["prefix_list"]
    nxt = str(row["next_action"])

    seq = prefix + [nxt]

    for a, b in zip(seq[:-1], seq[1:]):
        valid_bigrams.add((a, b))
        valid_last_to_next[a][b] += 1

    valid_task_action_counts[str(row["task_id"])][nxt] += 1

print("Valid bigrams:", len(valid_bigrams))

print("\nExample transitions:")
for k in list(valid_last_to_next.keys())[:8]:
    print(k, "->", valid_last_to_next[k].most_common(5))

# Goal:
# Synthetic data should follow the real next_action distribution.
# This is better for synthetic-only accuracy and for real+synthetic training.

target_actions = sorted([
    a for a in real_train["next_action"].astype(str).unique()
    if a in action_vocab
])

print("Target actions:", len(target_actions))

# Total synthetic size as percentage of real train size
SYNTHETIC_RATIO_OF_REAL = 0.80   # 80% of real train size
SYNTHETIC_TARGET_TOTAL = int(len(real_train) * SYNTHETIC_RATIO_OF_REAL)

# Minimum and maximum per class
MIN_SYNTH_PER_CLASS = 3
MAX_SYNTH_PER_CLASS_RATIO = 0.80  # no synthetic class should exceed 80% of its real count

real_total = sum(action_counts.values())

samples_per_action = {}

for action in target_actions:
    real_count = action_counts.get(action, 0)

    # proportional to real distribution
    proportional_n = int((real_count / real_total) * SYNTHETIC_TARGET_TOTAL)

    # give tiny classes a few examples, but do not over-amplify them
    n = max(MIN_SYNTH_PER_CLASS, proportional_n)

    # cap by real class count
    max_allowed = max(MIN_SYNTH_PER_CLASS, int(real_count * MAX_SYNTH_PER_CLASS_RATIO))

    n = min(n, max_allowed)

    samples_per_action[action] = n

plan_df = pd.DataFrame([
    {
        "action": action,
        "real_count": action_counts.get(action, 0),
        "real_ratio": action_counts.get(action, 0) / real_total,
        "planned_synthetic": samples_per_action[action],
        "synthetic_to_real_ratio": samples_per_action[action] / max(action_counts.get(action, 1), 1),
    }
    for action in target_actions
]).sort_values("real_count", ascending=False)

display(plan_df)

print("Real train rows:", len(real_train))
print("Synthetic target total:", SYNTHETIC_TARGET_TOTAL)
print("Planned synthetic total:", sum(samples_per_action.values()))

plan_df.to_csv(OUT_DIR / "generation_plan_similar_to_real.csv", index=False)

# Goal:
# Generate synthetic data similar to real distribution.
# Skip extremely rare classes because they are hard for the LLM to model correctly.

MIN_REAL_COUNT_TO_GENERATE = 10

target_actions = sorted([
    a for a in real_train["next_action"].astype(str).unique()
    if a in action_vocab and action_counts.get(a, 0) >= MIN_REAL_COUNT_TO_GENERATE
])

print("Target actions after rare-class filtering:", len(target_actions))

SYNTHETIC_RATIO_OF_REAL = 0.80
SYNTHETIC_TARGET_TOTAL = int(len(real_train) * SYNTHETIC_RATIO_OF_REAL)

MIN_SYNTH_PER_CLASS = 5
MAX_SYNTH_PER_CLASS_RATIO = 0.80

real_total_for_targets = sum(action_counts[a] for a in target_actions)

samples_per_action = {}

for action in target_actions:
    real_count = action_counts.get(action, 0)

    proportional_n = int((real_count / real_total_for_targets) * SYNTHETIC_TARGET_TOTAL)

    n = max(MIN_SYNTH_PER_CLASS, proportional_n)

    max_allowed = max(MIN_SYNTH_PER_CLASS, int(real_count * MAX_SYNTH_PER_CLASS_RATIO))

    n = min(n, max_allowed)

    samples_per_action[action] = n

plan_df = pd.DataFrame([
    {
        "action": action,
        "real_count": action_counts.get(action, 0),
        "planned_synthetic": samples_per_action[action],
        "synthetic_to_real_ratio": samples_per_action[action] / max(action_counts.get(action, 1), 1),
    }
    for action in target_actions
]).sort_values("real_count", ascending=False)

display(plan_df)

print("Real train rows:", len(real_train))
print("Planned synthetic total:", sum(samples_per_action.values()))

plan_df.to_csv(OUT_DIR / "generation_plan_similar_to_real_skip_rare.csv", index=False)

if not torch.cuda.is_available():
    raise RuntimeError("GPU not detected. Please enable GPU in your university Jupyter environment.")

LLM_PROVIDER = "huggingface"

HF_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# If you have 16GB+ VRAM, you can try:
# HF_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

USE_4BIT = True

print("Loading model:", HF_MODEL_NAME)
print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL_NAME,
    trust_remote_code=True
)

if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

model.eval()

print("Model loaded successfully.")

def row_to_example(row):
    prefix = row["prefix_list"]

    return {
        "task_id": str(row["task_id"]),
        "prefix": prefix[-35:],
        "next_action": str(row["next_action"]),
    }


def get_fewshot_examples(task_id=None, target_action=None, n=6):
    df = real_train.copy()

    # Prefer same task + same next action
    if task_id is not None and target_action is not None:
        exact = df[
            (df["task_id"].astype(str) == str(task_id)) &
            (df["next_action"].astype(str) == str(target_action))
        ]

        if len(exact) >= 2:
            sample_df = exact.sample(
                min(n, len(exact)),
                random_state=random.randint(1, 999999)
            )
            return [row_to_example(row) for _, row in sample_df.iterrows()]

    # Then same action
    if target_action is not None:
        same_action = df[df["next_action"].astype(str) == str(target_action)]

        if len(same_action) >= 2:
            sample_df = same_action.sample(
                min(n, len(same_action)),
                random_state=random.randint(1, 999999)
            )
            return [row_to_example(row) for _, row in sample_df.iterrows()]

    # Then same task
    if task_id is not None:
        same_task = df[df["task_id"].astype(str) == str(task_id)]

        if len(same_task) >= 2:
            sample_df = same_task.sample(
                min(n, len(same_task)),
                random_state=random.randint(1, 999999)
            )
            return [row_to_example(row) for _, row in sample_df.iterrows()]

    sample_df = df.sample(
        min(n, len(df)),
        random_state=random.randint(1, 999999)
    )

    return [row_to_example(row) for _, row in sample_df.iterrows()]

def get_good_previous_actions_for_target(target_action, top_n=8):
    prev_counts = Counter()

    for a, b in valid_bigrams:
        if b == target_action:
            prev_counts[a] += 1

    return [a for a, _ in prev_counts.most_common(top_n)]


def build_generation_prompt(task_id, target_action, n_samples, fewshot_examples):
    allowed_actions_text = ", ".join(action_vocab)
    examples_text = json.dumps(fewshot_examples[:6], indent=2)

    good_prev_actions = get_good_previous_actions_for_target(target_action, top_n=8)

    if good_prev_actions:
        prev_text = ", ".join(good_prev_actions)
    else:
        prev_text = "Use realistic previous actions from the examples."

    prompt = f"""
You generate synthetic UI-log next-action training samples for process annotation.

Return JSONL only.
JSONL means one valid JSON object per line.
Do not return a JSON array.
Do not use markdown.
Do not explain.

Generate exactly {n_samples} JSONL lines.

Each line must follow exactly this schema:
{{"task_id":"{task_id}","prefix":["PROCESS_START","ACTION"],"next_action":"{target_action}"}}

Critical rules:
1. next_action must be exactly "{target_action}".
2. Use only allowed actions.
3. Prefix must be realistic for UI annotation logs.
4. Prefix length must be between 6 and 40 actions.
5. Most prefixes should start with PROCESS_START.
6. Do not invent new actions.
7. Do not copy examples exactly.
8. Each JSONL line must be complete valid JSON.
9. No trailing commas.
10. No comments.

Very important:
The last action before "{target_action}" should usually be one of:
{prev_text}

Allowed actions:
{allowed_actions_text}

Few-shot real examples:
{examples_text}

Now output exactly {n_samples} JSONL lines:
"""
    return prompt.strip()

def call_huggingface_llm(prompt, max_new_tokens=1600):
    messages = [
        {
            "role": "system",
            "content": (
                "You generate JSONL data only. "
                "Each line must be one valid JSON object. "
                "No markdown. No explanation."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.40,
            top_p=0.85,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()


def call_llm(prompt):
    return call_huggingface_llm(prompt)

def extract_jsonl_objects(text):
    text = text.strip()

    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    objects = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if not line.startswith("{"):
            continue

        if line.endswith(","):
            line = line[:-1].strip()

        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                objects.append(obj)
        except Exception:
            pass

    if objects:
        return objects

    # Fallback extraction
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                objects.append(obj)
        except Exception:
            pass

    if objects:
        return objects

    raise ValueError("Could not parse JSONL objects from LLM output.")

def is_valid_sample(sample, target_action=None):
    if not isinstance(sample, dict):
        return False

    if "task_id" not in sample or "prefix" not in sample or "next_action" not in sample:
        return False

    prefix = sample["prefix"]
    next_action = str(sample["next_action"]).strip()

    if target_action is not None and next_action != target_action:
        return False

    if not isinstance(prefix, list):
        return False

    if len(prefix) < 3:
        return False

    if len(prefix) > 60:
        return False

    prefix = [str(a).strip() for a in prefix]

    if any(a not in action_vocab for a in prefix):
        return False

    if next_action not in action_vocab:
        return False

    return True


def transition_score(prefix, next_action):
    seq = prefix + [next_action]

    if len(seq) < 2:
        return 0.0

    total = 0
    valid = 0

    for a, b in zip(seq[:-1], seq[1:]):
        total += 1

        if (a, b) in valid_bigrams:
            valid += 1

    return valid / max(total, 1)


def normalize_generated_sample(sample, source_batch, target_action):
    prefix = [str(a).strip() for a in sample["prefix"]]
    next_action = str(sample["next_action"]).strip()

    return {
        "file_id": f"llm_synth_better_{source_batch}",
        "user_id": "synthetic",
        "task_id": str(sample["task_id"]),
        "pair_index": -1,
        "prefix_len": len(prefix),
        "prefix": json.dumps(prefix),
        "next_action": next_action,
        "split": "train",
        "source": "llm_synthetic",
        "target_action_requested": target_action,
        "transition_score": transition_score(prefix, next_action),
    }

task_ids = sorted(real_train["task_id"].astype(str).unique())

batches = []

MAX_PER_CALL = 10   # keep JSONL stable, but allow multiple calls per action/task

for action in target_actions:
    action_total = samples_per_action[action]

    # Choose only tasks where this action actually appears
    candidate_tasks = []

    for task_id in task_ids:
        if valid_task_action_counts[str(task_id)][action] > 0:
            candidate_tasks.append(task_id)

    if not candidate_tasks:
        continue

    # Distribute based on how often this action appears in each task
    task_counts = {
        str(task_id): valid_task_action_counts[str(task_id)][action]
        for task_id in candidate_tasks
    }

    total_task_count = sum(task_counts.values())

    # First calculate target samples per task proportionally
    task_sample_plan = {}

    for task_id, count in task_counts.items():
        n = round((count / total_task_count) * action_total)
        task_sample_plan[task_id] = int(n)

    # Fix rounding so total equals action_total
    current_total = sum(task_sample_plan.values())
    diff = action_total - current_total

    if diff != 0:
        # Add/subtract from tasks with highest real count
        sorted_tasks = sorted(task_counts.keys(), key=lambda t: task_counts[t], reverse=True)

        idx = 0
        while diff != 0 and len(sorted_tasks) > 0:
            task_id = sorted_tasks[idx % len(sorted_tasks)]

            if diff > 0:
                task_sample_plan[task_id] += 1
                diff -= 1
            else:
                if task_sample_plan[task_id] > 0:
                    task_sample_plan[task_id] -= 1
                    diff += 1

            idx += 1

    # Create batches; split large task counts into multiple calls
    for task_id, n_total in task_sample_plan.items():
        if n_total <= 0:
            continue

        remaining = n_total

        while remaining > 0:
            n = min(remaining, MAX_PER_CALL)

            batches.append({
                "task_id": str(task_id),
                "target_action": action,
                "n_samples": int(n),
            })

            remaining -= n

random.shuffle(batches)

batch_df = pd.DataFrame(batches)

print("Planned batches:", len(batch_df))
print("Approx synthetic target from batches:", batch_df["n_samples"].sum())

planned_by_action = (
    batch_df
    .groupby("target_action")["n_samples"]
    .sum()
    .rename("planned_from_batches")
    .to_frame()
)

# Compare with real distribution
compare_plan = plan_df.set_index("action").join(planned_by_action, how="left").fillna(0)
compare_plan["planned_from_batches"] = compare_plan["planned_from_batches"].astype(int)
compare_plan["batch_vs_requested_diff"] = (
    compare_plan["planned_from_batches"] - compare_plan["planned_synthetic"]
)

display(compare_plan.sort_values("real_count", ascending=False))

display(batch_df.head())

batch_df.to_csv(OUT_DIR / "generation_batches_similar_to_real_FIXED.csv", index=False)

synthetic_rows = []

for batch_idx, batch in enumerate(tqdm(batches)):
    task_id = batch["task_id"]
    target_action = batch["target_action"]
    n_samples = batch["n_samples"]

    fewshots = get_fewshot_examples(
        task_id=task_id,
        target_action=target_action,
        n=6
    )

    prompt = build_generation_prompt(
        task_id=task_id,
        target_action=target_action,
        n_samples=n_samples,
        fewshot_examples=fewshots
    )

    success = False
    raw = ""

    for attempt in range(MAX_RETRIES):
        try:
            raw = call_llm(prompt)
            parsed = extract_jsonl_objects(raw)

            before = len(synthetic_rows)

            for sample in parsed:
                if is_valid_sample(sample, target_action=target_action):
                    row = normalize_generated_sample(
                        sample,
                        source_batch=batch_idx,
                        target_action=target_action
                    )
                    synthetic_rows.append(row)

            added = len(synthetic_rows) - before

            print(
                f"Batch {batch_idx} | task={task_id} | target={target_action} | "
                f"requested={n_samples} | parsed={len(parsed)} | added={added}"
            )

            success = True
            break

        except Exception as e:
            print(f"Batch {batch_idx} failed attempt {attempt + 1}: {e}")
            print("Raw preview:")
            print(raw[:600] if raw else "No output")
            time.sleep(1)

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not success:
        print("Skipped batch:", batch)

    if len(synthetic_rows) > 0 and batch_idx % 5 == 0:
        pd.DataFrame(synthetic_rows).to_csv(LLM_SYNTHETIC_RAW_CSV, index=False)
        print("Checkpoint saved:", LLM_SYNTHETIC_RAW_CSV)

synthetic_raw = pd.DataFrame(synthetic_rows)
synthetic_raw.to_csv(LLM_SYNTHETIC_RAW_CSV, index=False)

print("Generated raw synthetic rows:", len(synthetic_raw))
display(synthetic_raw.head())
print("Saved:", LLM_SYNTHETIC_RAW_CSV)

import pandas as pd

raw_path = LLM_SYNTHETIC_RAW_CSV

synthetic_raw_check = pd.read_csv(raw_path)

print("Generated rows so far:", len(synthetic_raw_check))
display(synthetic_raw_check.head())

display(
    synthetic_raw_check["next_action"]
    .value_counts()
    .to_frame("raw_synthetic_count")
)

def extract_jsonl_objects(text):
    """
    Robust parser:
    1. Parses JSONL lines
    2. Parses full JSON array
    3. Parses one pretty JSON object
    4. Extracts balanced JSON objects from messy output
    """
    text = text.strip()

    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    objects = []

    # Try full JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass

    # Try JSONL line by line
    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.endswith(","):
            line = line[:-1].strip()

        if not line.startswith("{"):
            continue

        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                objects.append(obj)
        except Exception:
            pass

    if objects:
        return objects

    # Balanced object extraction
    candidates = []
    start = None
    depth = 0

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1

        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                objects.append(obj)
        except Exception:
            pass

    if objects:
        return objects

    raise ValueError("Could not parse JSONL objects from LLM output.")

def build_generation_prompt(task_id, target_action, n_samples, fewshot_examples):
    allowed_actions_text = ", ".join(action_vocab)
    examples_text = json.dumps(fewshot_examples[:5], indent=2)

    good_prev_actions = get_good_previous_actions_for_target(target_action, top_n=8)

    if good_prev_actions:
        prev_text = ", ".join(good_prev_actions)
    else:
        prev_text = "Use realistic previous actions from the examples."

    prompt = f"""
Generate synthetic UI-log next-action training samples.

Return JSONL only: one complete JSON object per line.
Do not return a JSON array.
Do not use markdown.
Do not explain.

Generate exactly {n_samples} lines.

Schema:
{{"task_id":"{task_id}","prefix":["PROCESS_START","ACTION"],"next_action":"{target_action}"}}

Rules:
1. next_action must be exactly "{target_action}".
2. Use only allowed actions.
3. Prefix length must be between 6 and 35.
4. Prefix must look like the few-shot examples.
5. Do not repeat the same action more than 3 times consecutively.
6. Do not create prefixes like ACTION, ACTION, ACTION, ACTION many times.
7. The final prefix action before next_action should usually be one of:
{prev_text}
8. Do not copy examples exactly.
9. No trailing commas.
10. Each line must be valid JSON.

Allowed actions:
{allowed_actions_text}

Few-shot examples:
{examples_text}

Output exactly {n_samples} JSONL lines now:
"""
    return prompt.strip()

def has_bad_repetition(prefix, max_repeat=3):
    if not prefix:
        return True

    count = 1

    for i in range(1, len(prefix)):
        if prefix[i] == prefix[i - 1]:
            count += 1
            if count > max_repeat:
                return True
        else:
            count = 1

    return False


before = len(synthetic_df)

synthetic_df = synthetic_df[
    ~synthetic_df["prefix_list"].apply(lambda p: has_bad_repetition(p, max_repeat=3))
].copy()

after = len(synthetic_df)

print("Removed bad repetition rows:", before - after)
print("Remaining:", after)

synthetic_df = pd.read_csv(LLM_SYNTHETIC_RAW_CSV)

synthetic_df["prefix_list"] = synthetic_df["prefix"].apply(parse_prefix)
synthetic_df["next_action"] = synthetic_df["next_action"].astype(str).str.strip()
synthetic_df["task_id"] = synthetic_df["task_id"].astype(str)

synthetic_df["transition_score"] = synthetic_df.apply(
    lambda row: transition_score(row["prefix_list"], row["next_action"]),
    axis=1
)

print("Raw synthetic:", synthetic_df.shape)

display(synthetic_df.head())
display(synthetic_df["transition_score"].describe().to_frame())

before = len(synthetic_df)

synthetic_df = synthetic_df[
    synthetic_df["next_action"].isin(action_vocab)
].copy()

synthetic_df = synthetic_df[
    synthetic_df["prefix_list"].apply(
        lambda p: len(p) > 0 and all(a in action_vocab for a in p)
    )
].copy()

filtered = synthetic_df[
    synthetic_df["transition_score"] >= MIN_TRANSITION_SCORE
].copy()

after = len(filtered)

print("Removed invalid or low-transition rows:", before - after)
print("Remaining:", after)

filtered["prefix"] = filtered["prefix_list"].apply(lambda x: json.dumps(x))

filtered["pair_key"] = (
    filtered["task_id"].astype(str)
    + "||"
    + filtered["prefix"].astype(str)
    + "||"
    + filtered["next_action"].astype(str)
)

before = len(filtered)
filtered = filtered.drop_duplicates("pair_key").copy()
after = len(filtered)

print("Removed duplicate synthetic rows:", before - after)
print("Remaining:", after)

real_df["task_id"] = real_df["task_id"].astype(str)
real_df["next_action"] = real_df["next_action"].astype(str)

real_df["prefix_list_temp"] = real_df["prefix"].apply(parse_prefix)
real_df["prefix_norm"] = real_df["prefix_list_temp"].apply(lambda x: json.dumps(x))

real_df["pair_key"] = (
    real_df["task_id"].astype(str)
    + "||"
    + real_df["prefix_norm"].astype(str)
    + "||"
    + real_df["next_action"].astype(str)
)

real_keys = set(real_df["pair_key"])

before = len(filtered)
filtered = filtered[~filtered["pair_key"].isin(real_keys)].copy()
after = len(filtered)

print("Removed leakage rows:", before - after)
print("Remaining:", after)

real_class_counts = Counter(real_train["next_action"].astype(str))

capped_parts = []

for action, group in filtered.groupby("next_action"):
    real_count = real_class_counts.get(action, 1)

    max_allowed = int(max(8, real_count * MAX_SYNTH_PER_CLASS_RATIO))

    keep_n = min(len(group), max_allowed)

    group = group.sample(
        keep_n,
        random_state=RANDOM_SEED
    )

    capped_parts.append(group)

if capped_parts:
    filtered_capped = pd.concat(capped_parts, ignore_index=True)
else:
    filtered_capped = pd.DataFrame()

print("Before cap:", len(filtered))
print("After cap:", len(filtered_capped))

display(filtered_capped["next_action"].value_counts().to_frame("synthetic_count"))

needed_cols = [
    "file_id",
    "user_id",
    "task_id",
    "pair_index",
    "prefix_len",
    "prefix",
    "next_action",
    "split",
    "source",
    "target_action_requested",
    "transition_score",
]

for col in needed_cols:
    if col not in filtered_capped.columns:
        if col == "file_id":
            filtered_capped[col] = "llm_synthetic"
        elif col == "user_id":
            filtered_capped[col] = "synthetic"
        elif col == "pair_index":
            filtered_capped[col] = -1
        elif col == "split":
            filtered_capped[col] = "train"
        elif col == "source":
            filtered_capped[col] = "llm_synthetic"
        elif col == "prefix_len":
            filtered_capped[col] = filtered_capped["prefix"].apply(lambda x: len(parse_prefix(x)))
        else:
            filtered_capped[col] = None

filtered_capped["split"] = "train"
filtered_capped["source"] = "llm_synthetic"

filtered_capped = filtered_capped[needed_cols].copy()

filtered_capped.to_csv(LLM_SYNTHETIC_FILTERED_CSV, index=False)

print("Saved:", LLM_SYNTHETIC_FILTERED_CSV)
print("Rows:", len(filtered_capped))

display(filtered_capped.head())