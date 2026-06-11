# Cell 2 — Imports
import json
import ast
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Cell 3 — Config and paths
RESULTS_PATH          = Path("results.csv")
JSON_ROOT             = Path("tea_pie_study_UI_logs")
SYNTH_JSONL_PATH      = Path("synth_prefix_f1_noassist.jsonl")

OUT_DIR   = Path("no_assistance_f1_full_json_plus_synth_jsonl")
OUT_DIR.mkdir(exist_ok=True)

PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

PREFIX_LENGTHS = [10, 20, 30, 40, 50, 60]
TARGET_COL     = "no-assistance-overall-f1"
RANDOM_SEED    = 42

np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Cell 4 — Load results CSV (target labels)
results_df = pd.read_csv(RESULTS_PATH, sep=";")

results_df["user-id"] = results_df["user-id"].astype(str).str.strip()

results_df[TARGET_COL] = (
    results_df[TARGET_COL]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

results_df[TARGET_COL] = pd.to_numeric(
    results_df[TARGET_COL],
    errors="coerce"
)

target_map = dict(zip(results_df["user-id"], results_df[TARGET_COL]))

print("Results:", results_df.shape)
print("Target:", TARGET_COL)

display(results_df[["user-id", TARGET_COL]].head())

# Cell 5 — Find JSON session files
json_files = list(JSON_ROOT.rglob("*.json"))

print("JSON files found:", len(json_files))
print(json_files[:10])

if len(json_files) == 0:
    raise FileNotFoundError("No JSON files found. Check JSON_ROOT path.")

# Cell 6 — Inspect a sample JSON file
sample_path = json_files[0]

with open(sample_path, "r", encoding="utf-8") as f:
    sample = json.load(f)

print("Sample:", sample_path)
print("Top-level keys:", sample.keys())

logs = sample.get("logs", [])

print("Number of logs:", len(logs))

if len(logs) > 0:
    print("First log keys:", logs[0].keys())
    print(json.dumps(logs[0], indent=2)[:1000])

# Cell 7 — Helper: extract action token from a UI log event
def extract_action_from_event(event):
    if not isinstance(event, dict):
        return None

    for key in ["action", "event_type", "type", "name"]:
        if key in event and event[key] is not None:
            return str(event[key]).strip()

    return None

# Cell 8 — Build real sessions dataframe from JSON logs
real_session_rows = []

for path in json_files:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        user_id = str(obj.get("userId", "")).strip()
        task_id = str(obj.get("taskId", "")).strip()
        logs    = obj.get("logs", [])

        actions = []
        for event in logs:
            action = extract_action_from_event(event)
            if action:
                actions.append(action)

        if len(actions) == 0:
            continue

        target_f1 = target_map.get(user_id, np.nan)

        if pd.isna(target_f1):
            continue

        real_session_rows.append({
            "sample_id":   str(path),
            "user_id":     user_id,
            "task_id":     task_id,
            "actions":     actions,
            "target_f1":   float(target_f1),
            "session_len": len(actions),
            "source":      "real"
        })

    except Exception as e:
        print("Failed:", path, e)

real_sessions = pd.DataFrame(real_session_rows)

print("Real sessions:", real_sessions.shape)
display(real_sessions.head())
display(real_sessions["session_len"].describe())

for L in PREFIX_LENGTHS:
    print(f"Real sessions >= {L}:", (real_sessions["session_len"] >= L).sum())

# Cell 9 — Build prefix samples from real sessions
def make_prefix_samples_from_actions(df, prefix_lengths, source_name):
    rows = []

    for _, row in df.iterrows():
        actions = row["actions"]

        for L in prefix_lengths:
            if len(actions) >= L:
                prefix = actions[:L]

                rows.append({
                    "sample_id":       row.get("sample_id", ""),
                    "user_id":         str(row.get("user_id", "unknown")),
                    "task_id":         str(row.get("task_id", "unknown")),
                    "prefix_len":      L,
                    "prefix_text":     " ".join(prefix),
                    "target_f1":       float(row["target_f1"]),
                    "source":          source_name,
                    "full_session_len": len(actions),
                })

    return pd.DataFrame(rows)


real_samples = make_prefix_samples_from_actions(
    real_sessions,
    PREFIX_LENGTHS,
    source_name="real"
)

print("Real prefix samples:", real_samples.shape)
display(real_samples.head())
display(real_samples["prefix_len"].value_counts().sort_index())

real_samples.to_csv(
    OUT_DIR / "real_f1_prefix_samples_10_60.csv",
    index=False
)

# Cell 10 — Train/test split by user (75/25)
unique_users = sorted(real_samples["user_id"].unique())

rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(unique_users)

n_test = max(1, int(len(unique_users) * 0.25))

test_users  = set(unique_users[:n_test])
train_users = set(unique_users[n_test:])

real_train_samples = real_samples[
    real_samples["user_id"].isin(train_users)
].copy()

real_test_samples = real_samples[
    real_samples["user_id"].isin(test_users)
].copy()

print("Train users:", sorted(train_users))
print("Test users:", sorted(test_users))
print("Real train:", real_train_samples.shape)
print("Real test:", real_test_samples.shape)

display(real_train_samples["prefix_len"].value_counts().sort_index())
display(real_test_samples["prefix_len"].value_counts().sort_index())

# Cell 11 — Load synthetic JSONL file
def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as e:
                print("Bad line:", line_no, e)

    return pd.DataFrame(rows)


synth_raw = load_jsonl(SYNTH_JSONL_PATH)

print("Synthetic raw:", synth_raw.shape)
display(synth_raw.head())
print(synth_raw.columns.tolist())

# Cell 12 — Build synthetic prefix samples
def extract_actions_from_prefix_events(prefix_events):
    actions = []

    if not isinstance(prefix_events, list):
        return actions

    for event in prefix_events:
        if isinstance(event, dict):
            action = event.get("action", None)
            if action is not None:
                actions.append(str(action).strip())
        elif isinstance(event, str):
            actions.append(event.strip())

    return [a for a in actions if a]


synth_rows = []

for _, row in synth_raw.iterrows():
    actions = extract_actions_from_prefix_events(row.get("prefix_events", []))

    if len(actions) == 0:
        continue

    target = row.get("target_no_assist_f1", np.nan)

    try:
        target = float(target)
    except Exception:
        target = np.nan

    if pd.isna(target):
        continue

    prefix_len = int(row.get("prefix_len", len(actions)))

    synth_rows.append({
        "sample_id":       str(row.get("sample_id", "")),
        "user_id":         str(row.get("userId", "synthetic")),
        "task_id":         str(row.get("taskId", "unknown")),
        "prefix_len":      prefix_len,
        "prefix_text":     " ".join(actions[:prefix_len]),
        "target_f1":       target,
        "source":          "synthetic",
        "full_session_len": len(actions),
    })

synth_samples = pd.DataFrame(synth_rows)

# Keep only required prefix lengths
synth_samples = synth_samples[
    synth_samples["prefix_len"].isin(PREFIX_LENGTHS)
].copy()

print("Synthetic prefix samples:", synth_samples.shape)
display(synth_samples.head())
display(synth_samples["prefix_len"].value_counts().sort_index())

synth_samples.to_csv(
    OUT_DIR / "synthetic_f1_prefix_samples_10_60.csv",
    index=False
)

# Cell 13 — Verify prefix counts across all sets
print("Real train prefix counts:")
display(real_train_samples["prefix_len"].value_counts().sort_index())

print("Real test prefix counts:")
display(real_test_samples["prefix_len"].value_counts().sort_index())

print("Synthetic prefix counts:")
display(synth_samples["prefix_len"].value_counts().sort_index())

# Cell 14 — Regression evaluation helpers
def safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return pearsonr(y_true, y_pred)[0]


def evaluate_regression(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pr   = safe_pearson(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    return mae, rmse, pr, r2

# Cell 15 — Baseline model trainer (Ridge / Random Forest)
def train_eval_baseline_model(model_name, train_df, test_df):
    results     = []
    predictions = []

    for L in PREFIX_LENGTHS:
        train_L = train_df[train_df["prefix_len"] == L].copy()
        test_L  = test_df[test_df["prefix_len"] == L].copy()

        if len(train_L) == 0 or len(test_L) == 0:
            print("Skipping:", model_name, "prefix", L)
            continue

        vectorizer = CountVectorizer(
            tokenizer=lambda x: x.split(),
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 3),
            min_df=1
        )

        X_train = vectorizer.fit_transform(train_L["prefix_text"])
        X_test  = vectorizer.transform(test_L["prefix_text"])

        y_train = train_L["target_f1"].astype(float).values
        y_test  = test_L["target_f1"].astype(float).values

        if model_name == "ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=300,
                random_state=RANDOM_SEED,
                min_samples_leaf=2,
                n_jobs=-1
            )
        else:
            raise ValueError(model_name)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae, rmse, pr, r2 = evaluate_regression(y_test, y_pred)

        results.append({
            "model":     model_name,
            "prefix_len": L,
            "mae":       mae,
            "rmse":      rmse,
            "pearson_r": pr,
            "r2":        r2,
            "n_train":   len(train_L),
            "n_test":    len(test_L),
        })

        for true, pred in zip(y_test, y_pred):
            predictions.append({
                "model":        model_name,
                "prefix_len":   L,
                "true_f1":      true,
                "predicted_f1": pred,
            })

    return pd.DataFrame(results), pd.DataFrame(predictions)

# Cell 16 — Run Ridge and Random Forest across all scenarios
scenario_train_sets = {
    "real_only":          real_train_samples,
    "synthetic_only":     synth_samples,
    "real_plus_synthetic": pd.concat(
        [real_train_samples, synth_samples],
        ignore_index=True
    )
}

baseline_results     = []
baseline_predictions = []

for scenario, train_df in scenario_train_sets.items():
    for model_name in ["ridge", "random_forest"]:
        print("\nRunning:", model_name, scenario)

        res_df, pred_df = train_eval_baseline_model(
            model_name=model_name,
            train_df=train_df,
            test_df=real_test_samples
        )

        res_df["scenario"]  = scenario
        pred_df["scenario"] = scenario

        baseline_results.append(res_df)
        baseline_predictions.append(pred_df)

baseline_results_df     = pd.concat(baseline_results, ignore_index=True)
baseline_predictions_df = pd.concat(baseline_predictions, ignore_index=True)

display(baseline_results_df)

baseline_results_df.to_csv(
    OUT_DIR / "ridge_random_forest_results_by_prefix.csv",
    index=False
)

baseline_predictions_df.to_csv(
    OUT_DIR / "ridge_random_forest_predictions.csv",
    index=False
)

# Cell 17 — Build vocabulary for LSTM
all_tokens = []

for df_part in [real_train_samples, real_test_samples, synth_samples]:
    for text in df_part["prefix_text"].astype(str):
        all_tokens.extend(text.split())

vocab      = ["<PAD>", "<UNK>"] + sorted(set(all_tokens))
token2id   = {tok: i for i, tok in enumerate(vocab)}

PAD_IDX    = token2id["<PAD>"]
UNK_IDX    = token2id["<UNK>"]
VOCAB_SIZE = len(vocab)

MAX_LEN    = 80
BATCH_SIZE = 64

print("Vocab size:", VOCAB_SIZE)

# Cell 18 — Dataset and LSTM model definition
class F1PrefixDataset(Dataset):
    def __init__(self, df, max_len=80):
        self.df      = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def encode_prefix(self, prefix_text):
        tokens = str(prefix_text).split()
        ids    = [token2id.get(tok, UNK_IDX) for tok in tokens]
        ids    = ids[-self.max_len:]
        length = max(1, len(ids))

        while len(ids) < self.max_len:
            ids.append(PAD_IDX)

        return ids, length

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x, length = self.encode_prefix(row["prefix_text"])

        return {
            "x":      torch.tensor(x, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "y":      torch.tensor(float(row["target_f1"]), dtype=torch.float32),
        }


def make_f1_loader(df, shuffle=False):
    return DataLoader(
        F1PrefixDataset(df, max_len=MAX_LEN),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )


class LSTMF1Regressor(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_idx,
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.30
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, lengths):
        emb = self.embedding(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (h, _) = self.lstm(packed)

        pred = self.regressor(h[-1]).squeeze(1)

        return torch.clamp(pred, 0.0, 1.0)
    

# Cell 19 — LSTM evaluation and training loop
def evaluate_lstm(model, loader):
    model.eval()

    y_true      = []
    y_pred      = []
    criterion   = nn.MSELoss()
    total_loss  = 0
    total_items = 0

    with torch.no_grad():
        for batch in loader:
            x       = batch["x"].to(DEVICE)
            lengths = batch["length"].to(DEVICE)
            y       = batch["y"].to(DEVICE)

            pred = model(x, lengths)
            loss = criterion(pred, y)

            bs           = y.size(0)
            total_loss  += loss.item() * bs
            total_items += bs

            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    mae, rmse, pr, r2 = evaluate_regression(y_true, y_pred)

    return total_loss / max(total_items, 1), mae, rmse, pr, r2, y_true, y_pred


def train_lstm_for_prefix(
    train_df,
    test_df,
    prefix_len,
    experiment_name,
    epochs=40,
    patience=8
):
    train_L = train_df[train_df["prefix_len"] == prefix_len].copy()
    test_L  = test_df[test_df["prefix_len"] == prefix_len].copy()

    if len(train_L) == 0 or len(test_L) == 0:
        print("Skipping LSTM:", experiment_name, prefix_len)
        return None

    train_loader = make_f1_loader(train_L, shuffle=True)
    test_loader  = make_f1_loader(test_L, shuffle=False)

    model = LSTMF1Regressor(VOCAB_SIZE, PAD_IDX).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    best_mae   = float("inf")
    best_state = None
    no_improve = 0
    history    = []

    for epoch in range(1, epochs + 1):
        model.train()

        total_train_loss  = 0
        total_train_items = 0

        for batch in train_loader:
            x       = batch["x"].to(DEVICE)
            lengths = batch["length"].to(DEVICE)
            y       = batch["y"].to(DEVICE)

            optimizer.zero_grad()
            pred = model(x, lengths)
            loss = criterion(pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs                = y.size(0)
            total_train_loss  += loss.item() * bs
            total_train_items += bs

        train_loss = total_train_loss / max(total_train_items, 1)

        val_loss, mae, rmse, pr, r2, _, _ = evaluate_lstm(model, test_loader)

        history.append({
            "epoch":     epoch,
            "train_loss": train_loss,
            "val_loss":  val_loss,
            "mae":       mae,
            "rmse":      rmse,
            "pearson_r": pr,
            "r2":        r2
        })

        print(
            f"{experiment_name} | prefix={prefix_len} | "
            f"epoch={epoch:02d} | mae={mae:.4f} | "
            f"rmse={rmse:.4f} | r={pr:.4f}"
        )

        if mae < best_mae:
            best_mae   = mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        OUT_DIR / f"history_{experiment_name}_prefix_{prefix_len}.csv",
        index=False
    )

    return model, test_L


# Cell 20 — Run LSTM across all scenarios and prefix lengths
lstm_results     = []
lstm_predictions = []

for scenario, train_df in scenario_train_sets.items():
    for L in PREFIX_LENGTHS:
        experiment_name = f"lstm_regressor_{scenario}"

        print("\n==============================")
        print(experiment_name, "prefix", L)
        print("==============================")

        out = train_lstm_for_prefix(
            train_df=train_df,
            test_df=real_test_samples,
            prefix_len=L,
            experiment_name=experiment_name,
            epochs=40,
            patience=8
        )

        if out is None:
            continue

        model, test_L = out

        test_loader = make_f1_loader(test_L, shuffle=False)

        loss, mae, rmse, pr, r2, y_true, y_pred = evaluate_lstm(model, test_loader)

        lstm_results.append({
            "model":     "lstm_regressor",
            "scenario":  scenario,
            "prefix_len": L,
            "mae":       mae,
            "rmse":      rmse,
            "pearson_r": pr,
            "r2":        r2,
            "n_train":   len(train_df[train_df["prefix_len"] == L]),
            "n_test":    len(test_L)
        })

        for t, p in zip(y_true, y_pred):
            lstm_predictions.append({
                "model":        "lstm_regressor",
                "scenario":     scenario,
                "prefix_len":   L,
                "true_f1":      t,
                "predicted_f1": p
            })

lstm_results_df     = pd.DataFrame(lstm_results)
lstm_predictions_df = pd.DataFrame(lstm_predictions)

display(lstm_results_df)

lstm_results_df.to_csv(
    OUT_DIR / "lstm_results_by_prefix.csv",
    index=False
)

lstm_predictions_df.to_csv(
    OUT_DIR / "lstm_predictions.csv",
    index=False
)


# Cell 21 — Combine all results and save
all_results = pd.concat(
    [baseline_results_df, lstm_results_df],
    ignore_index=True
)

all_results = all_results.sort_values(["prefix_len", "mae"])

display(all_results)

all_results.to_csv(
    OUT_DIR / "all_f1_results_by_prefix.csv",
    index=False
)


# Cell 22 — Aggregate results (mean over prefix lengths)
agg_results = (
    all_results
    .groupby(["model", "scenario"])
    .agg(
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        pearson_r=("pearson_r", "mean"),
        r2=("r2", "mean"),
        n_train=("n_train", "sum"),
        n_test=("n_test", "sum")
    )
    .reset_index()
    .sort_values("mae")
)

display(agg_results)

agg_results.to_csv(
    OUT_DIR / "aggregate_f1_results.csv",
    index=False
)


# Cell 23 — Plot MAE vs Prefix Length for each model
for model_name in ["ridge", "random_forest", "lstm_regressor"]:
    sub = all_results[all_results["model"] == model_name]

    plt.figure(figsize=(12, 7))

    for scenario in ["real_only", "synthetic_only", "real_plus_synthetic"]:
        s = sub[sub["scenario"] == scenario].sort_values("prefix_len")

        plt.plot(
            s["prefix_len"],
            s["mae"],
            marker="o",
            linewidth=2,
            label=scenario
        )

    plt.xlabel("Prefix Length")
    plt.ylabel("MAE")
    plt.title(f"No-Assistance F1 MAE vs Prefix Length - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = PLOTS_DIR / f"mae_vs_prefix_{model_name}.png"
    plt.savefig(path, dpi=300)
    plt.show()

    print("Saved:", path)


# Cell 24 — Summary of output files
print("Done.\n")
print("Main output files:")
print(OUT_DIR / "real_f1_prefix_samples_10_60.csv")
print(OUT_DIR / "synthetic_f1_prefix_samples_10_60.csv")
print(OUT_DIR / "ridge_random_forest_results_by_prefix.csv")
print(OUT_DIR / "lstm_results_by_prefix.csv")
print(OUT_DIR / "all_f1_results_by_prefix.csv")
print(OUT_DIR / "aggregate_f1_results.csv")

