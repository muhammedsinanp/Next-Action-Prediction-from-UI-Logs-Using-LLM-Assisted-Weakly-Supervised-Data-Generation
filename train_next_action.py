# =============================================================================
# Next-Action Prediction — Full Training Pipeline
# Models: Markov, N-gram, LSTM, Transformer
# Scenarios: Real Only | Synthetic Only | Real + Synthetic
# Metrics: Accuracy, Top-3, Weighted P/R/F1, Macro P/R/F1
# =============================================================================

import ast
import json
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ── Paths ─────────────────────────────────────────────────────────────────────
REAL_PATH     = "real_data.csv"
SYNTH_PATH    = "llm_synthetic_data.csv"

OUT_DIR     = Path("outputs");      OUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR   = OUT_DIR / "plots";    PLOTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR = OUT_DIR / "reports";  REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR  = OUT_DIR / "models";   MODELS_DIR.mkdir(exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
EPOCHS     = 50
PATIENCE   = 7
LR         = 1e-3
BATCH_SIZE = 64

LSTM_CFG = dict(embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.3)
TF_CFG   = dict(embed_dim=64, num_heads=4,   num_layers=2, dropout=0.1)

KNOWN_ACTIONS = {
    "PROCESS_START", "PROCESS_END",
    "STEP_MENTION_SUGGESTION", "STEP_METION", "STEP_MENTION",
    "STEP_ENTITIES", "STEP_RELATION_SUGGESTION", "STEP_RELATIONS",
    "TOKEN_SELECTED", "TOKEN_DESELECTED", "TOKEN_SELECTED_MULTI",
    "MENTION_CREATED", "MENTION_DELETED", "MENTION_SELECTED",
    "MENTION_DESELECTED", "MENTION_SUGGESTION_ACCEPTED",
    "MENTION_SUGGESTION_REJECTED", "MENTION_SUGGESTION_BOUNDS_UPDATED",
    "MENTION_TYPE_UPDATED", "MENTION_BOUNDS_UPDATED",
    "RELATION_CREATED", "RELATION_DELETED", "RELATION_SELECTED",
    "RELATION_DESELECTED", "RELATION_MARKED", "RELATION_UNMARKED",
    "RELATION_SUGGESTION_ACCEPTED", "RELATION_SUGGESTION_REJECTED",
    "RELATION_SUGGESTION_MARKED", "RELATION_TYPE_UPDATED",
    "ENTITY_GROUPED", "ENTITY_REMOVED", "UNDO_ACTION",
}


# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================

def parse_prefix_to_str(prefix_str):
    """Convert a prefix (JSON list or space-separated string) → space-separated string."""
    try:
        tokens = ast.literal_eval(str(prefix_str))
        if isinstance(tokens, list):
            return " ".join(str(t) for t in tokens)
    except Exception:
        pass
    return str(prefix_str).strip()


def load_and_clean(path, name):
    df = pd.read_csv(path)
    df["prefix"]      = df["prefix"].apply(parse_prefix_to_str)
    df["next_action"] = df["next_action"].astype(str)
    before = len(df)
    df = df[df["next_action"].isin(KNOWN_ACTIONS)].copy()
    df = df[df["prefix"].str.strip().str.len() > 0].copy()
    df["prefix_len"] = df["prefix"].str.split().str.len()
    print(f"{name}: {before} → {len(df)} rows after cleaning")
    return df.reset_index(drop=True)


def get_split(df, sp):
    return df[df["split"] == sp].reset_index(drop=True)


real  = load_and_clean(REAL_PATH,  "real")
synth = load_and_clean(SYNTH_PATH, "synth")

# Assign unique integer user_ids to synthetic rows (avoids overlap with real)
synth["user_id"] = range(100, 100 + len(synth))
synth["split"]   = "train"          # all synthetic rows used only for training

real_train = get_split(real, "train")
real_val   = get_split(real, "val")
real_test  = get_split(real, "test")

# Quick leakage check
real_test_prefixes = set(real_test["prefix"].str.strip())
leak_mask = synth["prefix"].str.strip().isin(real_test_prefixes)
if leak_mask.sum():
    print(f"Removing {leak_mask.sum()} synth rows that leak into real_test.")
    synth = synth[~leak_mask].reset_index(drop=True)

print(f"\nreal_train={len(real_train)}  real_val={len(real_val)}  real_test={len(real_test)}")
print(f"synth_train={len(synth)}")

# ── Vocabulary ────────────────────────────────────────────────────────────────
MAX_LEN = int(max(real["prefix_len"].max(), synth["prefix_len"].max()))

all_tokens = sorted({
    tok
    for seq in pd.concat([real["prefix"], synth["prefix"]]).dropna()
    for tok in seq.split()
})
all_labels = sorted(set(
    pd.concat([real["next_action"], synth["next_action"]]).dropna()
))

PAD, UNK  = "<PAD>", "<UNK>"
vocab     = [PAD, UNK] + all_tokens
token2idx = {t: i for i, t in enumerate(vocab)}
PAD_IDX   = token2idx[PAD]

label2idx = {a: i for i, a in enumerate(all_labels)}
idx2label = {i: a for a, i in label2idx.items()}

VOCAB_SIZE  = len(token2idx)
NUM_CLASSES = len(label2idx)

print(f"\nVOCAB_SIZE={VOCAB_SIZE}  NUM_CLASSES={NUM_CLASSES}  MAX_LEN={MAX_LEN}")

# ── Training scenarios ────────────────────────────────────────────────────────
scenarios = {
    "real":            (real_train,                                       real_val, real_test),
    "synthetic":       (synth,                                            real_val, real_test),
    "real+synthetic":  (pd.concat([real_train, synth], ignore_index=True), real_val, real_test),
}


# =============================================================================
# 2. PYTORCH DATASET & LOADERS
# =============================================================================

def encode_prefix(prefix_str):
    tokens = prefix_str.strip().split()
    ids    = [token2idx.get(t, token2idx[UNK]) for t in tokens]
    ids    = ids + [PAD_IDX] * (MAX_LEN - len(ids))
    return ids[:MAX_LEN]


class ActionDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(
            [encode_prefix(row["prefix"]) for _, row in df.iterrows()],
            dtype=torch.long,
        )
        self.y = torch.tensor(
            [label2idx[row["next_action"]] for _, row in df.iterrows()],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(train_df, val_df, test_df, bs=BATCH_SIZE):
    return (
        DataLoader(ActionDataset(train_df), batch_size=bs, shuffle=True,
                   worker_init_fn=lambda _: np.random.seed(SEED)),
        DataLoader(ActionDataset(val_df),   batch_size=bs, shuffle=False),
        DataLoader(ActionDataset(test_df),  batch_size=bs, shuffle=False),
    )


# =============================================================================
# 3. BASELINE MODELS (Markov & N-gram)
# =============================================================================

class MarkovModel:
    """First-order Markov: P(next | last_action_in_prefix)."""

    def __init__(self):
        self.trans         = defaultdict(Counter)
        self.global_counts = Counter()

    def fit(self, df):
        for _, row in df.iterrows():
            tokens = row["prefix"].strip().split()
            last   = tokens[-1] if tokens else PAD
            nxt    = row["next_action"]
            self.trans[last][nxt] += 1
            self.global_counts[nxt] += 1
        return self

    def predict_one(self, prefix_str):
        tokens = prefix_str.strip().split()
        last   = tokens[-1] if tokens else PAD
        if last in self.trans:
            return self.trans[last].most_common(1)[0][0]
        return self.global_counts.most_common(1)[0][0]

    def predict_topk(self, prefix_str, k=3):
        tokens = prefix_str.strip().split()
        last   = tokens[-1] if tokens else PAD
        pool   = self.trans.get(last, self.global_counts)
        preds  = [a for a, _ in pool.most_common(k)]
        fallback = self.global_counts.most_common(1)[0][0]
        while len(preds) < k:
            preds.append(fallback)
        return preds

    def predict(self, df):
        return [self.predict_one(r["prefix"]) for _, r in df.iterrows()]

    def predict_top3(self, df):
        return [self.predict_topk(r["prefix"], k=3) for _, r in df.iterrows()]


class NGramModel:
    """N-gram with back-off to shorter contexts."""

    def __init__(self, n=3):
        self.n      = n
        self.counts = defaultdict(Counter)

    def fit(self, df):
        for _, row in df.iterrows():
            tokens  = row["prefix"].strip().split()
            label   = row["next_action"]
            context = tuple(tokens[-(self.n - 1):]) if self.n > 1 else ()
            self.counts[context][label] += 1
        return self

    def _global(self):
        g = Counter()
        for c in self.counts.values():
            g.update(c)
        return g

    def predict_topk(self, prefix_str, k=3):
        tokens = prefix_str.strip().split()
        for n in range(self.n, 0, -1):
            ctx = tuple(tokens[-(n - 1):]) if n > 1 else ()
            if ctx in self.counts:
                preds = [a for a, _ in self.counts[ctx].most_common(k)]
                fallback = self._global().most_common(1)[0][0]
                while len(preds) < k:
                    preds.append(fallback)
                return preds
        fallback = self._global().most_common(1)[0][0]
        return [fallback] * k

    def predict_one(self, prefix_str):
        return self.predict_topk(prefix_str, k=1)[0]

    def predict(self, df):
        return [self.predict_one(r["prefix"]) for _, r in df.iterrows()]

    def predict_top3(self, df):
        return [self.predict_topk(r["prefix"], k=3) for _, r in df.iterrows()]


# =============================================================================
# 4. NEURAL MODELS (LSTM & Transformer)
# =============================================================================

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 num_classes, dropout=0.3, pad_idx=PAD_IDX):
        super().__init__()
        self.pad_idx   = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                 batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0.0)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        lengths = (x != self.pad_idx).sum(dim=1).clamp(min=1).cpu()
        emb     = self.dropout(self.embedding(x))
        packed  = nn.utils.rnn.pack_padded_sequence(
                      emb, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(self.dropout(h[-1]))


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 num_classes, max_len=MAX_LEN, dropout=0.1, pad_idx=PAD_IDX):
        super().__init__()
        self.pad_idx   = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        enc_layer      = nn.TransformerEncoderLayer(
                             d_model=embed_dim, nhead=num_heads,
                             dim_feedforward=embed_dim * 4,
                             dropout=dropout, batch_first=True)
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, L      = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        emb       = self.dropout(self.embedding(x) + self.pos_embed(positions))
        pad_mask  = (x == self.pad_idx)
        out       = self.encoder(emb, src_key_padding_mask=pad_mask)
        mask      = (~pad_mask).unsqueeze(-1).float()
        pooled    = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.dropout(pooled))


# =============================================================================
# 5. TRAINING & EVALUATION UTILITIES
# =============================================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_neural(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    top3_correct = 0
    total = 0
    for X, y in loader:
        logits = model(X.to(DEVICE))
        preds  = logits.argmax(dim=1).cpu().tolist()
        topk   = torch.topk(logits, k=min(3, logits.size(1)), dim=1).indices.cpu()
        top3_correct += topk.eq(y.unsqueeze(1)).any(dim=1).sum().item()
        total += len(y)
        all_preds  += preds
        all_labels += y.tolist()
    top3 = top3_correct / total
    return all_labels, all_preds, top3


def train_neural(model, train_loader, val_loader,
                 epochs=EPOCHS, lr=LR, patience=PATIENCE):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", patience=2, factor=0.5, verbose=False)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_state  = None
    wait        = 0

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_labels, val_preds, _ = eval_neural(model, val_loader)
        val_f1 = precision_recall_fscore_support(
            val_labels, val_preds, average="weighted", zero_division=0
        )[2]
        scheduler.step(val_f1)

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"  epoch {epoch:02d} | loss={tr_loss:.4f} | "
              f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def compute_all_metrics(y_true, y_pred, y_top3=None):
    """Return a dict with every requested metric."""
    acc = accuracy_score(y_true, y_pred)

    wp, wr, wf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    if y_top3 is not None:
        top3 = np.mean([t in preds for t, preds in zip(y_true, y_top3)])
    else:
        top3 = None

    return dict(
        accuracy=round(acc, 4),
        top3=round(top3, 4) if top3 is not None else None,
        weighted_precision=round(wp, 4),
        weighted_recall=round(wr, 4),
        weighted_f1=round(wf, 4),
        macro_precision=round(mp, 4),
        macro_recall=round(mr, 4),
        macro_f1=round(mf, 4),
    )


# =============================================================================
# 6. MAIN TRAINING LOOP
# =============================================================================

all_results = {}   # key: "{scenario}|{model}"  →  metrics + preds/labels

for scenario, (tr, va, te) in scenarios.items():
    print(f"\n{'='*65}")
    print(f"SCENARIO: {scenario.upper()}  "
          f"(train={len(tr)}  val={len(va)}  test={len(te)})")
    print(f"{'='*65}")

    # ── Markov ────────────────────────────────────────────────────────────────
    print("\n[Markov]")
    markov = MarkovModel().fit(tr)
    y_pred_m  = markov.predict(te)
    y_top3_m  = markov.predict_top3(te)
    y_true    = te["next_action"].tolist()
    metrics_m = compute_all_metrics(y_true, y_pred_m, y_top3_m)
    all_results[f"{scenario}|markov"] = dict(**metrics_m, preds=y_pred_m,
                                              labels=y_true, top3_preds=y_top3_m)
    print(f"  acc={metrics_m['accuracy']:.4f}  "
          f"top3={metrics_m['top3']:.4f}  "
          f"w-f1={metrics_m['weighted_f1']:.4f}  "
          f"mac-f1={metrics_m['macro_f1']:.4f}")

    # ── Best N-gram (k=3 with back-off) ───────────────────────────────────────
    print("\n[N-gram (n=3, back-off)]")
    ngram = NGramModel(n=3).fit(tr)
    y_pred_ng = ngram.predict(te)
    y_top3_ng = ngram.predict_top3(te)
    metrics_ng = compute_all_metrics(y_true, y_pred_ng, y_top3_ng)
    all_results[f"{scenario}|ngram"] = dict(**metrics_ng, preds=y_pred_ng,
                                             labels=y_true, top3_preds=y_top3_ng)
    print(f"  acc={metrics_ng['accuracy']:.4f}  "
          f"top3={metrics_ng['top3']:.4f}  "
          f"w-f1={metrics_ng['weighted_f1']:.4f}  "
          f"mac-f1={metrics_ng['macro_f1']:.4f}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_loaders(tr, va, te)

    # ── LSTM ──────────────────────────────────────────────────────────────────
    print("\n[LSTM]")
    torch.manual_seed(SEED)
    lstm = LSTMModel(
        vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES, **LSTM_CFG
    ).to(DEVICE)
    lstm = train_neural(lstm, train_loader, val_loader)
    y_true_ids, y_pred_ids, top3_lstm = eval_neural(lstm, test_loader)
    y_true_l = [idx2label[i] for i in y_true_ids]
    y_pred_l = [idx2label[i] for i in y_pred_ids]
    metrics_l = compute_all_metrics(y_true_l, y_pred_l)
    metrics_l["top3"] = round(top3_lstm, 4)
    all_results[f"{scenario}|lstm"] = dict(**metrics_l, preds=y_pred_l,
                                            labels=y_true_l)
    print(f"  acc={metrics_l['accuracy']:.4f}  "
          f"top3={metrics_l['top3']:.4f}  "
          f"w-f1={metrics_l['weighted_f1']:.4f}  "
          f"mac-f1={metrics_l['macro_f1']:.4f}")
    torch.save(lstm.state_dict(), MODELS_DIR / f"lstm_{scenario}.pt")

    # ── Transformer ───────────────────────────────────────────────────────────
    print("\n[Transformer]")
    torch.manual_seed(SEED)
    tf = TransformerModel(
        vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES,
        max_len=MAX_LEN, **TF_CFG
    ).to(DEVICE)
    tf = train_neural(tf, train_loader, val_loader)
    y_true_ids, y_pred_ids, top3_tf = eval_neural(tf, test_loader)
    y_true_t = [idx2label[i] for i in y_true_ids]
    y_pred_t = [idx2label[i] for i in y_pred_ids]
    metrics_t = compute_all_metrics(y_true_t, y_pred_t)
    metrics_t["top3"] = round(top3_tf, 4)
    all_results[f"{scenario}|transformer"] = dict(**metrics_t, preds=y_pred_t,
                                                   labels=y_true_t)
    print(f"  acc={metrics_t['accuracy']:.4f}  "
          f"top3={metrics_t['top3']:.4f}  "
          f"w-f1={metrics_t['weighted_f1']:.4f}  "
          f"mac-f1={metrics_t['macro_f1']:.4f}")
    torch.save(tf.state_dict(), MODELS_DIR / f"transformer_{scenario}.pt")


# =============================================================================
# 7. RESULTS TABLE  (all metrics)
# =============================================================================

SCENARIO_LABEL = {
    "real":           "Real Only",
    "synthetic":      "Synthetic Only",
    "real+synthetic": "Real + Synthetic",
}
MODEL_LABEL = {
    "markov":      "Markov",
    "ngram":       "Best N-gram",
    "lstm":        "LSTM",
    "transformer": "Transformer",
}

rows = []
for key, res in all_results.items():
    sc, mdl = key.split("|")
    rows.append({
        "Model":              MODEL_LABEL.get(mdl, mdl),
        "Training Setting":   SCENARIO_LABEL.get(sc, sc),
        "Accuracy":           res["accuracy"],
        "Top-3":              res["top3"],
        "Weighted Precision": res["weighted_precision"],
        "Weighted Recall":    res["weighted_recall"],
        "Weighted F1":        res["weighted_f1"],
        "Macro Precision":    res["macro_precision"],
        "Macro Recall":       res["macro_recall"],
        "Macro F1":           res["macro_f1"],
    })

results_df = pd.DataFrame(rows)

# Sort: Model order then scenario order
model_order    = {"Markov": 0, "Best N-gram": 1, "LSTM": 2, "Transformer": 3}
scenario_order = {"Real Only": 0, "Synthetic Only": 1, "Real + Synthetic": 2}
results_df["_mo"] = results_df["Model"].map(model_order)
results_df["_so"] = results_df["Training Setting"].map(scenario_order)
results_df = (results_df.sort_values(["_mo", "_so"])
                        .drop(columns=["_mo", "_so"])
                        .reset_index(drop=True))

print("\n\n" + "=" * 90)
print("FULL RESULTS TABLE")
print("=" * 90)
print(results_df.to_string(index=False))

results_df.to_csv(OUT_DIR / "all_metrics_results.csv", index=False)
print(f"\nSaved → {OUT_DIR / 'all_metrics_results.csv'}")


# =============================================================================
# 8. PER-CLASS CLASSIFICATION REPORTS
# =============================================================================

for key, res in all_results.items():
    sc, mdl = key.split("|")
    report = classification_report(
        res["labels"], res["preds"], output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(
        REPORTS_DIR / f"cls_report_{mdl}_{sc}.csv"
    )
print(f"Per-class reports saved to {REPORTS_DIR}")


# =============================================================================
# 9. VISUALISATIONS
# =============================================================================

# 9a. Bar charts: Accuracy & Weighted F1
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
colors     = ["steelblue", "darkorange", "seagreen"]
col_order  = ["Real Only", "Synthetic Only", "Real + Synthetic"]

for ax, metric in zip(axes, ["Accuracy", "Weighted F1"]):
    pivot = results_df.pivot(
        index="Model", columns="Training Setting", values=metric
    )[col_order]
    pivot.plot(kind="bar", ax=ax, color=colors, alpha=0.88, edgecolor="white")
    ax.set_title(f"{metric} — All Models × Scenarios", fontsize=13)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Scenario")
    ax.grid(axis="y", alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "accuracy_f1_bars.png", dpi=150, bbox_inches="tight")
plt.show()

# 9b. Confusion matrices — best model per scenario
best_by_scenario = {}
for sc_label in ["Real Only", "Synthetic Only", "Real + Synthetic"]:
    cands = {k: v for k, v in all_results.items()
             if SCENARIO_LABEL.get(k.split("|")[0]) == sc_label}
    best_key = max(cands, key=lambda k: cands[k]["accuracy"])
    best_by_scenario[sc_label] = (best_key, cands[best_key])

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
labels_list = [idx2label[i] for i in range(NUM_CLASSES)]

for ax, (sc_label, (key, res)) in zip(axes, best_by_scenario.items()):
    cm      = confusion_matrix(res["labels"], res["preds"], labels=labels_list)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cm_norm, ax=ax, cmap="Blues", annot=False,
                xticklabels=labels_list, yticklabels=labels_list,
                linewidths=0.3, vmin=0, vmax=1)
    model_name = MODEL_LABEL.get(key.split("|")[1], key.split("|")[1])
    ax.set_title(f"{sc_label}\n{model_name} (row-normalised)", fontsize=10)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.tick_params(axis="both", labelsize=6)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDone. All outputs saved to:", OUT_DIR)