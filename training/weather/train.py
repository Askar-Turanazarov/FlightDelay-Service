# Step 5. Trains the weather model: the project's FlightFeatureEngineer + CatBoost on the 13 flight columns and eight
# weather categories, on all of 2015 except 73 whole hidden days, then scores it on those days.
# Weather is what was actually observed, so the test score is the upper limit a perfect forecast could give.
# Environment: CTR (category combinations, 2 = the shipped model, 3 = the heavier one), DEVICE (GPU or CPU),
# ITERATIONS, ROWS (a small sample for a smoke run; nothing is saved then).
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
from common import BASE_FEATURES, CAT_FEATURES, FLIGHTS_WEATHER, MODEL, PREPARED, WEATHER, WORK, hidden_days, is_hidden, sys_path  # noqa: E402

sys_path()
from catboost import CatBoostClassifier, Pool  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from src.feature_engineering import FlightFeatureEngineer  # noqa: E402

ITERATIONS = int(os.environ.get("ITERATIONS", 2500))
ROWS = int(os.environ.get("ROWS", 0))
CTR = int(os.environ.get("CTR", 2))
DEVICE = os.environ.get("DEVICE", "GPU")
name = f"weather_ctr{CTR}" + ("_smoke" if ROWS else "")
CATS = CAT_FEATURES + WEATHER
FEATURES = BASE_FEATURES + WEATHER


def score(y, p):
    y = np.asarray(y)
    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    at = int(np.searchsorted(tp, np.ceil(0.82 * int(y.sum()))))
    return {"n": int(len(y)), "delay_share": round(float(y.mean()), 4), "auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "precision_at_recall_82": round(float(tp[at] / (at + 1)), 4)}


def by_month(frame, p):
    return {int(m): round(float(roc_auc_score(g.target, p[g.pos])), 4)
            for m, g in frame.assign(pos=np.arange(len(frame))).groupby("MONTH")}


def curve(pipe, X, y):
    clf = pipe.named_steps["clf"]
    pool_ = Pool(pipe.named_steps["feature_eng"].transform(X), y, cat_features=CATS)
    vals = clf.eval_metrics(pool_, ["AUC"], eval_period=500)["AUC"]
    trees = list(range(1, clf.tree_count_ + 1, 500)) + [clf.tree_count_]
    return {str(t): round(float(v), 4) for t, v in zip(trees, vals)}


df = pd.read_pickle(PREPARED)
# Weather is read as categories: the 1 GB file then fits in a few GB of memory.
w = pd.read_csv(FLIGHTS_WEATHER, usecols=["ARRIVAL_DELAY", "MONTH", "DAY", "SCHEDULED_DEPARTURE"] + WEATHER,
                dtype=dict.fromkeys(WEATHER, "category"))
w = w[w.ARRIVAL_DELAY.notna()].reset_index(drop=True)
# prepared.pkl is flights.csv without cancelled rows, in file order, so rows line up one to one. Checked, not assumed.
for c in ("MONTH", "DAY", "SCHEDULED_DEPARTURE"):
    if len(w) != len(df) or not np.array_equal(w[c].to_numpy(), df[c].to_numpy()):
        raise ValueError(f"rows do not line up on {c}")
for c in WEATHER:
    df[c] = w[c]
del w

hidden = is_hidden(df, hidden_days(df))
tr, te = df[~hidden], df[hidden].reset_index(drop=True)
if ROWS:
    tr = tr.sample(n=ROWS, random_state=42)
for c in WEATHER:
    tr, te = tr.assign(**{c: tr[c].astype(str)}), te.assign(**{c: te[c].astype(str)})
print(f"[{name}] train {len(tr):,}, test {len(te):,}")

train_dir = os.path.join(WORK, "catboost_info", name)
os.makedirs(train_dir, exist_ok=True)
pipe = Pipeline([
    ("feature_eng", FlightFeatureEngineer()),
    ("clf", CatBoostClassifier(
        iterations=ITERATIONS, learning_rate=0.03, depth=7, l2_leaf_reg=12, scale_pos_weight=4.2,
        task_type=DEVICE, **({"devices": "0"} if DEVICE == "GPU" else {}), bootstrap_type="Bernoulli", subsample=0.8,
        grow_policy="SymmetricTree", eval_metric="AUC", random_seed=42, random_strength=2,
        verbose=500, cat_features=CATS, max_ctr_complexity=CTR, train_dir=train_dir)),
])
t1 = time.time()
pipe.fit(tr[FEATURES], tr.target)
result = {"name": name, "train_rows": len(tr), "train_seconds": int(time.time() - t1)}

sample = tr.sample(n=min(1_000_000, len(tr)), random_state=7)
result["train_score"] = score(sample.target, pipe.predict_proba(sample[FEATURES])[:, 1])
p = pipe.predict_proba(te[FEATURES])[:, 1]
result["test"] = score(te.target, p)
result["by_month"] = by_month(te, p)
result["curve_test"] = curve(pipe, te[FEATURES], te.target)
result["curve_train"] = curve(pipe, sample[FEATURES], sample.target)
clf = pipe.named_steps["clf"]
imp = pd.Series(clf.get_feature_importance(), index=clf.feature_names_).sort_values(ascending=False)
result["importance"] = {k: round(float(v), 2) for k, v in imp.items()}
result["importance_weather_total"] = round(float(imp[WEATHER].sum()), 2)

if not ROWS:
    joblib.dump(pipe, MODEL, compress=3)
    result["pkl_mb"] = round(os.path.getsize(MODEL) / 2**20, 1)
try:
    import psutil
    result["peak_ram_gb"] = round(psutil.Process().memory_info().peak_wset / 2**30, 1)
except Exception:
    pass
with open(os.path.join(WORK, f"{name}.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(json.dumps({k: result[k] for k in ("name", "train_seconds", "train_score", "test", "importance_weather_total")}, indent=1))
