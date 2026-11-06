# Step 6. Calibrates the trained model (train.py) for the app: predictions on the 73 hidden days, an isotonic
# calibrator fitted on them, the threshold and risk bands, the model card and a load test in a clean process.
# The calibrator is stored as two arrays for np.interp, so the app needs no extra pickle.
# Output in WORK: calibration_weather.json (copy to models/) and model_card.json; the model is WORK/model_pipeline_weather.pkl.
import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
from common import BASE_FEATURES, FLIGHTS_WEATHER, MODEL, PREPARED, ROOT, WEATHER, WORK, hidden_days, is_hidden, sys_path  # noqa: E402

sys_path()
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402


def ece(y, p, bins=15):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    i = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    return float(sum((i == b).mean() * abs(y[i == b].mean() - p[i == b].mean()) for b in range(bins) if (i == b).any()))


def at(y, flag, k=None):
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    r = {"recall": tp / max(int(y.sum()), 1), "precision": tp / max(tp + fp, 1), "alert_share": float(flag.mean())}
    if k is not None:
        r["value_per_1000"] = 1000 * (k * tp - fp) / len(y)
    return r


FEATURES = BASE_FEATURES + WEATHER
THRESHOLD = 0.20                    # a miss costs four false alarms: 1 / (1 + 4)
BANDS = [0.12, 0.20, 0.35]          # low | usual | high | critical, on calibrated probability
CTR = int(os.environ.get("CTR", 2))  # as in train.py, for the model card
PREDS = os.path.join(WORK, "preds_hidden_days.npz")
CALIBRATION = os.path.join(WORK, "calibration_weather.json")

if not os.path.exists(PREDS):
    df = pd.read_pickle(PREPARED)
    w = pd.read_csv(FLIGHTS_WEATHER, usecols=["ARRIVAL_DELAY"] + WEATHER, dtype=dict.fromkeys(WEATHER, "category"))
    w = w[w.ARRIVAL_DELAY.notna()].reset_index(drop=True)
    if len(w) != len(df):
        raise ValueError("rows do not line up")
    hidden = is_hidden(df, hidden_days(df))
    te = df[hidden].reset_index(drop=True)
    for c in WEATHER:
        te[c] = w.loc[hidden, c].astype(str).to_numpy()
    del df, w
    pipe = joblib.load(MODEL)
    p = pipe.predict_proba(te[FEATURES])[:, 1]
    np.savez_compressed(PREDS, p=p.astype("float32"), y=te.target.to_numpy(), day=(te.MONTH * 100 + te.DAY).to_numpy().astype("int32"))
    del pipe, te

n = np.load(PREDS)
p, y, day = n["p"].astype("float64"), n["y"], n["day"]
card = {"model": "CatBoost, day + weather, trained on 292 days of 2015", "test": "73 hidden days, 1,116,989 flights",
        "auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4),
        "delay_share": round(float(y.mean()), 4), "raw_mean_p": round(float(p.mean()), 4), "raw_ece": round(ece(y, p), 4)}

# Honest numbers first: fit on half of the days, judge on the other half, ten times.
uniq, split = np.unique(day), []
for seed in range(10):
    fit = np.isin(day, np.random.default_rng(seed).permutation(uniq)[: len(uniq) // 2])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p[fit], y[fit])
    c = iso.predict(p[~fit])
    r = at(y[~fit], c >= THRESHOLD, 4)
    r.update(ece=ece(y[~fit], c), brier=brier_score_loss(y[~fit], c))
    split.append(r)
card["held_out_days"] = {k: round(float(np.mean([s[k] for s in split])), 4) for k in split[0]}
card["held_out_days"]["recall_sd"] = round(float(np.std([s["recall"] for s in split])), 4)

# The shipped calibrator uses all 73 days.
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
xs, ys = iso.X_thresholds_, iso.y_thresholds_
c = np.interp(p, xs, ys)
assert np.allclose(c, iso.predict(p))
edges = [0.0] + BANDS + [1.0]
names = ["low", "usual", "high", "critical"]
band = np.digitize(c, BANDS)
card["bands"] = [{"name": names[i], "from": edges[i], "to": edges[i + 1], "flight_share": round(float((band == i).mean()), 4),
                  "delay_share": round(float(y[band == i].mean()), 4)} for i in range(4)]
card["threshold"] = THRESHOLD
card["max_ctr_complexity"] = CTR
card["threshold_rule"] = "calibrated probability >= 0.20; right if a missed delay costs about four false alarms"

cats = pd.read_csv(FLIGHTS_WEATHER, usecols=WEATHER, dtype=dict.fromkeys(WEATHER, "category"), keep_default_na=False)
card["weather_values"] = {col: sorted(cats[col].cat.categories.tolist()) for col in WEATHER}
del cats

card["pkl_mb"] = round(os.path.getsize(MODEL) / 2**20, 1)
card["features_in_order"] = FEATURES
calib = {"x": [round(float(v), 6) for v in xs], "y": [round(float(v), 6) for v in ys], "threshold": THRESHOLD,
         "bands": BANDS, "base_rate": card["delay_share"]}
with open(CALIBRATION, "w", encoding="utf-8") as f:
    json.dump(calib, f)
with open(os.path.join(WORK, "model_card.json"), "w", encoding="utf-8") as f:
    json.dump(card, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in card.items() if k not in ("weather_values", "features_in_order")}, indent=1))

# Load test in a clean process from the project root, as the app would do it.
test = r'''
import json, sys, time, joblib, numpy as np, pandas as pd, psutil
t = time.time(); pipe = joblib.load(sys.argv[1]); cal = json.load(open(sys.argv[2]))
row = {"MONTH": 1, "DAY": 15, "DAY_OF_WEEK": 4, "AIRLINE": "AA", "ORIGIN_AIRPORT": "ORD", "DESTINATION_AIRPORT": "LGA",
       "SCHEDULED_DEPARTURE": 1700, "DISTANCE": 733, "SCHEDULED_TIME": 125, "SCHEDULED_ARRIVAL": 2005, "DEP_HOUR": 17,
       "ARR_HOUR": 20, "IS_WEEKEND": 0}
out = {}
for name, wx in (("clear", ["light", "good", "none", "none"] * 2), ("snow_storm", ["strong", "poor", "snow", "none", "light", "good", "none", "none"])):
    r = dict(row); r.update(zip(json.loads(sys.argv[3]), wx))
    raw = float(pipe.predict_proba(pd.DataFrame([r]))[0][1])
    out[name] = {"raw": round(raw, 4), "calibrated": round(float(np.interp(raw, cal["x"], cal["y"])), 4)}
out["load_seconds"] = round(time.time() - t, 1); out["rss_gb"] = round(psutil.Process().memory_info().rss / 2**30, 2)
print(json.dumps(out))
'''
res = subprocess.run([sys.executable, "-c", test, MODEL, CALIBRATION, json.dumps(WEATHER)], cwd=ROOT,
                     capture_output=True, text=True)
print(res.stdout, res.stderr[-2000:])
print("Copy to the app:", MODEL, "->", os.path.join(ROOT, "models"), "and", CALIBRATION, "->", os.path.join(ROOT, "models"))
