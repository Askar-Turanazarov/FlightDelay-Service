# Askar Turanazarov

# src/train_compact.py
# Trains the COMPACT variant of the model. Same data, split, features and CatBoost settings as
# train_final.py with one change: max_ctr_complexity=3 instead of the default 4. CatBoost then
# stores counter (CTR) tables for combinations of at most three categorical features, which cuts
# the pipeline from 433 MB to ~115 MB on disk and from ~1.5 GB to ~0.6 GB in RAM at the cost of
# ~0.007 ROC-AUC. This is the variant served by the public demo (Streamlit Community Cloud,
# 1 GB RAM). The full model from train_final.py stays the reference model of the diploma.
#
# Usage (GPU by default, set task_type="CPU" below otherwise):
#     kaggle datasets download -d usdot/flight-delays -p data --unzip
#     python src/train_compact.py
#
# Outputs:
#     models/model_pipeline_compact.pkl        sklearn Pipeline (FlightFeatureEngineer + CatBoost)
#     models/catboost_model_compact.cbm        the same CatBoost model in native format
#     reports/metrics/compact_model_results.txt metrics on the same 20 % hold-out as the full model

import os
import sys
import time

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineering import FlightFeatureEngineer  # noqa: E402
from src.train_final import load_flight_data_final          # noqa: E402  (same loader, same target)

MAX_CTR_COMPLEXITY = 3

BASE_FEATURES = [
    'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE',
    'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
    'SCHEDULED_DEPARTURE', 'DISTANCE', 'SCHEDULED_TIME',
    'SCHEDULED_ARRIVAL', 'DEP_HOUR', 'ARR_HOUR', 'IS_WEEKEND',
]
CAT_FEATURES = [
    'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE',
    'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ROUTE',
    'IS_WEEKEND', 'IS_NIGHT',
]
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.37, 0.40, 0.45, 0.50]

PKL_PATH = "models/model_pipeline_compact.pkl"
CBM_PATH = "models/catboost_model_compact.cbm"
REPORT_PATH = "reports/metrics/compact_model_results.txt"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ('feature_eng', FlightFeatureEngineer()),
        ('clf', CatBoostClassifier(
            iterations=2500,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=12,
            scale_pos_weight=4.2,
            max_ctr_complexity=MAX_CTR_COMPLEXITY,   # the only difference from train_final.py
            task_type="GPU",
            devices='0',
            bootstrap_type='Bernoulli',
            subsample=0.8,
            grow_policy='SymmetricTree',
            eval_metric='AUC',
            random_seed=42,
            random_strength=2,
            early_stopping_rounds=100,
            verbose=100,
            cat_features=CAT_FEATURES,
        )),
    ])


if __name__ == "__main__":
    print(f"\nCompact model: max_ctr_complexity={MAX_CTR_COMPLEXITY}")
    df = load_flight_data_final(2100000)
    X_train, X_test, y_train, y_test = train_test_split(
        df[BASE_FEATURES], df['target'], test_size=0.2, random_state=42, stratify=df['target'])
    print(f"train {X_train.shape}, test {X_test.shape}")

    pipeline = build_pipeline()
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_min, train_sec = divmod(int(time.time() - t0), 60)

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)
    joblib.dump(pipeline, PKL_PATH, compress=3)
    pipeline.named_steps['clf'].save_model(CBM_PATH, format="cbm")

    prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    y_true = y_test.values
    lines = [
        "COMPACT MODEL REPORT (max_ctr_complexity=3)",
        "-" * 50,
        f"TRAINING TIME: {train_min} min {train_sec} sec",
        f"TREES: {pipeline.named_steps['clf'].tree_count_}",
        f"ROC-AUC SCORE: {auc:.4f}",
        f"PKL SIZE: {os.path.getsize(PKL_PATH) / 2**20:.1f} MB (joblib compress=3)",
        f"CBM SIZE: {os.path.getsize(CBM_PATH) / 2**20:.1f} MB",
        "-" * 50,
        "CLASSIFICATION METRICS @ 0.50:",
        classification_report(y_true, (prob >= 0.5).astype(int)),
        "-" * 50,
        "THRESHOLD SWEEP (same grid as recall085_results.txt):",
        f"{'Threshold':<12} | {'Recall':<10} | {'Precision':<10}",
        "-" * 38,
    ]
    for thr in THRESHOLDS:
        pred = prob >= thr
        tp = int((pred & (y_true == 1)).sum())
        rec = tp / max(int((y_true == 1).sum()), 1)
        prec = tp / max(int(pred.sum()), 1)
        lines.append(f"{thr:<12} | {rec:<10.4f} | {prec:<10.4f}")
    lines += ["", "Reference (full model, train_final.py): ROC-AUC 0.7860, recall 0.8179 / precision 0.3058 @ 0.37."]
    report = "\n".join(lines) + "\n"
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"saved {PKL_PATH}, {CBM_PATH}, {REPORT_PATH}")
