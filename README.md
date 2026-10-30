# Flight Delay Prediction & Risk Analysis

**English** · [Oʻzbekcha](README.uz.md) · [Русский](README.ru.md)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost 1.2.8](https://img.shields.io/badge/CatBoost-1.2.8-FFCC00?logo=catboost&logoColor=black)](https://catboost.ai/)
[![Streamlit 1.53](https://img.shields.io/badge/Streamlit-1.53-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Hugging%20Face%20Hub-yellow)](https://huggingface.co/AskarT/flight-delay-catboost)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine-learning service that estimates the probability that a US domestic flight
will arrive **more than 15 minutes late** (the FAA definition of a delay) using only the information
known at planning time: airline, origin and destination airports, date, departure hour, distance and
scheduled flight time. On top of the forecast the app explains *why* (per-flight Shapley values) and
translates the risk into an **expected financial loss** using FAA cost assumptions.

**🚀 Live demo:** https://huggingface.co/spaces/AskarT/flight-delay-risk
*(free CPU Space: it sleeps after 48 h without visitors, the first load takes about a minute while the model is downloaded)*

![Prediction screen](docs/img/02_prediction.png)

| Explanation & economics | Model evidence |
|:--:|:--:|
| ![SHAP and loss breakdown](docs/img/03_shap_economics.png) | ![Model metrics tab](docs/img/04_model_evidence.png) |

---

## Contents

1. [What the service does](#what-the-service-does)
2. [Data](#data)
3. [Approach](#approach)
4. [Results](#results)
5. [Model artifacts](#model-artifacts)
6. [Run it](#run-it)
7. [Project structure](#project-structure)
8. [Known limitations & roadmap](#known-limitations--roadmap)
9. [Documentation & author](#documentation--author)

---

## What the service does

* **Forecast** — a CatBoost classifier returns the raw probability of a significant delay.
  The business decision threshold is **0.37** (tuned for recall, see [Results](#results)):
  at or above it the verdict is "a delay is likely".
* **Context** — the probability is compared with the average delay rate in the data (18.9 %),
  so a user sees "1.8× the average" instead of an abstract number. A *risk index* bar stretches
  the probability around the threshold for readability only; it is not a calibration.
* **Explanation** — Shapley values computed natively by CatBoost show which factors pushed this
  particular flight towards or away from a delay (airport, route, day, hour, airline, ...).
* **Economics** — expected loss = probability × total cost of a delay event, where the event cost
  combines direct airline operating costs (110 USD per delay minute), passenger time
  (FAA Value of Travel Time Savings, 62.50 USD per passenger-hour) and a 2.2× knock-on factor for
  reactionary delays across the network.
* **Three UI languages** — English, Uzbek (Latin) and Russian. The app opens in the browser's
  language (`uz*` → Uzbek, `ru*` → Russian, anything else → English) and has a switcher in the sidebar.

![Uzbek interface](docs/img/05_uzbek.png)

## Data

* **Source:** Kaggle — [2015 Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays)
  (US Department of Transportation, Bureau of Transportation Statistics).
  5.8 million US domestic flights of 2015, 565 MB `flights.csv` plus the lookup tables
  `airlines.csv` and `airports.csv`.
* **In this repository:** only the two small lookup tables and a route table derived from the data
  (`models/lookup_data.joblib`). The raw `flights.csv` is **not** committed; it is needed only to retrain.
  Download it with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api):

  ```bash
  kaggle datasets download -d usdot/flight-delays -p data --unzip
  ```

* **Training subset:** the first 2.1 million rows (80 % train / 20 % stratified test).
  Target: `ARRIVAL_DELAY > 15` minutes; positive rate ≈ 18.9 %.
* **Leakage control:** only columns known before departure are used
  (`MONTH, DAY, DAY_OF_WEEK, AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, SCHEDULED_DEPARTURE,
  SCHEDULED_ARRIVAL, SCHEDULED_TIME, DISTANCE`). Actual departure/arrival times, taxi times,
  tail numbers and delay causes are excluded.

## Approach

**Feature engineering** — a custom scikit-learn transformer `FlightFeatureEngineer`
(`src/feature_engineering.py`) is the first step of the pipeline, so the same logic is applied at
training and inference time:

| Feature | Idea |
|---|---|
| `hour_sin`, `hour_cos` | departure hour as a point on a circle (23:00 is close to 00:00) |
| `traffic_density` | number of departures from the origin airport in that hour of that day (runway/terminal congestion) |
| `AIRLINE_FLIGHT_COUNT` | flights of the airline from that airport that day (crew/aircraft rotation pressure, "domino effect") |
| `ROUTE` | origin–destination pair as a categorical feature (congested corridors) |
| `plan_speed` | distance / scheduled time: a tight schedule leaves no slack to recover in the air |
| `IS_NIGHT`, `IS_WEEKEND` | 00:00–05:59 departures; Saturday/Sunday |

The traffic statistics are learned on the training split only (`fit`) and looked up at inference.

**Model** — `CatBoostClassifier` inside `sklearn.pipeline.Pipeline`, trained on GPU
(`src/train_final.py`): 2500 iterations, learning rate 0.03, depth 7, `l2_leaf_reg=12`,
`scale_pos_weight=4.2` to counter the 81/19 class imbalance, Bernoulli subsampling 0.8,
early stopping on AUC (100 rounds). Categorical features are handled by CatBoost's ordered target
statistics: `MONTH, DAY, DAY_OF_WEEK, AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, ROUTE,
IS_WEEKEND, IS_NIGHT`. Training time: about 11 minutes on an RTX 4050.

**Threshold** — the default 0.5 catches only 66 % of delays. Because a missed delay (false
negative) is more expensive for an airline than a false alarm, the threshold was swept and fixed at
**0.37**, giving recall 0.82 at precision 0.31.

**Explainability** — `src/analytics.py` asks CatBoost for per-row SHAP values
(`get_feature_importance(type="ShapValues")`) and shows the seven strongest contributions.

## Results

Test set: 408 907 flights (20 % hold-out). Source of truth: [`reports/metrics/`](reports/metrics/).

| Metric | Value |
|---|---|
| ROC-AUC | **0.786** |
| Recall @ 0.37 (chosen threshold) | **0.82** |
| Precision @ 0.37 | 0.31 |
| Recall / Precision @ 0.50 | 0.66 / 0.40 |
| Accuracy @ 0.50 | 0.75 |

Threshold sweep (`reports/metrics/recall085_results.txt`):

| Threshold | 0.20 | 0.25 | 0.30 | 0.35 | **0.37** | 0.40 | 0.45 | 0.50 |
|---|---|---|---|---|---|---|---|---|
| Recall | 0.967 | 0.933 | 0.891 | 0.840 | **0.818** | 0.783 | 0.723 | 0.660 |
| Precision | 0.220 | 0.240 | 0.265 | 0.293 | **0.306** | 0.325 | 0.359 | 0.396 |

| Precision–recall curve | Feature importance |
|:--:|:--:|
| ![PR curve](reports/figures/pr_curve_optimal_threshold.png) | ![Feature importance](reports/figures/CatBoost%20Feature%20Importance.jpg) |

Earlier experiments (PCA, ensembles with LightGBM, 30/45-minute targets) are kept in
`notebooks/archive/` and `reports/figures/`; the final model outperformed all of them on ROC-AUC.

## Model artifacts

| File | Size | Where | Used by the app |
|---|---|---|---|
| `model_pipeline_main.pkl` — full pipeline (feature engineer + CatBoost), joblib `compress=3` | 433 MB | [Hugging Face Hub](https://huggingface.co/AskarT/flight-delay-catboost) | **yes** |
| `catboost_model_main.cbm` — the same CatBoost model, native format | 1.29 GB | local only | no |
| `lookup_data.joblib` — routes, airports, airlines for the UI | 0.1 MB | this repo | yes |

The model is large because CatBoost stores counter (CTR) tables for high-cardinality categorical
features such as `ROUTE` (≈ 4 200 values) and `DAY`.

**Loading logic** (`src/model_loader.py`): if `models/model_pipeline_main.pkl` exists locally it is
used; otherwise the file is downloaded once from the Hub with `huggingface_hub` and cached.
Override the source with environment variables `HF_MODEL_REPO` and `HF_MODEL_FILE`.

**Compatibility:** the pickle is bound to Python 3.10, catboost 1.2.8, scikit-learn 1.7.2,
numpy 2.2.6, pandas 2.3.3 (all pinned in `requirements.txt`) and to the class in
`src/feature_engineering.py`. Changing that file requires retraining.

## Run it

### Local (Python 3.10)

```bash
git clone https://github.com/Askar-Turanazarov/FlightDelay-Service.git
cd FlightDelay-Service
python -m venv venv
venv\Scripts\activate            # Windows   |   source venv/bin/activate on Linux/macOS
pip install -r requirements.txt
streamlit run src/app.py
```

On Windows you can simply double-click `START_PROJECT.bat`: it creates the venv, installs the
requirements and starts the app on the first free port from 8505.

The model is fetched from Hugging Face on the first start (433 MB). To use a local copy put it at
`models/model_pipeline_main.pkl`.

### Docker

```bash
docker build -t flt-delay-app .
docker run -p 8501:8501 flt-delay-app
```

Then open http://localhost:8501. Useful variants:

```bash
# use a local model file instead of downloading it
docker run -p 8501:8501 -v "$PWD/models:/app/models:ro" flt-delay-app

# a PyPI mirror if the default CDN is slow in your region
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t flt-delay-app .
```

The image (python 3.10-slim, non-root user, health check on `/_stcore/health`) is the same one that
runs in the Hugging Face Space. It is about 1.6 GB, almost all of it CatBoost, SciPy and pyarrow.

### Retrain

```bash
kaggle datasets download -d usdot/flight-delays -p data --unzip
pip install -r requirements-dev.txt
python src/train_final.py          # GPU by default; set task_type="CPU" in the script otherwise
```

EDA and the training story live in `notebooks/01_EDA_and_Training.ipynb`.

## Project structure

```
flight_delay_service/
├── src/
│   ├── app.py                 Streamlit UI
│   ├── analytics.py           risk index, relative risk, SHAP explanations
│   ├── feature_engineering.py FlightFeatureEngineer (part of the pickled pipeline)
│   ├── model_loader.py        local file or Hugging Face Hub download
│   ├── train_final.py         training script
│   └── i18n/                  en.py, uz.py, ru.py + locale detection
├── data/                      airlines.csv, airports.csv (flights.csv is downloaded, not committed)
├── models/                    lookup_data.joblib (+ model_pipeline_main.pkl, downloaded)
├── reports/figures/           PR curve, feature importance, confusion matrices
├── reports/metrics/           raw metric reports from training
├── notebooks/                 01_EDA_and_Training.ipynb, archive/ with experiments
├── docs/                      Report.pdf, Tech_report.pdf (Russian), img/ screenshots
├── logos/                     airline logos for the UI
├── Dockerfile · requirements.txt · requirements-dev.txt · START_PROJECT.bat
```

## Known limitations & roadmap

* **Model size.** 433 MB compressed / about 1.3 GB in RAM. A compact retrain
  (`max_ctr_complexity=1`, `DAY` and `ROUTE` not as categories) is planned.
* **Risk index is not calibration.** The main number shown is the raw probability; the index bar is
  a visual transform. A proper `CalibratedClassifierCV` step is on the roadmap.
* **2015 US data only.** Airports, airlines and traffic patterns are those of 2015; the app is a
  demonstration of the method, not an operational tool.
* **Traffic features need known dates.** `traffic_density` and `AIRLINE_FLIGHT_COUNT` are looked up
  from training statistics for the chosen airport/day/hour; unseen combinations fall back to 1.
* Planned: pytest for the transformer, loader and translations; GitHub Actions (lint + Docker build).

## Documentation & author

* `docs/Report.pdf` — diploma report, `docs/Tech_report.pdf` — technical report (both in Russian).
* Model card: https://huggingface.co/AskarT/flight-delay-catboost

**Askar Turanazarov** — Data Science diploma project, 2026. Released under the [MIT License](LICENSE).
