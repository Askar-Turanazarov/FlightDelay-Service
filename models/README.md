# Models

| File | Size | In repo | Notes |
|---|---|---|---|
| `lookup_data.joblib` | 0.1 MB | yes | routes (origin → destination, distance, scheduled time), airport and airline lists for the UI |
| `model_pipeline_compact.pkl` | 114 MB | **no** | compact sklearn pipeline (`FlightFeatureEngineer` + `CatBoostClassifier`, `max_ctr_complexity=3`), **used by default** and by the live demo; hosted on [Hugging Face Hub](https://huggingface.co/AskarT/flight-delay-catboost) |
| `model_pipeline_main.pkl` | 433 MB | **no** | full pipeline, the diploma reference model; same Hub repository; used with `MODEL_VARIANT=full` |
| `catboost_model_compact.cbm`, `catboost_model_main.cbm` | 326 MB, 1.29 GB | no | the same CatBoost models in native format, produced by `train_compact.py` / `train_final.py`, not used by the app |

The app (`src/model_loader.py`) picks the variant from `MODEL_VARIANT` (`compact` by default,
`full` for the diploma model), uses the matching pickle from this folder when it exists, otherwise
downloads it from the Hub on first start and caches it in `HF_HOME` (`~/.cache/huggingface` by
default). To point at another copy set `HF_MODEL_REPO` and/or `HF_MODEL_FILE`.

Manual download:

```bash
pip install huggingface_hub
hf download AskarT/flight-delay-catboost model_pipeline_compact.pkl --local-dir models
hf download AskarT/flight-delay-catboost model_pipeline_main.pkl --local-dir models      # full model, optional
```

Why two variants: the full model needs about 1.5 GB of RAM and does not fit the free hosting of the
demo; the compact one (0.6 GB) costs 0.007 ROC-AUC. Details and the other configurations tried:
`reports/metrics/compact_experiments.txt`, `reports/metrics/compact_model_results.txt`.

The pickle was produced with Python 3.10, catboost 1.2.8, scikit-learn 1.7.2, numpy 2.2.6 and
pandas 2.3.3 and contains the `FlightFeatureEngineer` class from `src/feature_engineering.py`.
Load it from the project root (or with the project on `sys.path`) with these versions installed.
