# Models

| File | Size | In repo | Notes |
|---|---|---|---|
| `lookup_data.joblib` | 0.1 MB | yes | routes (origin → destination, distance, scheduled time), airport and airline lists for the UI |
| `model_pipeline_main.pkl` | 433 MB | **no** | full sklearn pipeline (`FlightFeatureEngineer` + `CatBoostClassifier`); hosted on [Hugging Face Hub](https://huggingface.co/YOUR_HF_LOGIN/flight-delay-catboost) |
| `catboost_model_main.cbm` | 1.29 GB | no | the same CatBoost model in native format, produced by `train_final.py`, not used by the app |

The app (`src/model_loader.py`) uses `model_pipeline_main.pkl` from this folder when it exists,
otherwise downloads it from the Hub on first start and caches it in `HF_HOME`
(`~/.cache/huggingface` by default). To point at another copy set `HF_MODEL_REPO` and/or
`HF_MODEL_FILE`.

Manual download:

```bash
pip install huggingface_hub
hf download YOUR_HF_LOGIN/flight-delay-catboost model_pipeline_main.pkl --local-dir models
```

The pickle was produced with Python 3.10, catboost 1.2.8, scikit-learn 1.7.2, numpy 2.2.6 and
pandas 2.3.3 and contains the `FlightFeatureEngineer` class from `src/feature_engineering.py`.
Load it from the project root (or with the project on `sys.path`) with these versions installed.
