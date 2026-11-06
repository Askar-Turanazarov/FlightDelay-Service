# Models

| File | Size | In repo | Notes |
|---|---|---|---|
| `lookup_data.joblib` | 0.1 MB | yes | routes (origin → destination, distance, scheduled time), airport and airline lists for the UI |
| `calibration_weather.json` | 8 KB | yes | isotonic calibration of the weather model (`x`, `y`), its threshold 0.20, risk bands and base delay rate |
| `model_pipeline_weather.pkl` | 40 MB | **no** | weather pipeline (`FlightFeatureEngineer` + `CatBoostClassifier`, `max_ctr_complexity=2`, 13 flight + 8 weather inputs), **opened by default** and by the live demo; hosted on [Hugging Face Hub](https://huggingface.co/AskarT/flight-delay-catboost) |
| `model_pipeline_compact.pkl` | 114 MB | **no** | compact diploma pipeline (`max_ctr_complexity=3`), offered in the sidebar; same Hub repository |
| `model_pipeline_main.pkl` | 433 MB | **no** | full pipeline, the diploma reference model; same Hub repository; offered when the file is here or with `MODEL_VARIANT=full` |
| `catboost_model_compact.cbm`, `catboost_model_main.cbm` | 326 MB, 1.29 GB | no | the same CatBoost models in native format, produced by `train_compact.py` / `train_final.py`, not used by the app |

The app (`src/model_loader.py`) knows three variants: `weather`, `compact` and `full`. The visitor
picks one in the sidebar (or with `?model=` in the address); `MODEL_VARIANT` sets the one opened by
default (`weather` unless set). Each pickle is taken from this folder when it exists, otherwise
downloaded from the Hub the first time the variant is picked and cached in `HF_HOME`
(`~/.cache/huggingface` by default). `HF_MODEL_REPO` points at another repository and `HF_MODEL_FILE`
replaces the file of the default variant.

Manual download:

```bash
pip install huggingface_hub
hf download AskarT/flight-delay-catboost model_pipeline_weather.pkl --local-dir models
hf download AskarT/flight-delay-catboost model_pipeline_compact.pkl --local-dir models
hf download AskarT/flight-delay-catboost model_pipeline_main.pkl --local-dir models      # full model, optional
```

The weather model outputs a raw probability inflated by the class weight; the app maps it through
`calibration_weather.json` (`np.interp(raw, x, y)`). Its metrics: `reports/metrics/weather_model_results.txt`;
how it was built: `training/weather/README.md`.
Why the diploma has two variants: the full model needs about 1.5 GB of RAM and does not fit the free
hosting of the demo; the compact one (0.6 GB) costs 0.007 ROC-AUC. Details:
`reports/metrics/compact_experiments.txt`, `reports/metrics/compact_model_results.txt`.

The pickles were produced with Python 3.10, catboost 1.2.8, scikit-learn 1.7.2, numpy 2.2.6 and
pandas 2.3.3. They also load under Python 3.12 with the same packages (the live demo runs 3.12); for
the weather and compact models the predictions were checked to match. Each contains the
`FlightFeatureEngineer` class from `src/feature_engineering.py`: load it from the project root (or
with the project on `sys.path`).
