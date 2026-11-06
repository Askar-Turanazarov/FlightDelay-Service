# Training the weather model

These scripts produced `model_pipeline_weather.pkl` and `models/calibration_weather.json`, the default
model of the app. They join hourly METAR weather of 2015 to every flight, train the project's
`FlightFeatureEngineer` + CatBoost pipeline on 292 days, test it on 73 whole hidden days and calibrate it.
Results: `reports/metrics/weather_model_results.txt`.

## Before you start

* `data/flights.csv` from Kaggle (`kaggle datasets download -d usdot/flight-delays -p data --unzip`).
* Python 3.10 and `pip install -r requirements-dev.txt`: the versions are pinned because the pipeline is pickled.
* Internet access for steps 1 and 2 (Iowa Environmental Mesonet, no key needed).
* A CUDA GPU for step 5, or `DEVICE=CPU` (much slower).
* About 2 GB of free disk. The scripts were run on a laptop with 32 GB of RAM and an RTX 4050.

Everything generated goes to `training/weather/work/` (ignored by git). Set `WEATHER_WORK` to use
another folder, for example on a bigger disk.

## Steps

Run from the project root.

| Step | Command | Output in `work/` | Time measured |
|---|---|---|---|
| 1, optional | `python training/weather/map_stations.py` | rewrites `training/weather/stations.csv` | not timed |
| 2 | `python training/weather/download_metar.py` | `raw/<station>.csv`, 322 stations, 470 MB | not timed |
| audit, optional | `python training/weather/check_metar.py` | `station_coverage.csv` | not timed |
| 3 | `python training/weather/prepare.py` | `prepared.pkl`, `prep_info.json` | 26 s |
| 4 | `python training/weather/build_dataset.py` | `flights_weather_2015.csv` (1 GB), `build_report.json`, `weather_combinations.csv` | 10 min |
| 5 | `python training/weather/train.py` | `model_pipeline_weather.pkl`, `weather_ctr2.json` | 23.5 min of training on the GPU |
| 6 | `python training/weather/package.py` | `calibration_weather.json`, `model_card.json`, `preds_hidden_days.npz` | not timed |

Then copy `model_pipeline_weather.pkl` and `calibration_weather.json` into `models/`; the app takes
local files before the Hugging Face copy.

**Step 1.** `stations.csv` maps each airport of `data/airports.csv` to a METAR station and its time zone:
the station with the airport's own code, otherwise the nearest one with a 2015 archive within 15 km,
plus five manual fixes written in the script. It is already in the repository; rerun it only if you
want to rebuild it from the current IEM metadata.

**Step 4.** For each flight, the worst conditions of the hour at the origin in the scheduled departure
hour and at the destination in the scheduled arrival hour (local times converted to UTC), in four
categories each. The thresholds were fixed from aviation practice before any model was trained:

| Column | Values |
|---|---|
| `*_WIND` (max of wind and gust, kt) | light < 10, moderate 10–19, strong 20–33, gale ≥ 34 |
| `*_VISIBILITY` (statute miles) | good ≥ 5, reduced 1–5, poor < 1 |
| `*_PRECIP` | none, rain, snow, freezing |
| `*_HAZARD` | none, fog, thunderstorm, icing, other |

`unknown` marks a missing report (under 0.1 % of flights). The file also keeps every original column
of `flights.csv`, traffic counted from the schedule and an `IS_HIDDEN_DAY` mark; the model uses only
the 13 flight columns of the app and the eight weather columns.

**Step 5.** 2500 trees, depth 7, learning rate 0.03, `l2_leaf_reg=12`, `scale_pos_weight=4.2`,
`max_ctr_complexity=2`. The 73 hidden days are a 20 % sample of the calendar days with seed 42
(`common.hidden_days`); every step uses the same days. Environment: `CTR=3` trains the heavier
variant (0.6988 on the hidden days, 170 MB), `DEVICE=CPU` trains without a GPU, `ROWS=20000
ITERATIONS=30` makes a quick smoke run that saves nothing.

**Step 6.** The isotonic calibrator is fitted on the predictions for the 73 hidden days and stored as
two arrays for `np.interp`. The honest calibrated figures in the report come from fitting on half of
the days and judging on the other half, ten splits. The threshold 0.20 is right if a missed delay
costs about four false alarms; the risk bands are 0.12 / 0.20 / 0.35.

## Checked

* `prepare.py` and `build_dataset.py` run from scratch give an identical `prepared.pkl` table and the same
  `flights_weather_2015.csv`, byte for byte, as the files the published model was trained on.
* `package.py` run on the published `model_pipeline_weather.pkl` gives `models/calibration_weather.json`
  byte for byte and the same model card.
* `train.py` smoke run: the same 1,116,989 test flights (delay share 0.1794) and 4,597,019 training rows.
* CatBoost on a GPU is not guaranteed to be bit-for-bit reproducible, so a full retrain can differ
  from the published model in the last digits.

The model learned from the weather that was observed. The app takes a forecast instead, so in real
use its score is lower than the 0.698 measured here.

Weather data: Iowa Environmental Mesonet, Iowa State University (ASOS/METAR archive).
