# Paths and column lists shared by the weather model scripts.
# Every script runs from any folder: the project root is found from this file, and generated files go to WORK
# (training/weather/work by default, or the folder in the WEATHER_WORK environment variable).
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORK = os.environ.get("WEATHER_WORK") or os.path.join(HERE, "work")
RAW = os.path.join(WORK, "raw")                                  # one METAR file per station
STATIONS = os.path.join(HERE, "stations.csv")                    # airport -> METAR station and time zone
PREPARED = os.path.join(WORK, "prepared.pkl")                    # flights after the project's own loader
FLIGHTS_WEATHER = os.path.join(WORK, "flights_weather_2015.csv")  # every flight with the weather at both ends
MODEL = os.path.join(WORK, "model_pipeline_weather.pkl")

BASE_FEATURES = [
    "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE",
    "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE", "DISTANCE", "SCHEDULED_TIME",
    "SCHEDULED_ARRIVAL", "DEP_HOUR", "ARR_HOUR", "IS_WEEKEND",
]
CAT_FEATURES = [
    "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE",
    "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ROUTE",
    "IS_WEEKEND", "IS_NIGHT",
]
WEATHER = [f"{p}_{c}" for p in ("ORIGIN", "DEST") for c in ("WIND", "VISIBILITY", "PRECIP", "HAZARD")]


def hidden_days(frame):
    """The 73 calendar days (20 %) the model never trains on. frame is prepared.pkl, in its own row order."""
    return frame[["MONTH", "DAY"]].drop_duplicates().sample(frac=0.2, random_state=42)


def is_hidden(frame, days):
    return frame.set_index(["MONTH", "DAY"]).index.isin(days.set_index(["MONTH", "DAY"]).index)


def sys_path():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.makedirs(WORK, exist_ok=True)
