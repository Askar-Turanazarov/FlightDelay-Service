# Step 4. Builds flights_weather_2015.csv: every row of flights.csv plus airport codes restored for October,
# four weather categories at each end of the route, traffic counted from the schedule, and the hidden-day mark.
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from common import FLIGHTS_WEATHER, PREPARED, RAW, ROOT, STATIONS, WORK, hidden_days

CONFIDENT = 0.9
WEATHER = ["WIND", "VISIBILITY", "PRECIP", "HAZARD"]


def restored_codes(raw):
    """Same method as prepare.py: a numeric id maps to the IATA code the same scheduled flights use in other months."""
    o = raw.ORIGIN_AIRPORT.astype(str)
    d = raw.DESTINATION_AIRPORT.astype(str)
    num = o.str.fullmatch(r"\d+") & d.str.fullmatch(r"\d+")
    key = ["AIRLINE", "FLIGHT_NUMBER", "SCHEDULED_DEPARTURE", "DISTANCE"]
    ref = (raw[~num].groupby(key + ["ORIGIN_AIRPORT"]).size().rename("n").reset_index()
           .sort_values("n", ascending=False).drop_duplicates(key))
    m = raw[num][key + ["ORIGIN_AIRPORT"]].merge(ref, on=key, suffixes=("_id", "_iata"))
    votes = m.groupby(["ORIGIN_AIRPORT_id", "ORIGIN_AIRPORT_iata"]).size().rename("v").reset_index()
    votes["share"] = votes.v / votes.groupby("ORIGIN_AIRPORT_id").v.transform("sum")
    best = votes.sort_values("v", ascending=False).drop_duplicates("ORIGIN_AIRPORT_id")
    good = best[best.share >= CONFIDENT]
    good = good[~good.ORIGIN_AIRPORT_iata.duplicated(keep=False)]
    return {str(a): b for a, b in zip(good.ORIGIN_AIRPORT_id, good.ORIGIN_AIRPORT_iata)}


def tokens(text):
    return [t.lstrip("+-") for t in str(text).split()] if isinstance(text, str) else []


def precip_of(codes):
    found = {t for c in codes for t in tokens(c)}
    if any(t.startswith("VC") for t in found):
        found = {t for t in found if not t.startswith("VC") or "TS" in t}
    if any(re.search(r"FZRA|FZDZ|PL", t) for t in found):
        return "freezing"
    if any(re.search(r"SN|SG", t) and not re.match(r"BL|DR", t) for t in found):
        return "snow"
    if any(re.search(r"RA|DZ|UP", t) for t in found):
        return "rain"
    return "none"


def hazard_of(codes, iced):
    found = {t for c in codes for t in tokens(c)}
    if any("TS" in t for t in found):
        return "thunderstorm"
    if iced or any(re.search(r"FZ|PL", t) for t in found):
        return "icing"
    if any(t.endswith("FG") and not re.match(r"MI|BC|PR|VC", t) for t in found):
        return "fog"
    if any(re.search(r"SQ|FC|GR|GS|SS|DS|VA|PO|BLSN", t) for t in found):
        return "other"
    return "none"


def hourly(path):
    """One row per UTC hour with the worst conditions reported in it. Missing hours reuse up to two earlier ones."""
    w = pd.read_csv(path, low_memory=False, na_values=["T"], keep_default_na=True,
                    usecols=["station", "valid", "sknt", "gust", "vsby", "wxcodes", "ice_accretion_1hr"])
    if w.empty:
        return w, 0
    w["hour"] = pd.to_datetime(w.valid, utc=True).dt.floor("h")
    w["wind"] = w[["sknt", "gust"]].max(axis=1)
    iced = pd.to_numeric(w.ice_accretion_1hr, errors="coerce").fillna(0) > 0
    w["iced"] = iced
    rows = []
    for hour, g in w.groupby("hour"):
        codes = g.wxcodes.dropna().tolist()
        rows.append((hour, g.wind.max(), g.vsby.min(), precip_of(codes), hazard_of(codes, bool(g.iced.any()))))
    h = pd.DataFrame(rows, columns=["hour", "wind", "vsby", "PRECIP", "HAZARD"]).set_index("hour")
    span = pd.date_range(h.index.min(), h.index.max(), freq="h")
    h = h.reindex(span).ffill(limit=2)
    # Thresholds fixed from aviation practice before looking at any model result: knots and statute miles.
    h["WIND"] = np.select([h.wind.isna(), h.wind < 10, h.wind < 20, h.wind < 34], ["unknown", "light", "moderate", "strong"], "gale")
    h["VISIBILITY"] = np.select([h.vsby.isna(), h.vsby >= 5, h.vsby >= 1], ["unknown", "good", "reduced"], "poor")
    h["PRECIP"] = h.PRECIP.fillna("unknown")
    h["HAZARD"] = h.HAZARD.fillna("unknown")
    h = h[WEATHER].rename_axis("hour").reset_index()
    h["station"] = os.path.basename(path)[:-4]
    return h, len(w)


def local_to_utc(frame, tz_of_airport):
    """SCHEDULED_DEPARTURE is local time at the origin. Scheduled arrival is departure plus SCHEDULED_TIME."""
    hhmm = frame.SCHEDULED_DEPARTURE.astype(int)
    local = (pd.to_datetime(dict(year=frame.YEAR, month=frame.MONTH, day=frame.DAY))
             + pd.to_timedelta(hhmm // 100, unit="h") + pd.to_timedelta(hhmm % 100, unit="m"))
    tz = frame.ORIGIN_AIRPORT.map(tz_of_airport)
    out = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    for name, idx in tz.groupby(tz).groups.items():
        out.loc[idx] = (local.loc[idx].dt.tz_localize(name, ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC"))
    return out


# 1. Flights, every row and every original column.
flights = pd.read_csv(os.path.join(ROOT, "data", "flights.csv"), low_memory=False,
                      dtype={"ORIGIN_AIRPORT": str, "DESTINATION_AIRPORT": str})
print("flights", flights.shape, flush=True)
mapping = restored_codes(flights)
numeric = flights.ORIGIN_AIRPORT.str.fullmatch(r"\d+") | flights.DESTINATION_AIRPORT.str.fullmatch(r"\d+")
flights["AIRPORT_CODE_RESTORED"] = numeric.astype(int)
for c in ("ORIGIN_AIRPORT", "DESTINATION_AIRPORT"):
    flights.loc[numeric, c] = flights.loc[numeric, c].map(lambda v: mapping.get(v, v))

# 2. Weather per station and hour.
stations = pd.read_csv(STATIONS).dropna(subset=["station"])
parts, raw_rows, empty = [], {}, []
for path in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
    h, n = hourly(path)
    raw_rows[os.path.basename(path)[:-4]] = n
    if n == 0:
        empty.append(os.path.basename(path)[:-4])
    else:
        parts.append(h)
weather = pd.concat(parts, ignore_index=True)
print("weather hours", len(weather), flush=True)

# 3. Join each end of the route to its station and scheduled hour.
station_of = dict(zip(stations.iata, stations.station))
tz_of = dict(zip(stations.iata, stations.tz))
dep = local_to_utc(flights, tz_of)
arr = dep + pd.to_timedelta(flights.SCHEDULED_TIME, unit="m")
for prefix, airport, when in (("ORIGIN", flights.ORIGIN_AIRPORT, dep), ("DEST", flights.DESTINATION_AIRPORT, arr)):
    key = pd.DataFrame({"station": airport.map(station_of).values, "hour": when.dt.floor("h").values})
    key["hour"] = pd.to_datetime(key.hour, utc=True)
    got = key.merge(weather, how="left", on=["station", "hour"])
    for c in WEATHER:
        flights[f"{prefix}_{c}"] = got[c].fillna("unknown").values
    print(prefix, "joined", flush=True)

# 4. Traffic from the schedule: cancelled flights count too, so a storm does not lower the number.
flights["DEP_HOUR_LOCAL"] = (flights.SCHEDULED_DEPARTURE.astype(int) // 100) % 24
flights["SCHEDULED_AIRPORT_HOUR_FLIGHTS"] = flights.groupby(["ORIGIN_AIRPORT", "MONTH", "DAY", "DEP_HOUR_LOCAL"]).FLIGHT_NUMBER.transform("size")
flights["SCHEDULED_AIRLINE_DAY_FLIGHTS"] = flights.groupby(["ORIGIN_AIRPORT", "AIRLINE", "MONTH", "DAY"]).FLIGHT_NUMBER.transform("size")
flights = flights.drop(columns=["DEP_HOUR_LOCAL"])

# 5. The same 73 hidden days as train.py, for an honest split by date. Needs prepared.pkl (step 3).
prepared = pd.read_pickle(PREPARED)[["MONTH", "DAY"]]
days = hidden_days(prepared)
del prepared
flights["IS_HIDDEN_DAY"] = pd.MultiIndex.from_frame(flights[["MONTH", "DAY"]]).isin(pd.MultiIndex.from_frame(days)).astype(int)

# 6. Allowed combinations for the demo form, with their real frequency.
combos = pd.concat([flights[[f"{p}_{c}" for c in WEATHER]].set_axis(WEATHER, axis=1) for p in ("ORIGIN", "DEST")])
combos = combos.value_counts().rename("flights").reset_index()
combos.to_csv(os.path.join(WORK, "weather_combinations.csv"), index=False)

# 7. Checks written next to the file.
late = flights.ARRIVAL_DELAY > 15
flown = flights.ARRIVAL_DELAY.notna()
report = {
    "rows": len(flights), "columns": flights.shape[1], "cancelled": int(flights.CANCELLED.sum()),
    "flown": int(flown.sum()), "codes_mapped": len(mapping),
    "rows_with_restored_codes": int(numeric.sum()),
    "numeric_origin_left": int(flights.ORIGIN_AIRPORT.str.fullmatch(r"\d+").sum()),
    "numeric_dest_left": int(flights.DESTINATION_AIRPORT.str.fullmatch(r"\d+").sum()),
    "stations_downloaded": len(raw_rows), "stations_empty": empty,
    "hidden_days": len(days), "hidden_day_rows": int(flights.IS_HIDDEN_DAY.sum()),
}
for p in ("ORIGIN", "DEST"):
    for c in WEATHER:
        col = f"{p}_{c}"
        report[col] = {
            "share": flights[col].value_counts(normalize=True).round(4).to_dict(),
            "delay_share_flown": late[flown].groupby(flights.loc[flown, col]).mean().round(3).to_dict(),
            "cancelled_share": flights.groupby(col).CANCELLED.mean().round(3).to_dict(),
        }
with open(os.path.join(WORK, "build_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in report.items() if not k.startswith(("ORIGIN", "DEST"))}, indent=1), flush=True)

# 8. The file itself.
flights.to_csv(FLIGHTS_WEATHER, index=False)
print("written", round(os.path.getsize(FLIGHTS_WEATHER) / 2**20), "MB", flush=True)
