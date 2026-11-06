# Step 1, optional: stations.csv is already in the repository, with the manual fixes below.
# Finds the METAR station and time zone of every airport in airports.csv, using IEM station metadata.
# A station with the airport's own code wins; otherwise the nearest one with a 2015 archive, within MAX_KM.
import json
import math
import os
import time
import urllib.request

import pandas as pd

from common import ROOT, STATIONS

MAX_KM = 15
BASE = "https://mesonet.agron.iastate.edu/geojson/network/{}_ASOS.geojson"


def km(lat1, lon1, lat2, lon2):
    a, b = math.radians(lat1), math.radians(lat2)
    h = math.sin((b - a) / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


air = pd.read_csv(os.path.join(ROOT, "data", "airports.csv"))
stations = {}
for state in sorted(air.STATE.dropna().unique()):
    stations[state] = []
    for attempt in range(6):
        try:
            with urllib.request.urlopen(BASE.format(state), timeout=120) as r:
                stations[state] = json.load(r)["features"]
            break
        except Exception as e:
            print("retry", state, attempt + 1, e)
            time.sleep(10 * (attempt + 1))
    time.sleep(1)

rows = []
for a in air.itertuples():
    found, dist, how = None, None, "none"
    candidates = [f for f in stations.get(a.STATE, []) if (f["properties"].get("archive_begin") or "9999") <= "2015-01-01"]
    same = [f for f in candidates if f["properties"]["sid"] == a.IATA_CODE]
    if same:
        found, how = same[0], "same code"
    if pd.notna(a.LATITUDE):
        near = sorted(candidates, key=lambda f: km(a.LATITUDE, a.LONGITUDE, f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]))
        if near:
            d = km(a.LATITUDE, a.LONGITUDE, near[0]["geometry"]["coordinates"][1], near[0]["geometry"]["coordinates"][0])
            if found is None and d <= MAX_KM:
                found, how = near[0], "nearest"
            if found is not None:
                c = found["geometry"]["coordinates"]
                dist = round(km(a.LATITUDE, a.LONGITUDE, c[1], c[0]), 1)
    p = found["properties"] if found else {}
    rows.append({"iata": a.IATA_CODE, "state": a.STATE, "station": p.get("sid"), "tz": p.get("tzname"),
                 "how": how, "km": dist, "station_name": p.get("sname")})
res = pd.DataFrame(rows)
# Stations the metadata search cannot reach: no coordinates in airports.csv, or an archive date missing in the metadata.
for iata, station, tz in [("GUM", "PGUM", "Pacific/Guam"), ("PPG", "NSTU", "Pacific/Pago_Pago"), ("UST", "SGJ", "America/New_York")]:
    if res.loc[res.iata == iata, "station"].isna().all():
        res.loc[res.iata == iata, ["station", "tz", "how"]] = [station, tz, "manual"]
# The nearest or same-code station has next to no 2015 archive here; these report from the same airfield.
for iata, station in [("HNL", "PHNL"), ("YUM", "NYL")]:
    res.loc[res.iata == iata, ["station", "how"]] = [station, "manual"]
res.to_csv(STATIONS, index=False)
print(res.how.value_counts().to_string())
print("not found:", res[res.station.isna()][["iata", "state"]].values.tolist())
print("nearest:", res[res.how == "nearest"][["iata", "station", "km"]].values.tolist())
print("same code farther than 5 km:", res[(res.how == "same code") & (res.km > 5)][["iata", "station", "km"]].values.tolist())
print("no tz:", res[res.station.notna() & res.tz.isna()].iata.tolist())
