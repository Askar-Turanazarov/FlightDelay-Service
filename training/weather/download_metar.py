# Step 2. Downloads each station's METAR for 2015 from IEM, one request at a time. Finished files are skipped on a rerun.
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

from common import RAW, STATIONS

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
FIELDS = ["tmpc", "sknt", "gust", "vsby", "p01m", "wxcodes", "ice_accretion_1hr", "snowdepth", "metar"]
# One extra day on each side: a local date near New Year can fall in another UTC year.
PARAMS = [("year1", 2014), ("month1", 12), ("day1", 31), ("year2", 2016), ("month2", 1), ("day2", 3),
          ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("missing", "empty"), ("trace", "T"),
          ("report_type", 3), ("report_type", 4)] + [("data", f) for f in FIELDS]

stations = pd.read_csv(STATIONS).dropna(subset=["station"]).station.unique()
os.makedirs(RAW, exist_ok=True)
print(len(stations), "stations", flush=True)
for i, s in enumerate(stations, 1):
    path = os.path.join(RAW, f"{s}.csv")
    if os.path.exists(path):
        continue
    url = URL + "?" + urllib.parse.urlencode([("station", s)] + PARAMS)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                body = r.read()
            if not body.startswith(b"station,valid"):
                raise ValueError(body[:120])
            with open(path + ".part", "wb") as f:
                f.write(body)
            os.replace(path + ".part", path)
            print(i, s, body.count(b"\n") - 1, "rows", flush=True)
            break
        except Exception as e:
            print(i, s, "retry", attempt + 1, e, flush=True)
            time.sleep(30 * (attempt + 1))
    else:
        print(i, s, "FAILED", flush=True)
    time.sleep(1)
print("finished", flush=True)
