# Optional audit after step 2. What the raw METAR strings actually contain: wind shear, runway state,
# runway visual range, and how complete the fields are.
import glob
import json
import os
import re

import pandas as pd

from common import RAW, WORK

SHEAR = re.compile(r"\bWS(?!HFT)\b|\bLLWS\b")
RUNWAY_STATE = re.compile(r"\bR\d{2}[LRC]?/\d{6}\b")
RVR = re.compile(r"\bR\d{2}[LRC]?/[MP]?\d{4}")
total = shear = state = rvr = 0
rows = []
for path in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
    w = pd.read_csv(path, low_memory=False, usecols=["valid", "sknt", "vsby", "wxcodes", "p01m", "metar"])
    y = w[w.valid.str.startswith("2015")]
    m = y.metar.fillna("")
    s, st, r = m.str.contains(SHEAR).sum(), m.str.contains(RUNWAY_STATE).sum(), m.str.contains(RVR).sum()
    total += len(y)
    shear += int(s)
    state += int(st)
    rvr += int(r)
    rows.append({"station": os.path.basename(path)[:-4], "reports_2015": len(y), "hours_2015": y.valid.str[:13].nunique(),
                 "wind_filled": round(float(y.sknt.notna().mean()), 3) if len(y) else 0,
                 "visibility_filled": round(float(y.vsby.notna().mean()), 3) if len(y) else 0,
                 "precip_filled": round(float(y.p01m.notna().mean()), 3) if len(y) else 0})
cov = pd.DataFrame(rows)
cov.to_csv(os.path.join(WORK, "station_coverage.csv"), index=False)
print(json.dumps({"reports": total, "wind_shear_mentions": shear, "runway_state_groups": state, "rvr_groups": rvr,
                  "stations": len(cov), "stations_under_90pct_hours": cov[cov.hours_2015 < 0.9 * 8760].station.tolist(),
                  "stations_no_data": cov[cov.reports_2015 == 0].station.tolist()}, indent=1))
