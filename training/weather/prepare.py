# Step 3. Builds the flight table the next steps share: the project's own loader (src/train_final.py) on the whole
# data/flights.csv, October airport ids restored to IATA codes, and one stratified 20 % random holdout ("split").
# The weather model does not use that holdout: it is tested on whole hidden days (common.hidden_days).
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from common import PREPARED, ROOT, WORK, sys_path  # noqa: E402

sys_path()
os.chdir(ROOT)
from src.train_final import load_flight_data_final  # noqa: E402

CONFIDENT = 0.9

raw = pd.read_csv("data/flights.csv", low_memory=False, usecols=[
    "AIRLINE", "FLIGHT_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE", "DISTANCE"])
o = raw.ORIGIN_AIRPORT.astype(str)
d = raw.DESTINATION_AIRPORT.astype(str)
num = o.str.fullmatch(r"\d+") & d.str.fullmatch(r"\d+")
key = ["AIRLINE", "FLIGHT_NUMBER", "SCHEDULED_DEPARTURE", "DISTANCE"]
base = raw[~num]
ref = (base.groupby(key + ["ORIGIN_AIRPORT"]).size().rename("n").reset_index()
       .sort_values("n", ascending=False).drop_duplicates(key))
m = raw[num][key + ["ORIGIN_AIRPORT"]].merge(ref, on=key, suffixes=("_id", "_iata"))
votes = m.groupby(["ORIGIN_AIRPORT_id", "ORIGIN_AIRPORT_iata"]).size().rename("v").reset_index()
votes["share"] = votes.v / votes.groupby("ORIGIN_AIRPORT_id").v.transform("sum")
best = votes.sort_values("v", ascending=False).drop_duplicates("ORIGIN_AIRPORT_id")
good = best[best.share >= CONFIDENT]
good = good[~good.ORIGIN_AIRPORT_iata.duplicated(keep=False)]
mapping = {str(a): b for a, b in zip(good.ORIGIN_AIRPORT_id, good.ORIGIN_AIRPORT_iata)}
del raw, base, ref, m

df = load_flight_data_final(None)
df = df.drop(columns=["TAIL_NUMBER"]).reset_index(drop=True)
oct_rows = df.MONTH == 10
before = df.loc[oct_rows, "ORIGIN_AIRPORT"].str.fullmatch(r"\d+").mean()
for c in ("ORIGIN_AIRPORT", "DESTINATION_AIRPORT"):
    df.loc[oct_rows, c] = df.loc[oct_rows, c].map(lambda v: mapping.get(v, v))
left_o = int(df.loc[oct_rows, "ORIGIN_AIRPORT"].str.fullmatch(r"\d+").sum())
left_d = int(df.loc[oct_rows, "DESTINATION_AIRPORT"].str.fullmatch(r"\d+").sum())

idx = np.arange(len(df))
pool, hold = train_test_split(idx, test_size=0.2, random_state=42, stratify=df.target)
df["split"] = "pool"
df.loc[hold, "split"] = "hold"
df.to_pickle(PREPARED)

info = {
    "rows": len(df), "delay_share": round(float(df.target.mean()), 4),
    "pool": len(pool), "holdout": len(hold),
    "ids_total": int(best.shape[0]), "ids_mapped": len(mapping),
    "october_numeric_share_before": round(float(before), 4),
    "october_rows": int(oct_rows.sum()), "october_origin_left_numeric": left_o, "october_dest_left_numeric": left_d,
    "ids_left_out": best[~best.ORIGIN_AIRPORT_id.astype(str).isin(mapping)][["ORIGIN_AIRPORT_id", "ORIGIN_AIRPORT_iata", "share"]].astype(str).to_dict("records"),
}
with open(os.path.join(WORK, "prep_info.json"), "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
