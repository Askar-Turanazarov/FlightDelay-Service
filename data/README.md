# Data

| File | In repo | Purpose |
|---|---|---|
| `airlines.csv` | yes | IATA code → airline name (14 carriers) |
| `airports.csv` | yes | IATA code → airport name, city, state, coordinates (used for the map) |
| `flights.csv` | **no** (565 MB) | 5.8 M US domestic flights of 2015; needed only to retrain the model |

Source: Kaggle — [2015 Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays)
(US Department of Transportation). Download the full dataset into this folder with the
[Kaggle CLI](https://github.com/Kaggle/kaggle-api):

```bash
kaggle datasets download -d usdot/flight-delays -p data --unzip
```

`flights.csv` is listed in `.gitignore` and `.dockerignore`; never commit it.
The running app does not read it: the routes, airports and airlines it offers come from
`models/lookup_data.joblib`, which was derived from the training subset.
