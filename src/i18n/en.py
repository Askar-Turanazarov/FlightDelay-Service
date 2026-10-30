# English UI strings (reference language; other files must define the same keys).

STRINGS = {
    # --- page & sidebar ---
    "app.title": "📊 Flight delay risk analytics",
    "sidebar.header": "📝 Flight parameters",
    "sidebar.language": "Language",
    "sidebar.airline": "Airline",
    "sidebar.origin": "Origin airport",
    "sidebar.destination": "Destination airport",
    "sidebar.month": "Month",
    "sidebar.day": "Day",
    "sidebar.dep_hour": "Departure hour",
    "sidebar.unknown": "Unknown",
    "model.loading": "Loading the model (downloaded from Hugging Face on the first start)...",

    # --- tabs ---
    "tab.analysis": "🚀 Analysis & forecast",
    "tab.model": "📈 Model evidence",

    # --- top panel ---
    "route.header": "**📍 Route**",
    "flight.header": "**✈️ Flight details**",
    "flight.distance": "Distance: **{km} km**",
    "flight.time": "Flight time: **{duration}**",
    "airline.header": "**🛩 Airline**",
    "airline.iata": "IATA code: **{code}**",
    "airline.unknown": "Unknown airline",
    "duration.h_m": "{h} h {m} min",
    "duration.m": "{m} min",

    # --- prediction ---
    "button.analyze": "RUN RISK ANALYSIS",
    "risk.level": "Risk level",
    "risk.low": "🔵 Low",
    "risk.moderate": "🟡 Moderate",
    "risk.high": "🟠 High",
    "risk.critical": "🔴 Critical",
    "risk.index_bar": "Risk index (visual scale): {idx}",
    "metric.prob": "Delay probability",
    "metric.prob_help": "Raw model probability. Decision threshold: {thr}.",
    "metric.rel": "vs. average delay rate ({base})",
    "rel.lower": "below the average delay rate",
    "rel.typical": "around the average delay rate",
    "rel.higher": "above the average delay rate",
    "verdict.delay": "Model verdict: a delay is likely (probability ≥ {thr}).",
    "verdict.ontime": "Model verdict: a delay is unlikely (probability < {thr}).",
    "econ.risk_value": "Expected financial loss (risk value): **${value}**",
    "insight": "💡 **Insight:** the main factor today is **{feature}**, which {direction} the delay risk.",
    "direction.up": "increases",
    "direction.down": "decreases",

    # --- economics breakdown ---
    "econ.expander": "Breakdown of the potential loss",
    "econ.total": "Total loss if the delay occurs: **${value}**",
    "econ.formula": (
        "Expected loss = delay probability × total loss if the delay occurs.\n\n"
        "Here: `{expected} USD` at a probability of `{prob}` of a total loss of `{total} USD`."
    ),
    "econ.network": "Knock-on (network) effect",
    "econ.direct": "Direct airline costs",
    "econ.pax": "Passenger time loss",
    "econ.method": "Method: FAA Value of Travel Time Savings (VTTS) & airline operating costs.",

    # --- SHAP ---
    "shap.expander": "🔍 Why this result? (SHAP factor analysis)",
    "shap.intro": "Contribution of each factor to the final probability (Shapley values) for the flight from {origin} to {dest}:",
    "shap.axis_feature": "Feature",
    "shap.axis_contrib": "Contribution",
    "shap.caption": "Positive values (red) increase the delay risk, negative values (green) decrease it.",

    # --- feature names ---
    "feature.MONTH": "Month",
    "feature.DAY": "Day of month",
    "feature.DAY_OF_WEEK": "Day of week",
    "feature.AIRLINE": "Airline",
    "feature.ORIGIN_AIRPORT": "Origin airport",
    "feature.DESTINATION_AIRPORT": "Destination airport",
    "feature.DEP_HOUR": "Departure hour",
    "feature.ARR_HOUR": "Arrival hour",
    "feature.SCHEDULED_TIME": "Scheduled flight time",
    "feature.DISTANCE": "Distance",
    "feature.ROUTE": "Route",
    "feature.plan_speed": "Planned speed",
    "feature.SCHEDULED_ARRIVAL": "Scheduled arrival",
    "feature.SCHEDULED_DEPARTURE": "Scheduled departure",
    "feature.hour_sin": "Hour (sin)",
    "feature.hour_cos": "Hour (cos)",
    "feature.traffic_density": "Airport traffic (per hour)",
    "feature.AIRLINE_FLIGHT_COUNT": "Airline flights at airport (per day)",
    "feature.IS_NIGHT": "Night flight",
    "feature.IS_WEEKEND": "Weekend",

    # --- metrics tab ---
    "metrics.header": "Model quality (trained on 2.1 M of 5.8 M rows)",
    "metrics.loaded.compact": "Loaded model: **compact** (max_ctr_complexity=3, {size}) — the variant served by the public demo.",
    "metrics.loaded.full": "Loaded model: **full** (the diploma reference model, {size}).",
    "metrics.auc": "ROC-AUC on the test set: **{auc}**",
    "metrics.recall": "Recall at threshold 0.37: **{recall}** (precision {precision})",
    "metrics.variants_header": "Two model variants",
    "metrics.variants_text": "Both models share the data, the split and every training parameter; the only difference "
                             "is how deep CatBoost combines categorical features in its CTR tables "
                             "(`max_ctr_complexity`: 4 vs 3). The compact one needs 2.7× less RAM and fits a free host, "
                             "at a cost of 0.007 ROC-AUC. The full model remains the reference result of the diploma.",
    "metrics.col_metric": "Metric",
    "metrics.col_full": "Full model",
    "metrics.col_compact": "Compact model",
    "metrics.row_auc": "ROC-AUC",
    "metrics.row_recall": "Recall @ 0.37",
    "metrics.row_precision": "Precision @ 0.37",
    "metrics.row_size": "Pipeline size (pkl)",
    "metrics.row_ram": "RAM after loading",
    "metrics.row_ctr": "CTR combinations (max_ctr_complexity)",
    "metrics.row_script": "Training script",
    "metrics.figures_note": "The figures below were produced for the full model during the diploma work.",
    "metrics.pr_caption": "Choosing the classification threshold (precision–recall)",
    "metrics.fi_caption": "Feature importance (CatBoost)",
}
