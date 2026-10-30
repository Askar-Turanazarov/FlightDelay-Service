# Askar Turanazarov

# src/analytics.py
# Post-processing of CatBoost predictions for the UI:
#   * risk_index     - non-linear stretch of the raw probability for the visual gauge. It is a
#                      display transform, NOT a probability calibration: decisions and money are
#                      always computed from the raw model probability.
#   * relative_risk  - raw probability compared with the dataset base rate (18.9 % of flights in
#                      the 2015 data arrived more than 15 minutes late).
#   * get_local_shap - per-flight SHAP contributions via CatBoost's built-in ShapValues.

import numpy as np
import pandas as pd
import catboost as cb

BASE_DELAY_RATE = 0.189      # share of delayed flights in the training data
DECISION_THRESHOLD = 0.37    # chosen on the precision-recall curve for Recall ~ 0.82


class FlightAnalytics:
    """Helpers that turn a fitted pipeline (FlightFeatureEngineer + CatBoost) into UI-ready numbers."""

    def __init__(self, model, base_rate=BASE_DELAY_RATE, threshold=DECISION_THRESHOLD):
        self.model = model
        self.base_rate = base_rate
        self.threshold = threshold

    # ---------- probability-derived numbers ----------

    def is_delay_predicted(self, prob: float) -> bool:
        """Binary verdict at the tuned threshold (Recall-oriented)."""
        return prob >= self.threshold

    def risk_index(self, prob: float, contrast: float = 1.8) -> float:
        """
        Visual risk index in [0, 1] for the gauge / progress bar.

        A sigmoid applied to the scaled logit, centred on the decision threshold: the middle of
        the bar (0.5) corresponds exactly to `threshold`, and the typical 0.1-0.5 range of model
        outputs is spread over the whole bar so differences between flights are easier to see.
        Monotonic in `prob`, so ordering is preserved. It is not a calibrated probability.
        """
        eps = 1e-5

        def logit(p):
            return np.log((p + eps) / (1 - p + eps))

        return float(1 / (1 + np.exp(-contrast * (logit(prob) - logit(self.threshold)))))

    def relative_risk(self, prob: float):
        """
        Compare the raw probability with the dataset base rate.

        Returns (ratio, status) where ratio = prob / base_rate and status is one of
        'lower' (< 0.8x), 'typical' (0.8x-1.25x) or 'higher' (> 1.25x).
        """
        ratio = prob / self.base_rate
        if ratio < 0.8:
            status = "lower"
        elif ratio <= 1.25:
            status = "typical"
        else:
            status = "higher"
        return ratio, status

    # ---------- explainability ----------

    def get_local_shap(self, input_df: pd.DataFrame, top_n: int = 7) -> pd.DataFrame:
        """
        SHAP contributions of each feature to one prediction, using CatBoost's fast built-in
        get_feature_importance(type='ShapValues'). Returns the `top_n` features by absolute
        contribution with technical columns: 'feature', 'contribution', 'abs_contribution'.
        """
        if hasattr(self.model, "named_steps"):
            actual_model = self.model.named_steps["clf"]
            transformer = self.model.named_steps["feature_eng"]
            processed = transformer.transform(input_df)
            if isinstance(processed, np.ndarray):
                processed = pd.DataFrame(processed, columns=transformer.get_feature_names_out())
        else:
            actual_model = self.model
            processed = input_df

        # CatBoost keeps the categorical feature indices inside the model; an explicit Pool
        # avoids "Cannot convert 'AS' to float" on string categories.
        cat_features = actual_model.get_cat_feature_indices()
        pool = cb.Pool(data=processed, cat_features=cat_features)

        shap_values = actual_model.get_feature_importance(data=pool, type="ShapValues")[0]
        contributions = shap_values[:-1]  # last element is the expected value (bias)

        shap_df = pd.DataFrame({
            "feature": [str(c).split("__")[-1] for c in processed.columns],
            "contribution": contributions,
        })
        shap_df["abs_contribution"] = shap_df["contribution"].abs()
        return shap_df.sort_values("abs_contribution", ascending=False).head(top_n).reset_index(drop=True)
