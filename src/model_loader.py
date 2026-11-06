# Askar Turanazarov

# src/model_loader.py
# Registry of the model variants and where their files come from. A local copy in models/ wins;
# otherwise the pickle is downloaded once from the Hugging Face Hub into the standard HF cache. This
# lets the same code run on a laptop (local pkl), in Docker and on a free cloud host (no pkl in the
# image or repository).
#
# Three variants exist; the visitor picks one in the sidebar:
#   weather  model_pipeline_weather.pkl  40 MB,   ~0.4 GB RAM  flight + weather at both airports, whole of 2015,
#                                                               calibrated, ROC-AUC 0.698 on 73 unseen days
#   compact  model_pipeline_compact.pkl  ~115 MB, ~0.6 GB RAM  diploma model, max_ctr_complexity=3 (train_compact.py)
#   full     model_pipeline_main.pkl     433 MB,  ~1.5 GB RAM  diploma reference model (train_final.py)
# The full model does not fit the free demo host, so it is offered only where its file lies in models/
# or MODEL_VARIANT=full asks for it. MODEL_VARIANT sets the variant opened by default;
# HF_MODEL_REPO overrides the source repository and HF_MODEL_FILE the file of that default variant.

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

VARIANTS = {
    "weather": {"file": "model_pipeline_weather.pkl", "calibration": "calibration_weather.json", "weather": True},
    "compact": {"file": "model_pipeline_compact.pkl", "calibration": None, "weather": False},
    "full": {"file": "model_pipeline_main.pkl", "calibration": None, "weather": False},
}
DEFAULT_VARIANT = "weather"
ALWAYS_OFFERED = ("weather", "compact")

# Weather inputs of the weather model with the values a form may offer, in display order. "unknown" also
# exists in the training data (under 0.1 % of flights) and is never offered.
WEATHER_VALUES = {
    "WIND": ("light", "moderate", "strong", "gale"),
    "VISIBILITY": ("good", "reduced", "poor"),
    "PRECIP": ("none", "rain", "snow", "freezing"),
    "HAZARD": ("none", "fog", "thunderstorm", "icing", "other"),
}
# Column order the weather pipeline was trained on, after the 13 flight columns.
WEATHER_COLUMNS = [f"{end}_{kind}" for end in ("ORIGIN", "DEST") for kind in WEATHER_VALUES]

DEFAULT_HF_REPO = "AskarT/flight-delay-catboost"
ENV_VARIANT = "MODEL_VARIANT"
ENV_REPO = "HF_MODEL_REPO"
ENV_FILE = "HF_MODEL_FILE"


def default_variant() -> str:
    """'weather' unless the MODEL_VARIANT environment variable names another variant."""
    variant = os.getenv(ENV_VARIANT, DEFAULT_VARIANT).strip().lower()
    if variant not in VARIANTS:
        raise ValueError(f"{ENV_VARIANT}={variant!r}; expected one of {sorted(VARIANTS)}")
    return variant


def model_filename(variant: str) -> str:
    """File name of the pipeline pickle; HF_MODEL_FILE replaces it for the default variant only."""
    if variant == default_variant() and os.getenv(ENV_FILE):
        return os.environ[ENV_FILE]
    return VARIANTS[variant]["file"]


def local_model_path(variant: str) -> Path:
    return MODELS_DIR / model_filename(variant)


def available_variants() -> list:
    """Variants offered in the UI: weather and compact always, full only where it can actually be served."""
    return [v for v in VARIANTS
            if v in ALWAYS_OFFERED or v == default_variant() or local_model_path(v).is_file()]


def uses_weather(variant: str) -> bool:
    return VARIANTS[variant]["weather"]


def hf_repo_id() -> str:
    return os.getenv(ENV_REPO, DEFAULT_HF_REPO)


def model_source(variant: str) -> str:
    """'local' if the pkl sits in models/, otherwise 'hub'."""
    return "local" if local_model_path(variant).is_file() else "hub"


def resolve_model_path(variant: str) -> Path:
    """Return a filesystem path to the pipeline pickle, downloading it if needed."""
    local = local_model_path(variant)
    if local.is_file():
        return local

    from huggingface_hub import hf_hub_download  # imported lazily: not needed for local runs

    cached = hf_hub_download(repo_id=hf_repo_id(), filename=model_filename(variant))
    return Path(cached)


def load_calibration(variant: str):
    """Isotonic calibration of the variant ({'x', 'y', 'threshold', 'bands', 'base_rate'}) or None.
    A small JSON kept in git next to lookup_data.joblib; the raw output maps through np.interp(raw, x, y)."""
    name = VARIANTS[variant]["calibration"]
    if name is None:
        return None
    with open(MODELS_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)
