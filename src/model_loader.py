# Askar Turanazarov

# src/model_loader.py
# Locates the trained pipeline. A local copy in models/ wins; otherwise the file is
# downloaded once from the Hugging Face Hub into the standard HF cache. This lets the
# same code run on a laptop (local pkl), in Docker and on a free cloud host
# (no pkl in the image or repository).
#
# Two variants of the model exist, both trained on the same data and split:
#   compact  model_pipeline_compact.pkl  ~115 MB, ~0.6 GB RAM, ROC-AUC 0.779  (train_compact.py)
#   full     model_pipeline_main.pkl     433 MB,  ~1.5 GB RAM, ROC-AUC 0.786  (train_final.py)
# The compact one is the default because the public demo runs on a 1 GB host. Select the
# full model with MODEL_VARIANT=full; HF_MODEL_REPO / HF_MODEL_FILE override the source.

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

VARIANTS = {
    "compact": "model_pipeline_compact.pkl",
    "full": "model_pipeline_main.pkl",
}
DEFAULT_VARIANT = "compact"

DEFAULT_HF_REPO = "AskarT/flight-delay-catboost"
ENV_VARIANT = "MODEL_VARIANT"
ENV_REPO = "HF_MODEL_REPO"
ENV_FILE = "HF_MODEL_FILE"


def model_variant() -> str:
    """'compact' (default) or 'full', from the MODEL_VARIANT environment variable."""
    variant = os.getenv(ENV_VARIANT, DEFAULT_VARIANT).strip().lower()
    if variant not in VARIANTS:
        raise ValueError(f"{ENV_VARIANT}={variant!r}; expected one of {sorted(VARIANTS)}")
    return variant


def model_filename() -> str:
    """File name of the pipeline pickle: HF_MODEL_FILE if set, else the chosen variant's file."""
    return os.getenv(ENV_FILE) or VARIANTS[model_variant()]


def local_model_path() -> Path:
    return MODELS_DIR / model_filename()


def hf_repo_id() -> str:
    return os.getenv(ENV_REPO, DEFAULT_HF_REPO)


def model_source() -> str:
    """'local' if the pkl sits in models/, otherwise 'hub'."""
    return "local" if local_model_path().is_file() else "hub"


def resolve_model_path() -> Path:
    """Return a filesystem path to the pipeline pickle, downloading it if needed."""
    local = local_model_path()
    if local.is_file():
        return local

    from huggingface_hub import hf_hub_download  # imported lazily: not needed for local runs

    cached = hf_hub_download(repo_id=hf_repo_id(), filename=model_filename())
    return Path(cached)
