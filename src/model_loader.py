# Askar Turanazarov

# src/model_loader.py
# Locates the trained pipeline. A local copy in models/ wins; otherwise the file is
# downloaded once from the Hugging Face Hub into the standard HF cache. This lets the
# same code run on a laptop (local pkl), in Docker and in a Hugging Face Space
# (no pkl in the image or repository).

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILENAME = "model_pipeline_main.pkl"
LOCAL_MODEL_PATH = BASE_DIR / "models" / MODEL_FILENAME

# Override with environment variables when the model lives in another repo/file.
DEFAULT_HF_REPO = "AskarT/flight-delay-catboost"
ENV_REPO = "HF_MODEL_REPO"
ENV_FILE = "HF_MODEL_FILE"


def hf_repo_id() -> str:
    return os.getenv(ENV_REPO, DEFAULT_HF_REPO)


def model_source() -> str:
    """'local' if the pkl sits in models/, otherwise 'hub'."""
    return "local" if LOCAL_MODEL_PATH.is_file() else "hub"


def resolve_model_path() -> Path:
    """Return a filesystem path to the pipeline pickle, downloading it if needed."""
    if LOCAL_MODEL_PATH.is_file():
        return LOCAL_MODEL_PATH

    from huggingface_hub import hf_hub_download  # imported lazily: not needed for local runs

    cached = hf_hub_download(
        repo_id=hf_repo_id(),
        filename=os.getenv(ENV_FILE, MODEL_FILENAME),
    )
    return Path(cached)
