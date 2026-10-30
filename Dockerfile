# Flight Delay Risk Service - Streamlit app image
# Build:  docker build -t flt-delay-app .
# Run:    docker run -p 8501:8501 flt-delay-app
#         (the model is downloaded from Hugging Face Hub on first start;
#          to use a local copy instead: -v "$PWD/models:/app/models")
# The same image runs on Hugging Face Spaces (Docker SDK, app_port 8501).

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Non-root user with uid 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

WORKDIR /app

# Dependencies first so the layer is cached between code changes.
# PIP_INDEX_URL can point to a PyPI mirror when the default CDN is slow:
#   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t flt-delay-app .
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (filtered by .dockerignore: no venv, data, model weights, notebooks)
COPY --chown=user:user . .

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
