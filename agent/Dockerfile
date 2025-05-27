FROM docker.all-hands.dev/all-hands-ai/runtime:0.39-nikolaik

WORKDIR /workspace
COPY . /workspace

RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir -e . && python -m pip install hydra-core hydra-joblib-launcher

ENV PATH="/workspace/.venv/bin:${PATH}"