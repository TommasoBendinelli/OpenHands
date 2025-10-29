# OpenAI-compatible chat script

This repository provides a minimal Hydra-driven CLI for calling an OpenAI-compatible chat completion endpoint through `litellm`. The defaults in `conf/config.yaml` point at an SGLang deployment (`http://10.55.64.55:30000/v1`) with the `openai/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` model, but every field can be overridden at runtime.

## Usage

1. Install dependencies (for example via pip):
   ```bash
   pip install hydra-core==1.3.2 litellm==1.43.6
   ```
2. (Optional) If the target server requires authentication, export the expected credentials and either edit `conf/config.yaml` or supply `api.key_env=<VAR_NAME>` as a Hydra override when running. When `api.key_env` is not set the script passes a placeholder key (`sk-no-key-required`) so LiteLLM can call keyless backends:
   ```bash
   export OPENAI_API_KEY="sk-your-key"
   export OPENAI_ORG_ID="org-optional"
   ```
3. Run the script:
   ```bash
   python main.py
   ```
   Override any configuration value by appending `hydra` overrides, e.g.:
   ```bash
   python main.py request.messages[1].content="Tell me a joke" request.temperature=0.8
   ```
   Override the serving endpoint or provider by supplying values such as `api.base_url=https://my-host/v1` or `api.provider=openai`.

Each run stores its artifacts under `evaluation/outputs/<timestamp>/`, including:
- `config_resolved.yaml`: the resolved Hydra configuration for the run
- `response.json`: the raw API JSON payload (if `output.save_json` is true)
- `response.txt`: the extracted assistant message when available (if `output.save_text` is true)

Set `output.show_full_response=true` to print the full JSON payload returned by the API for debugging.
