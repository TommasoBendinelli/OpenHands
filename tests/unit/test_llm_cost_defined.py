from pathlib import Path

from openhands.core.config import AppConfig
from openhands.core.config.utils import load_from_toml


def _load_config() -> AppConfig:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / 'config.toml'
    if not config_path.exists():
        config_path = repo_root / 'config.template.toml'

    cfg = AppConfig()
    load_from_toml(cfg, str(config_path))
    return cfg


def test_cost_defined_for_models(monkeypatch):
    """Ensure every configured LLM has cost information available."""

    # Use the local model cost map bundled with litellm to avoid network calls
    monkeypatch.setenv('LITELLM_LOCAL_MODEL_COST_MAP', 'true')

    from litellm import completion_cost as litellm_completion_cost

    cfg = _load_config()

    for name, llm_conf in cfg.llms.items():
        model_name = llm_conf.model

        has_custom_cost = (
            llm_conf.input_cost_per_token is not None
            and llm_conf.output_cost_per_token is not None
        )

        try:
            litellm_completion_cost(model=model_name, prompt='', completion='')
            has_known_cost = True
        except Exception:
            has_known_cost = False

        assert has_custom_cost or has_known_cost, (
            f"Cost not defined for model '{model_name}' in config section '{name}'"
        )
