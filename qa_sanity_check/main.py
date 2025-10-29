#!/usr/bin/env python3
"""Minimal Hydra+LiteLLM client for SGLang chat completions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
import re


import hydra
import litellm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def _resolve_request(
    cfg: DictConfig,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    request = OmegaConf.to_container(cfg.request, resolve=True)
    if not isinstance(request, dict):
        raise TypeError("request config must be a mapping")

    try:
        messages = request.pop("messages")
    except KeyError as exc:
        raise RuntimeError("request.messages must be provided") from exc
    if not isinstance(messages, list):
        raise TypeError("request.messages must be a list")

    model = request.pop("model", None)
    if not model:
        raise RuntimeError("request.model must be a non-empty string")

    if cfg.request.instance_type == 'maxon':
        if (
            cfg.request.instance == '01_log_solution_within_1000'
            or cfg.request.instance == '04_specific_log_file'
        ):

            # Change system message
            messages[0][
                'content'
            ] = "You are an expert providing assistance about a support ticket."

            # breakpoint()

            base_dir = os.path.dirname(__file__)
            logfile_path = os.path.join(
                base_dir,
                f'tasks/maxon/{cfg.request.instance}/{cfg.request.ticket_file}.txt',
            )

            with open(logfile_path, 'r') as file:
                logfile_content = file.read()

            instruction = f"""The ticket includes a logfile that contains the information about the issue. You need to identify the issue and suggest solutions for it. Here is the content of the logfile: \n {logfile_content}
            """

            instruction += """\n If you provide recommendations for a solution, be as specific as possible."""

            # Change user message
            messages[1]['content'] = instruction

    return model, messages, request


def _call_model(cfg: DictConfig) -> tuple[dict[str, Any], str | None]:
    model, messages, params = _resolve_request(cfg)

    params.setdefault("api_base", cfg.api.base_url.rstrip("/"))

    # if cfg.api.provider:
    #     params["custom_llm_provider"] = cfg.api.provider

    key_env = cfg.api.key_env
    if key_env:
        api_key = os.getenv(key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {key_env} is not set")
        params["api_key"] = api_key
    else:
        # LiteLLM expects an api_key even for keyless endpoints. IMPORTANT, otherwise LITELLM ERROR!
        params.setdefault("api_key", "sk-no-key-required")

    response = litellm.completion(model=model, messages=messages, **params)

    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif isinstance(response, dict):
        data = response
    else:
        raise TypeError("Unexpected LiteLLM response type")

    message = data["choices"][0]["message"]["content"].strip()

    return data, message


def _save_run(cfg: DictConfig, payload: dict[str, Any], message: str | None) -> None:
    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    # breakpoint()

    # Append ground truth to payload
    base_dir = os.path.dirname(__file__)
    # base_path = Path('qa_sanity_check/tasks/maxon/')

    base_path = os.path.join(
        base_dir,
        f'tasks/maxon/',
    )

    def extract_ground_truth_from_task():
        data_path = (
            f'{base_path}/{cfg.request.instance}/{cfg.request.ticket_file}_gt.md'
        )
        with open(data_path, 'r') as f:
            content = f.read()
        # Extract everything after '# Expected Result'
        match = re.search(r'# Expected Result\s*(.*)', content, flags=re.DOTALL)
        if match:
            expected_result = match.group(1).strip()
        return expected_result

    if cfg.request.instance == '01_log_solution_within_1000':
        payload['ground_truth'] = extract_ground_truth_from_task()

    if cfg.request.instance == '04_specific_log_file':
        payload['ground_truth'] = extract_ground_truth_from_task()

    if cfg.output.save_json:
        (run_dir / "response.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if message is not None and cfg.output.save_text:
        (run_dir / "response.txt").write_text(f"{message}\n", encoding="utf-8")

    OmegaConf.save(cfg, str(run_dir / "config_resolved.yaml"), resolve=True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    data, message = _call_model(cfg)

    print(json.dumps(data, indent=2, ensure_ascii=False))
    print(message)

    _save_run(cfg, data, message)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # pragma: no cover - concise CLI error surfacing
        print(error, file=sys.stderr)
        sys.exit(1)
