"""
llm_client.py – April 2025
Python 3.12 compatible utility for GPT‑4o‑mini & Gemini 2.0 models
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())  # automatically walks up folders
import pandas as pd
from omegaconf import OmegaConf

# ---------- OpenAI ---------------------------------------------------------
try:
    import openai  # SDK ≥1.75.0

    _openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except ImportError:
    _openai_client = None

# ---------- Gemini / Google Gen AI ----------------------------------------
try:
    import google.genai as genai  # unified SDK ≥1.11.0

    _genai_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
except ImportError:
    _genai_client = None

# --------------------------------------------------------------------------

Provider = Literal['openai', 'gemini']


def llm_complete(
    instruction: str,
    provider: Provider = 'openai',
    model: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Return an LLM response as plain text.

    Parameters
    ----------
    instruction : str
        The user instruction.
    provider : "openai" | "gemini"
        Which backend to call.
    model : str | None
        Override the default model (see table above).
    **kwargs
        Extra settings forwarded to the SDK:
          * OpenAI → temperature, max_tokens, etc.
          * Gemini → generation_config=dict(...), safety_settings=[...], etc.

    Raises
    ------
    ImportError / RuntimeError / ValueError if a dependency is missing
    or a provider string is wrong.
    """
    if provider == 'openai':
        if _openai_client is None:
            raise ImportError('`openai` package not installed.')
        model = model or 'gpt-4o-mini'
        rsp = _openai_client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': instruction}],
            **kwargs,
        )
        return rsp.choices[0].message.content.strip()

    elif provider == 'gemini':
        if _genai_client is None:
            raise ImportError('`google-genai` package not installed.')
        if not os.getenv('GOOGLE_API_KEY'):
            raise RuntimeError('GOOGLE_API_KEY is missing.')
        model = model or 'gemini-2.0-flash'  # ← fastest; swap for pro if needed
        rsp = _genai_client.models.generate_content(
            model=model,
            contents=instruction,
            **kwargs,
        )
        # unified SDK exposes .text on successful responses
        return rsp.text.strip()

    else:
        raise ValueError(f'Unknown provider: {provider!r}')


def _iter_result_folders(root: Path, after: datetime) -> Iterable[Path]:
    """Yield sub‑folders whose *name* looks like a timestamp newer than *after*."""
    for p in root.iterdir():
        if not p.is_dir():
            continue

        # If it does not start with YYYY skip
        if not p.name.startswith('20'):
            continue
        ts = datetime.strptime(p.name, '%Y-%m-%d_%H-%M-%S')

        if ts > after:
            yield p


def _load_experiment(folder: Path) -> tuple[dict, dict]:
    meta, out = {}, {}
    meta_path = folder / METADATA_JSON
    output_path = folder / OUTPUT_JSON

    cfg = OmegaConf.load(folder / '.hydra' / 'config.yaml')
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as e:
            print(f'[WARN] Bad JSON in {meta_path}: {e}')
    else:
        print(f'[INFO] {meta_path} missing')

    if output_path.exists():
        try:
            with output_path.open(encoding='utf-8') as f:
                out = {i: json.loads(line) for i, line in enumerate(f) if line.strip()}
        except json.JSONDecodeError as e:
            print(f'[WARN] Bad JSON line in {output_path}: {e}')
    else:
        print(f'[INFO] {output_path} missing')

    return meta, out, cfg


# Get all the entries evaluation/evaluation_outputs/outputs
ROOT_DIR = Path('evaluation/evaluation_outputs/outputs')
CUTOFF = datetime.strptime('2025-04-30_15-32-20', '%Y-%m-%d_%H-%M-%S')
METADATA_JSON = 'metadata.json'
OUTPUT_JSON = 'output.jsonl'

solutions = {
    'find_peaks': 'The correct feature is to use the number of peaks in the signal to tell the class of the signal.',
    'predict_ts_stationarity': 'The correct feature is to tell whether the time series is stationary or not.',
}


def main():
    folders = sorted(_iter_result_folders(ROOT_DIR, CUTOFF))
    # Iterate over the folders
    res = {}
    entries_df = []
    for folder in folders:
        # Open metadata
        metadata, outputs, cfg = _load_experiment(folder)

        msg_txt = '\n'.join([x['message'] for x in outputs[0]['history'][2:]])
        breakpoint()
        metric = outputs[0]['test_result']['result']['metric']
        res = {
            'instance': cfg['instance'],
            'metric': metric,
            'solution': solutions[cfg['instance']],
            'messages': msg_txt,
        }

    df = pd.DataFrame.from_records([res])
    breakpoint()


if __name__ == '__main__':
    main()
