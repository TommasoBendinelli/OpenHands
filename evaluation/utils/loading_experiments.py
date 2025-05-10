from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Optional, Tuple
import re
from omegaconf import OmegaConf
import json

METADATA_JSON = 'metadata.json'
OUTPUT_JSON = 'output.jsonl'


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


# Regex that captures both the timestamp and the run-index
_TS_WITH_IDX_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(?P<idx>\d+)$'
)


def get_folders_in_range(
    base_dir: Path, after: Optional[datetime], before: Optional[datetime]
) -> list[Path]:
    """
    Get list of subfolders matching the timestamp+index regex and falling within a datetime range.

    Parameters:
        base_dir (Path): Root path to search in.
        after (Optional[datetime]): Start of the range (inclusive). Pass None to skip lower bound.
        before (Optional[datetime]): End of the range (inclusive). Pass None to skip upper bound.

    Returns:
        List[Path]: List of subfolder paths within the range.
    """

    matching_paths = []

    if after == None:
        after = datetime.strptime('2025-05-04_00-00-32', '%Y-%m-%d_%H-%M-%S')

    if before == None:
        before = datetime.today()

    for date_folder in base_dir.iterdir():
        if not date_folder.is_dir():
            continue

        try:
            folder_date = datetime.strptime(date_folder.name, "%Y-%m-%d")
        except ValueError:
            continue  # skip folders that don't match top-level date format

        # Early skip if outside overall date range
        if after and folder_date.date() < after.date():
            continue
        if before and folder_date.date() > before.date():
            continue

        for subfolder in date_folder.iterdir():
            if not subfolder.is_dir():
                continue

            match = _TS_WITH_IDX_RE.match(subfolder.name)
            if not match:
                continue

            ts_str = match.group("ts")
            try:
                ts_dt = datetime.strptime(ts_str, "%Y-%m-%d_%H-%M-%S")
            except ValueError:
                continue

            if (not after or ts_dt >= after) and (not before or ts_dt <= before):
                matching_paths.append(subfolder)

    return matching_paths
