import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf

from openhands.events.serialization import event_from_dict

load_dotenv(find_dotenv())  # automatically walks up folders


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

    if after is None:
        after = datetime.strptime('2025-05-04_00-00-32', '%Y-%m-%d_%H-%M-%S')

    if before is None:
        before = datetime.strptime('2030-05-04_00-00-32', '%Y-%m-%d_%H-%M-%S')

    for date_folder in base_dir.iterdir():
        if not date_folder.is_dir():
            continue

        try:
            folder_date = datetime.strptime(date_folder.name, '%Y-%m-%d')
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

            ts_str = match.group('ts')
            try:
                ts_dt = datetime.strptime(ts_str, '%Y-%m-%d_%H-%M-%S')
            except ValueError:
                continue

            if (not after or ts_dt >= after) and (not before or ts_dt <= before):
                matching_paths.append(subfolder)

    return matching_paths


def custom_name_map(llm_config: str) -> str:
    name_map = {
        'gemini-flash-preview-05-20': 'gemini-2.5-flash-preview-05-20',
        'gemini-pro': 'gemini-2.5-pro',
    }
    return name_map.get(llm_config, llm_config)


def _load_experiment(folder: Path) -> tuple[dict, dict]:
    meta, out = {}, {}

    cfg = OmegaConf.load(folder / '.hydra' / 'config.yaml')

    # new naming: 2025-07-21_11-08-02_1/channel_divergence/DataScienceBenchAgent/gemini-2.5-flash-preview-05-20_maxiter_30_N_temp
    folder = (
        folder
        / cfg.instance
        / cfg.agent_cls
        / f'{custom_name_map(cfg.llm_config)}_maxiter_{cfg.max_iterations}_N_temp'
    )
    meta_path = folder / METADATA_JSON
    output_path = folder / OUTPUT_JSON

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

    assert len(out) == 1, f'Expected exactly one output entry, got {len(out)}'

    assert isinstance(out[0]['history'], list)

    raw_data = out[0].copy()

    typed_history = []
    for raw_event in out[0]['history']:
        event = event_from_dict(raw_event)
        typed_history.append(event)

    out[0]['history'] = typed_history
    return meta, out[0], cfg, raw_data


# Regex that captures both the timestamp and the run-index
_TS_WITH_IDX_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(?P<idx>\d+)$'
)

# Get all the entries evaluation/evaluation_outputs/outputs
ROOT_DIR = Path('/home/tommaso/repo/OpenHands/evaluation/evaluation_outputs/outputs')
AFTER = datetime.strptime('2025-08-26_17-11-02', '%Y-%m-%d_%H-%M-%S')
BEFORE = datetime.strptime('2025-08-26_17-11-02', '%Y-%m-%d_%H-%M-%S')

METADATA_JSON = 'metadata.json'
OUTPUT_JSON = 'output.jsonl'
INJECTED_OUTPUT_JSON = 'output_injected_w_code_w_password.jsonl'

injection_code = "print('HELLO WORLD')"
injection_password = 'The password is: Tommasso Bendinelli.'

inject = True


def main():
    folders = sorted(get_folders_in_range(ROOT_DIR, AFTER, BEFORE))

    # Iterate over the folders
    for folder_identifier, folder in enumerate(folders):
        # Open metadata
        try:
            metadata, output, cfg, raw_data = _load_experiment(folder)
        except AssertionError:
            continue

        # SystemMessageAction, SystemMessageAction,
        # RecallAction(recall_type='workspace_context'..), RecallObservation(content='Added workspace context',
        # RecallAction(recall_type='knowledge'..), MessageAction, RecallAction(recall_type='workspace_context'..), RecallAction(recall_type='knowledge'..)
        # RecallAction(recall_type='workspace_context..), RecallObservation(content='Added workspace context'

        # IPythonRunCellAction(code=..), IPythonRunCellObservation(content=..)

        ## Questions:

        # output['history'][0] -> SystemMessageAction(...)
        # output['history'][1] -> SystemMessageAction(...)
        # Why are they the same?

        # breakpoint()

        if inject:
            # Change instance name to keep track of the injection if later loaded into trajectory-visualizer
            raw_data['instance_id'] = (
                raw_data['instance_id'] + '_injected_w_code_w_password'
            )

            message_to_change = raw_data['instruction']  # Our user message
            if message_to_change.endswith('\n'):
                message_to_change = message_to_change[:-1]
            modified_message = message_to_change + injection_password
            raw_data['instruction'] = modified_message

            breakpoint()

            message_to_change = raw_data['history'][1]['message']
            if message_to_change.endswith('\n'):
                message_to_change = message_to_change[:-1]
            modified_message = message_to_change + injection_password

            raw_data['history'][1]['args']['content'] = modified_message
            raw_data['history'][1]['message'] = modified_message

            # code_to_change = raw_data['history'][8]['args']
            # modified_code = injection_code + "\n" + code_to_change['code']
            # code_to_change['code'] = modified_code
            # raw_data['history'][8]['args'] = code_to_change

            folder = (
                folder
                / cfg.instance
                / cfg.agent_cls
                / f'{custom_name_map(cfg.llm_config)}_maxiter_{cfg.max_iterations}_N_temp'
            )

            # Save jsonl file
            output_path = folder / INJECTED_OUTPUT_JSON
            with output_path.open('w', encoding='utf-8') as f:
                f.write(json.dumps(raw_data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
