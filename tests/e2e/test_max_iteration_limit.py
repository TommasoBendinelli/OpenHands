import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.runtime.conftest import TEST_IN_CI


@pytest.mark.skipif(
    not TEST_IN_CI,
    reason='This test requires network access and Docker to run.',
)
@pytest.mark.parametrize(
    'llm_config',
    [
        'gpt-4o-mini',
        'gemini-flash-lite',
        'open_router_claude',
    ],
)
def test_iteration_limit_end_to_end(llm_config):
    timestamp = '1970-01-01_00-00-00'

    day = timestamp.split('_')[0]
    max_iterations = 2
    root_dir = (
        Path('evaluation/evaluation_outputs/outputs')
        / day
        / f'{timestamp}_0'
        / 'channel_corr_hard'
        / 'DataScienceBenchAgent'
    )

    if root_dir.exists():
        shutil.rmtree(root_dir)
    env = os.environ.copy()
    env['DEBUG'] = '1'
    cmd = [
        'python',
        '-m',
        'evaluation.benchmarks.data_science_bench.run_infer',
        'number_of_experiments=1',
        'eval_n_limit=1',
        'class_type=explorative_data_analysis',
        'instance=channel_corr_hard',
        'constraints=0',
        f'llm_config={llm_config}',
        'feedback_iterations=5',
        'cheating_attempt=False',
        'warm_against_cheating=False',
        'max_budget_per_task=1',
        'prompt_variation=0',
        'seed=20',
        'keep_going_until_succeed=True',
        'native_tool_calling=False',
        'is_plotting_enabled=True',
        'give_structure_hint=False',
        'disable_numbers=False',
        'is_read_csv_banned=False',
        f'max_iterations={max_iterations}',
        'identifier_experiment=baseline',
        f'timestamp={timestamp}',
    ]
    subprocess.run(cmd, check=True, env=env)
    base_dir = list(root_dir.iterdir())[0]
    output_file = base_dir / 'output.jsonl'
    with open(output_file) as f:
        data = json.loads(f.readline())
    assert 'current iteration: 2, max iteration: 2' in (data.get('error') or '').lower()
