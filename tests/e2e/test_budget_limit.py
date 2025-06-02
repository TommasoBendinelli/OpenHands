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
def test_budget_limit_end_to_end(llm_config):
    timestamp = '1970-01-01_00-00-00'

    day = timestamp.split('_')[0]
    steps = 100
    root_dir = (
        Path('evaluation/evaluation_outputs/outputs')
        / day
        / f'{timestamp}_0'
        / 'channel_corr_hard'
        / 'DataScienceBenchAgent'
    )

    # Delete the output directory if it exists
    if root_dir.exists():
        shutil.rmtree(root_dir)
    env = os.environ.copy()
    env['DEBUG'] = '1'
    max_budget_per_task = 0.001
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
        'solution_iterations=5',
        'cheating_attempt=False',
        'warm_against_cheating=False',
        f'max_budget_per_task={max_budget_per_task}',
        'prompt_variation=0',
        'seed=20',
        'keep_going_until_succeed=True',
        'native_tool_calling=False',
        'is_plotting_enabled=True',
        'give_structure_hint=False',
        'disable_numbers=False',
        'is_read_csv_banned=False',
        f'max_iterations={steps}',
        'identifier_experiment=baseline',
        f'timestamp={timestamp}',
    ]

    subprocess.run(cmd, check=True, env=env)
    # Find the directory inside the base directory
    base_dir = list(root_dir.iterdir())
    assert len(base_dir) == 1, (
        f'Expected exactly one directory in {root_dir}, but found {len(base_dir)}'
    )
    base_dir = base_dir[0]

    output_file = base_dir / 'output.jsonl'

    with open(output_file) as f:
        line = f.readline()
        data = json.loads(line)

    costs = [x['cost'] for x in data['metrics']['costs']]
    # Budget stops exactly before the last message

    assert costs[-2] < max_budget_per_task, (
        f'Expected the last cost to be less than {max_budget_per_task}, but got {costs[-2]}'
    )
