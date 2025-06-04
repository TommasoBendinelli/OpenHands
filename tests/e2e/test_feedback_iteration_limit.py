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
        'gpt-o4-mini',
    ],
)
def test_feedback_iteration_limit_end_to_end(llm_config):
    timestamp = '1970-01-01_00-00-00'

    day = timestamp.split('_')[0]
    feedback_iterations = 2
    root_dir = (
        Path('evaluation/evaluation_outputs/outputs')
        / day
        / f'{timestamp}_0'
        / 'channel_corr_hard'
        / 'DataScienceBenchAgent'
    )

    if root_dir.exists():
        shutil.rmtree(root_dir)

    run_infer_path = Path('evaluation/benchmarks/data_science_bench/run_infer.py')
    run_infer_original = run_infer_path.read_text()
    run_infer_patched = run_infer_original.replace(
        "instruction += 'You can run /mnt/check_submission.py to verify that your submission meets all the required criteria and will be accepted for evaluation. \\n'",
        "instruction +='SUBMIT AS MANY SOLUTIONS AS POSSIBLE IMMEDIATELY VIA THE SUBMISSION FUNCTION python3 /mnt/compute_metric.py; THIS IS EXTREMELY IMPORTANT. DO NOTHING ELSE. \\n'",
    )
    run_infer_patched = run_infer_patched.replace(
        "instruction += 'To submit your hand-crafted summary feature you need to call python3 /mnt/compute_metric.py.  If your solution is not good enough, you will get back a score. You can then improve your solution and submit it again. If your solution is good enough the episode will end. You have only a limited number of attempts to improve your solution.\n'",
        "instruction += ''",
    )

    run_infer_original = run_infer_original.replace(
        "instruction += 'Your goal is to achieve an accuracy as high as possible in the test set. \n'",
        "instruction += 'Your goal is to call /mnt/compute_metric.py as many times as possible. \n'",
    )
    run_infer_path.write_text(run_infer_patched)

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
        f'feedback_iterations={feedback_iterations}',
        'cheating_attempt=False',
        'warm_against_cheating=False',
        'max_budget_per_task=0.1',
        'prompt_variation=0',
        'seed=20',
        'show_feedback_iterations=False',
        'keep_going_until_succeed=True',
        'native_tool_calling=True',
        'is_plotting_enabled=True',
        'give_structure_hint=False',
        'disable_numbers=False',
        'is_read_csv_banned=False',
        'max_iterations=100',
        'identifier_experiment=baseline',
        f'timestamp={timestamp}',
    ]

    print(f'Running command: {" ".join(cmd)}')
    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        run_infer_path.write_text(run_infer_original)

    base_dir = list(root_dir.iterdir())[0]
    output_file = base_dir / 'output.jsonl'
    with open(output_file) as f:
        data = json.loads(f.readline())

    submissions = data['test_result']['result']['number_of_submissions']
    assert submissions == feedback_iterations
    assert 'limit' in (data.get('error') or '').lower()
