import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.runtime.conftest import TEST_IN_CI


@pytest.mark.skipif(
    not TEST_IN_CI,
    reason='This test requires network access and Docker to run.',
)
def test_feedback_iterations_limit(tmp_path):
    script_src = Path(
        'evaluation/benchmarks/data_science_bench/tasks/explorative_data_analysis/compute_metric.py'
    )
    script_path = tmp_path / 'compute_metric.py'
    shutil.copy(script_src, script_path)

    # Set a small limit
    content = script_path.read_text()
    content = content.replace('RUN_COUNTER_LIMIT = 20', 'RUN_COUNTER_LIMIT = 2')
    script_path.write_text(content)

    env = os.environ.copy()
    env['PYTHONPATH'] = str(tmp_path)

    for _ in range(2):
        subprocess.run(
            [
                sys.executable,
                '-c',
                'from compute_metric import _update_run_counter; _update_run_counter()',
            ],
            check=True,
            cwd=tmp_path,
            env=env,
        )

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                sys.executable,
                '-c',
                'from compute_metric import _update_run_counter; _update_run_counter()',
            ],
            check=True,
            cwd=tmp_path,
            env=env,
        )
