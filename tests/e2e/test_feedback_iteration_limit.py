import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.runtime.conftest import TEST_IN_CI

CANONICAL_INSTRUCTION = (
    'SUBMIT AS MANY SOLUTIONS AS POSSIBLE IMMEDIATELY VIA THE SUBMISSION '
    'FUNCTION python3 /mnt/compute_metric.py; THIS IS EXTREMELY IMPORTANT. '
    'DO NOTHING ELSE.\n'
)


class _InstructionStripper(ast.NodeTransformer):
    """Remove assignments to ``instruction`` after the first one."""

    def __init__(self) -> None:
        self.first_seen = False
        super().__init__()

    def _mutates_instruction(self, node: ast.AST) -> bool:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        else:
            return False
        return any(isinstance(t, ast.Name) and t.id == 'instruction' for t in targets)

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        if self._mutates_instruction(node):
            if self.first_seen:
                return None
            self.first_seen = True
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST | None:
        if self._mutates_instruction(node):
            if self.first_seen:
                return None
            self.first_seen = True
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if self._mutates_instruction(node):
            if self.first_seen:
                return None
            self.first_seen = True
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        if not node.body:
            node.body = [ast.Pass()]
        if node.orelse is not None and len(node.orelse) == 0:
            node.orelse = [ast.Pass()]
        return node


def patch_run_infer(path: Path) -> str:
    """Rewrite ``run_infer.py`` so ``instruction`` is a constant string.

    Parameters
    ----------
    path:
        The path to ``run_infer.py``.

    Returns
    -------
    str
        The original file contents.
    """

    original = path.read_text()
    tree = ast.parse(original)
    stripper = _InstructionStripper()
    tree = stripper.visit(tree)
    ast.fix_missing_locations(tree)

    # Find where ``instruction`` was first declared
    first_assign = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == 'instruction' for t in targets):
                first_assign = node
                break
    if first_assign is None:
        return original

    # Build new assignment node
    new_assign = ast.parse(
        'instruction = (\n'
        "    'SUBMIT AS MANY SOLUTIONS AS POSSIBLE IMMEDIATELY VIA THE SUBMISSION '"
        "    'FUNCTION python3 /mnt/compute_metric.py; THIS IS EXTREMELY IMPORTANT. '"
        "    'DO NOTHING ELSE.\\n'\n"
        ')\n'
    ).body[0]

    # Insert after the first assignment
    parent_map = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    parent = parent_map[first_assign]
    body = parent.body if hasattr(parent, 'body') else []
    idx = body.index(first_assign) + 1
    body.insert(idx, new_assign)
    ast.fix_missing_locations(parent)

    new_code = ast.unparse(tree)
    path.write_text(new_code)
    return original


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
    run_infer_original = patch_run_infer(run_infer_path)

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
