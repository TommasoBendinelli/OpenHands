import subprocess
import sys
from pathlib import Path

# Directory that contains all tasks
TASKS_ROOT = Path('evaluation/benchmarks/error_bench/tasks/explorative_data_analysis')
# Folders to ignore while scanning for tasks
SKIP_DIRS = {'old', '__pycache__', 'utils'}


def run_solution(solution_path: Path) -> str:
    """Execute the solution.py file and return its combined stdout / stderr output."""
    result = subprocess.run(
        [sys.executable, solution_path.name],
        cwd=solution_path.parent,  # run inside the task directory
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


def main() -> None:
    """Iterate over all tasks and ensure each reports `Test accuracy: 100%`."""
    tasks = []
    time_series_task = []
    tabular_task = []
    for task_dir in TASKS_ROOT.glob('*'):
        # Skip anything that is not an actual task directory
        if not task_dir.is_dir() or task_dir.name in SKIP_DIRS:
            continue

        solution_file = task_dir / 'solution.py'
        if not solution_file.exists():
            raise FileNotFoundError(f'Solution file not found in {task_dir}')

        # First generate the dataset
        generate_file = task_dir / 'generate_dataset.py'
        if generate_file.exists():
            print(f'Generating dataset for {task_dir.name}...')
            result = subprocess.run(
                [sys.executable, generate_file.name],
                cwd=task_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f'Failed to generate dataset for {task_dir.name}.\n'
                    f'Output:\n{result.stdout + result.stderr}'
                )
        else:
            raise FileNotFoundError(f'Dataset generation file not found in {task_dir}')
        output = run_solution(solution_file)
        if (
            'Test accuracy: 100%' in output
            or 'Test accuracy: 1.00' in output
            or 'Test accuracy: 100.00%' in output
            or r'100% accuracy' in output
            or 'Test accuracy: 1.0' in output
            or 'Test  accuracy : 100.00%' in output
            or 'Test  accuracy: 1.000' in output
            or 'Test  accuracy: 1.00' in output
        ):
            print(f'{task_dir.name}: 100% accuracy')
        else:
            raise ValueError(
                f'{task_dir.name} does not reach 100% accuracy.\n'
                f'Script output:\n{output}'
            )

        # Also open the metadata file to check whether it's tabular data or a time series problem
        metadata_file = task_dir / 'metadata.json'
        if not metadata_file.exists():
            raise FileNotFoundError(f'Metadata file not found in {task_dir}')

        # Open json file
        import json

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        if metadata['is_time_series']:
            time_series_task.append(task_dir.name)
        else:
            tabular_task.append(task_dir.name)

        tasks.append(task_dir.name)

    # Sort task alphabetically
    tasks.sort()

    # Give a , separate list of the tasks names
    print('\nAll tasks passed successfully.')
    print('Tasks:', ','.join(tasks))

    # Also create a list of the task names, like a python list
    print([f'{task}' for task in tasks])
    print([f'{task}' for task in time_series_task])
    print([f'{task}' for task in tabular_task])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'{exc}')
        sys.exit(1)
