from pathlib import Path
import subprocess
import sys

# Directory that contains all tasks
TASKS_ROOT = Path("evaluation/benchmarks/error_bench/tasks/explorative_data_analysis")
# Folders to ignore while scanning for tasks
SKIP_DIRS = {"old", "__pycache__", "utils"}


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
    for task_dir in TASKS_ROOT.glob("*"):
        # Skip anything that is not an actual task directory
        if not task_dir.is_dir() or task_dir.name in SKIP_DIRS:
            continue

        solution_file = task_dir / "solution.py"
        if not solution_file.exists():
            raise FileNotFoundError(f"Solution file not found in {task_dir}")

        output = run_solution(solution_file)
        if "Test accuracy: 100%" in output or "Test accuracy: 1.00" in output or "Test accuracy: 100.00%" in output or r"100% accuracy" in output or "Test accuracy: 1.0" in output or "Test  accuracy : 100.00%" in output or "Test  accuracy: 1.000" in output:
            print(f"{task_dir.name}: 100% accuracy")
        else:
            raise ValueError(
                f"{task_dir.name} does not reach 100% accuracy.\n" f"Script output:\n{output}"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{exc}")
        sys.exit(1)
