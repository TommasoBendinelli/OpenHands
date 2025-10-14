#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import shutil
import subprocess
import yaml
import hydra
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path
import os, json


@hydra.main(config_path="config", config_name="main")
def main(cfg):
    BASE_DIR = Path(to_absolute_path(cfg.base_dir))
    TASK = cfg.task

    TODAY = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXP_ROOT = BASE_DIR / "evaluation" / TODAY / TASK
    INSTANCE_DIR = BASE_DIR / "tasks" / TASK
    EXP_ROOT.mkdir(parents=True, exist_ok=True)

    for dir_path in sorted(Path(INSTANCE_DIR).glob("*/"))[:10]:
        name = dir_path.name
        run_dir = EXP_ROOT / name
        prompt_file = dir_path / "prompt.txt"

        if not prompt_file.exists():
            continue

        run_dir.mkdir(parents=True, exist_ok=True)

        if cfg.diagrams:
            src = dir_path / "diagrams"
            shutil.copytree(src, run_dir / "diagrams")

        if cfg.data_broken:
            src = dir_path / "data_broken.pkl"
            shutil.copy2(src, run_dir / src.name)

        if cfg.data_clean:
            src = dir_path / "data_healthy.pkl"
            shutil.copy2(src, run_dir / src.name)

        metadata_path = os.path.join(dir_path, "metadata_task.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            print(f"No metadata.json found in {folder}")

        instruction = f""""You are given a simulation and your task is to determine whether any physically implausible events occur at any time point. Select the correct answer in the list. Note that some answers also require you to return the time something happened."""

        if cfg.include_time_until_everything_is_good:
            instruction += f"""There was nothing wrong until time {metadata['modification']['start_time']}"""

        prompt = f"""{instruction} \n {'\n'.join(f'{chr(65 + i)}) {msg}' for i, msg in enumerate(metadata["multiple_choices"]))}"""

        prompt += (
            '\n Please provide your response in the following format:\n'
            'Final answer: <selected option text with the missing information filled inside the curly brackets {}> Do not remove the curly brackets {}.\n'
        )

        prompt += f"""\n Save your answer in results.json in the current directory."""

        output_log = run_dir / "output.log"

        if cfg.model == "gpt-5-codex":
            # Run codex
            cmd = [
                "codex",
                "exec",
                "--model",
                cfg.model,
                "--sandbox",
                cfg.sandbox,
                "-c",
                f"model_reasoning_effort={cfg.effort}",
            ]
        if cfg.model == "gemini":
            # "--show-memory-usage"
            cmd = ["gemini", "--sandbox", "--approval-mode=auto_edit", "--debug"]

        if cfg.model == "claude":
            # cmd = ["claude", "--verbose", "--permission-mode", "acceptEdits"]
            cmd = ["claude", "--verbose", "--dangerously-skip-permissions"]

        print(f"==> Running {name} in {run_dir}")

        with open(output_log, "w") as log:
            result = subprocess.run(
                cmd,
                cwd=run_dir,
                input=prompt,  # <-- replaces stdin=file
                text=True,  # interprets input as text instead of bytes
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        success = result.returncode == 0 and (run_dir / "results.json").exists()
        if not success:
            raise RuntimeError(
                f"Codex execution failed for {name}. Check {output_log} for details."
            )

        # Save config as YAML
        base_cfg = OmegaConf.to_container(cfg, resolve=True)
        run_config = {
            **base_cfg,
            "source_folder": str(dir_path),
            "run_folder": str(run_dir),
            "timestamp": datetime.now().isoformat(),
            "sandbox_mode": cfg.sandbox,
            "prompt": prompt,
        }

        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(run_config, f, sort_keys=False)


if __name__ == "__main__":
    main()
