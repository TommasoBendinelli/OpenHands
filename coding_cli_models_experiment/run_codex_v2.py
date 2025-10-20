#!/usr/bin/env python3
import json
import shutil
import subprocess
import sys
from datetime import datetime
from itertools import islice
from pathlib import Path

import hydra
import yaml
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


def _normalize_cli_args():
    for idx, arg in enumerate(sys.argv[1:], start=1):
        if arg == '--raise_on_error':
            sys.argv[idx] = 'raise_on_error=true'
        elif arg.startswith('--raise_on_error='):
            value = arg.split('=', 1)[1]
            normalized = str(value).lower() in {'1', 'true', 'yes', 'on'}
            sys.argv[idx] = f'raise_on_error={str(normalized).lower()}'


_normalize_cli_args()


def _copy_tree(src: Path, dst: Path) -> bool:
    """Copy a directory into dst, overwriting dst if needed."""

    if not src.exists():
        return False

    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    except TypeError:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    return True


def _warn_or_raise(cfg, message: str, exc_cls=RuntimeError):
    if getattr(cfg, 'raise_on_error', False):
        raise exc_cls(message)
    print(message)


def _iter_instances(base_dir: Path, task: str):
    legacy_root = base_dir / 'tasks' / task
    if legacy_root.exists():
        for dir_path in sorted(p for p in legacy_root.iterdir() if p.is_dir()):
            metadata_path = dir_path / 'metadata_task.json'
            if not metadata_path.exists():
                continue

            yield {
                'name': dir_path.name,
                'source_folder': dir_path,
                'metadata_path': metadata_path,
                'data_clean': dir_path / 'data_healthy.pkl',
                'data_broken': dir_path / 'data_broken.pkl',
                'diagram_candidates': [dir_path / 'diagrams', dir_path / 'diagram'],
            }
        return

    cleaned_candidates = [
        base_dir / 'cleaned_data',
        base_dir.parent / 'cleaned_data',
    ]
    cleaned_root = next((path for path in cleaned_candidates if path.exists()), None)
    if cleaned_root is None:
        raise FileNotFoundError('Could not locate a dataset directory (tasks or cleaned_data).')

    task_roots = []
    direct_task_root = cleaned_root / task
    if direct_task_root.exists():
        task_roots.append(direct_task_root)

    for label_dir in sorted(p for p in cleaned_root.iterdir() if p.is_dir() and p.name.startswith('label_')):
        task_dir = label_dir / task
        if task_dir.exists():
            task_roots.append(task_dir)

    if not task_roots:
        raise FileNotFoundError(
            f'Could not locate task "{task}" under cleaned_data. Checked {cleaned_root}'
        )

    seen = set()

    for task_root in task_roots:
        if task_root in seen:
            continue
        seen.add(task_root)

        first_level_dirs = sorted(p for p in task_root.iterdir() if p.is_dir())
        has_label_dirs = any(p.name.startswith('label_') for p in first_level_dirs)

        label_dirs = first_level_dirs if has_label_dirs else [task_root]

        for label_dir in label_dirs:
            if has_label_dirs and not label_dir.name.startswith('label_'):
                continue

            timestamp_dirs = (
                sorted(p for p in label_dir.iterdir() if p.is_dir())
                if has_label_dirs
                else first_level_dirs
            )

            for timestamp_dir in timestamp_dirs:
                if not timestamp_dir.is_dir():
                    continue

                scenario_dirs = sorted(
                    p for p in timestamp_dir.iterdir() if p.is_dir() and p.name.startswith('scenario_')
                )

                for scenario_dir in scenario_dirs:
                    healthy_dir = scenario_dir / 'healthy'
                    if not healthy_dir.exists():
                        continue

                    healthy_metadata = healthy_dir / 'metadata_task.json'
                    healthy_data = healthy_dir / 'data_healthy.pkl'
                    diagram_candidates = [scenario_dir / 'diagrams', scenario_dir / 'diagram']

                    rec_dirs = sorted(
                        p for p in scenario_dir.iterdir() if p.is_dir() and p.name.startswith('rec_')
                    )

                    if rec_dirs:
                        for rec_dir in rec_dirs:
                            metadata_path = rec_dir / 'metadata_task.json'
                            if not metadata_path.exists():
                                continue

                            name = '__'.join(rec_dir.relative_to(cleaned_root).parts)
                            yield {
                                'name': name,
                                'source_folder': rec_dir,
                                'metadata_path': metadata_path,
                                'data_clean': healthy_data,
                                'data_broken': rec_dir / 'data_broken.pkl',
                                'diagram_candidates': diagram_candidates,
                            }
                    elif healthy_metadata.exists():
                        name = '__'.join(healthy_dir.relative_to(cleaned_root).parts)
                        yield {
                            'name': name,
                            'source_folder': healthy_dir,
                            'metadata_path': healthy_metadata,
                            'data_clean': healthy_data,
                            'data_broken': None,
                            'diagram_candidates': diagram_candidates,
                        }


@hydra.main(config_path='config', config_name='main')
def main(cfg):
    BASE_DIR = Path(to_absolute_path(cfg.base_dir))
    TASK = cfg.task

    TODAY = datetime.now().strftime('%Y%m%d_%H%M%S')
    EXP_ROOT = BASE_DIR / 'evaluation' / TODAY / TASK
    EXP_ROOT.mkdir(parents=True, exist_ok=True)

    for instance in islice(_iter_instances(BASE_DIR, TASK), 10):
        name = instance['name']
        run_dir = EXP_ROOT / name
        raise_errors = getattr(cfg, 'raise_on_error', False)

        metadata_path = instance['metadata_path']
        if not metadata_path.exists():
            _warn_or_raise(
                cfg,
                f'No metadata_task.json found in {instance["source_folder"]}, skipping.',
                FileNotFoundError,
            )
            if not raise_errors:
                continue

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        run_dir.mkdir(parents=True, exist_ok=True)

        if cfg.diagrams:
            for candidate in instance.get('diagram_candidates', []):
                if _copy_tree(candidate, run_dir / 'diagrams'):
                    break

        if cfg.data_broken and instance.get('data_broken') and instance['data_broken'].exists():
            shutil.copy2(instance['data_broken'], run_dir / instance['data_broken'].name)
        elif cfg.data_broken:
            _warn_or_raise(cfg, f'No data_broken.pkl found for {name}', FileNotFoundError)
            if not raise_errors:
                continue

        if cfg.data_clean and instance.get('data_clean') and instance['data_clean'].exists():
            shutil.copy2(instance['data_clean'], run_dir / instance['data_clean'].name)
        elif cfg.data_clean:
            _warn_or_raise(cfg, f'No data_healthy.pkl found for {name}', FileNotFoundError)
            if not raise_errors:
                continue

        description = metadata.get('description')
        if description:
            (run_dir / 'description.md').write_text(description, encoding='utf-8')

        xml_description = metadata.get('xml_description')
        if xml_description:
            (run_dir / 'xml_description.xml').write_text(xml_description, encoding='utf-8')

        instruction = (
            "You are given a simulation and your task is to determine whether any physically "
            "implausible events occur at any time point. Select the correct answer in the list. "
            "Note that some answers also require you to return the time something happened."
        )

        if cfg.include_time_until_everything_is_good and metadata.get('modification'):
            instruction += f" There was nothing wrong until time {metadata['modification']['start_time']}"


        prompt_sections = [instruction]

        question = metadata.get('question')
        if question:
            prompt_sections.append(question)

        if description:
            prompt_sections.append(f"System description:\n{description}")

        if xml_description:
            prompt_sections.append(f"XML description:\n{xml_description}")

        prompt = '\n\n'.join(prompt_sections)
        prompt += '\n\n' + '\n'.join(
            f"{chr(65 + i)}) {msg}" for i, msg in enumerate(metadata['multiple_choices'])
        )

        prompt += (
            '\n Please provide your response in the following format:\n'
            'Final answer: <selected option text with the missing information filled inside the curly brackets {}> Do not remove the curly brackets {}.\n'
        )

        if cfg.diagrams:
            prompt += (
                '\n You can consult the diagrams png in the diagrams folder for additional context.'
            )

        if cfg.explicitly_tell_to_use_matlplolib:
            prompt += f"\n You should use matplotlib to inspect the data!"


        prompt += """\n Save your answer in results.json in the current directory."""

        output_log = run_dir / 'output.log'

        if cfg.model == 'gpt-5-codex':
            # Run codex
            cmd = [
                'codex',
                'exec',
                '--model',
                cfg.model,
                '--sandbox',
                cfg.sandbox,
                '-c',
                f'model_reasoning_effort={cfg.effort}',
            ]
        if cfg.model == 'gemini':
            # "--show-memory-usage"
            cmd = [
                'gemini',
                '--sandbox',
                '--approval-mode=auto_edit',
                '--debug',
                '--output-format',
                'json',
            ]

        if cfg.model == 'claude':
            # cmd = ["claude", "--verbose", "--permission-mode", "acceptEdits"]
            cmd = [
                'claude',
                '--verbose',
                '--dangerously-skip-permissions',
                '--output-format',
                'stream-json',
            ]

        print(f'==> Running {name} in {run_dir}')

        with open(output_log, 'w') as log:
            result = subprocess.run(
                cmd,
                cwd=run_dir,
                input=prompt,  # <-- replaces stdin=file
                text=True,  # interprets input as text instead of bytes
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        success = result.returncode == 0 and (run_dir / 'results.json').exists()
        if not success:
            message = f'Codex execution failed for {name}. Check {output_log} for details.'
            _warn_or_raise(cfg, message)
            if not raise_errors:
                continue

        # Save config as YAML
        base_cfg = OmegaConf.to_container(cfg, resolve=True)
        run_config = {
            **base_cfg,
            'source_folder': str(instance['source_folder']),
            'run_folder': str(run_dir),
            'timestamp': datetime.now().isoformat(),
            'sandbox_mode': cfg.sandbox,
            'prompt': prompt,
        }

        with open(run_dir / 'config.yaml', 'w') as f:
            yaml.safe_dump(run_config, f, sort_keys=False)


if __name__ == '__main__':
    main()
