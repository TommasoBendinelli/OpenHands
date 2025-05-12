import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.utils.loading_experiments import _load_experiment, get_folders_in_range

# import hydra
# from omegaconf import DictConfig, OmegaConf
# from omegaconf import OmegaConf
# from hydra.core.config_store import ConfigStore
# from hydra.core.hydra_config import HydraConfig


def generate_function_call_regex(function_name: str) -> str:
    """
    Generate a regex pattern to detect a function or method call in Python code.

    Args:
        function_name (str): The function or method name (e.g., 'print', 'pd.read_csv', 'df.mean').

    Returns:
        str: A regex string pattern to detect that function call.
    """
    # Escape periods (for method or attribute access)
    escaped_name = re.escape(function_name)
    # Match function call pattern, allowing optional spaces before the parenthesis
    pattern = rf'\b{escaped_name}\s*\('
    return pattern


# @dataclass
# class ExperimentConfig:
#     root_dir: str = "evaluation/evaluation_outputs/outputs"
#     after: str = "2025-05-06_19-40-12"
#     before: str = "2025-05-06_19-43-24"
#     functions_to_not_use: list[str] = field(default_factory=lambda: ["np.mean"])
#     output_file: str = "instruction_violations.json"
#     metadata_json: str = 'metadata.json'
#     output_json: str = 'output.jsonl'


# cs = ConfigStore.instance()
# cs.store(name="experiment_config", node=ExperimentConfig)


# @hydra.main(config_name="experiment_config")
# def main(cfg: ExperimentConfig):


def main():
    # HydraConfig.get().job.override_dirname = ""
    # run_dir = Path("results") / datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    # run_dir.mkdir(parents=True, exist_ok=True)

    # Get 005_results.csv
    # results = pd.read_csv("end_results/005_results.csv")
    # seed_list = [21, 23, 24, 25]  # replace with your actual list of seeds

    # breakpoint()

    # filtered = results[
    #     (results['solution_iterations'] == 1) & (results['seed'].isin(seed_list))
    # ]

    # # For each task, average across seed and prompt_variation and groupby task
    # filtered = filtered.groupby(
    #     ['instance']
    # ).agg(
    #     {
    #         'perfect_accuracy': 'mean',
    #         'number_of_submissions': 'mean',
    #         'accumulated_cost': 'mean',
    #         'instruction_violations': 'sum',
    #     }
    # ).reset_index()

    # breakpoint()

    # assert len(filtered) == 144

    root_dir = Path('evaluation/evaluation_outputs/outputs')
    after_dt = datetime.strptime('2025-05-09_11-41-20', '%Y-%m-%d_%H-%M-%S')
    before_dt = datetime.strptime('2025-05-09_12-11-32', '%Y-%m-%d_%H-%M-%S')

    file_name_with_constraints = 'results_instruction_following_temperatue1.csv'
    file_name_with_constraints_no_violations = (
        'results_instruction_following_no_violations_temperatue1.csv'
    )

    runs = sorted(get_folders_in_range(root_dir, after_dt, before_dt))

    res = {}
    entries_df = []
    for folder in runs:
        metadata, outputs, cfg = _load_experiment(folder)

        # Select results with include_constraints=True
        if not cfg['include_constraints']:
            continue

        if not outputs:
            continue

        generated_code = []
        generated_code_run = []
        for i in outputs[0]['history']:
            for key in i.keys():
                if 'args' in key:
                    if 'code' in i['args'].keys():
                        generated_code.append(i['args']['code'])

        generated_code_run = '\n'.join(generated_code)
        if generated_code_run == '':
            print(
                f'No Python Code generated in the selected trajectory. Folder: {folder}'
            )
            python_used = False
            # raise ValueError(f"No Python Code was generated in the selected trajectory. Folder: {folder}")
        else:
            python_used = True

        ## Get all data
        instance = cfg['instance']
        contraints = cfg['constraints']
        llm_config = cfg['llm_config']
        llm_hints = cfg['hints']

        res[str(folder)] = {}
        res[str(folder)]['metadata'] = metadata
        res[str(folder)]['metrics'] = []
        assert len(outputs) == 1, 'Multiple outputs found'
        # for key, output in outputs.items():

        metric = outputs[0]['test_result']['result']['metric']
        if 'number_of_submissions' in outputs[0]['test_result']['result']:
            number_of_submissions = outputs[0]['test_result']['result'][
                'number_of_submissions'
            ]
        elif 'number_of_iteractions' in outputs[0]['test_result']['result']:
            number_of_submissions = outputs[0]['test_result']['result'][
                'number_of_iterations'
            ]
        else:
            number_of_submissions = np.nan

        if not number_of_submissions:
            number_of_submissions = 0
        res[str(folder)]['metrics'].append(metric)
        res[str(folder)]['number_of_submissions'] = number_of_submissions
        if 'llm_metrics' in outputs[0]['history'][-1]:
            accumulated_cost = outputs[0]['history'][-1]['llm_metrics'][
                'accumulated_cost'
            ]
        elif 'llm_metrics' in outputs[0]['history'][-2]:  # To check why this is needed
            accumulated_cost = outputs[0]['history'][-2]['llm_metrics'][
                'accumulated_cost'
            ]
        else:
            accumulated_cost = np.nan

        # res[str(folder)]['accumulated_cost'] = accumulated_cost
        scores = list(res[str(folder)]['metrics'])

        # If metrics are empty, set them to 0
        if not res[str(folder)]['metrics'][0]:
            res[str(folder)]['metrics'][0] = 0

        # df = pd.DataFrame(res[str(folder)]['metrics'], columns=['metric'])
        current_dict = {
            'metric': scores,
        }
        df = pd.DataFrame.from_dict(current_dict)

        if (
            outputs[0]['error']
            and "RuntimeError: Agent reached maximum budget" in outputs[0]['error']
        ):
            outputs[0]['error'] = None
            use_max_budget = True
        else:
            use_max_budget = False

        # Which functions to not use
        df['include_constraints'] = cfg['include_constraints']
        df['instruction_constraint'] = cfg['instruction_constraint']

        df['error'] = outputs[0]['error']
        # Add contraints as column
        df['constraints'] = contraints
        # Add llm_config as column
        df['llm_hints'] = llm_hints
        df['llm_config'] = llm_config
        df['instance'] = instance
        df['folder'] = folder.name
        df['max_budget_per_task'] = cfg['max_budget_per_task']
        df['solution_iterations'] = cfg['solution_iterations']
        df['seed'] = cfg['seed']
        df['prompt_variation'] = cfg['prompt_variation']

        # Complete failure"]
        # df['invalid_or_no_submission'] = df["metric"].isna()
        df['perfect_accuracy'] = df['metric'] == 1
        df['number_of_submissions'] = number_of_submissions
        df['accumulated_cost'] = accumulated_cost

        # Convert folder.name to datetime
        # df["timestamp"] = datetime.strptime(folder.name.split("_")[0], "%Y-%m-%d")
        date_folder = folder.name.split('_')[0] + '_' + folder.name.split('_')[1]

        df['time'] = datetime.strptime(date_folder, '%Y-%m-%d_%H-%M-%S')

        # Count the number of times each function is found in the generated code
        # for func in functions_to_not_use:
        regex = generate_function_call_regex(cfg['instruction_constraint'])
        count = len(re.findall(regex, generated_code_run))
        # functions_found[func] = count
        if count > 0:
            print(
                f'Model did not follow instruction: {cfg["instruction_constraint"]} found {count} times in generated code'
            )
        if count == 0:
            instruction_violations = False
        else:
            instruction_violations = count

        df['instruction_violations'] = instruction_violations
        df['python_used'] = python_used
        entries_df.append(df)

    results_all = pd.concat(entries_df, ignore_index=True, axis=0)

    breakpoint()

    # Average across seeds and prompt_variations
    results_all_with_constraints_averaged = (
        results_all.groupby('instance')[['metric']].mean().reset_index()
    )

    # Save config json
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / file_name_with_constraints

    # Save datatfram as csv
    results_all_with_constraints_averaged.to_csv(output_file, index=False)

    # Only when there are no constraint violations
    results_all = results_all[results_all['instruction_violations'] == 0]
    results_all_with_constraints_averaged_no_violations = (
        results_all.groupby('instance')[['metric']].mean().reset_index()
    )

    output_file_no_violations = output_dir / file_name_with_constraints_no_violations
    # Save datatfram as csv
    results_all_with_constraints_averaged_no_violations.to_csv(
        output_file_no_violations, index=False
    )


if __name__ == '__main__':
    main()
