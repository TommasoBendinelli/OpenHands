"""
llm_client.py – April 2025
Python 3.12 compatible utility for GPT‑4o‑mini & Gemini 2.0 models
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

load_dotenv(find_dotenv())  # automatically walks up folders
# # ---------- OpenAI ---------------------------------------------------------
try:
    import openai  # SDK ≥1.75.0

    _openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except ImportError:
    _openai_client = None

# ---------- Gemini / Google Gen AI ----------------------------------------
try:
    import google.genai as genai  # unified SDK ≥1.11.0

    _genai_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
except ImportError:
    _genai_client = None

# --------------------------------------------------------------------------

Provider = Literal['openai', 'gemini']


def clean_metrics(df: pd.DataFrame, nan_option: str) -> pd.DataFrame:
    """
    Handle NaNs in the 'metric' column according to nan_option.
    """
    df['metric'] = df['metric'].apply(lambda x: max(x) if len(x) > 0 else np.nan)
    if nan_option == '-1':
        df = df.dropna(subset=['metric'])
    elif nan_option == '0':
        # Replace NaNs with 0
        df['metric'] = df['metric'].fillna(0)

    return df


def llm_complete(
    instruction: str,
    provider: Provider = 'openai',
    model: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Return an LLM response as plain text.

    Parameters
    ----------
    instruction : str
        The user instruction.
    provider : "openai" | "gemini"
        Which backend to call.
    model : str | None
        Override the default model (see table above).
    **kwargs
        Extra settings forwarded to the SDK:
          * OpenAI → temperature, max_tokens, etc.
          * Gemini → generation_config=dict(...), safety_settings=[...], etc.

    Raises
    ------
    ImportError / RuntimeError / ValueError if a dependency is missing
    or a provider string is wrong.
    """
    if provider == 'openai':
        if _openai_client is None:
            raise ImportError('`openai` package not installed.')
        model = model or 'gpt-4o-mini'
        rsp = _openai_client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': instruction}],
            **kwargs,
        )
        return rsp.choices[0].message.content.strip()

    elif provider == 'gemini':
        if _genai_client is None:
            raise ImportError('`google-genai` package not installed.')
        if not os.getenv('GOOGLE_API_KEY'):
            raise RuntimeError('GOOGLE_API_KEY is missing.')
        model = model or 'gemini-2.0-flash'  # ← fastest; swap for pro if needed
        rsp = _genai_client.models.generate_content(
            model=model,
            contents=instruction,
            **kwargs,
        )
        # unified SDK exposes .text on successful responses
        return rsp.text.strip()

    else:
        raise ValueError(f'Unknown provider: {provider!r}')


# Regex that captures both the timestamp and the run-index
_TS_WITH_IDX_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(?P<idx>\d+)$'
)


def _load_experiment(folder: Path) -> tuple[dict, dict]:
    meta, out = {}, {}
    meta_path = folder / METADATA_JSON
    output_path = folder / OUTPUT_JSON

    cfg = OmegaConf.load(folder / '.hydra' / 'config.yaml')
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as e:
            print(f'[WARN] Bad JSON in {meta_path}: {e}')
    else:
        print(f'[INFO] {meta_path} missing')

    if output_path.exists():
        try:
            with output_path.open(encoding='utf-8') as f:
                out = {i: json.loads(line) for i, line in enumerate(f) if line.strip()}
        except json.JSONDecodeError as e:
            print(f'[WARN] Bad JSON line in {output_path}: {e}')
    else:
        print(f'[INFO] {output_path} missing')

    return meta, out, cfg


def get_folders_in_range(
    base_dir: Path, after: Optional[datetime], before: Optional[datetime]
) -> List[Path]:
    """
    Get list of subfolders matching the timestamp+index regex and falling within a datetime range.

    Parameters:
        base_dir (Path): Root path to search in.
        after (Optional[datetime]): Start of the range (inclusive). Pass None to skip lower bound.
        before (Optional[datetime]): End of the range (inclusive). Pass None to skip upper bound.

    Returns:
        List[Path]: List of subfolder paths within the range.
    """

    matching_paths = []

    if after is None:
        after = datetime.strptime('2025-05-04_00-00-32', '%Y-%m-%d_%H-%M-%S')

    if before is None:
        before = datetime.strptime('2030-05-04_00-00-32', '%Y-%m-%d_%H-%M-%S')

    for date_folder in base_dir.iterdir():
        if not date_folder.is_dir():
            continue

        try:
            folder_date = datetime.strptime(date_folder.name, '%Y-%m-%d')
        except ValueError:
            continue  # skip folders that don't match top-level date format

        # Early skip if outside overall date range
        if after and folder_date.date() < after.date():
            continue
        if before and folder_date.date() > before.date():
            continue

        for subfolder in date_folder.iterdir():
            if not subfolder.is_dir():
                continue

            match = _TS_WITH_IDX_RE.match(subfolder.name)
            if not match:
                continue

            ts_str = match.group('ts')
            try:
                ts_dt = datetime.strptime(ts_str, '%Y-%m-%d_%H-%M-%S')
            except ValueError:
                continue

            if (not after or ts_dt >= after) and (not before or ts_dt <= before):
                matching_paths.append(subfolder)

    return matching_paths


# Get all the entries evaluation/evaluation_outputs/outputs
ROOT_DIR = Path('evaluation/evaluation_outputs/outputs')
AFTER = datetime.strptime('2025-05-08_20-55-30', '%Y-%m-%d_%H-%M-%S')
BEFORE = None  # datetime.strptime('2025-05-06_11-47-22', '%Y-%m-%d_%H-%M-%S')
# AFTER = None
# BEFORE = None
METADATA_JSON = 'metadata.json'
OUTPUT_JSON = 'output.jsonl'

solutions = {
    'find_peaks': 'The correct feature is to use the number of peaks in the signal to tell the class of the signal.',
    'predict_ts_stationarity': 'The correct feature is to tell whether the time series is stationary or not.',
    'frequency_band': 'The correct feature that separates the two classes is the frequency band (signals from the first class have 0-4 Hz, while the second class has 20-50 Hz).',
    'set_points': 'The correct feature is to consider whether there are set points in the signal or not.',
}


def main():
    runs = sorted(get_folders_in_range(ROOT_DIR, AFTER, BEFORE))

    # Iterate over the folders
    res = {}
    entries_df = []
    for folder in runs:
        # Open metadata
        metadata, outputs, cfg = _load_experiment(folder)

        if not outputs:
            continue

        messages = '\n'.join([x['message'] for x in outputs[0]['history'][2:]])
        # Remove sklearn.metrics from the messages
        messages = re.sub(r'sklearn.metrics', '', messages)
        metric = outputs[0]['test_result']['result']['metric']

        if 'sklearn' in messages:
            is_sklearn = True
        else:
            is_sklearn = False

        # solution = solutions[cfg['instance']]

        instance = cfg['instance']
        contraints = cfg['constraints']
        llm_config = cfg['llm_config']
        llm_hints = cfg['hints']
        res[str(folder)] = {}
        res[str(folder)]['metadata'] = metadata
        res[str(folder)]['metrics'] = []
        assert len(outputs) == 1, 'Multiple outputs found'
        # for key, output in outputs.items():

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

        scores = list(res[str(folder)]['metrics'])
        # msg_history = outputs[0]["history"]
        # accumulated_costs_sum = sum([x['cost'] for x in outputs[0]['metrics']['costs']])

        def compute_cost_per_score(
            msgs: list[dict],
            costs: list[dict],
            scores: list[float],
            number_of_submissions: int,
        ) -> float:
            """
            Compute the cost per score.
            """
            cost_associated_with_score = []
            to_go_idx = 0
            # conten_msg = [x['content'] for x in msgs if 'content' in x]
            for idx, score in enumerate(scores, 1):
                # Search in which msg there is "idx: score"
                for idx_msg, msg in enumerate(msgs[to_go_idx:], to_go_idx):
                    if 'content' not in msg:
                        continue
                    if f'Accuracy on test set {idx}:' in msg['content']:
                        to_go_idx = idx
                        break

                # if number_of_submissions > 0:
                #     assert to_go_idx != 0, "to_go_idx should not be 0"

                # Get the idx - 1 and sum the costs up to that point
                if to_go_idx == 0 and len(scores) > 1:
                    cost_sum = np.nan
                elif to_go_idx == 0 and len(scores) == 1:
                    cost_sum = sum(x['cost'] for x in costs[:])
                else:
                    cost_sum = sum(x['cost'] for x in costs[:to_go_idx])

                cost_associated_with_score.append(cost_sum)

            return cost_associated_with_score

        msgs = outputs[0]['history']
        costs = outputs[0]['metrics']['costs']
        cost_associated_with_scores = compute_cost_per_score(
            msgs, costs, scores[0], number_of_submissions=number_of_submissions
        )
        res[str(folder)]['accumulated_cost'] = accumulated_cost

        current_dict = {
            'metric': scores,
            'cost_associated_with_scores': [cost_associated_with_scores],
        }
        df = pd.DataFrame.from_dict(current_dict)

        if outputs[0]['error'] and "RuntimeError: Agent reached maximum budget" in outputs[0]['error']:
            outputs[0]['error'] = None 
            use_max_budget = True
        else:
            use_max_budget = False

        df['error'] = outputs[0]['error']
        df['use_max_budget'] = use_max_budget
        # Add contraints as column
        df['constraints'] = contraints
        # Add llm_config as column
        df['hints'] = llm_hints
        df['llm_config'] = llm_config
        df['instance'] = instance
        df['folder'] = folder.name
        df['is_skelarn'] = is_sklearn
        df['max_budget_per_task'] = cfg['max_budget_per_task']
        df['solution_iterations'] = cfg['solution_iterations']
        df['seed'] = cfg['seed']
        df['prompt_variation'] = cfg['prompt_variation']
        df['number_of_iterations'] = len(outputs[0]['history'])
        df['cumulative_number_of_completion_tokens'] = outputs[0]['metrics'][
            'accumulated_token_usage'
        ]['completion_tokens']

        # Complete failure"]
        # df['invalid_or_no_submission'] = df["metric"].isna()
        df['perfect_accuracy'] = df['metric'].apply(
            lambda x: 1 if len(x) > 0 and max(x) == 1 else 0
        )
        df['number_of_submissions'] = number_of_submissions
        df['final_accumulated_cost'] = accumulated_cost

        # Convert folder.name to datetime
        # df["timestamp"] = datetime.strptime(folder.name.split("_")[0], "%Y-%m-%d")
        date_folder = folder.name.split('_')[0] + '_' + folder.name.split('_')[1]

        df['time'] = datetime.strptime(date_folder, '%Y-%m-%d_%H-%M-%S')

        if False:
            # Call Gemini and ask him whether the solution was A) proposed or B)
            GEMINI_INSTRUCTION = f"""
            Given the following messages, was the approach '{solution}' ever mentioned in the conversation? Answer Yes or No only
            {messages}
            """
            gemini_response_question_1 = llm_complete(
                GEMINI_INSTRUCTION,
                provider='gemini',
                model='gemini-2.5-flash-preview-04-17',
            )

            GEMINI_INSTRUCTION = f"""
            Given the following messages, was the approach '{solution}' implemented in code? Answer Yes or No only
            {messages}
            """
            gemini_response_question_2 = llm_complete(
                GEMINI_INSTRUCTION,
                provider='gemini',
                model='gemini-2.5-flash-preview-04-17',
            )

            df['proposed'] = gemini_response_question_1
            df['implemented'] = gemini_response_question_2

            GEMINI_INSTRUCTION = f"""
            Given the following messages, can you list all the methods that were tried to solve the problem? Answer with a list of methods
            {messages}
            """
            gemini_response_question_3 = llm_complete(
                GEMINI_INSTRUCTION,
                provider='gemini',
                model='gemini-2.5-flash-preview-04-17',
            )
            df['methods'] = gemini_response_question_3
        entries_df.append(df)

    results_all = pd.concat(entries_df, ignore_index=True, axis=0).iloc[-40:]
    breakpoint()
    # Group by task
    for instance, group_df in results_all.groupby('instance'):
        # is_sklearn used?
        is_sklearn = group_df['is_skelarn'].unique()[0]
        
    results_all['best_metric'] = results_all['metric'].apply(
        lambda x: max(x) if len(x) > 0 else np.nan
    )
    results_005 = results_all.loc[results_all['max_budget_per_task'] == 0.05]
    results_05 = results_all.loc[results_all['max_budget_per_task'] == 0.50]
    results_005 = results_all.loc[results_all['max_budget_per_task'] == 0.05]

    # Keep only the first score for each task
    results_005['first_score'] = results_005['metric'].apply(
        lambda x: x[0] if len(x) > 0 else np.nan
    )
    results_005['max_score'] = results_005['metric'].apply(
        lambda x: max(x) if len(x) > 0 else np.nan
    )
    results_005['first_cost'] = results_005['cost_associated_with_scores'].apply(
        lambda x: x[0] if (len(x) > 0 and min(x) < 0.005) else np.nan
    )

    # Keep the ones which are not Nan with first cost tilde
    results_005_to_group = results_005.loc[~results_005['first_cost'].isna()]

    # Group by task and get max of the first score and sum of the first cost
    results_005_to_group = (
        results_005_to_group.groupby('instance')
        .agg(
            first_score=('first_score', 'max'),
            first_cost=('first_cost', 'sum'),
            number_of_submissions=('instance', 'size'),
        )
        .reset_index()
    )

    # Group by task and get mean of the max_score and mean of the first cost
    results_005_all = (
        results_005.groupby('instance')
        .agg(
            max_score=('max_score', 'mean'),
            first_cost=('final_accumulated_cost', 'mean'),
            number_of_submissions=('number_of_submissions', 'mean'),
        )
        .reset_index()
    )
    print(results_005_all)

    tasks = sorted(results_all['instance'].unique())
    NAN_OPT = '-1'  # -1, drop

    # Run the hypothesis test about different prompts
    for task in tasks:
        group_50 = {}
        task_df = results_005[results_005['instance'] == task]
        # grouped = task_df.groupby('prompt_variation')

        groups_df = []
        short_look = {}
        for idx, group_df in task_df.groupby('prompt_variation'):
            # print(group_df)
            # Drop nan
            group_df = clean_metrics(group_df, NAN_OPT)

            groups_df.append(group_df['metric'].reset_index(drop=True))
            short_look[idx] = {
                'mean': group_df['metric'].mean(),
                'std': group_df['metric'].std(),
            }

            group_50[idx] = group_df['metric'].reset_index(drop=True)
        # short_look_df = pd.DataFrame.from_dict(short_look, orient='index')
        # print(short_look_df)
        pd.concat(group_50, axis=1).to_csv(
            Path('end_results') / f'005_prompts_variations_{task}.csv', index=False
        )

        # Build a long-form DataFrame
        values = np.concatenate([x.values for x in groups_df])
        labels = np.concatenate(
            [[f'G{i}' for _ in grp] for i, grp in enumerate(groups_df)]
        )
        df = pd.DataFrame({'value': values, 'group': labels})
        if len(df) < 2:
            print(f'Not enough data for {task}')
            continue
        # Run Tukey HSD
        tukey = pairwise_tukeyhsd(df['value'], df['group'])
        print(tukey.summary())
        # One-way ANOVA

        F_stat, p_value = stats.f_oneway(*groups_df)
        print(f'F = {F_stat:.3f}, p = {p_value:.4f}')
        # breakpoint()

    def compute_best_run_per_seed(task_df: pd.DataFrame, task: str) -> pd.DataFrame:
        # Assert that the budget should be 0.05
        assert (
            task_df['max_budget_per_task'].unique()[0] == 0.05
        ), 'Budget should be 0.05 for all'
        res = []
        for idx, group_df in task_df.groupby('seed'):
            # Compute mean and std

            group_df = clean_metrics(group_df, NAN_OPT)
            # Compute metric and accumulated cost

            best_result_seed = group_df['metric'].max()
            sum_compute_result = group_df['final_accumulated_cost'].sum()
            df_tmp = pd.DataFrame(
                {
                    'metric': best_result_seed,
                    'final_accumulated_cost': sum_compute_result,
                },
                index=[0],
            )
            res.append(df_tmp)

        return pd.concat(res, axis=0, ignore_index=True)

    # Run the hypothesis about poor exploration
    result_005 = {}
    for task in tasks:
        # Get final accuracy
        results_per_seeds = compute_best_run_per_seed(
            results_005[results_005['instance'] == task], task
        )
        mean_result = results_per_seeds.mean()
        std_result = results_per_seeds.std()
        result_005[task] = {
            'mean': mean_result['metric'],
            'std': std_result['metric'],
            'mean_cost': mean_result['final_accumulated_cost'],
            'std_cost': std_result['final_accumulated_cost'],
        }
    result_005_df = pd.DataFrame.from_dict(result_005, orient='index')
    # Save the results
    # result_005_df.to_csv(Path("end_results") / "005_results.csv")
    print(result_005_df)
    # Get the 05 experiments after 2025-05-06 18:00:00
    result_005_df = results_005.loc[
        (
            results_005['time']
            > datetime.strptime('2025-05-07 13:00:00', '%Y-%m-%d %H:%M:%S')
        )
    ]
    results_05 = results_05.loc[
        results_05['time']
        > datetime.strptime('2025-05-08 12:01:20', '%Y-%m-%d %H:%M:%S')
    ]
    # Budge should below 10
    print(results_05)
    results_05 = results_05.loc[results_05['number_of_submissions'] < 11]
    # Group by instance
    res_05 = {}
    for task, df_task in results_05.groupby('instance'):
        # Compute mean metric and std and accumulated cost
        df_task = clean_metrics(df_task, NAN_OPT)
        mean_result = df_task['metric'].mean()
        std_result = df_task['metric'].std()
        mean_cost = df_task['final_accumulated_cost'].mean()
        std_cost = df_task['final_accumulated_cost'].std()
        # Get the 25% and 75% quantile
        # q1 = df_task['metric'].quantile(0.25)
        # q3 = df_task['metric'].quantile(0.75)
        res_05[task] = {
            'mean': mean_result,
            'std': std_result,
            'mean_cost': mean_cost,
            'std_cost': std_cost,
        }

    result_05_df = pd.DataFrame.from_dict(res_05, orient='index')
    print(result_05_df)
    # Save these results
    result_05_df.to_csv(Path('end_results') / '05_results.csv')
    result_005_df.to_csv(Path('end_results') / '005_results.csv')


if __name__ == '__main__':
    main()
