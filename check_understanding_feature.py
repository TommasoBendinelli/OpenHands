from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from evaluation.utils.loading_experiments import _load_experiment, get_folders_in_range

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


# Get all the entries evaluation/evaluation_outputs/outputs
ROOT_DIR = Path('evaluation/evaluation_outputs/outputs')
AFTER = datetime.strptime('2025-05-10_03-19-54', '%Y-%m-%d_%H-%M-%S')
BEFORE = datetime.strptime('2025-05-10_03-49-36', '%Y-%m-%d_%H-%M-%S')
# AFTER = None
# BEFORE = None
METADATA_JSON = 'metadata.json'
OUTPUT_JSON = 'output.jsonl'

# solutions = {
#     'find_peaks': 'The correct feature is to use the number of peaks in the signal to tell the class of the signal.',
#     'predict_ts_stationarity': 'The correct feature is to tell whether the time series is stationary or not.',
#     'frequency_band': 'The correct feature that separates the two classes is the frequency band (signals from the first class have 0-4 Hz, while the second class has 20-50 Hz).',
#     'set_points': 'The correct feature is to consider whether there are set points in the signal or not.',
# }

solutions = {
    'channel_corr': 'Correlation between the two signals.',
    'channel_divergence': 'Whether the average gap between two channels is constant or not',
    'common_frequency': 'Whether the two signals have a common frequency or not.',
    'dominant_feature': 'The dominant feature of the signal, i.e. the feature is larger than the others.',
    'feature_compare': 'Whether feature 1 is greater than feature 2.',
    'find_peaks': 'Number of peaks in the signal.',
    'frequency_band': 'Frequency band (signals from the first class have 0-4 Hz, while the second class has 20-50 Hz).',
    'interaction_sign': 'Whether the product of the two signals is positive or negative.',
    'kidney_flag': 'Whether creatinine >1.4  AND eGFR < 60.',
    'periodic_presence': 'Whether the signal is periodic or not.',
    'phase_alignment': 'Whether the phase of the two signals is aligned or not.',
    'predict_ts_stationarity': 'Whether the time series is stationary or not.',
    'set_points': 'Whether there are change points in the signal or not.',
    'simulatenus_spike': 'Whether there is at least one spike that happens at the same time in both signals.',
    'spike_presence': 'Whether there is a spike in the signal or not.',
    'sum_threshold': 'The sum of the first three featurees.',
    'variance_burst': 'Whether there is a variance burst or not.',
}


def main():
    runs = sorted(get_folders_in_range(ROOT_DIR, AFTER, BEFORE))

    results_iteration = []

    run_it = False

    if run_it:
        for i in range(0, 4):
            # Iterate over the folders
            res = {}
            entries_df = []
            for folder in runs:
                print(f'Processing folder: {folder}')
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

                solution = solutions[cfg['instance']]

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
                elif (
                    'llm_metrics' in outputs[0]['history'][-2]
                ):  # To check why this is needed
                    accumulated_cost = outputs[0]['history'][-2]['llm_metrics'][
                        'accumulated_cost'
                    ]
                else:
                    accumulated_cost = np.nan

                scores = list(res[str(folder)]['metrics'])

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

                if (
                    outputs[0]['error']
                    and 'RuntimeError: Agent reached maximum budget'
                    in outputs[0]['error']
                ):
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
                date_folder = (
                    folder.name.split('_')[0] + '_' + folder.name.split('_')[1]
                )

                df['time'] = datetime.strptime(date_folder, '%Y-%m-%d_%H-%M-%S')

                # Check if the the model understands the underlying structure of the dataset
                GEMINI_INSTRUCTION = f"""
                A classification task is given to a model, where the two classes of the dataset are clearly separated by one specific feature. Given the following messages determine if the model used this feature to solve the task in the submission that lead to 100% accuracy. The feature is '{solution}'. The messages are: {messages}.

                Respond in the following format:

                1. Understood feature: <yes|no>
                2. Justification: <explain why you think the model understood or not the feature>.
                """

                gemini_response_question_1 = llm_complete(
                    GEMINI_INSTRUCTION,
                    provider='gemini',
                    model='gemini-2.5-flash-preview-04-17',
                )

                # breakpoint()

                match_yes_no = re.search(
                    r'1\.\s*\*{0,2}Understood feature:\*{0,2}\s*(yes|no)',
                    gemini_response_question_1,
                    re.IGNORECASE,
                )
                yes_no = match_yes_no.group(1) if match_yes_no else None

                match_justification = re.search(
                    r'2\.\s*\*{0,2}Justification:\*{0,2}\s*(.*)',
                    gemini_response_question_1,
                    re.IGNORECASE | re.DOTALL,
                )
                justification = (
                    match_justification.group(1).strip()
                    if match_justification
                    else None
                )

                df['understood_feature'] = yes_no
                df['justification'] = justification
                df['gemini_response_question_1'] = gemini_response_question_1

                entries_df.append(df)

            results_all = pd.concat(entries_df, ignore_index=True, axis=0)

            # Save the results to a csv file
            output_dir = Path('results')
            output_dir.mkdir(parents=True, exist_ok=True)
            check_understanding_feature = f'check_understanding_feature_{i}.csv'
            output_file = output_dir / check_understanding_feature
            results_all.to_csv(output_file, index=False)

            results_all['iteration'] = i
            results_iteration.append(results_all)

        # Concatenate all iterations
        results_all_iterations = pd.concat(results_iteration, ignore_index=True, axis=0)
        # Save the results to a csv file
        check_understanding_feature = 'check_understanding_feature_all_iterations.csv'
        output_file = output_dir / check_understanding_feature
        results_all_iterations.to_csv(output_file, index=False)

    else:
        # Load the results from the csv file
        output_dir = Path('results')
        check_understanding_feature = 'check_understanding_feature_all_iterations.csv'
        output_file = output_dir / check_understanding_feature
        results_all_iterations = pd.read_csv(output_file)

        # results_all_iterations = results_all_iterations[results_all_iterations['is_skelarn'] == False]

        for iteration_value, group_df in results_all_iterations.groupby('iteration'):
            group_df = group_df.reset_index(drop=True)
            print(
                group_df[
                    [
                        'instance',
                        'folder',
                        'understood_feature',
                        'justification',
                        'metric',
                        'is_skelarn',
                        'perfect_accuracy',
                    ]
                ]
            )

    breakpoint()


if __name__ == '__main__':
    main()
