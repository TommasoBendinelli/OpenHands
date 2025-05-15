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
from collections import Counter
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from typing import Iterable, Tuple
from textwrap import indent

load_dotenv(find_dotenv())  # automatically walks up folders
# # # ---------- OpenAI ---------------------------------------------------------
# try:
#     import openai  # SDK ≥1.75.0

#     _openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
# except ImportError:
#     _openai_client = None

    
def fill_the_table(
    model_results_df: pd.DataFrame,
    important_columns: tuple[str, ...] = (
        "perfect_accuracy_at_first",      # One-Shot 100 %
        "mean_perfect_accuracy",          # Acc₁₀₀ %
        "mean_almost_perfect_accuracy",   # Acc₉₉ %
    ),
    model_order: tuple[str, ...] = (
        "Gemini Flash 2.5",
        "Gemini Pro 2.5",
        "GPT-4.1",
        "o4-mini",
        "Claude 3.7 Sonnet",
    ),
    show_percent: bool = True,            # ⇦ NEW FLAG
) -> str:
    """
    Build a LaTeX table with accuracy metrics for time-series and tabular tasks.

    Parameters
    ----------
    show_percent : bool, default True
        • True  → format 0.263158 as ``26.32\\%``
        • False → format 0.263158 as ``0.2632``
    """

    # ---------- helpers -------------------------------------------------------
    def fmt(x: float) -> str:
        """Format either as percentage or plain decimal."""
        return (f"{round(x * 100, 2):.2f}\\%" if show_percent
                else f"{x:.4f}")

    # ---------- wide pivot ----------------------------------------------------
    pivot = (
        model_results_df
        .pivot(index="llm_config",
               columns="is_time_series_task",   # 0 ↔ tabular, 1 ↔ time-series
               values=list(important_columns))
    )

    # ---------- locate maxima -------------------------------------------------
    bold_mask: dict[tuple[str, int], pd.Series] = {}
    for structure in (0, 1):
        for col in important_columns:
            vals = pivot[col][structure]
            bold_mask[(col, structure)] = vals == vals.max()

    # ---------- build each row -----------------------------------------------
    rows = []
    n_metrics = len(important_columns)

    for model in model_order:
        parts = [model]

        # collect cells in (0, 1) order
        for structure in (0, 1):
            for col in important_columns:
                cell = fmt(pivot.at[model, (col, structure)])
                if bold_mask[(col, structure)].get(model, False):
                    cell = f"\\textbf{{{cell}}}"
                parts.append(cell)

        # swap slices so headings match data
        ts_chunk = " & ".join(parts[1 + n_metrics : 1 + 2 * n_metrics])  # structure == 1
        tb_chunk = " & ".join(parts[1 : 1 + n_metrics])                  # structure == 0
        rows.append(f"    {model} &\n        {ts_chunk} &\n        {tb_chunk} \\\\")
    body = "\n".join(rows)

    # ---------- headings ------------------------------------------------------
    metric_headers = {
        "perfect_accuracy_at_first":      r"OneShotAcc\textsubscript{100\%}",
        "mean_perfect_accuracy":          r"Acc\textsubscript{100\%}",
        "mean_almost_perfect_accuracy":   r"Acc\textsubscript{99\%}",
    }
    header_cells = " & ".join(metric_headers.values())

    first_end  = 1 + n_metrics
    second_end = 1 + 2 * n_metrics
    cmidrules  = rf"\cmidrule(lr){{2-{first_end}}}\cmidrule(lr){{{first_end+1}-{second_end}}}"
    col_spec   = "@{}l" + "c" * (2 * n_metrics) + "@{}"

    # ---------- wrap in table skeleton ----------------------------------------
    latex = rf"""
\begin{{table}}[ht]
    \centering
    \caption{{Entry me)}}
    \label{{tab:agent-results-all}}
    \begin{{tabular}}{{{col_spec}}}
        \toprule
        & \multicolumn{{{n_metrics}}}{{c}}{{\textbf{{Time-Series Structure}}}}
        & \multicolumn{{{n_metrics}}}{{c}}{{\textbf{{Tabular Structure}}}} \\
        {cmidrules}
        \textbf{{Model}} & {header_cells} & {header_cells} \\
        \midrule
{indent(body, '        ')}
        \bottomrule
    \end{{tabular}}
\end{{table}}
""".strip("\n")

    return latex

def fill_costs_pycalls_table(
    model_results_df: pd.DataFrame,
    important_columns: tuple[str, ...] = (
        "cost_when_right",                                # Costright
        "number_of_python_calls_before_first_submission", # #PyCalls_before
        "number_of_python_calls_when_right_at_first",     # #PyCalls_right
    ),
    model_order: tuple[str, ...] = (
        "Gemini Flash 2.5",
        "Gemini Pro 2.5",
        "GPT-4.1",
        "o4-mini",
        "Claude 3.7 Sonnet",
    ),
) -> str:
    """
    Build a LaTeX table with *cost* and *Python-call* metrics for
    time-series (is_time_series_task == 0) and tabular (== 1) problems.

    ── Changes from the original ─────────────────────────────────────────────
    • Time-Series / Tabular columns are still flipped back into the
      correct order.
    • All bold-highlighting code has been removed.
    """

    # ---------- helpers -------------------------------------------------------
    def fmt_cost(x: float) -> str:   # 0.0269
        return f"{x:.4f}"

    def fmt_calls(x: float) -> str:  # 5.63
        return f"{x:.2f}"

    _format = {
        "cost_when_right": fmt_cost,
        "number_of_python_calls_before_first_submission": fmt_calls,
        "number_of_python_calls_when_right_at_first": fmt_calls,
    }

    # ---------- wide pivot ----------------------------------------------------
    # 0 ↔ tabular, 1 ↔ time-series (see slice swap below)
    pivot = (
        model_results_df
        .pivot_table(
            index="llm_config",
            columns="is_time_series_task",
            values=list(important_columns),
            aggfunc="mean",
        )
    )

    # ---------- build each row (no bolding) ----------------------------------
    rows = []
    n_metrics = len(important_columns)

    for model in model_order:
        parts = [model]

        # collect cells in (0, 1) order
        for structure in (0, 1):
            for col in important_columns:
                raw = pivot.at[model, (col, structure)]
                parts.append(_format[col](raw))

        # ── slice swap so columns match headings ─────────────────────────────
        ts_chunk = " & ".join(parts[1 + n_metrics : 1 + 2 * n_metrics])  # structure == 1
        tb_chunk = " & ".join(parts[1 : 1 + n_metrics])                  # structure == 0
        rows.append(f"    {model} &\n        {ts_chunk} & {tb_chunk} \\\\")
    body = "\n".join(rows)

    # ---------- headings ------------------------------------------------------
    metric_headers = {
        "cost_when_right":                                r"Cost\textsubscript{right}",
        "number_of_python_calls_before_first_submission": r"\#PyCalls\textsubscript{before}",
        "number_of_python_calls_when_right_at_first":     r"\#PyCalls\textsubscript{right}",
    }
    header_cells = " & ".join(metric_headers[col] for col in important_columns)

    first_end  = 1 + n_metrics
    second_end = 1 + 2 * n_metrics
    cmidrules  = rf"\cmidrule(lr){{2-{first_end}}}\cmidrule(lr){{{first_end+1}-{second_end}}}"
    col_spec   = "@{}l" + "c" * (2 * n_metrics) + "@{}"

    # ---------- wrap in table skeleton ----------------------------------------
    latex = rf"""
\begin{{table}}[ht]
    \centering
    \caption{{Additional Metrics (Cost \& Python Calls)}}
    \label{{tab:additional-metrics}}
    \begin{{tabular}}{{{col_spec}}}
        \toprule
        & \multicolumn{{{n_metrics}}}{{c}}{{\textbf{{Time-Series Structure}}}}
        & \multicolumn{{{n_metrics}}}{{c}}{{\textbf{{Tabular Structure}}}} \\
        {cmidrules}
        \textbf{{Model}} & {header_cells} & {header_cells} \\
        \midrule
{indent(body, '        ')}
        \bottomrule
    \end{{tabular}}
\end{{table}}
""".strip("\n")

    return latex





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

def get_metric_before_violation(before: str, outputs: list[dict]):
    # find all occurrences of "'id': <digits>"
    match_id = int(re.findall(r"'id'\s*:\s*(\d+)", before)[-1])

    # Collect all the accuracy on the test set Accuracy on test set
    outputs = [x for x in outputs[0]['history'] if 'content' in x and 'Accuracy on test set' in x['content'] and x['id'] < match_id]
    # Keep only those before the match



    accuracies = []
    pattern = re.compile(r"Accuracy on test set \d+:\s*([0-9]*\.?[0-9]+)")

    for x in outputs:
        content = x.get('content', '')
        # find all numbers after “Accuracy on test set <digit>:”
        for match in pattern.findall(content):
            accuracies.append(float(match))


    return accuracies, [outputs], match_id

# Get all the entries evaluation/evaluation_outputs/outputs
ROOT_DIR = Path('evaluation/evaluation_outputs/outputs')
AFTER = datetime.strptime('2025-05-12_00-00-00', '%Y-%m-%d_%H-%M-%S')
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

datasets = ['channel_corr', 'channel_divergence', 'cofounded_group_outlier', 'common_frequency', 'dominant_feature', 'find_peaks', 'ground_mean_threashold', 'outlier_ratio', 'periodic_presence', 'predict_ts_stationarity', 'row_max_abs', 'row_variance',  'sign_rotated_generator', 'simultanus_spike', 'sum_threshold', 'variance_burst', 'zero_crossing']

time_series_datasets = ['simultanus_spike', 'channel_corr', 'channel_divergence', 'variance_burst', 'common_frequency', 'predict_ts_stationarity', 'zero_crossing', 'find_peaks', 'periodic_presence']


tabular_datasets = ['sum_threshold']



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
        
        # Check if there are API errors
        if outputs[0]['error'] == 'BadRequestError: litellm.BadRequestError: OpenAIException - Unsupported parameter: \'stop\' is not supported with this model.':
            continue

        if outputs[0]['error'] == 'STATUS$ERROR_LLM_INTERNAL_SERVER_ERROR':
            continue

        if outputs[0]['error'] == 'RuntimeError: There was an unexpected error while running the agent: APIConnectionError. You can refresh the page or ask the agent to try again.':
            continue
        
        if outputs[0]['error'] == 'RuntimeError: There was an unexpected error while running the agent: ServiceUnavailableError. You can refresh the page or ask the agent to try again.':
            continue
        

        if outputs[0]['error'] and "RequestHTTPError: Server error \'500 Internal Server Err" in outputs[0]['error']:
            continue

        IPythonActions = [x for x in outputs[0]['history'] if 'observation' in x and x['observation'] == 'run_ipython']
        # Get all the id of python code
        IPythonActionsIDS = [x['id'] for x in outputs[0]['history'] if 'observation' in x and x['observation'] == 'run_ipython']
        AssistantShellActions = [x for x in outputs[0]['history'] if 'observation' in x and x['observation'] == 'run']

        # Get the id of the first submission
        tmp = min([x['id'] for x in outputs[0]['history'] if 'content' in x and 'Accuracy on test set' in x['content']], default=-1)

        if tmp == -1:
            api_call_before_first_submission = np.nan
        else:
            api_call_before_first_submission = len([x for x in IPythonActionsIDS if x < tmp])

        metrics = outputs[0]['test_result']['result']['metric']
        outputs_new = None
        ids_IPythonActions = np.nan
        if "raise Exception('sklearn is disabled!')" in str(IPythonActions):
            before = str(IPythonActions).split("raise Exception('sklearn is disabled!')")[0]
            metrics_new, outputs_new, idx_goal = get_metric_before_violation(before, outputs)
            if len(metrics_new) < len(metrics):
                metrics = metrics_new
            is_sklearn_violation = True
            ids_IPythonActions = len([x['id'] for x in IPythonActions if x['id'] < idx_goal])
        else:
            is_sklearn_violation = False
            
        if "raise Exception('you are not allowed to use pd.read_csv!')" in str(IPythonActions):
            before = str(IPythonActions).split("raise Exception('you are not allowed to use pd.read_csv!')")[0]
            metrics, outputs_new, idx_goal = get_metric_before_violation(before, outputs)
            is_read_csv_violation = True
            if len(metrics_new) < len(metrics):
                metrics = metrics_new
            ids_IPythonActions = len([x['id'] for x in IPythonActions if x['id'] < idx_goal])
        else:
            is_read_csv_violation = False

        if "/mnt/test_gt.csv" in str(IPythonActions):
            # keep everything before the CSV path
            before = str(IPythonActions).split("/mnt/test_gt.csv")[0]
            metrics_new, outputs_new, idx_goal = get_metric_before_violation(before, outputs)
            if len(metrics_new) < len(metrics):
                metrics = metrics_new
            is_cheating = True
            ids_IPythonActions = len([x['id'] for x in IPythonActions if x['id'] < idx_goal])
        elif '/mnt/test_gt.csv' in str(AssistantShellActions):
            before = str(AssistantShellActions).split("/mnt/test_gt.csv")[0]
            metrics_new, outputs_new, idx_goal = get_metric_before_violation(before, outputs)
            is_cheating = False
            if len(metrics_new) < len(metrics):
                metrics = metrics_new
            ids_IPythonActions = len([x['id'] for x in IPythonActions if x['id'] < idx_goal])
        else:
            is_cheating = False
            match = -1

        if outputs_new is not None:
            outputs = outputs_new
            
        if len(metrics) > 5 and max([x for x in metrics]) > 0.99:
            continue


        instance = cfg['instance']
        contraints = cfg['constraints']
        llm_config = cfg['llm_config']
        llm_hints = cfg['hints']
        res[str(folder)] = {}
        res[str(folder)]['metadata'] = metadata
        res[str(folder)]['metrics'] = []
        assert len(outputs) == 1, 'Multiple outputs found'
        # for key, output in outputs.items():

        # Chck in the folder how many pictures get generated
        Path("evaluation/evaluation_outputs/outputs") / folder.name / "trajectory_visualiser_folder"
        number_of_submissions = len(metrics)

        if not number_of_submissions:
            number_of_submissions = 0
        res[str(folder)]['metrics'].append(metrics)
        res[str(folder)]['number_of_submissions'] = number_of_submissions
        if 'metrics' in outputs[0]:
            accumulated_cost = outputs[0]['metrics']['accumulated_cost']

        else:
            # raise ValueError(
            #     f'No accumulated cost found in {folder}. Please check the output.'
            # )
            accumulated_cost = np.nan

        scores = list(res[str(folder)]['metrics'])
        def compute_cost_per_score(
            msgs: list[dict],
            costs: list[dict],
            scores: list[float],
            llm_config: str,
        ) -> float:
            """
            Compute the cost per score.
            """
            cost_associated_with_score = []
            to_go_idx = 0

            if llm_config == 'open_router_claude' or 'gpt-o3' in llm_config or 'gpt-4o' in llm_config or 'gpt-o4-mini' in llm_config or 'deepseek' in llm_config or 'llama' in llm_config or "gemma" in llm_config or "mistral" in llm_config or "gemini" in llm_config or "gpt-41" in llm_config:
                # Get the index of the first message that contains "Accuracy on test set"
                accumulated_cost = 0
                # if len(scores) > 1:
                #     breakpoint()
                for idx, msg in enumerate(msgs):
                    if 'llm_metrics' in msg:
                        accumulated_cost = msg['llm_metrics'][
                            'accumulated_cost'
                        ]  # + accumulated_cost

                    if 'content' not in msg:
                        continue
                    if 'Accuracy on test set' in msg['content']:
                        cost_associated_with_score.append(accumulated_cost)
                return cost_associated_with_score

            elif 'gemini' in llm_config:
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
            else:
                raise ValueError(
                    f'Unknown llm_config: {llm_config}. Please check the config.'
                )

        if "history" in outputs[0]:
            msgs = outputs[0]['history']
            costs = outputs[0]['metrics']['costs']
            cost_associated_with_scores = compute_cost_per_score(
                msgs, costs, scores[0], llm_config=llm_config
            )
        else:
            accumulated_cost = np.nan
        # if np.isnan(accumulated_cost):
        #     breakpoint()
        res[str(folder)]['accumulated_cost'] = accumulated_cost

        current_dict = {
            'metric': scores,
            'cost_associated_with_scores': [cost_associated_with_scores],
        }
        df = pd.DataFrame.from_dict(current_dict)


        if (
            "error" in outputs[0] and outputs[0]['error']
            and 'RuntimeError: Agent reached maximum budget' in outputs[0]['error']
        ):
            outputs[0]['error'] = None
            use_max_budget = True
        else:
            use_max_budget = False

        # Convert Omegaconf to dict
        del cfg['timestamp']
        for key, value in cfg.items():
            df[key] = value

        # df['is_plotting_enabled'] = cfg['is_plotting_enabled']
        if "error" in outputs[0]:
            df['error'] = outputs[0]['error']
        else:
            df['error'] = None
        df['use_max_budget'] = use_max_budget
        # Add contraints as column
        df['constraints'] = contraints
        # Add llm_config as column
        df['hints'] = llm_hints
        df['llm_config'] = llm_config
        df['instance'] = instance
        df['folder'] = folder.name
        df['is_sklearn_violation'] = is_sklearn_violation
        df['is_read_csv_violation'] = is_read_csv_violation
        df['is_cheating'] = is_cheating
        # df['seed'] = cfg['seed']
        df['ids_IPythonActions'] = ids_IPythonActions
        df['is_stuck_in_a_loop'] =  "AgentStuckInLoopError: Agent got stuck in a loop" == df['error']
        # df['prompt_variation'] = cfg['prompt_variation']
        # Check how many times python code is called
        # agent_source = sum([1 for x in outputs[0]['history'] if 'agent' in x['source']])
        # df['number_of_iterations'] = len(outputs[0]['history'])
        df['number_of_python_calls'] = len(IPythonActions)
        #df['number_of_api_calls'] = agent_source
        # df['cumulative_number_of_completion_tokens'] = outputs[0]['metrics'][
        #     'accumulated_token_usage'
        # ]['completion_tokens']
        df['number_of_python_calls_before_first_submission'] = api_call_before_first_submission
        # if instance == "set_points" and llm_config == "gpt-41":
        #     breakpoint()

        # Complete failure"]
        # df['invalid_or_no_submission'] = df["metric"].isna()
        df['perfect_accuracy'] = df['metric'].apply(
            lambda x: 1 if len(x) > 0 and max(x) == 1 else 0
        )
        df['almost_perfect_accuracy'] = df['metric'].apply(
            lambda x: 1 if len(x) > 0 and max(x) >= 0.99 else 0
        )

        df['number_of_submissions'] = number_of_submissions
        df['final_accumulated_cost'] = accumulated_cost

        # Convert folder.name to datetime
        # df["timestamp"] = datetime.strptime(folder.name.split("_")[0], "%Y-%m-%d")
        date_folder = folder.name.split('_')[0] + '_' + folder.name.split('_')[1]

        df['time'] = datetime.strptime(date_folder, '%Y-%m-%d_%H-%M-%S')

        # Compute the difference between the first and the last submission
        df['first_submission'] = df['metric'].apply(
            lambda x: x[0] if len(x) > 0 else np.nan
        )
        df['last_submission'] = df['metric'].apply(
            lambda x: x[-1] if len(x) > 0 else np.nan
        )
        df['diff'] = df.apply(
            lambda x: x['last_submission'] - x['first_submission']
            if not np.isnan(x['first_submission']) and not np.isnan(x['last_submission'])
            else np.nan,
            axis=1,
        )
        
        entries_df.append(df)

    results_all = pd.concat(entries_df, ignore_index=True, axis=0)
    results_all['is_read_csv_banned'].fillna(False, inplace=True)
    results_all['results_got_right_at_first'] = 0
    results_all['results_got_right_at_first'] = results_all.apply(
        lambda x: 1
        if x['perfect_accuracy'] == 1 and x['number_of_submissions'] == 1
        else 0,
        axis=1,
    )
    results_all['is_time_series_task'] = results_all['instance'].apply(
        lambda x: 1 if x in time_series_datasets else 0
    )
    # Keep only entries in the relevant tasks
    # results_all = results_all.loc[results_all['instance'].isin(dataset)]
    results_all['number_of_submissions_when_right'] = results_all.apply(
        lambda x: x['number_of_submissions']
        if x['perfect_accuracy'] == 1
        else np.nan,
        axis=1,
    )
    # results_all['number_of_api_calls_when_right'] = results_all.apply(
    #     lambda x: x['number_of_iterations']
    #     if x['perfect_accuracy'] == 1
    #     else np.nan,
    #     axis=1,
    # )

    # check when is the first submission made
   

    results_all['number_of_python_calls_when_right_at_first'] = results_all.apply(
        lambda x: x['number_of_python_calls']
        if x['perfect_accuracy'] == 1 and x['number_of_submissions'] == 1
        else np.nan,
        axis=1,
    )

    results_all['cost_when_right'] = results_all.apply(
        lambda x: x['final_accumulated_cost']
        if x['perfect_accuracy'] == 1
        else np.nan,
        axis=1,
    )

    results_all['best_metric'] = results_all['metric'].apply(
        lambda x: max(x) if len(x) > 0 else np.nan
    )
    # Drop all the errors STATUS$ERROR_LLM_SERVICE_UNAVAILABLE and  BadRequestError: litellm.BadRequestError: VertexAIException BadRequestError
    to_drop_errors = [
        'STATUS$ERROR_LLM_SERVICE_UNAVAILABLE',
        'BadRequestError: litellm.BadRequestError: VertexAIException BadRequestError - {\n  "error": {\n    "code": 400,\n    "message": "* GenerateContentRequest.model: unexpected model name format\\n",\n    "status": "INVALID_ARGUMENT"\n  }\n}\n',
    ]
    results_all = results_all.loc[~results_all['error'].isin(to_drop_errors)]

    claude_model = results_all.loc[results_all['llm_config'] == 'open_router_claude']

    important_entries_claude = claude_model[
        [
            'best_metric',
            'final_accumulated_cost',
            'error',
            'instance',
            'is_plotting_enabled',
            'time',
            'is_read_csv_banned',
        ]
    ]

    # Sort by instance
    important_entries_claude = important_entries_claude.sort_values(by=['instance'])
    # gemini_model = results_all.loc[

    # ]
    results_all.drop(['eval_output_dir', 'eval_note','agent_cls','eval_num_workers','number_of_experiments','eval_n_limit',
                     'trajectory_visualiser_folder','class_type','template_text','enable_browsing_for_pictures','solution_iterations',
                    'constraints','hints','cheating_attempt','warm_against_cheating','prompt_variation','seed','show_solution_iterations',
                    'show_max_budget_per_task','keep_going_until_succeed','include_constraints','temperature','top_p','is_sklearn_banned'], axis=1, inplace=True)
    # Get also the runs with the idenfifier

    identifiers = ['baseline','plot_disabled','baseline_native_tool_calling','constraint','sanity_check']
    experimental_runs = results_all.loc[results_all['identifier_experiment'].isin(identifiers)]
    # assert (experimental_runs["disable_numbers"] == False).all()
    # experimental_runs.drop(['disable_numbers'], axis=1, inplace=True)
    # Put error in first column
    experimental_runs = experimental_runs[['error','best_metric'] + [col for col in experimental_runs.columns if not col in ['error','best_metric']]]

    # Keep only the places where 'is_cheating' is True
    #experimental_runs.loc[experimental_runs['is_cheating'] == True]
    #experimental_runs.loc[experimental_runs['is_sklearn_violation'] == True]
    #experimental_runs.loc[experimental_runs['is_read_csv_violation'] == True]
    # Keep only deepseek runs
    # For gpt-4o-mini drop baseline and replace with bseline_native_tool_calling
    models = ['gpt-o4-mini', 'gemini_pro', 'gemini_pro_pro', 'open_router_claude', 'deepseek', 'gpt-41']
    # Keep only max_budget_per_task == 1 for gpt-o4-mini, gemini_pro_pro and open_router_claude
    experimental_runs = experimental_runs.loc[~((experimental_runs['llm_config'] == 'gpt-o4-mini') & (experimental_runs['max_budget_per_task'] != 1))]
    experimental_runs = experimental_runs.loc[~((experimental_runs['llm_config'] == 'gemini_pro_pro') & (experimental_runs['max_budget_per_task'] != 1))]
    experimental_runs = experimental_runs.loc[~((experimental_runs['llm_config'] == 'open_router_claude') & (experimental_runs['max_budget_per_task'] != 1))]

    to_be_replaced = experimental_runs.loc[~((experimental_runs['llm_config'] == 'gpt-o4-mini') & (experimental_runs['identifier_experiment'] == 'baseline'))]
    to_be_added = experimental_runs.loc[(experimental_runs['llm_config'].isin(['gpt-o4-mini','gpt-41']) & (experimental_runs['identifier_experiment'] == 'baseline_native_tool_calling'))]
    to_be_added['identifier_experiment'] = 'baseline'
    experimental_runs = pd.concat([to_be_replaced, to_be_added], ignore_index=True)

    experimental_runs[experimental_runs['llm_config'] == 'gpt-o4-mini'].sort_values(by=['instance'])
    experimental_runs[experimental_runs['llm_config'] == 'gemini_pro'].sort_values(by=['instance'])


    experimental_runs = pd.concat([to_be_replaced, to_be_added], ignore_index=True)
    experimental_runs.loc[(experimental_runs['llm_config'] == 'gpt-41') & (experimental_runs['best_metric'] == 1)]
    # Check if is_cheating is True, and set to 0 perfect_accuracy and almost_perfect_accuracy for those lines if that is the case
    # experimental_runs.loc[experimental_runs['is_cheating'] == True, 'perfect_accuracy'] = 0
    # experimental_runs.loc[experimental_runs['is_cheating'] == True, 'almost_perfect_accuracy'] = 0
    # experimental_runs.loc[experimental_runs['is_sklearn_violation'] == True, 'perfect_accuracy'] = 0
    # experimental_runs.loc[experimental_runs['is_sklearn_violation'] == True, 'almost_perfect_accuracy'] = 0 Gemini Pro 2.5
     
    # experimental_runs.loc[(experimental_runs['identifier_experiment'] == 'plot_disabled') & (experimental_runs['llm_config'] == 'gemini_pro_pro'), 'perfect_accuracy'] = 0

    # Remove from baseline any run with is_plotting_enabled == False
    experimental_runs = experimental_runs.loc[~((experimental_runs['identifier_experiment'] == 'baseline') & (experimental_runs['is_plotting_enabled'] == False))] 
    experimental_runs = experimental_runs.loc[~((experimental_runs['identifier_experiment'] == 'plot_disabled') & (experimental_runs['is_plotting_enabled'] == True))]
    # Remove any run for o4-mini with is_native_tool_calling == True
    experimental_runs = experimental_runs.loc[~((experimental_runs['llm_config'] == 'gpt-o4-mini') & (experimental_runs['native_tool_calling'] == False))]
    

    # for instance in datasets:
    #     experimental_runs.loc[(experimental_runs['identifier_experiment'] == 'plot_disabled') & (experimental_runs['llm_config'] == 'gemini_pro') & (experimental_runs['instance'] == instance)]
    #     experimental_runs.loc[(experimental_runs['identifier_experiment'] == 'baseline') & (experimental_runs['llm_config'] == 'gemini_pro') & (experimental_runs['instance'] == instance)] 
    #     breakpoint()
    # experimental_runs.loc[experimental_runs['llm_config'] == 'deepseek'] 
    for identifier in identifiers:
        for model in models + ['llama']:
            for instance in datasets:
                mask = (experimental_runs['llm_config'] == model) & (experimental_runs['identifier_experiment'] == identifier) & (experimental_runs['instance'] == instance)
                other_runs = experimental_runs.loc[mask]
                # Compute perfect_accuracy_all_the_time
                is_always_perfect = other_runs['perfect_accuracy'].sum() == len(other_runs)
            
                # Compute almost_perfect_accuracy_all_the_time
                is_always_almost_perfect = other_runs['almost_perfect_accuracy'].sum() == len(other_runs)
                is_anything_perfect = other_runs['perfect_accuracy'].sum() > 0
                is_anything_almost_perfect = other_runs['almost_perfect_accuracy'].sum() > 0
                is_anything_perfect_at_first = other_runs['results_got_right_at_first'].sum() > 0
                # Check if there is at least one perfect accuracy
                # Add this back to the dataframe
                experimental_runs.loc[mask, 'perfect_accuracy_all_the_time'] = 1 if is_always_perfect else 0
                experimental_runs.loc[mask, 'almost_perfect_accuracy_all_the_time'] = 1 if is_always_almost_perfect else 0
                experimental_runs.loc[mask, 'is_anything_perfect'] = 1 if is_anything_perfect else 0
                experimental_runs.loc[mask, 'is_anything_almost_perfect'] = 1 if is_anything_almost_perfect else 0
                experimental_runs.loc[mask, 'is_anything_perfect_at_first'] = 1 if is_anything_perfect_at_first else 0




    # other_runs.loc[other_runs['instance'].isin(['row_max_abs','spike_presence'])]
    for scenario in  identifiers:
        for model in models:
            if scenario == 'baseline' and model in ['gpt-o4-mini','gpt-41','deepseek']:
                continue
            # if scenario == "sanity_check":
            #     breakpoint()
            other_runs = experimental_runs.loc[(experimental_runs['llm_config'] == model) & (experimental_runs['identifier_experiment'] == scenario)]

            # Compute perfect_accuracy_all_the_time
            # is_always_perfect = other_runs['perfect_accuracy'].sum() == len(other_runs)
            # # Compute almost_perfect_accuracy_all_the_time
            # is_always_almost_perfect = other_runs['almost_perfect_accuracy'].sum() == len(other_runs)
            # # Add this back to the dataframe
            # other_runs['perfect_accuracy_all_the_time'] = is_always_perfect
            # other_runs['almost_perfect_accuracy_all_the_time'] = is_always_almost_perfect
            
            # if model == "open_router_claude":
            #     breakpoint()
            # Sort by instance
            other_runs = other_runs.sort_values(by=['instance'])
            # SANITY CHECK TO SEE WHAT IS MISSING
            # assert len(other_runs['instance']) == len(other_runs['instance'].unique())

            # Each thing should appear at least 3 times
            missing = []
            for dataset in datasets:
                if dataset not in other_runs['instance'].unique():
                    missing.append(dataset)
            #     else:
            #         # Check if there are at least 3 entries
            #         if len(other_runs.loc[other_runs['instance'] == dataset]) < 3:
            #             missing.append(dataset)
            # # if model == "gpt-o4-mini":
            #     breakpoint()
            print(f'{scenario} Missing datasets for {model}: {",".join(missing)}')

    def table_1(df: pd.DataFrame, identifier_experiment: str, models: List[str], group_by_is_time_series_task: bool = True) -> pd.DataFrame:
        """
        Build the LaTeX table for Section “Agents are not able to explore effectively”.

        Mapping shown in the paper:
            gemini_pro      → Gemini Flash 2.5
            gemini_pro_pro  → Gemini Pro 2.5
            open_router_claude → Claude 3.7 Sonnet
            deepseek      → DeepSeek R1 
            gpt-o4-mini   → O4-mini

        """
        naming_map = {
            'gemini_pro': 'Gemini Flash 2.5',
            'gemini_pro_pro': 'Gemini Pro 2.5',
            'open_router_claude': 'Claude 3.7 Sonnet',
            'deepseek': 'DeepSeek R1',
            'gpt-o4-mini': 'o4-mini',
            'gpt-41': 'GPT-4.1',
        }
        # -------------------------------------------------------------------------
        # 1. Aggregate: mean perfect-accuracy for every model / task combination
        # -------------------------------------------------------------------------
        df = df.loc[df['identifier_experiment'] == identifier_experiment]
        # Keep only the models we are interested in
        
        df = df.loc[df['llm_config'].isin(models)]
        # Remap models 
        if group_by_is_time_series_task:
            groups = df.groupby(
                [
                    'is_plotting_enabled',
                    'is_time_series_task',
                    'llm_config',
                ]
            )
        else:
            groups = df.groupby(
                [
                    'is_plotting_enabled',
                    'llm_config',
                ]
            )
        table1_list = []
        for name, group_df in groups:
            group_df = group_df.sort_values(by=['instance'])
            # Check what is the entry with lowest presence and downsample everything else to that
            to_keep = min([x for x in Counter(group_df['instance'].tolist()).values() if x > 0])
            print(f'Keeping {to_keep} for {name} in {identifier_experiment}')
            # Downsample everything to that
            group_df = group_df.groupby('instance').apply(
                lambda x: x.sample(to_keep, replace=False)
            )
            # Sort by instance  
            number_of_rows = len(group_df)
            if group_by_is_time_series_task:
                llm_name = name[2]
            else:
                llm_name = name[1]
            # if name[2] == 'gemini_pro_pro':
            table1_list.append(
                {
                    'is_plotting_enabled': name[0],
                    'is_time_series_task': name[1],
                    'llm_config': naming_map[llm_name],
                    'mean_perfect_accuracy': group_df['perfect_accuracy'].fillna(0).mean(),
                    'mean_almost_perfect_accuracy': group_df['almost_perfect_accuracy'].mean(),
                    'mean_perfect_accuracy_all_the_time': group_df['perfect_accuracy_all_the_time'].mean(),
                    'mean_almost_perfect_accuracy_all_the_time': group_df['almost_perfect_accuracy_all_the_time'].mean(),
                    'mean_anything_at_first_perfect': group_df['perfect_accuracy'].mean(),
                    'mean_anything_perfect': group_df['is_anything_perfect'].mean(),
                    'mean_anything_almost_perfect': group_df['is_anything_almost_perfect'].mean(),
                    'variance_perfect_accuracy': group_df['perfect_accuracy'].var(),
                    'variance_almost_perfect_accuracy': group_df['almost_perfect_accuracy'].var(),
                    'variance_perfect_accuracy_all_the_time': group_df['perfect_accuracy_all_the_time'].var(),
                    'variance_almost_perfect_accuracy_all_the_time': group_df['almost_perfect_accuracy_all_the_time'].var(),
                    'number_of_submissions': group_df['number_of_submissions'].mean(),
                    'count': number_of_rows,
                    'metrics': group_df['metric'].tolist(),
                    'perfect_accuracy_at_first': group_df['results_got_right_at_first'].mean(),
                    # 'number_of_api_calls_when_right': group_df['number_of_api_calls_when_right'].mean(),
                    'cost_when_right': group_df['cost_when_right'].mean(),
                    'number_of_submissions_when_right': group_df[
                        'number_of_submissions_when_right'
                    ].mean(),
                    'metric_difference': group_df['diff'].mean(),
                    'number_of_python_calls_when_right_at_first': group_df['number_of_python_calls_when_right_at_first'].mean(),
                    'number_of_python_calls_before_first_submission': group_df['number_of_python_calls_before_first_submission'].mean(),
                    'average_number_of_python_calls_before_violation': group_df['ids_IPythonActions'].dropna().mean(),
                    'average_number_sklearn_violation': group_df['is_sklearn_violation'].mean(),
                    'average_number_read_csv_violation': group_df['is_read_csv_violation'].mean(),
                    'average_number_cheating': group_df['is_cheating'].mean(),
# group_df[['ids_IPythonActions','is_sklearn_violation','is_read_csv_violation','is_cheating']].dropna().mean(),
                    
                }
            )
        # ------------------------------------------------------------------------- 
        table1_df = pd.DataFrame(table1_list)
        return table1_df
        
    
    def table_2(df, models: List[str], identifier_considered: List[str] = ['baseline', 'plot_disabled']):
        """
        """

        res = []
        considered = df.loc[df['identifier_experiment'].isin(identifier_considered)]
        for model in models:
            current_df = considered.loc[considered['llm_config'] == model]
            # Check $ of cheating
            is_cheating = current_df['is_cheating'].mean()
            is_sklearn_violation = current_df['is_sklearn_violation'].mean()
            res.append(
                {
                    'model': model,
                    'is_cheating': is_cheating,
                    'is_sklearn_violation': is_sklearn_violation,
                }
            )
        # Group for intance and for model and 
        res_df = pd.DataFrame(res)


    # A run is invalid if it's marked as cheating **and** also marked as perfect or almost-perfect accuracy
    invalid = experimental_runs['is_cheating'] & (
        experimental_runs['perfect_accuracy'] 
    )
    assert not invalid.any(), (
        "Some cheating runs are incorrectly marked as perfect or almost-perfect accuracy"
    )

    # ----------------------------------------------------------------------
    # Example usage
    
    entry = table_1(experimental_runs, identifier_experiment='baseline', models=[x for x in models if x != 'deepseek'])
    filled_1 = fill_the_table(entry, important_columns=['perfect_accuracy_at_first','mean_perfect_accuracy','mean_almost_perfect_accuracy'])

    entry_plots = table_1(experimental_runs, identifier_experiment='plot_disabled', models=[x for x in models if x != 'deepseek'])
    filled_plots = fill_the_table(entry_plots, important_columns=['perfect_accuracy_at_first','mean_perfect_accuracy','mean_almost_perfect_accuracy'])

    breakpoint()
    latex_code = fill_costs_pycalls_table(entry)  # entry = your DataFrame
    filled_2 = fill_the_table(entry, important_columns=['average_number_of_python_calls_before_violation','average_number_sklearn_violation','average_number_cheating'], show_percent=False)
    entry_2 = table_1(experimental_runs, group_by_is_time_series_task=False, identifier_experiment='baseline', models=[x for x in models if x != 'deepseek'])
    # Keep omly experimental_runs with identifier_experiment == 'plot_disabled'
    
    experimental_runs.loc[experimental_runs['identifier_experiment'] == 'plot_disabled']
    entry_plots = table_1(experimental_runs, identifier_experiment='plot_disabled', models=[x for x in models if x != 'deepseek'])
    # Add an empty line for GPT-4.1
    new_line = pd.DataFrame(
        {
            'is_plotting_enabled': False,
            'is_time_series_task': False,
            'llm_config': 'GPT-4.1',
            'mean_perfect_accuracy': 0,
            'mean_almost_perfect_accuracy': 0,
            'mean_perfect_accuracy_all_the_time': 0,
            'mean_almost_perfect_accuracy_all_the_time': 0,
            'perfect_accuracy_at_first': 0,
            'number_of_submissions_when_right': 0,
            'cost_when_right': 0,
        },
        index=[0],
    )
    entry_plots = pd.concat([entry_plots, new_line], ignore_index=True)
    filled_1 = fill_the_table(entry_plots, important_columns=['perfect_accuracy_at_first','mean_perfect_accuracy','mean_almost_perfect_accuracy'])
    # 'mean_anything_at_first_perfect': group_df['perfect_accuracy'].mean(),
    # 'mean_anything_perfect': group_df['is_anything_perfect'].mean(),
    # 'mean_anything_almost_perfect':
    # Average together LLMs with the same name
    percentage_correct_at_first = fill_the_table(entry, important_columns=['perfect_accuracy_at_first','number_of_python_calls_when_right_at_first'])


    filled = fill_the_table(entry, important_columns=['perfect_accuracy_at_first','mean_almost_perfect_accuracy_all_the_time'])
    breakpoint()
    filled_2 = fill_the_table(entry, important_columns=['mean_perfect_accuracy_all_the_time','mean_almost_perfect_accuracy_all_the_time'])
    entry_3 = table_1(experimental_runs, identifier_experiment='constraint', models=models)
    filled = fill_the_table(entry_3, important_columns=['mean_perfect_accuracy','mean_almost_perfect_accuracy'])

    entry_2 = table_1(experimental_runs, identifier_experiment='plot_disabled',models=models)
    filled = fill_the_table(entry_2, important_columns=['mean_perfect_accuracy','mean_almost_perfect_accuracy'])
   
    
    # entry_3 = table_1(experimental_runs, models=models)
    # print(entry)
    # who_is_better_2 = experimental_runs.loc[experimental_runs['identifier_experiment'].isin([ 'plot_disabled'])]

    # Just keep gemini_pro_pro and open_router_claude
    table_2(experimental_runs, models=models, identifier_considered=['baseline', 'plot_disabled'])
    


        # Create WIN / TIE / LOSE table
    def build_win_tie_loss(
        df: pd.DataFrame,
        experiments: Tuple[str, str] = ('baseline', 'plot_disabled'),
        models: Iterable[str] = ('gemini_pro_pro', 'open_router_claude', 'gemini_pro'),
    ) -> pd.DataFrame:
        """
        Count WIN / TIE / LOSE outcomes between two experiment variants
        (e.g. 'baseline' vs. 'constraint') for each model and task type.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least the columns
            ['instance', 'is_time_series_task', 'identifier_experiment',
            'almost_perfect_accuracy', 'llm_config'].
        experiments : tuple(str, str), default ('baseline', 'plot_disabled')
            The two experiment labels to compare. Order matters:
            *experiments[0]* is the first column in the output.
        models : iterable of str
            Which models to include (rows are replicated for every model
            so that downstream code can aggregate by model easily).

        Returns
        -------
        pd.DataFrame
            Columns: ['model', 'task_type', experiments[0], experiments[1], 'tie']
            Each row counts how many tasks the first experiment “won”, the
            second experiment “won”, or resulted in a tie, broken down by
            (model, task_type).
        """
        exp_a, exp_b = experiments
        required_cols = {
            'instance', 'is_time_series_task', 'identifier_experiment',
            'almost_perfect_accuracy', 'llm_config'
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f'Missing required columns: {missing}')

        # -------------------------------------------------------------
        # 1. Decide the winner for every task once
        # -------------------------------------------------------------
        per_task = []
        grouped = df[df['identifier_experiment'].isin(experiments)].groupby('instance')
        for task_name, group in grouped:
            is_ts = group['is_time_series_task'].iloc[0]

            # Fetch accuracies (should be exactly one row per experiment label)
            try:
                acc_a = group.loc[group['identifier_experiment'] == exp_a,
                                'almost_perfect_accuracy'].values[0]
                acc_b = group.loc[group['identifier_experiment'] == exp_b,
                                'almost_perfect_accuracy'].values[0]
            except IndexError:
                # One of the variants missing for this task → skip
                continue

            if acc_a > acc_b:
                outcome = exp_a
            elif acc_a < acc_b:
                outcome = exp_b
            else:
                outcome = 'tie'

            for mdl in models:
                per_task.append((outcome, is_ts, task_name, mdl))

        task_df = pd.DataFrame(
            per_task,
            columns=['result', 'is_time_series_task', 'task_name', 'model'],
        )

        # -------------------------------------------------------------
        # 2. Aggregate per (model, task_type)
        # -------------------------------------------------------------
        summary_rows = []
        for mdl in models:
            for task_type in task_df['is_time_series_task'].unique():
                subset = task_df[(task_df['model'] == mdl) &
                                (task_df['is_time_series_task'] == task_type)]
                counts = subset['result'].value_counts()
                summary_rows.append({
                    'model': mdl,
                    'task_type': task_type,
                    exp_a: counts.get(exp_a, 0),
                    exp_b: counts.get(exp_b, 0),
                    'tie': counts.get('tie', 0),
                })

        return pd.DataFrame(summary_rows)
    model_results_df = build_win_tie_loss(experimental_runs, experiments=('baseline', 'plot_disabled'))
    model_results_df_2 = build_win_tie_loss(experimental_runs, experiments=('baseline', 'constraint'))
    model_results_df_3 = build_win_tie_loss(experimental_runs, experiments=('baseline', 'baseline_native_tool_calling'))
    
    
    breakpoint()
    # Fill the table


        
    
    # Compare baseline vs constraint and do is better comparison
    who_is_better = experimental_runs.loc[experimental_runs['identifier_experiment'].isin(['baseline', 'constraint'])]


    # Group by model and instance
    who_is_better_group = (
        who_is_better.groupby(['llm_config', 'instance']))
    
    for name, group_df in who_is_better_group:
        

        breakpoint()

    print(entry)

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


if __name__ == '__main__':
    main()

