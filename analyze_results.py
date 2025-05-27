"""
llm_client.py – April 2025
Python 3.12 compatible utility for GPT‑4o‑mini & Gemini 2.0 models
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from textwrap import indent
from typing import List, Optional

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf

load_dotenv(find_dotenv())  # automatically walks up folders


def round_table(latex: str, decimals: int = 2) -> str:
    """
    Round every floating-point number that appears in a LaTeX tabular string.

    Parameters
    ----------
    latex : str
        The LaTeX source (e.g. the value of your `latex` variable).
    decimals : int, optional
        How many decimals to keep (default: 2).

    Returns
    -------
    str
        A new LaTeX string with all floats rounded.
    """
    # Match an isolated float (e.g. 0.330000, 1.000000, .5).
    float_re = re.compile(
        r"""
        (?<![0-9.])      # not preceded by another digit/dot
        (?:\d*\.\d+)     # a decimal number
        (?![0-9.])       # not followed by another digit/dot
        """,
        re.VERBOSE,
    )

    return float_re.sub(
        lambda m: f"{float(m.group()):.{decimals}f}",
        latex,
    )


def make_overleaf_table(
    entry_with: pd.DataFrame,
    entry_without: pd.DataFrame,
    metric: str = "Acc100_mean",
    round_decimals: int = 2,
    na_rep: str = "–",
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """
    Build a LaTeX table ready for Overleaf.

    Parameters
    ----------
    entry_with, entry_without : pd.DataFrame
        Output of your `group_table(...)` call, containing the columns
        ['Model', 'instance', metric].
    metric : str, default 'Acc100_mean'
        Which performance column to show ('OneShotAcc_mean', 'Acc99_mean', …).
    round_decimals : int, default 2
        How many decimals to keep.
    na_rep : str, default '–'
        What to print when a cell is NaN.
    caption, label : str or None
        Optional LaTeX caption / label strings.

    Returns
    -------
    str  –  LaTeX code block.
    """

    # Sanity checks
    required_cols = {"Model", "instance", metric}
    assert required_cols.issubset(entry_with.columns), (
        f"entry_with missing columns: {required_cols - set(entry_with.columns)}"
    )
    assert required_cols.issubset(entry_without.columns), (
        f"entry_without missing columns: {required_cols - set(entry_without.columns)}"
    )

    # 1 · pivot both tables so that rows = tasks, cols = models
    piv_w  = entry_with.pivot(index="instance", columns="Model", values=metric)
    piv_wo = entry_without.pivot(index="instance", columns="Model", values=metric)

    # 2 · harmonise task & model ordering
    all_tasks  = sorted(set(piv_w.index).union(piv_wo.index))
    all_models = sorted(set(piv_w.columns).union(piv_wo.columns))
    piv_w  = piv_w.reindex(index=all_tasks,  columns=all_models)
    piv_wo = piv_wo.reindex(index=all_tasks, columns=all_models)

    # 3 · add a second level to the column index: ('Model', 'With'/'Without')

    piv_w .columns = pd.MultiIndex.from_tuples([(m, "with")    for m in piv_w.columns ])
    piv_wo.columns = pd.MultiIndex.from_tuples([(m, "w/o") for m in piv_wo.columns])

    # 4 · concatenate → table; order columns as (Model₁,With) (Model₁,Without) …
    table = pd.concat([piv_w, piv_wo], axis=1)
    table = table.loc[:, sorted(table.columns, key=lambda x: (x[0], x[1]))]
    table = table.round(round_decimals)
    # 5 · emit LaTeX
    n_models   = len(all_models)
    col_format = "l" + "c" * (2 * n_models)   # 1 left-aligned index + 2 per model
    latex = table.to_latex(
        multicolumn=True,
        multirow=False,
        escape=False,          # keep underscores in task names
        na_rep=na_rep,
        column_format=col_format,
        caption=caption,
        label=label,
    )

    latex = round_table(latex)
    # Replace any number from 0.000000 to 0.00

    return latex


def create_accuracy(
    model_results_df: pd.DataFrame,
    important_columns: tuple[str, ...] = (
        'OneShotAcc',  # One-Shot 100 %
        'mean_perfect_accuracy',  # Acc₁₀₀ %
        'mean_Acc99',  # Acc₉₉ %
    ),
    model_order: tuple[str, ...] = (
        'Gemini Flash 2.5',
        'Gemini Pro 2.5',
        'GPT-4.1',
        'o4-mini',
        'Claude 3.7 Sonnet',
    ),
    show_percent: bool = True,  # ⇦ NEW FLAG
) -> str:
    """
    Build a LaTeX table with accuracy metrics for time-series and tabular tasks.

    Parameters
    ----------
    show_percent : bool, default True
        • True  → format 0.263158 as ``26.32\\%``
        • False → format 0.263158 as ``0.2632``
    """

    # Sanity checks
    required = {'llm_config', 'is_time_series_task', *important_columns}
    missing = required - set(model_results_df.columns)
    assert not missing, f"create_accuracy missing columns: {missing}"

    # ---------- helpers -------------------------------------------------------
    def fmt(x: float) -> str:
        """Format either as percentage or plain decimal."""
        return f'{round(x * 100, 2):.2f}\\%' if show_percent else f'{x:.4f}'

    # ---------- wide pivot ----------------------------------------------------
    pivot = model_results_df.pivot(
        index='llm_config',
        columns='is_time_series_task',  # 0 ↔ tabular, 1 ↔ time-series
        values=list(important_columns),
    )

    # Ensure all required models are present
    missing_models = set(model_order) - set(pivot.index)
    assert not missing_models, f"Missing models in pivot: {missing_models}"

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
                    cell = f'\\textbf{{{cell}}}'
                parts.append(cell)

        # swap slices so headings match data
        ts_chunk = ' & '.join(
            parts[1 + n_metrics : 1 + 2 * n_metrics]
        )  # structure == 1
        tb_chunk = ' & '.join(parts[1 : 1 + n_metrics])  # structure == 0
        rows.append(f'    {model} &\n        {ts_chunk} &\n        {tb_chunk} \\\\')
    body = '\n'.join(rows)

    # ---------- headings ------------------------------------------------------
    metric_headers = {
        'OneShotAcc': r'OneShotAcc\textsubscript{100\%}',
        'mean_perfect_accuracy': r'Acc\textsubscript{100\%}',
        'mean_Acc99': r'Acc\textsubscript{99\%}',
    }
    header_cells = ' & '.join(metric_headers.values())

    first_end = 1 + n_metrics
    second_end = 1 + 2 * n_metrics
    cmidrules = (
        rf'\cmidrule(lr){{2-{first_end}}}\cmidrule(lr){{{first_end+1}-{second_end}}}'
    )
    col_spec = '@{}l' + 'c' * (2 * n_metrics) + '@{}'

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
""".strip('\n')

    return latex


def fill_costs_pycalls_table(
    model_results_df: pd.DataFrame,
    important_columns: tuple[str, ...] = (
        'cost_when_right',  # Costright
        'number_of_python_calls_before_first_submission',  # #PyCalls_before
        'number_of_python_calls_when_right_at_first',  # #PyCalls_right
    ),
    model_order: tuple[str, ...] = (
        'Gemini Flash 2.5',
        'Gemini Pro 2.5',
        'GPT-4.1',
        'o4-mini',
        'Claude 3.7 Sonnet',
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

    # Sanity checks
    required_cols = {
        'llm_config',
        'is_time_series_task',
        *important_columns,
    }
    missing = required_cols - set(model_results_df.columns)
    assert not missing, f"fill_costs_pycalls_table missing columns: {missing}"

    # ---------- helpers -------------------------------------------------------
    def fmt_cost(x: float) -> str:  # 0.0269
        return f'{x:.4f}'

    def fmt_calls(x: float) -> str:  # 5.63
        return f'{x:.2f}'

    _format = {
        'cost_when_right': fmt_cost,
        'number_of_python_calls_before_first_submission': fmt_calls,
        'number_of_python_calls_when_right_at_first': fmt_calls,
    }

    # ---------- wide pivot ----------------------------------------------------
    # 0 ↔ tabular, 1 ↔ time-series (see slice swap below)
    pivot = model_results_df.pivot_table(
        index='llm_config',
        columns='is_time_series_task',
        values=list(important_columns),
        aggfunc='mean',
    )

    missing_models = set(model_order) - set(pivot.index)
    assert not missing_models, f"Missing models in pivot: {missing_models}"

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
        ts_chunk = ' & '.join(
            parts[1 + n_metrics : 1 + 2 * n_metrics]
        )  # structure == 1
        tb_chunk = ' & '.join(parts[1 : 1 + n_metrics])  # structure == 0
        rows.append(f'    {model} &\n        {ts_chunk} & {tb_chunk} \\\\')
    body = '\n'.join(rows)

    # ---------- headings ------------------------------------------------------
    metric_headers = {
        'cost_when_right': r'Cost\textsubscript{right}',
        'number_of_python_calls_before_first_submission': r'\#PyCalls\textsubscript{before}',
        'number_of_python_calls_when_right_at_first': r'\#PyCalls\textsubscript{right}',
    }
    header_cells = ' & '.join(metric_headers[col] for col in important_columns)

    first_end = 1 + n_metrics
    second_end = 1 + 2 * n_metrics
    cmidrules = (
        rf'\cmidrule(lr){{2-{first_end}}}\cmidrule(lr){{{first_end+1}-{second_end}}}'
    )
    col_spec = '@{}l' + 'c' * (2 * n_metrics) + '@{}'

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
""".strip('\n')

    return latex


def clean_metrics(df: pd.DataFrame, nan_option: str) -> pd.DataFrame:
    """
    Handle NaNs in the 'metric' column according to nan_option.
    """
    assert 'metric' in df.columns, "Column 'metric' missing"
    assert nan_option in {'-1', '0', 'none'}

    df['metric'] = df['metric'].apply(lambda x: max(x) if len(x) > 0 else np.nan)
    if nan_option == '-1':
        df = df.dropna(subset=['metric'])
    elif nan_option == '0':
        # Replace NaNs with 0
        df['metric'] = df['metric'].fillna(0)

    return df


# Regex that captures both the timestamp and the run-index
_TS_WITH_IDX_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(?P<idx>\d+)$'
)


def _load_experiment(folder: Path) -> tuple[dict, dict]:
    assert folder.is_dir(), f"Folder does not exist: {folder}"
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

    assert base_dir.is_dir(), f"Base directory does not exist: {base_dir}"
    assert after is None or isinstance(after, datetime)
    assert before is None or isinstance(before, datetime)

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
    assert outputs and isinstance(outputs[0], dict), "outputs must contain dicts"
    assert 'history' in outputs[0], "outputs[0] must contain 'history'"

    # find all occurrences of "'id': <digits>"
    match_id = int(re.findall(r"'id'\s*:\s*(\d+)", before)[-1])

    # Collect all the accuracy on the test set Accuracy on test set
    outputs = [
        x
        for x in outputs[0]['history']
        if 'content' in x
        and 'Accuracy on test set' in x['content']
        and x['id'] < match_id
    ]
    # Keep only those before the match

    accuracies = []
    pattern = re.compile(r'Accuracy on test set \d+:\s*([0-9]*\.?[0-9]+)')

    for x in outputs:
        content = x.get('content', '')
        # find all numbers after “Accuracy on test set <digit>:”
        for match in pattern.findall(content):
            accuracies.append(float(match))

    return accuracies, [outputs], match_id


def filter_actions_by_observation(history: list[dict], obs_type: str) -> list[dict]:
    """
    Return all entries in `history` whose 'observation' field equals `obs_type`.
    """
    assert isinstance(history, list)
    return [entry for entry in history if entry.get('observation') == obs_type]


def get_action_ids_by_observation(history: list[dict], obs_type: str) -> list[int]:
    """
    Return the list of 'id' fields for entries in `history` matching `obs_type`.
    """
    filtered = filter_actions_by_observation(history, obs_type)
    return [entry['id'] for entry in filtered]


# Get all the entries evaluation/evaluation_outputs/outputs
ROOT_DIR = Path('evaluation/evaluation_outputs/outputs')
AFTER = datetime.strptime('2025-05-12_00-00-00', '%Y-%m-%d_%H-%M-%S')
CHECKING_MISSING_ENTRIES = False
BEFORE = None  # datetime.strptime('2025-05-06_11-47-22', '%Y-%m-%d_%H-%M-%S')
# AFTER = None
# BEFORE = None
METADATA_JSON = 'metadata.json'
OUTPUT_JSON = 'output.jsonl'
VALID_MODELS = ['gemini_pro_pro', 'gemini_pro', 'gemini_lite',
       'open_router_claude', 'gpt-4o', 'gpt-4o-mini', 'gpt-o3',
       'open_router_gpt-4o', 'gpt-o3-mini', 'gpt-o4-mini', 'deepseek',
       'llama', 'gemma-3-27b-it', 'mistral-small-24b-instruct-2503',
       'gpt-41']

MODELS_TO_DROP = ["gemini_lite", "gpt-o3", "gpt-4o-mini"]

FATAL_ERRORS = {
    "BadRequestError: litellm.BadRequestError: OpenAIException - Unsupported parameter: 'stop' is not supported with this model.",
    'STATUS$ERROR_LLM_INTERNAL_SERVER_ERROR',
    'RuntimeError: There was an unexpected error while running the agent: APIConnectionError. You can refresh the page or ask the agent to try again.',
    'RuntimeError: There was an unexpected error while running the agent: ServiceUnavailableError. You can refresh the page or ask the agent to try again.',
}


solutions = {
    'find_peaks': 'The correct feature is to use the number of peaks in the signal to tell the class of the signal.',
    'predict_ts_stationarity': 'The correct feature is to tell whether the time series is stationary or not.',
    'frequency_band': 'The correct feature that separates the two classes is the frequency band (signals from the first class have 0-4 Hz, while the second class has 20-50 Hz).',
    'set_points': 'The correct feature is to consider whether there are set points in the signal or not.',
}

TASKS = [
    'channel_corr_easy',
    'channel_corr_hard',
    'cofounded_group_outlier',
    'common_frequency',
    'dominant_feature',
    'find_peaks',
    'ground_mean_threashold',
    'outlier_ratio',
    'periodic_presence',
    'predict_ts_stationarity',
    'row_max_abs',
    # 'row_variance',
    'sign_rotated_generator',
    'simultanus_spike',
    'sum_threshold',
    'variance_burst',
    'zero_crossing',
]

time_series_datasets = [
    'simultanus_spike',
    'channel_corr',
    'channel_divergence',
    'variance_burst',
    'common_frequency',
    'predict_ts_stationarity',
    'zero_crossing',
    'find_peaks',
    'periodic_presence',
]

NAMING_MAP  = {
            'gemini_pro': 'Gemini Flash 2.5',
            'gemini_pro_pro': 'Gemini Pro 2.5',
            'open_router_claude': 'Claude 3.7 Sonnet',
            'deepseek': 'DeepSeek R1',
            'gpt-o4-mini': 'o4-mini',
            'gpt-41': 'GPT-4.1',
        }

tabular_datasets = ['sum_threshold']


def main():
    runs = sorted(get_folders_in_range(ROOT_DIR, AFTER, BEFORE))

    # Iterate over the folders
    res = {}
    entries_df = []
    for folder_identifier, folder in enumerate(runs):
        # Open metadata
        metadata, outputs, cfg = _load_experiment(folder)




        if not outputs:
            continue

        if 'identifier_experiment' not in cfg:
            continue

        err = outputs[0].get('error')

        if err in FATAL_ERRORS or (
            err and "RequestHTTPError: Server error '500 Internal Server Err" in err
        ):
            continue

        history = outputs[0]['history']
        IPythonActions = filter_actions_by_observation(history, 'run_ipython')
        IPythonActionsIDS = get_action_ids_by_observation(history, 'run_ipython')
        assert len(IPythonActions) == len(
            IPythonActionsIDS
        ), 'IPythonActions and IPythonActionsIDS should be the same length'
        AssistantShellActions = filter_actions_by_observation(history, 'run')

        # Get the id of the first submission
        tmp = min(
            [
                x['id']
                for x in outputs[0]['history']
                if 'content' in x and 'Accuracy on test set' in x['content']
            ],
            default=-1,
        )

        if tmp == -1:
            api_call_before_first_submission = np.nan
        else:
            api_call_before_first_submission = len(
                [x for x in IPythonActionsIDS if x < tmp]
            )

        metrics = outputs[0]['test_result']['result']['metric']
        outputs_new = None
        ids_IPythonActions = np.nan
        str_IPythonActions = str(IPythonActions)

        before, sep, _ = str_IPythonActions.partition(
            "raise Exception('sklearn is disabled!')"
        )
        if sep:
            metrics_new, outputs_new, idx_goal = get_metric_before_violation(
                before, outputs
            )
            if len(metrics_new) < len(metrics):
                metrics = metrics_new
            is_sklearn_violation = True
            ids_IPythonActions = len(
                [x['id'] for x in IPythonActions if x['id'] < idx_goal]
            )
        else:
            is_sklearn_violation = False

        before, sep, _ = str_IPythonActions.partition(
            "raise Exception('you are not allowed to use pd.read_csv!')"
        )
        if sep:
            metrics, outputs_new, idx_goal = get_metric_before_violation(
                before, outputs
            )
            is_read_csv_violation = True
            if len(metrics_new) < len(metrics):
                metrics = metrics_new
            ids_IPythonActions = len(
                [x['id'] for x in IPythonActions if x['id'] < idx_goal]
            )
        else:
            is_read_csv_violation = False

        before, sep, _ = str_IPythonActions.partition('/mnt/test_gt.csv')
        if sep:
            # keep everything before the CSV path
            metrics_new, outputs_new, idx_goal = get_metric_before_violation(
                before, outputs
            )
            if len(metrics_new) < len(metrics):
                metrics = metrics_new
            is_cheating = True
            ids_IPythonActions = len(
                [x['id'] for x in IPythonActions if x['id'] < idx_goal]
            )
        elif '/mnt/test_gt.csv' in str(AssistantShellActions):
            before = str(AssistantShellActions).split('/mnt/test_gt.csv')[0]
            metrics_new, outputs_new, idx_goal = get_metric_before_violation(
                before, outputs
            )
            is_cheating = False
            if len(metrics_new) < len(metrics):
                metrics = metrics_new
            ids_IPythonActions = len(
                [x['id'] for x in IPythonActions if x['id'] < idx_goal]
            )
        else:
            is_cheating = False
            # match = -1

        if outputs_new is not None:
            outputs = outputs_new

        if len(metrics) > 5 and max([x for x in metrics]) > 0.99:
            continue

        # instance = cfg['instance']
        # contraints = cfg['constraints']
        # llm_config = cfg['llm_config']
        # llm_hints = cfg['hints']
        res[str(folder)] = {}
        res[str(folder)]['metadata'] = metadata
        res[str(folder)]['metrics'] = []
        assert len(outputs) == 1, 'Multiple outputs found'
        # for key, output in outputs.items():

        # Chck in the folder how many pictures get generated
        Path(
            'evaluation/evaluation_outputs/outputs'
        ) / folder.name / 'trajectory_visualiser_folder'
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
            assert len(msgs) == len(costs), "msgs and costs length mismatch"
            cost_associated_with_score = []
            to_go_idx = 0

            if (
                llm_config == 'open_router_claude'
                or 'gpt-o3' in llm_config
                or 'gpt-4o' in llm_config
                or 'gpt-o4-mini' in llm_config
                or 'deepseek' in llm_config
                or 'llama' in llm_config
                or 'gemma' in llm_config
                or 'mistral' in llm_config
                or 'gemini' in llm_config
                or 'gpt-41' in llm_config
            ):
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

        if 'history' in outputs[0]:
            msgs = outputs[0]['history']
            costs = outputs[0]['metrics']['costs']
            cost_associated_with_scores = compute_cost_per_score(
                msgs, costs, scores[0], llm_config=cfg['llm_config']
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
            'error' in outputs[0]
            and outputs[0]['error']
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
        if 'error' in outputs[0]:
            df['error'] = outputs[0]['error']
        else:
            df['error'] = None
        df['use_max_budget'] = use_max_budget

        df['folder'] = folder.name
        df['is_sklearn_violation'] = is_sklearn_violation
        df['is_read_csv_violation'] = is_read_csv_violation
        df['is_cheating'] = is_cheating
        df['ids_IPythonActions'] = ids_IPythonActions
        df['is_stuck_in_a_loop'] = (
            'AgentStuckInLoopError: Agent got stuck in a loop' == df['error']
        )

        df['number_of_python_calls'] = len(IPythonActions)
        df['number_of_python_calls_before_first_submission'] = (
            api_call_before_first_submission
        )

        df['Acc100'] = df['metric'].apply(
            lambda x: 1 if len(x) > 0 and max(x) == 1 else 0
        )
        df['Acc99'] = df['metric'].apply(
            lambda x: 1 if len(x) > 0 and max(x) >= 0.99 else 0
        )

        df['number_of_submissions'] = number_of_submissions
        df['final_accumulated_cost'] = accumulated_cost

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
            if not np.isnan(x['first_submission'])
            and not np.isnan(x['last_submission'])
            else np.nan,
            axis=1,
        )
        # Basic sanity checks for per-run metrics
        assert df['Acc100'].isin([0, 1]).all()
        assert df['Acc99'].isin([0, 1]).all()
        assert df['number_of_submissions'].iloc[0] >= 0
        assert (
            np.isnan(df['final_accumulated_cost'].iloc[0])
            or df['final_accumulated_cost'].iloc[0] >= 0
        )
        assert df['diff'].dropna().between(-1, 1).all()
        assert df['metric'].apply(lambda xs: all(0 <= v <= 1 for v in xs)).all()

        df['folder_identifier'] = folder_identifier
        # if "channel_divergence" in df['instance'].iloc[0] and "41" in df['llm_config'].iloc[0]:
        #     breakpoint()
          # Rename channel_divergence to channel_corr_easy
        assert len(df) == 1, (
            f"df should have only one row, but has {len(df)} rows. df: {df}"
        )
        if df['instance'].iloc[0] == 'channel_divergence':
            df['instance'] = 'channel_corr_easy'

        if df['instance'].iloc[0] == 'channel_corr':
            df['instance'] = 'channel_corr_hard'


        # If identifier baseline_native_tool_calling and model is gpt-o4-mini or gpt-41 then set it to baseline
        if df['identifier_experiment'].iloc[0] == 'baseline_native_tool_calling' and df[
            'llm_config'
        ].iloc[0] in ['gpt-o4-mini', 'gpt-41']:
            df['identifier_experiment'] = 'baseline'

        if "open_router_gemini_pro_pro" in df['llm_config'].iloc[0]:
            df['llm_config'] = 'gemini_pro_pro'
        if "open_router_gemini_pro" in df['llm_config'].iloc[0]:
            df['llm_config'] = 'gemini_pro'


        # If the first row’s llm_config contains any of those, drop all matching rows
        matches = [cfg for cfg in MODELS_TO_DROP if cfg in df['llm_config'].iloc[0]]
        if matches:
            df = df.loc[~df['llm_config'].isin(matches)]

        assert (df['Acc100'] <= df['Acc99']).all(), ("The perfect accuracy should be less than or equal to the almost perfect accuracy")




        # if folder_identifier == 1293:
        #     breakpoint()

        entries_df.append(df)

    all_results = pd.concat(entries_df, ignore_index=True, axis=0)
    all_results['is_read_csv_banned'].fillna(False, inplace=True)
    all_results['OneShotAcc'] = 0
    all_results['OneShotAcc'] = all_results.apply(
        lambda x: 1
        if x['Acc100'] == 1 and x['number_of_submissions'] == 1
        else 0,
        axis=1,
    )
    all_results['is_time_series_task'] = all_results['instance'].apply(
        lambda x: 1 if x in time_series_datasets else 0
    )
    # Keep only entries in the relevant tasks
    # results_all = results_all.loc[results_all['instance'].isin(dataset)]
    all_results['number_of_submissions_when_right'] = all_results.apply(
        lambda x: x['number_of_submissions'] if x['Acc100'] == 1 else np.nan,
        axis=1,
    )

    # check when is the first submission made

    all_results['number_of_python_calls_when_right_at_first'] = all_results.apply(
        lambda x: x['number_of_python_calls']
        if x['Acc100'] == 1 and x['number_of_submissions'] == 1
        else np.nan,
        axis=1,
    )

    all_results['cost_when_right'] = all_results.apply(
        lambda x: x['final_accumulated_cost'] if x['Acc100'] == 1 else np.nan,
        axis=1,
    )
    # all_results.loc[all_results['llm_config'].isin(['gemini_pro_pro'])]

    all_results['best_metric'] = all_results['metric'].apply(
        lambda x: max(x) if len(x) > 0 else np.nan
    )
    # Global sanity checks on aggregated metrics
    assert all_results['Acc100'].between(0, 1).all()
    assert all_results['Acc99'].between(0, 1).all()
    assert all_results['OneShotAcc'].between(0, 1).all()
    assert all_results['number_of_submissions'].ge(0).all()
    assert all_results['final_accumulated_cost'].dropna().ge(0).all()
    assert all_results['cost_when_right'].dropna().ge(0).all()
    assert all_results['best_metric'].dropna().between(0, 1).all()
    # Drop all the errors STATUS$ERROR_LLM_SERVICE_UNAVAILABLE and  BadRequestError: litellm.BadRequestError: VertexAIException BadRequestError
    to_drop_errors = [
        'STATUS$ERROR_LLM_SERVICE_UNAVAILABLE',
        'BadRequestError: litellm.BadRequestError: VertexAIException BadRequestError - {\n  "error": {\n    "code": 400,\n    "message": "* GenerateContentRequest.model: unexpected model name format\\n",\n    "status": "INVALID_ARGUMENT"\n  }\n}\n',
    ]
    all_results = all_results.loc[~all_results['error'].isin(to_drop_errors)]


    all_results.drop(
        [
            'eval_output_dir',
            'eval_note',
            'agent_cls',
            'eval_num_workers',
            'number_of_experiments',
            'eval_n_limit',
            'trajectory_visualiser_folder',
            'class_type',
            'template_text',
            'enable_browsing_for_pictures',
            'solution_iterations',
            'constraints',
            'hints',
            'cheating_attempt',
            'warm_against_cheating',
            'prompt_variation',
            'show_solution_iterations',
            'show_max_budget_per_task',
            'keep_going_until_succeed',
            'include_constraints',
            'temperature',
            'top_p',
            'is_sklearn_banned',
        ],
        axis=1,
        inplace=True,
    )
    # Get also the runs with the idenfifier

    identifiers = [
        'baseline',
        'plot_disabled',
        # 'baseline_native_tool_calling',
        'constraint',
        'sanity_check',
    ]

    assert (all_results['llm_config'].isin(VALID_MODELS)).all(), (
        "The following models are not in the list of valid models: "
        f"{all_results['llm_config'][~all_results['llm_config'].isin(VALID_MODELS)].unique()}"
    )
    assertion_mask = (all_results['identifier_experiment'].isin(["baseline_native_tool_calling"])) & (
        all_results['llm_config'].isin(['gpt-o4-mini', 'gpt-41'])
    )

    assert not assertion_mask.any(), (
        "There are runs with identifier_experiment 'baseline_native_tool_calling' and llm_config 'gpt-o4-mini' or 'gpt-41'. They should be 'baseline'."
    )
    all_results_mask = (all_results['identifier_experiment'].isin(identifiers)) & (
        all_results['instance'].isin(TASKS)
    )
    paper_results = all_results.loc[
        all_results_mask
    ]
    # ]
    paper_results = paper_results[
        ['error', 'best_metric']
        + [col for col in paper_results.columns if col not in ['error', 'best_metric']]
    ]

    models = [
        'gpt-o4-mini',
        'gemini_pro',
        'gemini_pro_pro',
        'open_router_claude',
        'deepseek',
        'gpt-41',
    ]
    # Keep only max_budget_per_task == 1 for gpt-o4-mini, gemini_pro_pro and open_router_claude
    paper_results = paper_results.loc[
        ~(
            (paper_results['llm_config'] == 'gpt-o4-mini')
            & (paper_results['max_budget_per_task'] != 1)
        )
    ]
    paper_results = paper_results.loc[
        ~(
            (paper_results['llm_config'] == 'gemini_pro_pro')
            & (paper_results['max_budget_per_task'] != 1)
        )
    ]
    paper_results = paper_results.loc[
        ~(
            (paper_results['llm_config'] == 'open_router_claude')
            & (paper_results['max_budget_per_task'] != 1)
        )
    ]


    # paper_results = pd.concat([to_be_replaced, to_be_added], ignore_index=True)
    paper_results.loc[
        (paper_results['llm_config'] == 'gpt-41') & (paper_results['max_budget_per_task'] > 0.2)
    ]

    # Remove from baseline any run with is_plotting_enabled == False
    paper_results = paper_results.loc[
        ~(
            (paper_results['identifier_experiment'] == 'baseline')
            & (~paper_results['is_plotting_enabled'])
        )
    ]
    paper_results = paper_results.loc[
        ~(
            (paper_results['identifier_experiment'] == 'plot_disabled')
            & (paper_results['is_plotting_enabled'])
        )
    ]


    for identifier in identifiers:
        for model in models + ['llama']:
            for instance in TASKS:
                mask = (
                    (paper_results['llm_config'] == model)
                    & (paper_results['identifier_experiment'] == identifier)
                    & (paper_results['instance'] == instance)
                )
                other_runs = paper_results.loc[mask]
                # Compute perfect_accuracy_all_the_time
                is_always_perfect = other_runs['Acc100'].sum() == len(
                    other_runs
                )

                # Compute almost_perfect_accuracy_all_the_time
                is_always_almost_perfect = other_runs[
                    'Acc99'
                ].sum() == len(other_runs)
                is_anything_perfect = other_runs['Acc100'].sum() > 0
                is_anything_almost_perfect = (
                    other_runs['Acc99'].sum() > 0
                )
                is_anything_perfect_at_first = (
                    other_runs['OneShotAcc'].sum() > 0
                )
                # Check if there is at least one perfect accuracy
                # Add this back to the dataframe
                paper_results.loc[mask, 'perfect_accuracy_all_the_time'] = (
                    1 if is_always_perfect else 0
                )
                paper_results.loc[mask, 'almost_perfect_accuracy_all_the_time'] = (
                    1 if is_always_almost_perfect else 0
                )
                paper_results.loc[mask, 'is_anything_perfect'] = (
                    1 if is_anything_perfect else 0
                )
                paper_results.loc[mask, 'is_anything_almost_perfect'] = (
                    1 if is_anything_almost_perfect else 0
                )
                paper_results.loc[mask, 'is_anything_perfect_at_first'] = (
                    1 if is_anything_perfect_at_first else 0
                )

    df_to_assert = paper_results.fillna({
    "perfect_accuracy_all_the_time": 0,
    'Acc100': 1,          # or whatever makes sense
    "almost_perfect_accuracy_all_the_time": 0,
    'Acc99': 1
})

    assert (df_to_assert["perfect_accuracy_all_the_time"] <= df_to_assert['Acc100']).all(), (f"The always perfect accuracy should be less than or equal to the perfect accuracy, found: {df_to_assert[df_to_assert['perfect_accuracy_all_the_time'] > df_to_assert['Acc100']]}")
    assert (df_to_assert["almost_perfect_accuracy_all_the_time"] <= df_to_assert['Acc99']).all(), ("The always perfect accuracy should be less than or equal to the perfect accuracy")

    # other_runs.loc[other_runs['instance'].isin(['row_max_abs','spike_presence'])]
    if CHECKING_MISSING_ENTRIES:
        for scenario in identifiers:
            for model in models:
                if scenario == 'baseline' and model in [
                    'gpt-o4-mini',
                    'gpt-41',
                    'deepseek',
                ]:
                    continue
                # if scenario == "sanity_check":
                #     breakpoint()
                other_runs = paper_results.loc[
                    (paper_results['llm_config'] == model)
                    & (paper_results['identifier_experiment'] == scenario)
                ]

                # Sort by instance
                other_runs = other_runs.sort_values(by=['instance'])
                # SANITY CHECK TO SEE WHAT IS MISSING
                # assert len(other_runs['instance']) == len(other_runs['instance'].unique())

                # Each thing should appear at least 3 times
                missing = []
                for dataset in TASKS:
                    if dataset not in other_runs['instance'].unique():
                        missing.append(dataset)
                #     else:
                #         # Check if there are at least 3 entries
                #         if len(other_runs.loc[other_runs['instance'] == dataset]) < 3:
                #             missing.append(dataset)
                # # if model == "gpt-o4-mini":
                #     breakpoint()
                print(f'{scenario} Missing datasets for {model}: {",".join(missing)}')

    def group_table(
        df: pd.DataFrame,
        identifier_experiment: str,
        models: List[str],
        group_by_is_time_series_task: bool = True,
    ) -> pd.DataFrame:
        """
        Build the LaTeX table for Section “Agents are not able to explore effectively”.

        Mapping shown in the paper:
            gemini_pro      → Gemini Flash 2.5
            gemini_pro_pro  → Gemini Pro 2.5
            open_router_claude → Claude 3.7 Sonnet
            deepseek      → DeepSeek R1
            gpt-o4-mini   → O4-mini

        """
        # Sanity checks
        required_cols = {
            'identifier_experiment',
            'llm_config',
            'is_plotting_enabled',
            'is_time_series_task',
            'instance',
            'Acc100',
            'Acc99',
            'perfect_accuracy_all_the_time',
            'almost_perfect_accuracy_all_the_time',
            'number_of_submissions',
            'metric',
        }
        missing = required_cols - set(df.columns)
        assert not missing, f"group_table missing columns: {missing}"

        # Map the models to the names in the paper

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
            to_keep = min(
                [x for x in Counter(group_df['instance'].tolist()).values() if x > 0]
            )

            print(f'Keeping {to_keep} for {name} in {identifier_experiment}')
            # Downsample everything to that
            group_df = group_df.groupby('instance').apply(
                lambda x, keep=to_keep: x.sample(keep, replace=False)
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
                    'llm_config': NAMING_MAP[llm_name],
                    'mean_perfect_accuracy': group_df['Acc100']
                    .fillna(0)
                    .mean(),
                    'mean_Acc99': group_df[
                        'Acc99'
                    ].mean(),
                    'mean_perfect_accuracy_all_the_time': group_df[
                        'perfect_accuracy_all_the_time'
                    ].mean(),
                    'mean_Acc99_all_the_time': group_df[
                        'almost_perfect_accuracy_all_the_time'
                    ].mean(),
                    'mean_anything_at_first_perfect': group_df[
                        'Acc100'
                    ].mean(),
                    'mean_anything_perfect': group_df['is_anything_perfect'].mean(),
                    'mean_anything_almost_perfect': group_df[
                        'is_anything_almost_perfect'
                    ].mean(),
                    'variance_perfect_accuracy': group_df['Acc100'].var(),
                    'variance_almost_perfect_accuracy': group_df[
                        'Acc99'
                    ].var(),
                    'variance_perfect_accuracy_all_the_time': group_df[
                        'perfect_accuracy_all_the_time'
                    ].var(),
                    'variance_almost_perfect_accuracy_all_the_time': group_df[
                        'almost_perfect_accuracy_all_the_time'
                    ].var(),
                    'number_of_submissions': group_df['number_of_submissions'].mean(),
                    'count': number_of_rows,
                    'metrics': group_df['metric'].tolist(),
                    'OneShotAcc': group_df[
                        'OneShotAcc'
                    ].mean(),
                    # 'number_of_api_calls_when_right': group_df['number_of_api_calls_when_right'].mean(),
                    'cost_when_right': group_df['cost_when_right'].mean(),
                    'number_of_submissions_when_right': group_df[
                        'number_of_submissions_when_right'
                    ].mean(),
                    'metric_difference': group_df['diff'].mean(),
                    'number_of_python_calls_when_right_at_first': group_df[
                        'number_of_python_calls_when_right_at_first'
                    ].mean(),
                    'number_of_python_calls_before_first_submission': group_df[
                        'number_of_python_calls_before_first_submission'
                    ].mean(),
                    'average_number_of_python_calls_before_violation': group_df[
                        'ids_IPythonActions'
                    ]
                    .dropna()
                    .mean(),
                    'average_number_sklearn_violation': group_df[
                        'is_sklearn_violation'
                    ].mean(),
                    'average_number_read_csv_violation': group_df[
                        'is_read_csv_violation'
                    ].mean(),
                    'average_number_cheating': group_df['is_cheating'].mean(),
                    # group_df[['ids_IPythonActions','is_sklearn_violation','is_read_csv_violation','is_cheating']].dropna().mean(),
                }
            )
            # if name[1] == 'open_router_claude':
            #     raise ValueError('Issue')
            #     breakpoint()
        # -------------------------------------------------------------------------
        table1_df = pd.DataFrame(table1_list)
        # Ensure aggregated metrics are within expected bounds
        assert table1_df['mean_perfect_accuracy'].between(0, 1).all()
        assert table1_df['mean_Acc99'].between(0, 1).all()
        assert table1_df['mean_anything_perfect'].between(0, 1).all()
        assert table1_df['variance_perfect_accuracy'].ge(0).all()
        assert table1_df['variance_almost_perfect_accuracy'].ge(0).all()
        assert (
            table1_df['mean_perfect_accuracy_all_the_time']
            <= table1_df['mean_perfect_accuracy']
        ).all()
        assert (
            table1_df['mean_Acc99_all_the_time']
            <= table1_df['mean_Acc99']
        ).all()
        return table1_df

    def table_2(
        df,
        models: List[str],
        identifier_considered: Optional[List[str]] = None,
    ):
        """ """
        if identifier_considered is None:
            identifier_considered = ['baseline', 'plot_disabled']
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
        # res_df = pd.DataFrame(res)

    # A run is invalid if it's marked as cheating **and** also marked as perfect or almost-perfect accuracy
    invalid = paper_results['is_cheating'] & (paper_results['Acc100'])
    assert (
        not invalid.any()
    ), 'Some cheating runs are incorrectly marked as perfect or almost-perfect accuracy'

    # ----------------------------------------------------------------------
    # Example usage

    def df_to_overleaf_table(
        df: pd.DataFrame,
        value_col: str = "OneShotAcc_mean",
        caption: str | None = None,
        label: str | None = None,
        fmt: str = "%.3f",
    ) -> str:
        """
        Pivot `df` so rows are `instance`, columns are `llm_config`,
        and cells contain `value_col`.  Returns booktabs-ready LaTeX.

        Parameters
        ----------
        df : DataFrame shaped like
            ['llm_config', 'instance', value_col, ...]
        value_col : str
            Metric to show in each cell (e.g. 'OneShotAcc_mean')
        caption : str | None
            Optional \\caption{...} text
        label : str | None
            Optional \\label{...} tag
        fmt : str
            C-style numeric format, e.g. '%.2f'
        """
        assert value_col in df.columns, f"{value_col} column missing"

        # Replace llm_config with the name in the paper
        df['llm_config'] = df['llm_config'].replace(NAMING_MAP)
        # Rename "llm_config" to "Model"
        df.rename(columns={"llm_config": "Model"}, inplace=True)
        # Replace _ with \_ in the instance names
        df['instance']  = df['instance'].apply(
            lambda x: x.replace('_', '\\_')
        )
        # Remap Claude 3.7 Sonnet to Sonnet 3.7
        df['Model'] = df['Model'].replace(
            {'Claude 3.7 Sonnet': 'Claude 3.7 Sonnet'}
        )

        # Remap DeepSeek R1 to R1
        df['Model'] = df['Model'].replace(
            {'DeepSeek R1': 'R1'}
        )


        tbl = (
            df
            .pivot(index="instance", columns="Model", values=value_col)
            .sort_index()
            .loc[:, sorted(df["Model"].unique())]           # ordered columns
        )

        # --- render as LaTeX ---
        latex = tbl.to_latex(
            na_rep="-",
            escape=False,         # keep underscores in model names
            bold_rows=False,
            column_format="l" + "c" * tbl.shape[1],  # 1 left + N centered cols
            float_format=lambda x: fmt % x,
            caption=caption,
            label=label,
        )
        return latex

    def group_by_model_and_instance(
        df: pd.DataFrame,
        models: Optional[List[str]] = None,
        identifier_experiment: Optional[str] = None,
        population_var: bool = True,        # ddof = 0 instead of 1 → avoids NaN for n=1
        max_seeds: Optional[int] = None,
        ) -> pd.DataFrame:
        """
        Group the DataFrame by model and instance, and compute the mean and variance (Appendix A.1).
        """
        required_cols = {"llm_config", "instance", "OneShotAcc", "Acc100", "Acc99", "seed"}
        missing = required_cols - set(df.columns)
        assert not missing, f"group_by_model_and_instance missing columns: {missing}"
        # --- filtering ----------------------------------------------------------
        if identifier_experiment is not None:
            df = df.loc[df["identifier_experiment"] == identifier_experiment]

        if models is not None:
            df = df.loc[df["llm_config"].isin(models)]

        if max_seeds is not None:
            # For each llm_config and instance, keep only the last max_seeds rows
            df = (
                df.groupby(["llm_config", "instance"])
                .apply(lambda x: x.tail(max_seeds))
                .reset_index(drop=True)
            )


        # --- aggregation helpers -----------------------------------------------
        ddof = 0 if population_var else 1              # choose population or sample var
        var = lambda s: s.var(ddof=ddof)               # noqa: E731  (tiny helper)

        grouped = (
            df.groupby(["llm_config", "instance"])
            .agg(
                OneShotAcc_mean = ("OneShotAcc", "mean"),
                Acc100_mean     = ("Acc100",    "mean"),
                Acc99_mean      = ("Acc99",     "mean"),
                n               = ("seed",      "count"),
            )
            .reset_index()
        )
        assert grouped[["OneShotAcc_mean", "Acc100_mean", "Acc99_mean"]].apply(
            lambda s: s.between(0, 1).all()
        ).all()

        latex_code = df_to_overleaf_table(
            grouped,
            value_col="OneShotAcc_mean",
            caption="Instance-level accuracy (one-shot)",
            label="tab:instance_accuracy",
            fmt="%.2f",
        )

        return grouped, latex_code

    models = [x for x in models if x != 'deepseek']
    entry_with, _ = group_by_model_and_instance(
        paper_results,
        models=models,
        identifier_experiment='baseline',
        max_seeds=3,
    )

    entry_without, _ = group_by_model_and_instance(
        paper_results,
        models=models,
        identifier_experiment='plot_disabled',
        max_seeds=3,
    )

    # Round everything to 2 decimal places
    entry_with = entry_with.round(2)
    entry_without = entry_without.round(2)
    latex_table_final_oneshot = make_overleaf_table(entry_with=entry_with, entry_without=entry_without, metric="OneShotAcc_mean")
    latex_table_final_acc100 = make_overleaf_table(entry_with=entry_with, entry_without=entry_without, metric="Acc100_mean")
    latex_table_final_acc99 = make_overleaf_table(entry_with=entry_with, entry_without=entry_without, metric="Acc99_mean")

    deep_seek_output = paper_results.loc[paper_results['llm_config'] == 'deepseek']

    entry = group_table(
        paper_results,
        identifier_experiment='baseline',
        models=[x for x in models if x != 'deepseek'],
    )
    filled_1 = create_accuracy(
        entry,
        important_columns=[
            'OneShotAcc',
            'mean_perfect_accuracy',
            'mean_Acc99',
        ],
    )

    entry_plots = group_table(
        paper_results,
        identifier_experiment='plot_disabled',
        models=[x for x in models if x != 'deepseek'],
    )
    filled_plots = create_accuracy(
        entry_plots,
        important_columns=[
            'OneShotAcc',
            'mean_perfect_accuracy',
            'mean_Acc99',
        ],
    )
    print(filled_plots)
    breakpoint()
    latex_code = fill_costs_pycalls_table(entry)  # entry = your DataFrame
    print(latex_code)
    filled_2 = create_accuracy(
        entry,
        important_columns=[
            'average_number_of_python_calls_before_violation',
            'average_number_sklearn_violation',
            'average_number_cheating',
        ],
        show_percent=False,
    )
    entry_2 = group_table(
        paper_results,
        group_by_is_time_series_task=False,
        identifier_experiment='baseline',
        models=[x for x in models if x != 'deepseek'],
    )
    # Keep omly experimental_runs with identifier_experiment == 'plot_disabled'

    # paper_results.loc[paper_results['identifier_experiment'] == 'plot_disabled']
    entry_plots = group_table(
        paper_results,
        identifier_experiment='plot_disabled',
        models=[x for x in models if x != 'deepseek'],
    )
    # Add an empty line for GPT-4.1
    # new_line = pd.DataFrame(
    #     {
    #         'is_plotting_enabled': False,
    #         'is_time_series_task': False,
    #         'llm_config': 'GPT-4.1',
    #         'mean_perfect_accuracy': 0,
    #         'mean_Acc99': 0,
    #         'mean_perfect_accuracy_all_the_time': 0,
    #         'mean_Acc99_all_the_time': 0,
    #         'OneShotAcc': 0,
    #         'number_of_submissions_when_right': 0,
    #         'cost_when_right': 0,
    #     },
    #     index=[0],
    # )
    # entry_plots = pd.concat([entry_plots, new_line], ignore_index=True)
    filled_1 = create_accuracy(
        entry_plots,
        important_columns=[
            'OneShotAcc',
            'mean_perfect_accuracy',
            'mean_Acc99',
        ],
    )
    print(filled_1)
    # 'mean_anything_at_first_perfect': group_df['Acc100'].mean(),
    # 'mean_anything_perfect': group_df['is_anything_perfect'].mean(),
    # 'mean_anything_almost_perfect':
    # Average together LLMs with the same name
    percentage_correct_at_first = create_accuracy(
        entry,
        important_columns=[
            'OneShotAcc',
            'number_of_python_calls_when_right_at_first',
        ],
    )
    print(percentage_correct_at_first)

    filled = create_accuracy(
        entry,
        important_columns=[
            'OneShotAcc',
            'mean_Acc99_all_the_time',
        ],
    )

    filled_2 = create_accuracy(
        entry,
        important_columns=[
            'mean_perfect_accuracy_all_the_time',
            'mean_Acc99_all_the_time',
        ],
    )
    print(filled_2)
    # entry_3 = group_table(
    #     paper_results, identifier_experiment='constraint', models=models
    # )
    # filled = create_accuracy(
    #     entry_3,
    #     important_columns=['mean_perfect_accuracy', 'mean_Acc99'],
    # )

    entry_2 = group_table(
        paper_results, identifier_experiment='plot_disabled', models=models
    )
    filled = create_accuracy(
        entry_2,
        important_columns=['mean_perfect_accuracy', 'mean_Acc99'],
    )

    print(filled)

    # entry_3 = table_1(experimental_runs, models=models)
    # print(entry)
    # who_is_better_2 = experimental_runs.loc[experimental_runs['identifier_experiment'].isin([ 'plot_disabled'])]

    # Just keep gemini_pro_pro and open_router_claude
    table_2(
        paper_results,
        models=models,
        identifier_considered=['baseline', 'plot_disabled'],
    )

    # Create WIN / TIE / LOSE table
    # def build_win_tie_loss(
    #     df: pd.DataFrame,
    #     experiments: Tuple[str, str] = ('baseline', 'plot_disabled'),
    #     models: Iterable[str] = ('gemini_pro_pro', 'open_router_claude', 'gemini_pro'),
    # ) -> pd.DataFrame:
    #     """
    #     Count WIN / TIE / LOSE outcomes between two experiment variants
    #     (e.g. 'baseline' vs. 'constraint') for each model and task type.

    #     Parameters
    #     ----------
    #     df : pd.DataFrame
    #         Must contain at least the columns
    #         ['instance', 'is_time_series_task', 'identifier_experiment',
    #         'Acc99', 'llm_config'].
    #     experiments : tuple(str, str), default ('baseline', 'plot_disabled')
    #         The two experiment labels to compare. Order matters:
    #         *experiments[0]* is the first column in the output.
    #     models : iterable of str
    #         Which models to include (rows are replicated for every model
    #         so that downstream code can aggregate by model easily).

    #     Returns
    #     -------
    #     pd.DataFrame
    #         Columns: ['model', 'task_type', experiments[0], experiments[1], 'tie']
    #         Each row counts how many tasks the first experiment “won”, the
    #         second experiment “won”, or resulted in a tie, broken down by
    #         (model, task_type).
    #     """
    #     exp_a, exp_b = experiments
    #     required_cols = {
    #         'instance',
    #         'is_time_series_task',
    #         'identifier_experiment',
    #         'Acc99',
    #         'llm_config',
    #     }
    #     missing = required_cols - set(df.columns)
    #     if missing:
    #         raise KeyError(f'Missing required columns: {missing}')

    #     # -------------------------------------------------------------
    #     # 1. Decide the winner for every task once
    #     # -------------------------------------------------------------
    #     per_task = []
    #     grouped = df[df['identifier_experiment'].isin(experiments)].groupby('instance')
    #     for task_name, group in grouped:
    #         is_ts = group['is_time_series_task'].iloc[0]

    #         # Fetch accuracies (should be exactly one row per experiment label)
    #         try:
    #             acc_a = group.loc[
    #                 group['identifier_experiment'] == exp_a, 'Acc99'
    #             ].values[0]
    #             acc_b = group.loc[
    #                 group['identifier_experiment'] == exp_b, 'Acc99'
    #             ].values[0]
    #         except IndexError:
    #             # One of the variants missing for this task → skip
    #             continue

    #         if acc_a > acc_b:
    #             outcome = exp_a
    #         elif acc_a < acc_b:
    #             outcome = exp_b
    #         else:
    #             outcome = 'tie'

    #         for mdl in models:
    #             per_task.append((outcome, is_ts, task_name, mdl))

    #     task_df = pd.DataFrame(
    #         per_task,
    #         columns=['result', 'is_time_series_task', 'task_name', 'model'],
    #     )

    #     # -------------------------------------------------------------
    #     # 2. Aggregate per (model, task_type)
    #     # -------------------------------------------------------------
    #     summary_rows = []
    #     for mdl in models:
    #         for task_type in task_df['is_time_series_task'].unique():
    #             subset = task_df[
    #                 (task_df['model'] == mdl)
    #                 & (task_df['is_time_series_task'] == task_type)
    #             ]
    #             counts = subset['result'].value_counts()
    #             summary_rows.append(
    #                 {
    #                     'model': mdl,
    #                     'task_type': task_type,
    #                     exp_a: counts.get(exp_a, 0),
    #                     exp_b: counts.get(exp_b, 0),
    #                     'tie': counts.get('tie', 0),
    #                 }
    #             )

    #     return pd.DataFrame(summary_rows)

    # model_results_df = build_win_tie_loss(
    #     experimental_runs, experiments=('baseline', 'plot_disabled')
    # )
    # model_results_df_2 = build_win_tie_loss(
    #     experimental_runs, experiments=('baseline', 'constraint')
    # )
    # model_results_df_3 = build_win_tie_loss(
    #     experimental_runs, experiments=('baseline', 'baseline_native_tool_calling')
    # )

    # # Fill the table

    # # Compare baseline vs constraint and do is better comparison
    # who_is_better = experimental_runs.loc[
    #     experimental_runs['identifier_experiment'].isin(['baseline', 'constraint'])
    # ]

    # Group by model and instance
    # who_is_better_group = who_is_better.groupby(['llm_config', 'instance'])

    # for name, group_df in who_is_better_group:
    #     breakpoint()

    # print(entry)

    # results_all['best_metric'] = results_all['metric'].apply(
    #     lambda x: max(x) if len(x) > 0 else np.nan
    # )
    # results_005 = results_all.loc[results_all['max_budget_per_task'] == 0.05]
    # # results_05 = results_all.loc[results_all['max_budget_per_task'] == 0.50]
    # results_005 = results_all.loc[results_all['max_budget_per_task'] == 0.05]

    # # Keep only the first score for each task
    # results_005['first_score'] = results_005['metric'].apply(
    #     lambda x: x[0] if len(x) > 0 else np.nan
    # )
    # results_005['max_score'] = results_005['metric'].apply(
    #     lambda x: max(x) if len(x) > 0 else np.nan
    # )
    # results_005['first_cost'] = results_005['cost_associated_with_scores'].apply(
    #     lambda x: x[0] if (len(x) > 0 and min(x) < 0.005) else np.nan
    # )

    # # Keep the ones which are not Nan with first cost tilde
    # results_005_to_group = results_005.loc[~results_005['first_cost'].isna()]

    # # Group by task and get max of the first score and sum of the first cost
    # results_005_to_group = (
    #     results_005_to_group.groupby('instance')
    #     .agg(
    #         first_score=('first_score', 'max'),
    #         first_cost=('first_cost', 'sum'),
    #         number_of_submissions=('instance', 'size'),
    #     )
    #     .reset_index()
    # )

    # # Group by task and get mean of the max_score and mean of the first cost
    # results_005_all = (
    #     results_005.groupby('instance')
    #     .agg(
    #         max_score=('max_score', 'mean'),
    #         first_cost=('final_accumulated_cost', 'mean'),
    #         number_of_submissions=('number_of_submissions', 'mean'),
    #     )
    #     .reset_index()
    # )
    # print(results_005_all)


if __name__ == '__main__':
    main()
