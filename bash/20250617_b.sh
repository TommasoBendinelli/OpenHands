#!/usr/bin/env bash
set -euo pipefail

instance=channel_corr_easy
models=gemini-flash-preview-05-20

# Either export once…
export DEBUG=0          # remove this line if you prefer the 'env' variant below

cmd=(
  python -m evaluation.benchmarks.data_science_bench.run_infer
  number_of_experiments=1
  eval_n_limit=1
  class_type=explorative_data_analysis
  instance="$instance"
  constraints=0
  llm_config="$models"
  feedback_iterations=5
  cheating_attempt=False
  warm_against_cheating=False
  max_budget_per_task=1
  prompt_variation=0
  seed=20
  keep_going_until_succeed=True
  native_tool_calling=False
  is_plotting_enabled=True
  give_structure_hint=False
  disable_numbers=False
  is_read_csv_banned=False
  identifier_experiment="debugging"
)

echo "${cmd[@]}"   # prints a single tidy line
"${cmd[@]}"        # executes it
