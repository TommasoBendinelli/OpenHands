#!/usr/bin/env bash
# Exit immediately if a command fails and treat unset vars as errors

# instance="row_variance"
# models=


python -m pdb evaluation/benchmarks/error_bench/run_infer.py \
  number_of_experiments=1 \
  eval_n_limit=1 \
  class_type=explorative_data_analysis \
  instance="row_variance" \
  constraints=0 \
  llm_config="gemini_pro" \
  solution_iterations=5 \
  cheating_attempt=False \
  warm_against_cheating=False \
  max_budget_per_task=0.2 \
  prompt_variation=0 \
  seed=20 \
  keep_going_until_succeed=True \
  native_tool_calling=False \
  is_plotting_enabled=True \
  give_structure_hint=False \
  disable_numbers=False \
  is_read_csv_banned=False \
  identifier_experiment="baseline"

# Repeat 5 times
for i in {1..5}; do
  python evaluation/benchmarks/error_bench/run_infer.py \
    number_of_experiments=1 \
    eval_n_limit=1 \
    class_type=explorative_data_analysis \
    instance="$instance" \
    constraints=0 \
    llm_config="$models" \
    solution_iterations=5 \
    cheating_attempt=False \
    warm_against_cheating=False \
    max_budget_per_task=0.2 \
    prompt_variation=0 \
    seed=20 \
    keep_going_until_succeed=True \
    native_tool_calling=False \
    is_plotting_enabled=True,False \
    give_structure_hint=False \
    disable_numbers=False \
    is_read_csv_banned=False \
    identifier_experiment="baseline" \
    -m
done
#   number_of_experiments=1 \
#   eval_n_limit=1 \
#   class_type=explorative_data_analysis \
#   instance=channel_corr \
#   constraints=0 \
#   llm_config=gpt-o4-mini \
#   solution_iterations=5 \
#   cheating_attempt=False \
#   warm_against_cheating=False \
#   max_budget_per_task=0.5 \
#   prompt_variation=0 \
#   seed=20 \
#   keep_going_until_succeed=True \
#   native_tool_calling=False \
#   is_plotting_enabled=True \
#   give_structure_hint=False \
#   disable_numbers=True \
#   is_read_csv_banned=False \
#   identifier_experiment="baseline"


# # PLOT DISABLED
# python evaluation/benchmarks/error_bench/run_infer.py \
#   number_of_experiments=1 \
#   eval_n_limit=1 \
#   class_type=explorative_data_analysis \
#   instance=$instance \
#   constraints=0 \
#   llm_config=$models \
#   solution_iterations=5 \
#   cheating_attempt=False \
#   warm_against_cheating=False \
#   max_budget_per_task=1 \
#   prompt_variation=0 \
#   seed=20 \
#   keep_going_until_succeed=True \
#   native_tool_calling=False \
#   is_plotting_enabled=False \
#   give_structure_hint=False \
#   disable_numbers=False \
#   identifier_experiment="plot_disabled" \
#   -m

# # HINT VS NO HINT
# python evaluation/benchmarks/error_bench/run_infer.py \
#   number_of_experiments=1 \
#   eval_n_limit=1 \
#   class_type=explorative_data_analysis \
#   instance=$instance \
#   constraints=0 \
#   llm_config=$models \
#   solution_iterations=5 \
#   cheating_attempt=False \
#   warm_against_cheating=False \
#   max_budget_per_task=1 \
#   prompt_variation=0 \
#   seed=20 \
#   keep_going_until_succeed=True \
#   native_tool_calling=False \
#   is_plotting_enabled=True \
#   give_structure_hint=True \
#   disable_numbers=False \
#   identifier_experiment="hint_activated" \
#   -m

# # CONTRAINTS VS NO CONSTRAINTS

# python evaluation/benchmarks/error_bench/run_infer.py \
#   number_of_experiments=1 \
#   eval_n_limit=1 \
#   class_type=explorative_data_analysis \
#   instance=$instance \
#   constraints=0 \
#   llm_config=$models \
#   solution_iterations=5 \
#   cheating_attempt=False \
#   warm_against_cheating=False \
#   max_budget_per_task=1 \
#   prompt_variation=0 \
#   seed=20 \
#   keep_going_until_succeed=True \
#   native_tool_calling=False \
#   is_plotting_enabled=True \
#   give_structure_hint=False \
#   disable_numbers=False \
#   is_read_csv_banned=True \
#   identifier_experiment="constraint" \
#   -m
