instance=spike_presence,sum_threshold,zero_crossing

# 5*19*4 USD
models=gemini_pro_pro

# BASELINE


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
#   is_read_csv_banned=False \
#   identifier_experiment="baseline" \
#   -m




# # PLOT DISABLED
python evaluation/benchmarks/error_bench/run_infer.py \
  number_of_experiments=1 \
  eval_n_limit=1 \
  class_type=explorative_data_analysis \
  instance=$instance \
  constraints=0 \
  llm_config=$models \
  solution_iterations=5 \
  cheating_attempt=False \
  warm_against_cheating=False \
  max_budget_per_task=1 \
  prompt_variation=0 \
  seed=20 \
  keep_going_until_succeed=True \
  native_tool_calling=False \
  is_plotting_enabled=False \
  give_structure_hint=False \
  disable_numbers=False \
  identifier_experiment="plot_disabled" \
  -m

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
