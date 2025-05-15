instance=channel_corr,channel_divergence,cofounded_group_outlier,common_frequency,dominant_feature,find_peaks,ground_mean_threashold,outlier_ratio,periodic_presence,predict_ts_stationarity,row_max_abs,row_variance,set_points,sign_rotated_generator,simultanus_spike,spike_presence,sum_threshold,variance_burst,zero_crossing

# 5*19*4 USD
models=gemma-3-27b-it #,gemini_pro,open_router_claude #,deepseek,gpt-o4-mini

# BASELINE

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
  is_read_csv_banned=False \
  identifier_experiment="plot_disabled" \
  -m



# python evaluation/benchmarks/error_bench/run_infer.py \
#   number_of_experiments=1 \
#   eval_n_limit=1 \
#   class_type=explorative_data_analysis \
#   instance=channel_corr \
#   constraints=0 \
#   llm_config=llama \
#   solution_iterations=5 \
#   cheating_attempt=False \
#   warm_against_cheating=False \
#   max_budget_per_task=1 \
#   prompt_variation=0 \
#   seed=20 \
#   keep_going_until_succeed=True \
#   native_tool_calling=True \
#   is_plotting_enabled=True \
#   give_structure_hint=False \
#   disable_numbers=False \
#   is_read_csv_banned=False \
#   identifier_experiment="baseline" \
#   -m



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
#   native_tool_calling=True \
#   is_plotting_enabled=True \
#   give_structure_hint=False \
#   disable_numbers=False \
#   is_read_csv_banned=False \
#   identifier_experiment="baseline_native_tool_calling" \
#   -m




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

