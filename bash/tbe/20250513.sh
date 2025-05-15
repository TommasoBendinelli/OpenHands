instance=row_variance,simultanus_spike,channel_corr,outlier_ratio,channel_divergence,variance_burst,sign_rotated_generator,ground_mean_threashold,dominant_feature,set_points,common_frequency,predict_ts_stationarity,zero_crossing,sum_threshold,spike_presence,cofounded_group_outlier,find_peaks,row_max_abs,periodic_presence


# 5*19*4 USD
models=gemini_pro_pro,gemini_pro,open_router_claude,deepseek,gpt-o4-mini

# BASELINE


python evaluation/benchmarks/error_bench/run_infer.py \
  number_of_experiments=1 \
  eval_n_limit=1 \
  class_type=explorative_data_analysis \
  instance=simultanus_spike \
  constraints=0 \
  llm_config=deepseek \
  solution_iterations=5 \
  cheating_attempt=False \
  warm_against_cheating=False \
  max_budget_per_task=1 \
  prompt_variation=0 \
  seed=20 \
  keep_going_until_succeed=True \
  native_tool_calling=False \
  is_plotting_enabled=True \
  give_structure_hint=False \
  disable_numbers=False \
  is_read_csv_banned=False \
  identifier_experiment="prototype" \
