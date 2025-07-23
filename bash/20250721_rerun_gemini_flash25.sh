instance=channel_corr_easy,channel_corr_hard,cofounded_group_outlier,common_frequency,dominant_feature,find_peaks,ground_mean_threashold,outlier_ratio,periodic_presence,predict_ts_stationarity,row_max_abs,sign_rotated_generator,simultanus_spike,sum_threshold,variance_burst,zero_crossing

models=gemini-flash-preview-05-20

## 4.2.1 Baseline Results

python evaluation/benchmarks/data_science_bench/run_infer.py \
  number_of_experiments=1 \
  eval_n_limit=1 \
  class_type=explorative_data_analysis \
  instance=$instance \
  constraints=0 \
  llm_config=$models \
  feedback_iterations=5 \
  cheating_attempt=False \
  warm_against_cheating=False \
  max_budget_per_task=1 \
  prompt_variation=0 \
  keep_going_until_succeed=True \
  native_tool_calling=False \
  is_plotting_enabled=True \
  give_structure_hint=False \
  disable_numbers=False \
  is_read_csv_banned=False \
  identifier_experiment="baseline" \
  -m

## 4.4 Impact of Plot Generation on Performance

python evaluation/benchmarks/data_science_bench/run_infer.py \
  number_of_experiments=1 \
  eval_n_limit=1 \
  class_type=explorative_data_analysis \
  instance=$instance \
  constraints=0 \
  llm_config=$models \
  feedback_iterations=5 \
  cheating_attempt=False \
  warm_against_cheating=False \
  max_budget_per_task=1 \
  prompt_variation=0 \
  keep_going_until_succeed=True \
  native_tool_calling=False \
  is_plotting_enabled=False \
  give_structure_hint=False \
  disable_numbers=False \
  identifier_experiment="plot_disabled" \
  -m
