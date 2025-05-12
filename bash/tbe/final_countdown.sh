instance=row_variance #,simultanus_spike,channel_corr,outlier_ratio,channel_divergence,variance_burst,sign_rotated_generator,ground_mean_threashold,dominant_feature,set_points,common_frequency,predict_ts_stationarity,zero_crossing,sum_threshold,spike_presence,cofounded_group_outlier,find_peaks,row_max_abs,periodic_presence


#!/usr/bin/env bash
models=("gemini_pro_pro" ) # "gemini_lite" "open_router_claude" "deepseek")

# BASELINE
for model in "${models[@]}"; do
  # Pick the budget: 1 for the pro models, 0.05 for the lite model
  if [[ "$model" == *lite* ]]; then
    budget=0.5
  else
    budget=1
  fi

  python evaluation/benchmarks/error_bench/run_infer.py \
    number_of_experiments=1 \
    eval_n_limit=1 \
    class_type=explorative_data_analysis \
    instance="$instance" \
    constraints=0 \
    hints=0 \
    llm_config="$model" \
    solution_iterations=5 \
    cheating_attempt=False \
    warm_against_cheating=False \
    max_budget_per_task="$budget" \
    prompt_variation=0 \
    seed=20 \
    keep_going_until_succeed=True \
    native_tool_calling=False \
    is_plotting_enabled=True \
    give_structure_hint=True \
    disable_numbers=False \
    is_read_csv_banned=False \
    identifier_experiment="baseline" \
    -m
done


# PLOT DISABLED
for model in "${models[@]}"; do
  # Pick the budget: 1 for the pro models, 0.05 for the lite model
  if [[ "$model" == *lite* ]]; then
    budget=0.5
  else
    budget=1
  fi

  python evaluation/benchmarks/error_bench/run_infer.py \
    number_of_experiments=1 \
    eval_n_limit=1 \
    class_type=explorative_data_analysis \
    instance="$instance" \
    constraints=0 \
    hints=0 \
    llm_config="$model" \
    solution_iterations=5 \
    cheating_attempt=False \
    warm_against_cheating=False \
    max_budget_per_task="$budget" \
    prompt_variation=0 \
    seed=20 \
    keep_going_until_succeed=True \
    native_tool_calling=False \
    is_plotting_enabled=False \
    give_structure_hint=True \
    disable_numbers=False \
    identifier_experiment="plot_disabled" \
    -m
done

# HINT VS NO HINT
for model in "${models[@]}"; do
  # Pick the budget: 1 for the pro models, 0.05 for the lite model
  if [[ "$model" == *lite* ]]; then
    budget=0.5
  else
    budget=1
  fi

  python evaluation/benchmarks/error_bench/run_infer.py \
    number_of_experiments=1 \
    eval_n_limit=1 \
    class_type=explorative_data_analysis \
    instance="$instance" \
    constraints=0 \
    hints=0 \
    llm_config="$model" \
    solution_iterations=5 \
    cheating_attempt=False \
    warm_against_cheating=False \
    max_budget_per_task="$budget" \
    prompt_variation=0 \
    seed=20 \
    keep_going_until_succeed=True \
    native_tool_calling=False \
    is_plotting_enabled=False \
    give_structure_hint=True \
    disable_numbers=False \
    identifier_experiment="plot_disabled" \
    -m
done