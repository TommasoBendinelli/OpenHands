#!/usr/bin/env bash

# list of all instances
instances=(
variance_burst
  common_frequency
  dominant_feature
  find_peaks
  ground_mean_threashold
  outlier_ratio
  periodic_presence
  predict_ts_stationarity
  row_max_abs
  row_variance
  set_points
  sign_rotated_generator
  simultanus_spike
  spike_presence
  sum_threshold
  variance_burst
  zero_crossing
)

models=gemini_pro

# for instance in "${instances[@]}"; do
#   echo "Running experiment for: $instance"
python evaluation/benchmarks/error_bench/run_infer.py \
  number_of_experiments=1 \
  eval_n_limit=1 \
  class_type=explorative_data_analysis \
  instance="$instances" \
  constraints=0 \
  llm_config="$models" \
  solution_iterations=5 \
  cheating_attempt=False \
  warm_against_cheating=False \
  max_budget_per_task=0.5 \
  prompt_variation=0 \
  seed=20 \
  keep_going_until_succeed=True \
  native_tool_calling=False \
  is_plotting_enabled=False \
  give_structure_hint=False \
  disable_numbers=False \
  is_read_csv_banned=False \
  identifier_experiment="plot_disabled"  -m
