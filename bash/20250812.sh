instance=cofounded_group_outlier

# channel_corr_easy
# cofounded_group_outlier
# periodic_presence

models=gemini-pro

python -m evaluation.benchmarks.data_science_bench.run_infer \
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
  replay_trajectory_path="" \
  restore_trajectory_path="evaluation/evaluation_outputs/outputs/2025-08-08/2025-08-08_17-10-47_0/cofounded_group_outlier/DataScienceBenchAgent/gemini-2.5-pro_maxiter_30_N_temp/output.jsonl" \

# evaluation/evaluation_outputs/outputs/2025-08-08/2025-08-08_17-10-47_0/cofounded_group_outlier/DataScienceBenchAgent/gemini-2.5-pro_maxiter_30_N_temp/output.jsonl

# TODO: Check weird behaviour when replay_trajectory_path is not empty. Does not go into breakpoints of step() function.
