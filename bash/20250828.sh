instance=channel_corr_easy

models=gemini-pro #open_router_claude

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
  max_iterations=30 \
  check_password=true \
  replay_trajectory_path="evaluation/evaluation_outputs/outputs/2025-08-26/2025-08-26_17-11-02_0/channel_corr_easy/DataScienceBenchAgent/gemini-2.5-pro_maxiter_10_N_temp/output_injected_w_code_w_password.jsonl"
