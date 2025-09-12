instance=maxon_logs

models=gemini-pro

# name of the log file.txt
ticket_file=28041,27738,27979,28133,28268,28271,28340

python -m evaluation.benchmarks.maxon.run_infer \
  number_of_experiments=1 \
  class_type=logs \
  instance=$instance \
  ticket_file=$ticket_file \
  llm_config=$models \
  max_budget_per_task=1 \
  keep_going_until_succeed=True \
  native_tool_calling=False \
  is_plotting_enabled=True \
  disable_numbers=False \
  identifier_experiment="propotypying" \
  -m
