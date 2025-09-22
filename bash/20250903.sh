instance=simulink_simulation

models=gemini-pro

simulation_example=fault_tolerant_fuel_system

python -m evaluation.benchmarks.simulink.run_infer \
  number_of_experiments=1 \
  class_type=simulink \
  instance=$instance \
  llm_config=$models \
  max_budget_per_task=1 \
  keep_going_until_succeed=True \
  native_tool_calling=False \
  is_plotting_enabled=True \
  disable_numbers=False \
  identifier_experiment="propotypying" \
