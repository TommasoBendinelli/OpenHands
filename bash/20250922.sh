instance=simulink_simulation

models=gemini-pro

simulation_example=BouncingBall/20250924_142654__e3c81612

# bouncing_ball/20250922_105357__1e4b0c63

level=data_features_diagram

python -m evaluation.benchmarks.simulink.run_infer \
  number_of_experiments=1 \
  class_type=simulink \
  instance=$instance \
  llm_config=$models \
  simulation_example=$simulation_example \
  max_budget_per_task=1 \
  keep_going_until_succeed=True \
  native_tool_calling=False \
  is_plotting_enabled=True \
  disable_numbers=False \
  identifier_experiment="propotypying" \
  level=$level \
  max_iterations=20 \
