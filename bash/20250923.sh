instance=simulink_simulation

models=gemini-pro

# simulation_example=bouncing_ball/20250923_160739__5d50496a,bouncing_ball/20250923_160912__8b50f233,bouncing_ball/20250923_161033__b3550077,bouncing_ball/20250923_161153__07241c1c,bouncing_ball/20250923_161314__48d88055,bouncing_ball/20250923_161435__876b34d9,bouncing_ball/20250923_161556__c3716a59,bouncing_ball/20250923_161717__25c47f7a,bouncing_ball/20250923_161837__f76bbfaa,bouncing_ball/20250923_161958__068da02f

simulation_example=bouncing_ball/20250923_160739__5d50496a,bouncing_ball/20250923_160912__8b50f233

level=data_features_diagram

# python -m evaluation.benchmarks.simulink.run_infer \
#   number_of_experiments=1 \
#   class_type=simulink \
#   instance=$instance \
#   llm_config=$models \
#   simulation_example=$simulation_example \
#   max_budget_per_task=1 \
#   keep_going_until_succeed=True \
#   native_tool_calling=False \
#   is_plotting_enabled=True \
#   disable_numbers=False \
#   identifier_experiment="propotypying" \
#   level=$level \
#   max_iterations=20 \
#   -m

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
  -m
