models=open_router_claude

# instance=03_context
# ticket_file=UseCase_Delta,UseCase_Error6001_DoubleAssignment,UseCase_FileNotFound,UseCase_MotorDoesNotStart,UseCase_Recalculation

# python -m evaluation.benchmarks.maxon.run_infer \
#   number_of_experiments=1 \
#   class_type=logs \
#   instance=$instance \
#   ticket_file=$ticket_file \
#   llm_config=$models \
#   max_budget_per_task=5 \
#   keep_going_until_succeed=True \
#   native_tool_calling=False \
#   is_plotting_enabled=True \
#   disable_numbers=False \
#   identifier_experiment="propotypying" \
#   -m

instance=04_specific_log_file
ticket_file=UseCase_EncoderError,UseCase_ErrorSummary,UseCase_ShopOrder_Summary,UseCase_ShopOrderLoadingTime,UseCase_SpecificValue

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

# insance=02_specific_motor_type

# python -m evaluation.benchmarks.maxon.run_infer \
#   number_of_experiments=1 \
#   class_type=logs \
#   instance=$instance \
#   llm_config=$models \
#   max_budget_per_task=1 \
#   keep_going_until_succeed=True \
#   native_tool_calling=False \
#   is_plotting_enabled=True \
#   disable_numbers=False \
#   identifier_experiment="propotypying" \
