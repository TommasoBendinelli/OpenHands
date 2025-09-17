#!/bin/bash

models="open_router_claude"

# Define instances and their ticket files
declare -A configs

# Instance 03
configs["03_context"]="UseCase_Delta,UseCase_Error6001_DoubleAssignment,UseCase_FileNotFound,UseCase_MotorDoesNotStart,UseCase_Recalculation"

# Instance 04
configs["04_specific_log_file"]="UseCase_EncoderError,UseCase_ErrorSummary,UseCase_ShopOrder_Summary,UseCase_ShopOrderLoadingTime,UseCase_SpecificValue"

# Instance 02 (uncomment if needed)
configs["02_specific_motor_type"]=""

# Loop through all configs
for instance in "${!configs[@]}"; do
  ticket_file="${configs[$instance]}"

  echo "Running instance=$instance with ticket_file=$ticket_file"

  python -m evaluation.benchmarks.maxon.run_infer \
    number_of_experiments=1 \
    class_type=logs \
    instance="$instance" \
    ticket_file="$ticket_file" \
    llm_config="$models" \
    max_budget_per_task=5 \
    keep_going_until_succeed=True \
    native_tool_calling=False \
    is_plotting_enabled=True \
    disable_numbers=False \
    identifier_experiment="prototyping" \
    -m
done
