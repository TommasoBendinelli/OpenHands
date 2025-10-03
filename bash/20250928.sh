instance=simulink_simulation

# models=gemini-pro
# models=open_router_claude
models=open_router_gpt-5

level=data_features_diagram

# INSTANCE_DIR=/home/tommaso/repo/OpenHands/evaluation/benchmarks/simulink/tasks/MassSpringDamperWithPIDController

INSTANCE_DIR=/home/tommaso/repo/OpenHands/evaluation/benchmarks/simulink/tasks/BouncingBall

for full_example_path in "$INSTANCE_DIR"/*; do
    simulation_example="${full_example_path#*/tasks/}"

    echo "Running simulation for: $simulation_example"

    python -m evaluation.benchmarks.simulink.run_infer \
      number_of_experiments=1 \
      class_type=simulink \
      instance="$instance" \
      llm_config="$models" \
      simulation_example="$simulation_example" \
      max_budget_per_task=1 \
      keep_going_until_succeed=True \
      native_tool_calling=False \
      is_plotting_enabled=True \
      disable_numbers=False \
      identifier_experiment="propotypying" \
      level="$level" \
      max_iterations=50
done

# simulation_example=BouncingBall/20250924_143228__be5eb787

# python -m evaluation.benchmarks.simulink.run_infer \
#   number_of_experiments=1 \
#   class_type=simulink \
#   instance="$instance" \
#   llm_config="$models" \
#   simulation_example="$simulation_example" \
#   max_budget_per_task=1 \
#   keep_going_until_succeed=True \
#   native_tool_calling=False \
#   is_plotting_enabled=True \
#   disable_numbers=False \
#   identifier_experiment="propotypying" \
#   level="$level" \
#   max_iterations=50
