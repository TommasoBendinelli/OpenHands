# instance=schurter_tickets
# models=sglang_deepseek
# prompt_id=1,2,3,4,5,6,7,8,9,10,11,12,14,18,19,21,22,23,26,27,29,31,32,34,35,36

# instance=${instance}_${prompt_id}

# python -m evaluation.benchmarks.schurter.run_infer \
#   number_of_experiments=1 \
#   instance=$instance \
#   prompt_id=$prompt_id \
#   llm_config=$models \
#   max_budget_per_task=1 \
#   keep_going_until_succeed=True \
#   native_tool_calling=False \
#   is_plotting_enabled=True \
#   disable_numbers=False \
#   identifier_experiment="propotypying" \
#   -m

instance=schurter_tickets
models=sglang_deepseek
prompt_ids="1 2 3 4 5 6 7 8 9 10 11 12 14 18 19 21 22 23 26 27 29 31 32 34 35 36"

for pid in $prompt_ids; do
    python -m evaluation.benchmarks.schurter.run_infer \
      number_of_experiments=1 \
      instance=${instance}_${pid} \
      prompt_id=$pid \
      llm_config=$models \
      max_budget_per_task=1 \
      keep_going_until_succeed=True \
      native_tool_calling=False \
      is_plotting_enabled=True \
      disable_numbers=False \
      identifier_experiment="prototyping" \
      -m
done
