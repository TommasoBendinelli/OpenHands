python   evaluation/benchmarks/error_bench/run_infer.py   number_of_experiments=1   eval_n_limit=1   class_type=explorative_data_analysis   instance=simultanus_spike,sum_threshold  constraints=0   hints=0   llm_config="gemini_pro"   solution_iterations=10   cheating_attempt=False   warm_against_cheating=False   max_budget_per_task=1   prompt_variation=0 seed=20  keep_going_until_succeed=True native_tool_calling=False  is_plotting_enabled=False  give_structure_hint=True  disable_numbers=False -m

python   evaluation/benchmarks/error_bench/run_infer.py   number_of_experiments=1   eval_n_limit=1   class_type=explorative_data_analysis   instance=simultanus_spike,sum_threshts=0   hints=0   llm_config="gemini_pro"   soold,variance_burst,zero_crossing,predict_ts_stationarity,row_variance,row_max_abs    constraints=0   hints=0   llm_config="gemini_pro_pro"   solution_iterations=10   cheating_attempt=False   warm_against_cheating=False   max_budget_per_task=1   prompt_variat=True  disable_numbers=False tion=0 seed=20  keep_going_until_succeed=True native_tool_calling=False  is_plotting_enabled=True,False  give_structure_hint=True,False  disable_numbers=False -m


#!/usr/bin/env bash
models=("gemini_pro_pro" "gemini_pro" "gemini_lite")

for model in "${models[@]}"; do
  # Pick the budget: 1 for the pro models, 0.05 for the lite model
  if [[ "$model" == *lite* ]]; then
    budget=0.05
  else
    budget=1
  fi

  python evaluation/benchmarks/error_bench/run_infer.py \
    number_of_experiments=1 \
    eval_n_limit=1 \
    class_type=explorative_data_analysis \
    instance=row_variance,row_variance \
    constraints=0 \
    hints=0 \
    llm_config="$model" \
    solution_iterations=10 \
    cheating_attempt=False \
    warm_against_cheating=False \
    max_budget_per_task="$budget" \
    prompt_variation=0 \
    seed=20 \
    keep_going_until_succeed=True \
    native_tool_calling=False \
    is_plotting_enabled=True,False \
    give_structure_hint=True,False \
    disable_numbers=False \
    identifier_experiment="paper"
    -m
done

python   evaluation/benchmarks/error_bench/run_infer.py   number_of_experiments=1   eval_n_limit=1   class_type=explorative_data_analysis   instance=periodic_presence    constraints=0   hints=0   llm_config="gemini_pro"   solution_iterations=10   cheating_attempt=False   warm_against_cheating=False   max_budget_per_task=1   prompt_variation=0 seed=20  keep_going_until_succeed=True native_tool_calling=False  is_plotting_enabled=False  give_structure_hint=True  disable_numbers=True
