python -m pdb evaluation/benchmarks/error_bench/run_infer.py number_of_experiments=1   eval_n_limit=1  class_type=explorative_data_analysis instance=predict_ts_stationarity  constraints=0  hints=0  llm_config="gemini"  solution_iterations=5  seed=6,7,8,9,10    cheating_attempt=True warm_against_cheating=False   max_budget_per_task=0.25  -m


python -m pdb evaluation/benchmarks/error_bench/run_infer.py number_of_experiments=1   eval_n_limit=1  class_type=explorative_data_analysis instance=predict_ts_stationarity  constraints=0  hints=0  llm_config="gemini"  solution_iterations=1  seed=6,7,8,9,10   cheating_attempt=True warm_against_cheating=False   max_budget_per_task=0.05  -m


python -m pdb evaluation/benchmarks/error_bench/run_infer.py number_of_experiments=1   eval_n_limit=1  class_type=explorative_data_analysis instance=predict_ts_stationarity  constraints=0  hints=0  llm_config="gemini"  solution_iterations=1  seed=1,2,3,4,5,10,11,12,13,14,15,16,17,18,19,20,21   cheating_attempt=True warm_against_cheating=False   max_budget_per_task=0.05  -m


python evaluation/benchmarks/error_bench/run_infer.py \
  number_of_experiments=1 \
  eval_n_limit=1 \
  class_type=explorative_data_analysis \
  instance=predict_ts_stationarity \
  constraints=0 \
  hints=0 \
  llm_config="gemini" \
  solution_iterations=1 \
  cheating_attempt=True \
  warm_against_cheating=False \
  max_budget_per_task=0.05 \
  prompt_variation=0,1,2,3 \
  'seed=range(10,31)' -m


python evaluation/benchmarks/error_bench/run_infer.py   number_of_experiments=1   eval_n_limit=1   class_type=explorative_data_analysis   instance=predict_ts_stationarity   constraints=0   hints=0   llm_config="gemini"   solution_iterations=1   cheating_attempt=False   warm_against_cheating=False   max_budget_per_task=0.5   prompt_variation=0,1,2,3,4,5,6,7,8,9,10   'seed=range(45,46)' -m

python evaluation/benchmarks/error_bench/run_infer.py   number_of_experiments=1   eval_n_limit=1   class_type=explorative_data_analysis   instance=predict_ts_stationarity   constraints=0   hints=0   llm_config="gemini"   solution_iterations=10   cheating_attempt=False   warm_against_cheating=False   max_budget_per_task=0.5   prompt_variation=0,1,2,3,4,5,6,7,8,9,10   'seed=range(45,46)' -m


python evaluation/benchmarks/error_bench/run_infer.py   number_of_experiments=1   eval_n_limit=1   class_type=explorative_data_analysis   instance=predict_ts_stationarity   constraints=0   hints=0   llm_config="gemini"   solution_iterations=1   cheating_attempt=False   warm_against_cheating=False   max_budget_per_task=0.05   prompt_variation=0  seed=334 include_constraints=True
