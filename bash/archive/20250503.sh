python -m pdb evaluation/benchmarks/error_bench/run_infer.py number_of_experiments=1   eval_n_limit=1  class_type=explorative_data_analysis instance=predict_ts_stationarity  constraints=0,1,2  hints=0  llm_config="gemini"  solution_iterations=1,2,5,20  seed=6,7,8,9,10    cheating_attempt=True warm_against_cheating=False   max_budget_per_task=0.05,0.20  -m


python -m pdb evaluation/benchmarks/error_bench/run_infer.py number_of_experiments=1   eval_n_limit=1  class_type=explorative_data_analysis instance=predict_ts_stationarity  constraints=2  hints=0  llm_config="gemini"  solution_iterations=2  seed=6,7,8,9,10    cheating_attempt=True warm_against_cheating=False   max_budget_per_task=0.05  -m

python -m pdb evaluation/benchmarks/error_bench/run_infer.py number_of_experiments=1   eval_n_limit=1  class_type=explorative_data_analysis instance=predict_ts_stationarity  constraints=0  hints=0  llm_config="gemini"  solution_iterations=5  seed=6,7,8,9,10    cheating_attempt=True warm_against_cheating=False   max_budget_per_task=0.25  -m


python -m pdb evaluation/benchmarks/error_bench/run_infer.py number_of_experiments=1   eval_n_limit=1  class_type=explorative_data_analysis instance=predict_ts_stationarity  constraints=0  hints=0  llm_config="gemini"  solution_iterations=1  seed=6,7,8,9,10    cheating_attempt=True warm_against_cheating=False   max_budget_per_task=0.05 disable_numbers=True  -m
