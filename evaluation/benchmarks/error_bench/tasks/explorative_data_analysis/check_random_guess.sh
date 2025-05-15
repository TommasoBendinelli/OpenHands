instance=row_variance,simultanus_spike #,channel_corr,outlier_ratio,channel_divergence,variance_burst,sign_rotated_generator,ground_mean_threashold,dominant_feature,set_points,common_frequency,predict_ts_stationarity,zero_crossing,sum_threshold,spike_presence,cofounded_group_outlier,find_peaks,row_max_abs,periodic_presence

models=gemini_pro_pro,open_router_claude  # "gemini_lite" "open_router_claude" "deepseek")


python   evaluation/benchmarks/error_bench/run_infer.py   number_of_experiments=1   eval_n_limit=1   class_type=explorative_data_analysis   instance=$instance    constraints=0   hints=0   llm_config=$models   solution_iterations=10   cheating_attempt=False   warm_against_cheating=False   max_budget_per_task=1   prompt_variation=0 seed=20  keep_going_until_succeed=True native_tool_calling=False  is_plotting_enabled=False  give_structure_hint=False  disable_numbers=True identifier_experiment="check_random_guess"   -m