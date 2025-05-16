# How to run
1. Git clone this repository
```bash
git clone
```
2. Install the required packages
```bash
pip install -e .
```
3. Configure the config.toml with the models and the keys


# How to run experiments
3. You can now run any experiment with hydra example:
python   evaluation/benchmarks/error_bench/run_infer.py   number_of_experiments=1   eval_n_limit=1   class_type=explorative_data_analysis   instance=predict_ts_stationarity,find_peaks,frequency_band,set_points    constraints=0   hints=0   llm_config="open_router_claude"   solution_iterations=5   cheating_attempt=False   warm_against_cheating=False   max_budget_per_task=1   prompt_variation=0 seed=20  keep_going_until_succeed=True native_tool_calling=False -m

Hydra is located in the folder hydra_config the important parameters are:
- instance: the instance to run (which dataaset)
- solution_iterations: the number of calls to the oracle
- native_tool_calling: whether to use the native tool calling or not
- plotting_

# Where to find the datasets

