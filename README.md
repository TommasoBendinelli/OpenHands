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

Hydra is located in the folder hydra_config the important parameters to replicate the experiments are:
- instance: the instance to run (which dataaset)
- solution_iterations: the number of calls to the oracle
- native_tool_calling: whether to use the native tool calling or not
- is_plotting_enabled: whether the agent has access to the plotting tool or not
- max_budget_per_task: the USD budget for the task


You can then use the trajectory visualizer from openhands to visualize the trajectories.
Additionally results are saved in the folder: evaluation/evaluation_outputs/outputs
# Where to find the datasets
The 17 datasets of EDx17 are stored `evaluation/benchmarks/error_bench/tasks/explorative_data_analysis`. 
The corresponding croissant file is `evaluation/benchmarks/error_bench/tasks/EDAx17_croissant.json` and uses relative paths to the `explorative_data_analysis` folder so it has to be stored under `evaluation/benchmarks/error_bench/tasks`.

We will ensure long term hosting and access of the data upon acceptance through huggingface. 

Each of the folders in `evaluation/benchmarks/error_bench/tasks/explorative_data_analysis` contains one of the existing datasets comprising of four `.csv` files and a `generate_dataset.py` which can be used to regenerate the data.
In case there is issues with the croissant dataset you can regenerate the `croissant.json` with the `croissant/create_croissant_dataset.py`
script.

# Code Base
Our repository is based on the large OpenHands repository, please ignore files not mentioned by the README and follow only instructions found in this readme to run the experiments. For specific issues we refer to the OpenHands repository (https://github.com/All-Hands-AI/OpenHands)
