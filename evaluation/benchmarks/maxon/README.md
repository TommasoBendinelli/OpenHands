
## Run Inference

To run the data science benchmark inference, execute the `run_infer.py` script using `hydra`. You can specify various parameters using the `hydra_config/main.yaml` file or by overriding them directly in the command line.

First, ensure you have the necessary dependencies installed:
```bash
pip install -e .
```

Then, you can run the inference script as a Python module from the root of the OpenHands repository.
```bash
python3 -m evaluation.benchmarks.data_science_bench.run_infer
```
is the base command to run the inference. You can customize the configuration by using the `hydra_config/main.yaml` file or by passing parameters directly in the command line.
Important parameters include:
instance=channel_corr_easy -> Specifies the instance to run.
max_budget= -> Set to amount of USD you want to spend
