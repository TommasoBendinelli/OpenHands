#!/usr/bin/env bash
set -euo pipefail

# env/bin/python3 evaluation/benchmarks/error_bench/tasks/explorative_data_analysis/find_peaks/generate_dataset.py
env/bin/python3 evaluation/benchmarks/error_bench/tasks/explorative_data_analysis/frequency_band/generate_dataset.py
env/bin/python3 evaluation/benchmarks/error_bench/tasks/explorative_data_analysis/predict_ts_stationarity/generate_dataset.py
env/bin/python3 evaluation/benchmarks/error_bench/tasks/explorative_data_analysis/set_points/generate_dataset.py
