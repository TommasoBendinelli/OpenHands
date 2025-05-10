from pathlib import Path
def main():
    breakpoint()
    x = ",".join([x.name for x in list(Path("evaluation/benchmarks/error_bench/tasks/explorative_data_analysis").glob("*")) if (not x.name in ["old","__pycache__"]) and (not x.is_file())])


if __name__ == '__main__':
    main()