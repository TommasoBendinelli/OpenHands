from pathlib import Path
def main():
    x = ",".join([x.name for x in list(Path("evaluation/benchmarks/error_bench/tasks/explorative_data_analysis").glob("*")) if (not x.name in ["old","__pycache__"]) and (not x.is_file())])
    print(x)
    # Now print with '' between each name
    x = ",".join(["'"+x.name+"'" for x in list(Path("evaluation/benchmarks/error_bench/tasks/explorative_data_analysis").glob("*")) if (not x.name in ["old","__pycache__"]) and (not x.is_file())])
    print(x)

if __name__ == '__main__':
    main()