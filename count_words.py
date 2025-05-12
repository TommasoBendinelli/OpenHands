from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Optional, Tuple
import re
from omegaconf import OmegaConf
import json
import ast
from collections import Counter
from evaluation.utils.loading_experiments import get_folders_in_range, _load_experiment


def count_ngrams(text):
    """
    Count n-grams in the given text.

    Args:
        text (str): Input text to analyze.

    Returns:
        list of tuples: Sorted list of n-grams and their counts.
    """
    # Create a vectorizer that extracts 1- to 4-grams
    vec = CountVectorizer(ngram_range=(5, 10), analyzer='char')
    X = vec.fit_transform([text])

    # Sum counts and map back to n-grams
    counts = X.toarray().sum(axis=0)
    ngrams = vec.get_feature_names_out()

    # Pair and sort
    freqs = sorted(zip(ngrams, counts), key=lambda x: -x[1])

    return freqs


def clean_code(code: str) -> str:
    cleaned_lines = []
    for i, line in enumerate(code.splitlines()):
        try:
            ast.parse(line)
            cleaned_lines.append(line)
        except SyntaxError:
            print(f"Skipping faulty line {i+1}: {line!r}")
            continue
    return "\n".join(cleaned_lines)


def extract_function_calls(code: str) -> list:
    # Parse code
    fixed_code = clean_code(all_generated_code)
    tree = ast.parse(fixed_code)

    # Step 1: Get all custom function definitions
    custom_funcs = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    # Step 2: Identify library function calls
    called_functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Handle function call styles: direct, attribute, deep attribute
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                # Support nested attributes: e.g., ms.train_test_split or model.fit
                attrs = []
                while isinstance(func, ast.Attribute):
                    attrs.append(func.attr)
                    func = func.value
                if isinstance(func, ast.Name):
                    attrs.append(func.id)
                    full_name = ".".join(reversed(attrs))
                    func_name = full_name
                else:
                    continue
            else:
                continue

            if func_name.split(".")[0] not in custom_funcs:
                called_functions.append(func_name)

    return called_functions


def remove_jupyter_bash_lines(code: str) -> str:
    lines = code.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Remove if the line contains a Jupyter shell command (e.g., starts with '!' after '=')
        if re.search(r'=\s*!', line):  # e.g., test_data_string = !head ...
            continue
        elif stripped.startswith(
            '!'
        ):  # or starts with ! directly (e.g., !pip install ...)
            continue
        else:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


if __name__ == "__main__":

    # Get all the entries evaluation/evaluation_outputs/outputs
    ROOT_DIR = Path('evaluation/evaluation_outputs/outputs')
    AFTER = datetime.strptime('2025-05-10_00-05-01', '%Y-%m-%d_%H-%M-%S')
    BEFORE = datetime.strptime('2025-05-13_00-41-21', '%Y-%m-%d_%H-%M-%S')
    BEFORE = None

    runs = sorted(get_folders_in_range(ROOT_DIR, AFTER, BEFORE))

    messages = []
    for folder in runs:

        # Open metadata
        metadata, outputs, cfg = _load_experiment(folder)

        if not outputs:
            continue

        for i in outputs[0]['history']:
            for key in i.keys():
                if 'args' in key:
                    if 'code' in i['args'].keys():
                        # 'pip install pandas' cannot be handeled by ast
                        if "pip install" in i['args']['code']:
                            # remove the line
                            i['args']['code'] = i['args']['code'].replace(
                                "pip install pandas", ""
                            )
                        if "%" in i['args']['code']:
                            # remove the line
                            i['args']['code'] = i['args']['code'].replace("%", "")

                        i['args']['code'] = remove_jupyter_bash_lines(i['args']['code'])
                        messages.append(i['args']['code'])

    # Concatenate all string items in the list into one string
    all_generated_code = "\n".join(messages)

    called_functions = extract_function_calls(all_generated_code)

    # Step 4: Count the calls
    call_counts = Counter(called_functions)

    # Step 5: Print results
    print("Library/Built-in Function Calls:")
    for func, count in call_counts.items():
        print(f"- {func}: {count}")

    # Top 5 most common function calls
    top_5_calls = call_counts.most_common(15)
    print(f"\nTop 15 Most Common Function Calls: {top_5_calls}")
