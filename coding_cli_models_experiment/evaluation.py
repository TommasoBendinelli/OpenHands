import json
import re
import os
import warnings


def evaluate_model_answer(model_answer: str, metadata: json, tol: float = 1e-6):
    """
    Compare a provided answer against ground truth.

    Args:
        provided (str): The provided answer string.
        ground_truth (str): The ground truth string.
        tol (float): Allowed tolerance for numeric difference.

    Returns:
        dict: Results with text match, numeric match, difference, and relative error.
    """

    breakpoint()

    ground_truth = metadata['correct_choice'].format(
        f'{{{metadata["first_diff_time"]}}}'
    )

    # Remove \n from ground truth
    ground_truth = ground_truth.replace('\n', ' ').strip()


    # Extract numbers inside {}
    def extract_num(s):
        return float(re.search(r'\{([0-9.eE+-]+)\}', s).group(1))

    # Check if ground truth has numeric values, i.e. curly braces {}
    if re.search(r'\{[0-9.eE+-]+\}', ground_truth):
        num_truth = extract_num(ground_truth)
        try:
            num_provided = extract_num(model_answer)
        except AttributeError:
            print(model_answer)
            # Replace missing numeric values with NaN
            num_provided = float('nan')
            warnings.warn("Could not extract numeric value from one of the strings.")
            # raise ValueError('Could not extract numeric value from one of the strings.')

        # Remove numbers {float} and compare text parts
        text_provided = re.sub(r'\{[0-9.eE+-]+\}', '{}', model_answer)
        text_truth = re.sub(r'\{[0-9.eE+-]+\}', '{}', ground_truth)

        text_match = text_truth in text_provided or text_provided in text_truth
        difference = num_provided - num_truth
        relative_error = difference / num_truth if num_truth != 0 else float('inf')
        numeric_match = abs(difference) <= tol
    else:
        text_match = ground_truth in model_answer or model_answer in ground_truth
        numeric_match = None
        difference = None
        relative_error = None
        num_provided = None
        num_truth = None

    return {
        'text_match': text_match,
        'numeric_match': numeric_match,
        'difference': difference,
        'relative_error': relative_error,
        'provided_value': num_provided,
        'ground_truth_value': num_truth,
        'ground_truth': ground_truth,
        'provided_answer': model_answer,
    }


def get_all_folder_names(dir):
    folder_names = [
        os.path.join(dir, name)
        for name in os.listdir(dir)
        if os.path.isdir(os.path.join(dir, name)) and not name.startswith(".")
    ]
    return folder_names


def main():
    ## PATHS
    answers_dir = "/home/tommaso/repo/OpenHands/gpt_codex_experiment/tasks/BallBouncingBetweenWalls/"
    results_dir = "/home/tommaso/repo/OpenHands/gpt_codex_experiment/evaluation/20251009_171034/BallBouncingBetweenWalls"
    ## ----

    folder_names = get_all_folder_names(answers_dir)

    results_all = {}

    for folder in folder_names:
        # Open answer metadata.json
        metadata_path = os.path.join(
            answers_dir, os.path.basename(folder), "metadata_task.json"
        )
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            print(f"No metadata.json found in {folder}")
            continue

        # Open results.json
        results_path = os.path.join(
            results_dir, os.path.basename(folder), "results.json"
        )

        task_id = folder.split("/")[-1]

        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
        else:
            print(f"No results.json found in {folder}")
            continue

        # breakpoint()

        model_answer = next(iter(results.values()))
        eval_results = evaluate_model_answer(model_answer, metadata)

        results_all[folder] = eval_results

        print(f"Evaluated {folder}: {eval_results}")

    # Iterate through evaluation results and include summary statistics
    total = len(results_all)
    text_matches = sum(1 for res in results_all.values() if res['text_match'])
    numeric_matches = sum(
        1 for res in results_all.values() if res['numeric_match'] is True
    )

    eval_summary = {
        'total': total,
        'text_matches': text_matches,
        'numeric_matches': numeric_matches,
    }

    results_all['summary'] = eval_summary

    # Save evaluation results to eval_results.json
    eval_results_path = os.path.join(results_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(results_all, f, indent=4)


if __name__ == "__main__":
    main()
