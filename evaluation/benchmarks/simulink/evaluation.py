import re
import json


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

    # breakpoint()

    ground_truth = metadata['correct_answer'].format(
        f"{{{metadata['first_diff_time']}}}"
    )

    # Extract numbers inside {}
    extract_num = lambda s: float(re.search(r"\{([0-9.eE+-]+)\}", s).group(1))

    try:
        num_provided = extract_num(model_answer)
        num_truth = extract_num(ground_truth)
    except AttributeError:
        breakpoint()
        raise ValueError("Could not extract numeric value from one of the strings.")

    # Remove numbers {float} and compare text parts
    text_provided = re.sub(r"\{[0-9.eE+-]+\}", "{}", model_answer)
    text_truth = re.sub(r"\{[0-9.eE+-]+\}", "{}", ground_truth)

    text_match = text_truth in text_provided or text_provided in text_truth
    difference = num_provided - num_truth
    relative_error = difference / num_truth if num_truth != 0 else float("inf")
    numeric_match = abs(difference) <= tol

    return {
        "text_match": text_match,
        "numeric_match": numeric_match,
        "difference": difference,
        "relative_error": relative_error,
        "provided_value": num_provided,
        "ground_truth_value": num_truth,
    }
