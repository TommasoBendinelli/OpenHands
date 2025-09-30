import json
import re
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

    # breakpoint()

    ground_truth = metadata['correct_answer'].format(
        f'{{{metadata["first_diff_time"]}}}'
    )

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
        'provided_answer': model_answer
    }
