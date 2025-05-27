#!/usr/bin/env python3

import json
import random
from pathlib import Path

import numpy as np
import openml
import pandas as pd


def generate_data(percentage_of_corruption):
    """
    Fetches the first `num_datasets` from OpenML, then saves them locally as CSV.
    Also demonstrates injecting "two pawns on rank 8" errors specifically into KRKPA7
    when loading via task.get_X_and_y() (NumPy arrays).
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    task_description = """
    Use baseline.csv (a historically-clean, trusted sample from the same data source) to learn the valid patterns, value ranges, and categories for every column. Then scan to_check.csv—a new batch of data—to flag any cell that violates those learned rules.  You should check for functional constraints violation. Take also a look the column descriptions. They might be helpful! Return your result in a JSON format with only the following key: \n    'motivation' and the description of the corruption. If there is no corruption, return 'clean' as the value of the key 'motivation'. \n    'code_to_detect' is the code that takes as input the dataset and return the corrupted cells. This is the signature of the function:\n    def detect_corruption(df: pd.DataFrame) -> pd.DataFrame:\n        pass\n    Where the input is a pandas DataFrame and the output is a boolean DataFrame with the same shape as the input DataFrame. The function should return True for the cells that are corrupted and False for the cells that are not corrupted. DON'T BE SCARED TO EXPLORE THE DATASET! Before answering trying to list as many hypothesis as possible. Don't be afraid of listing hypothesis before answering! You should explore extensively the data before answering. You should start this task by listing all the functional dependecies that you can think of in this data and test each hypothesis you have until you find the one that is violated. Hint: take a look at stlmt
    """
    suite = openml.study.get_suite('OpenML-CC18')
    # https://www.openml.org/search?type=study&sort=tasks_included&study_type=task&id=99

    # 2) Loop through tasks in that suite
    task_id = suite.tasks[0]
    # for task_id in suite.tasks:
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    print(dataset.description)

    # Instead of a DataFrame, we do the numeric approach:
    X_df, _, is_attribute, _ = dataset.get_data(
        dataset_format='dataframe', target=dataset.default_target_attribute
    )
    X_df = X_df.drop(columns=['wkcti'])
    train_indices, test_indices = task.get_train_test_split_indices()
    X_clean = X_df.loc[train_indices]
    # Use the test set for corruption data
    # Sample percentage_of_corruption% of the test set for corruption
    X_corrupted = X_df.loc[test_indices]

    corrupted_indices = np.random.choice(
        X_corrupted.index,
        size=int(len(X_corrupted) * percentage_of_corruption),
        replace=False,
    )
    X_corrupted.loc[corrupted_indices, 'stlmt'] = 't'
    X_corrupted.loc[corrupted_indices, 'wknck'] = 't'

    # Create a tmp folder to save the datasets in the same directory where the script is located
    local_path = Path(__file__).parent
    tmp = local_path / 'tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    X_clean.to_csv(tmp / 'baseline.csv', index=False)
    X_corrupted.to_csv(tmp / 'to_check.csv', index=False)
    # (X_corrupted[['wkna8', 'cntxt']] == "t").all(axis=1).any()
    # breakpoint()
    bool_df = pd.DataFrame(np.zeros(X_df.shape), columns=X_df.columns, index=X_df.index)
    bool_df.loc[corrupted_indices, 'stlmt'] = 1
    bool_df.loc[corrupted_indices, 'wknck'] = 1

    # Save the boolean DataFrame
    bool_df.to_csv(tmp / 'solution.csv', index=False)
    # Save also a json with the task description
    metadata = {
        'task_description': task_description,
        'dataset_description': dataset_description,
        'correct_answer': 'Functional depedency between stlmt and wknck violated',
        'percentage_of_corruption': percentage_of_corruption,
    }
    with open(tmp / 'metadata.json', 'w') as f:
        f.write(json.dumps(metadata, indent=4))


dataset_description = """
    Feature
    Meaning (all Boolean <t/f> unless noted)

    1.  bkblk   Black king is not in the way of the plan.
    2.  bknwy   Black king is not in the white rook’s way.
    3.  bkon8   Black king sits on the 8th rank helping the rook.
    4.  bkona   Black king sits on the a‑file helping the rook.
    5.  bkspr   Black king can support its rook.
    6.  bkxbq   Black king is safe from any attack by a promoted pawn.
    7.  bkxcr   Black king can attack the critical square (b7).
    8.  bkxwp   Black king can attack the white pawn on a7.
    9.  blxwp   A Black piece attacks the white pawn from the left (x = –1).
    10. bxqsq   One or more Black pieces control the queening square (a8).
    11. cntxt   White king is on a board edge and not on a8.
    12. dsopp   The two kings stand in normal opposition.
    13. dwipd   White king is too far from the intersect point (files g/l).
    14. hdchk   A hidden check gives Black a useful delaying tactic.
    15. katri   Black king controls the intersect point (files b/n/w).
    16. mulch   Black can renew the check to good effect.
    17. qxmsq   The promoted pawn attacks a mating square.
    18. r2ar8   Black rook lacks safe access to file a or rank 8.
    19. reskd   White king can be re‑skewered after a delay.
    20. reskr   Black rook alone can renew the skewer threat.
    21. rimmx   Black rook can be captured safely.
    22. rkxwp   Rook bears on the white pawn from the left (x = –1).
    23. rxmsq   Rook safely attacks a mating square.
    24. simpl   A very simple (trivial) pattern applies.
    25. skach   White king can be skewered after a series of checks.
    26. skewr   A potential skewer (not a fork) exists.
    27. skrxp   Rook can execute a skewer or Black king attacks the pawn.
    28. spcop   A special‑opposition pattern is present.
    29. stlmt   White king is stalemated.
    30. thrsk   A skewer threat is lurking.
    31. wkcti   White king cannot control the intersect point.
    32. wkna8   White king is actually on square a8.
    33. wknck   White king is in check.
    34. wkovl   White king is overloaded (too many duties).
    35. wkpos   White king sits where it could be skewered.
    36. wtoeg   White king is one square away from the edge (files n/t).
    """

if __name__ == '__main__':
    percentage_of_corruption = [1]
    for i in percentage_of_corruption:
        generate_data(percentage_of_corruption=i)
