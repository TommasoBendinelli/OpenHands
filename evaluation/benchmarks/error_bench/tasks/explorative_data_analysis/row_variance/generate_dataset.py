"""
row_variance.py  –  perfectly separable version
label 1  ⇔  variance(first 8 features) > 0.7
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets   # noqa: E402


N_FEATS = 10
N_VARS  = 8           # how many cols the rule looks at


def generate_sample(high_var: bool) -> np.ndarray:
    """Create one row that is definitely on the correct side of THRESH."""
    x = np.random.normal(0, 1, size=N_FEATS)

    if high_var:
        # inflate a random half of the first 8 features
        mask = np.random.choice([0, 1], size=N_VARS, p=[0.5, 0.5]).astype(bool)
        x[:N_VARS][mask] *= np.random.uniform(2.0, 5.0)   # big stretch → huge variance
    else:
        # *shrink* all first-8 features so variance is tiny
        x[:N_VARS] *= np.random.uniform(0.2, 0.4)

    return x


def create_dataset(n_rows=2_000, out_csv='variance_tabular.csv') -> pd.DataFrame:
    rows, labels = [], []
    for _ in range(n_rows // 2):
        rows.append(generate_sample(high_var=True));  labels.append(1)
        rows.append(generate_sample(high_var=False)); labels.append(0)

    cols = [f'feat{i+1}' for i in range(N_FEATS)]
    df = pd.DataFrame(rows, columns=cols)
    df['label'] = labels
    return df


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    out_dir = Path(__file__).resolve().parent

    train = create_dataset()
    test  = create_dataset()
    save_datasets(train, test, out_dir)

    # sanity-check: histogram of row variance
    v_train = train.iloc[:, :N_VARS].var(axis=1, ddof=0)
    plt.hist(v_train[train.label == 0], bins=40, alpha=.6, label='label 0')
    plt.hist(v_train[train.label == 1], bins=40, alpha=.6, label='label 1')
    plt.title('Row variance cleanly separates the classes')
    plt.legend(); plt.tight_layout(); plt.show()
    plt.savefig(out_dir / 'variance_example.png')