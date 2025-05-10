#!/usr/bin/env python
"""
kidney_flag_ood.py
Binary classification with an out‑of‑distribution test split.

Ground‑truth rule
-----------------
    label 1  ⇐   (creatinine > 1.4 mg/dL)  AND  (eGFR < 60 mL/min)

Train–test design
-----------------
• TRAIN  : positive rows also have  BUN high   &  Albumin low
           negative rows have       BUN low    &  Albumin high
  -> BUN / Albumin act as *perfect* shortcuts.

• TEST   : BUN and Albumin are sampled **independently** of the label.
  -> shortcuts break; only the real rule holds.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import save_datasets                      # noqa: E402


# -------------   Ground‑truth labelling function   -----------------
def is_abnormal(row: np.ndarray) -> int:
    """Return 1 if (creatinine>1.4) AND (eGFR<60)."""
    creat, egfr = row[0], row[1]
    return int((creat > 1.4) and (egfr < 60))


# -------------   Sample generators   -------------------------------
def sample_train(flag: bool):
    """
    TRAIN distribution with *perfect* shortcuts:
        • positives → high BUN, low Albumin
        • negatives → low  BUN, high Albumin
    """
    if flag:                                  # label 1
        creat = np.random.uniform(1.6, 3.0)
        egfr  = np.random.uniform(15, 55)
        bun   = np.random.uniform(35, 60)     # HIGH shortcut
        alb   = np.random.uniform(2.0, 3.2)   # LOW  shortcut
    else:                                     # label 0
        creat = np.random.uniform(0.6, 1.3)
        egfr  = np.random.uniform(70, 120)
        bun   = np.random.uniform(7, 18)      # LOW  shortcut
        alb   = np.random.uniform(3.6, 5.0)   # HIGH shortcut
    return np.array([creat, egfr, bun, alb])


def sample_test(flag: bool):
    """
    TEST distribution: keep the same creat/eGFR regimes
    but RANDOMISE BUN & Albumin so the shortcuts disappear.
    """
    if flag:
        creat = np.random.uniform(1.5, 3.0)
        egfr  = np.random.uniform(10, 55)
    else:
        creat = np.random.uniform(0.6, 1.3)
        egfr  = np.random.uniform(70, 120)
    # BUN & Albumin drawn *independently* of the label
    bun = np.random.uniform(7, 60)
    alb = np.random.uniform(2.0, 5.0)
    return np.array([creat, egfr, bun, alb])


# -------------   Dataset builders   --------------------------------
def make_dataframe(n_rows: int, sampler):
    """Helper: build a balanced DataFrame using the given sampler."""
    rows, labels = [], []
    for _ in range(n_rows // 2):
        rows.append(sampler(True));  labels.append(1)
        rows.append(sampler(False)); labels.append(0)
    df = pd.DataFrame(rows, columns=['creatinine', 'eGFR', 'BUN', 'albumin'])
    df['label'] = labels
    return df.sample(frac=1).reset_index(drop=True)   # shuffle


def create_datasets(n_train=2_000, n_test=1_000, out_dir: Path | str = '.'):
    train_df = make_dataframe(n_train, sample_train)
    test_df  = make_dataframe(n_test,  sample_test)
    save_datasets(train_df, test_df, Path(out_dir))
    return train_df, test_df


# -------------   Quick sanity plot   -------------------------------
def sanity_plot(df: pd.DataFrame, title: str, path: Path):
    plt.figure(figsize=(5, 5))
    pos = df['label'] == 1
    plt.scatter(df.loc[~pos, 'creatinine'], df.loc[~pos, 'eGFR'],
                alpha=0.4, label='label 0')
    plt.scatter(df.loc[pos,  'creatinine'], df.loc[pos,  'eGFR'],
                alpha=0.4, label='label 1')
    plt.axvline(1.4, ls='--'); plt.axhline(60, ls='--')
    plt.xlabel('creatinine'); plt.ylabel('eGFR')
    plt.title(title); plt.legend(); plt.tight_layout(); plt.show()
    plt.savefig(path)


# -------------   Main script   -------------------------------------
if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    train_df, test_df = create_datasets(out_dir=here)

    sanity_plot(train_df, 'TRAIN distribution (shortcuts present)',
                here / 'kidney_train_example.png')
    sanity_plot(test_df, 'TEST distribution (shortcuts broken)',
                here / 'kidney_test_example.png')

    # Extra: show how BUN shortcut collapses in test
    plt.figure(figsize=(6, 4))
    plt.hist(train_df['BUN'][train_df['label']==0], bins=40,
             alpha=0.6, label='train label 0')
    plt.hist(train_df['BUN'][train_df['label']==1], bins=40,
             alpha=0.6, label='train label 1')
    plt.hist(test_df['BUN'][test_df['label']==0], bins=40,
             alpha=0.3, label='test label 0', histtype='step')
    plt.hist(test_df['BUN'][test_df['label']==1], bins=40,
             alpha=0.3, label='test label 1', histtype='step')
    plt.axvline(25, ls='--', c='k', lw=0.7)
    plt.title('BUN shortcut works in TRAIN but not in TEST')
    plt.xlabel('BUN'); plt.legend(); plt.tight_layout(); plt.show()
    plt.savefig(here / 'bun_shortcut_shift.png')