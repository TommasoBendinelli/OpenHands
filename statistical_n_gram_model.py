#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple statistical n-gram language model.
Usage example is at the bottom of the file.
"""

import math
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# ---------- preprocessing ----------------------------------------------------

TOKEN_RE = re.compile(r"[a-zA-Z']+|[0-9]+|[^\w\s]")  # keep punctuation as tokens


def tokenize(text: str) -> List[str]:
    """Lower-case and split text into tokens."""
    return TOKEN_RE.findall(text.lower())


# ---------- model ------------------------------------------------------------


class NGramModel:
    def __init__(self, n: int = 3, k: float = 1.0) -> None:
        """
        n : order of the model (e.g. 3 for trigram)
        k : smoothing constant (k=1 is Laplace, k=0 is MLE)
        """
        if n < 1:
            raise ValueError('n must be >= 1')
        self.n = n
        self.k = k
        self.context_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()
        self.total_contexts: Dict[Tuple[str, ...], int] = defaultdict(int)

    # ---- training -----------------------------------------------------------

    def fit(self, tokens: List[str]) -> None:
        """
        Build the n-gram counts from a list of tokens.
        Adds (n-1) </s> padding at the end for sentence completion.
        """
        padded = tokens + ['</s>'] * (self.n - 1)
        for i in range(len(padded) - self.n + 1):
            ngram = tuple(padded[i : i + self.n])
            context, word = ngram[:-1], ngram[-1]
            self.context_counts[context][word] += 1
            self.total_contexts[context] += 1
            self.vocab.add(word)
        # Include all context shells so unseen words get smoothed prob.
        self.vocab.update(padded[: self.n - 1])

    # ---- probabilities ------------------------------------------------------

    def prob(self, context: Tuple[str, ...], word: str) -> float:
        """
        P(word | context) with add-k smoothing.
        context length must be n-1.
        """
        context = self._fix_context(context)
        vocab_size = len(self.vocab)
        count_wc = self.context_counts[context][word]
        denom = self.total_contexts[context]
        return (count_wc + self.k) / (denom + self.k * vocab_size)

    # ---- generation ---------------------------------------------------------

    def sample_next(self, context: Tuple[str, ...]) -> str:
        """
        Draw a random next token from the context distribution.
        """
        context = self._fix_context(context)
        # Build cumulative distribution
        r = random.random()
        cum = 0.0
        for word in self.vocab:
            cum += self.prob(context, word)
            if r <= cum:
                return word
        return random.choice(tuple(self.vocab))  # fallback

    # ---- helpers ------------------------------------------------------------

    def _fix_context(self, context: Tuple[str, ...]) -> Tuple[str, ...]:
        if len(context) != self.n - 1:
            raise ValueError(f'context must have length {self.n - 1}')
        return context

    # ---- evaluation (optional) ---------------------------------------------

    def perplexity(self, tokens: List[str]) -> float:
        """
        Compute perplexity of a held-out sequence.
        """
        padded = tokens + ['</s>'] * (self.n - 1)
        log_prob = 0.0
        count = 0
        for i in range(len(padded) - self.n + 1):
            context = tuple(padded[i : i + self.n - 1])
            word = padded[i + self.n - 1]
            log_prob += math.log2(self.prob(context, word))
            count += 1
        return 2 ** (-log_prob / count)


# ---------- quick demo -------------------------------------------------------

if __name__ == '__main__':
    sample_text = """
        import pandas as pd
        import numpy as np

        train_df = pd.read_csv("/workspace/train.csv")

        threshold_high = 14.0
        threshold_low = -7.0

        def predict(value_999):
            if value_999 > threshold_high or value_999 < threshold_low:
                return 0
            else:
                return 1

        train_df['predicted_label'] = train_df['999'].apply(predict)

        accuracy = (train_df['predicted_label'] == train_df['label']).mean()

        print(f"Accuracy on train set with rule (value_999 > {threshold_high} or value_999 < {threshold_low}): {accuracy*100:.2f}%")

        # Check misclassified samples
        misclassified = train_df[train_df['predicted_label'] != train_df['label']]
        print(f"Number of misclassified samples: {len(misclassified)}")
        if len(misclassified) > 0:
            print("Misclassified samples details (feature 999):")
            print(misclassified[['999', 'label', 'predicted_label']])

    """

    tokens = tokenize(sample_text)
    n = 2  # trigram model
    k = 1.0  # Laplace smoothing
    model = NGramModel(n, k)
    model.fit(tokens)

    ctx = 'max'  # must be length n-1
    print(f"P('brown' | {ctx}) = {model.prob(ctx, 'brown'):.4f}")

    # generate 10 words
    generated = list(ctx)
    for _ in range(10):
        next_word = model.sample_next(tuple(generated[-(n - 1) :]))
        generated.append(next_word)
        if next_word == '</s>':
            break
    print('Generated:', ' '.join(generated))
