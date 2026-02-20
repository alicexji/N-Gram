from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple, Set


class BackoffNGramModel:
    """
    Stupid backoff n-gram
    Trains counts for all orders 1..n from the same training set
    Uses current order when an n-gram is seen, otherwise backs off to shorter context
    For example, if 7 gram fails is uninformative, we try 6...then 5... 4
    Uses add-alpha ONLY for unigrams to avoid zero as denominator
    Maps out of context variables (OOV) to <UNK>
    """

    def __init__(self, n: int, beta: float = 0.4, unigram_alpha: float = 0.1):
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n
        self.beta = beta
        self.unigram_alpha = unigram_alpha

        self.BOS = "<s>"
        self.EOS = "</s>"
        self.UNK = "<UNK>"

        self.vocab: Set[str] = set()
        self.vocab_size: int = 0

        # counts_by_order[k][gram] where gram is a tuple of length k
        self.counts_by_order: Dict[int, Dict[Tuple[str, ...], int]] = {
            k: defaultdict(int) for k in range(1, n + 1)
        }
        # context_counts_by_order[k][context] where context is length (k-1), for k>=2
        self.context_counts_by_order: Dict[int, Dict[Tuple[str, ...], int]] = {
            k: defaultdict(int) for k in range(2, n + 1)
        }
        self.total_unigrams = 0

    def _pad(self, tokens: List[str]) -> List[str]:
        return [self.BOS] * (self.n - 1) + tokens + [self.EOS]

    def train_from_file(self, train_path: str) -> None:
        # Build vocab from training only
        with open(train_path, "rb") as f:
            for line in f:
                toks = line.decode("utf-8", errors="ignore").strip().split()
                for t in toks:
                    self.vocab.add(t)

        self.vocab.update([self.BOS, self.EOS, self.UNK])
        self.vocab_size = len(self.vocab)

        # Count all k-grams for k=1..n
        with open(train_path, "rb") as f:
            for line in f:
                toks = line.decode("utf-8", errors="ignore").strip().split()
                toks = [t if t in self.vocab else self.UNK for t in toks]
                padded = self._pad(toks)

                L = len(padded)
                for i in range(L):
                    for k in range(1, self.n + 1):
                        if i - k + 1 < 0:
                            continue
                        gram = tuple(padded[i - k + 1 : i + 1])
                        self.counts_by_order[k][gram] += 1
                        if k == 1:
                            self.total_unigrams += 1
                        elif k >= 2:
                            ctx = gram[:-1]
                            self.context_counts_by_order[k][ctx] += 1

    def _ml_prob(self, k: int, context: Tuple[str, ...], token: str) -> float:
        """MLE for k-gram (k>=2), assuming gram exists."""
        gram = context + (token,)
        num = self.counts_by_order[k].get(gram, 0)
        den = self.context_counts_by_order[k].get(context, 0)
        return num / den if den > 0 else 0.0

    def _unigram_prob(self, token: str) -> float:
        """Add-alpha smoothed unigram."""
        c = self.counts_by_order[1].get((token,), 0)
        num = c + self.unigram_alpha
        den = self.total_unigrams + self.unigram_alpha * self.vocab_size
        return num / den if den > 0 else 0.0

    def prob(self, context_list: List[str], token: str) -> float:
        # Map OOV
        token = token if token in self.vocab else self.UNK
        ctx_tokens = [t if t in self.vocab else self.UNK for t in context_list]

        # Use up to n-1 context tokens
        ctx_tokens = ctx_tokens[-(self.n - 1):]

        # Try highest order down to bigram; then unigram
        backoff_factor = 1.0

        for k in range(self.n, 1, -1):  # k = n, n-1, ..., 2
            need = k - 1
            if len(ctx_tokens) < need:
                continue
            ctx = tuple(ctx_tokens[-need:])
            gram = ctx + (token,)
            if self.counts_by_order[k].get(gram, 0) > 0:
                p = self._ml_prob(k, ctx, token)
                return max(backoff_factor * p, 1e-12)

            backoff_factor *= self.beta  # unseen -> back off

        # Unigram base case (smoothed)
        p1 = self._unigram_prob(token)
        return max(backoff_factor * p1, 1e-12)

    def perplexity(self, eval_path: str) -> float:
        log_sum = 0.0
        N = 0
        with open(eval_path, "rb") as f:
            for line in f:
                toks = line.decode("utf-8", errors="ignore").strip().split()
                toks = [t if t in self.vocab else self.UNK for t in toks]
                padded = self._pad(toks)

                for i in range(self.n - 1, len(padded)):
                    context = padded[i - (self.n - 1) : i]
                    gt = padded[i]  # ground-truth next token
                    p = self.prob(context, gt)  # P(gt | context)
                    log_sum += math.log(p)
                    N += 1

        return math.exp(-log_sum / N) if N > 0 else float("inf")
