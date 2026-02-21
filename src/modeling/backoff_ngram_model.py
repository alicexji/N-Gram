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

        # probability cache for memoization, otherwise backoff is too expensive
        # key: (context_tuple, token) → probability
        self._prob_cache: Dict[Tuple[Tuple[str, ...], str], float] = {}

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
        """
        Compute P(token | context) using stupid backoff.

        Strategy:
        1. Try highest-order n-gram first.
        2. If unseen, multiply by beta and back off to shorter context.
        3. If we back off all the way, use smoothed unigram probability.
        4. Cache results to avoid recomputing the same probability repeatedly.
        """
        # Map OOV to <UNK> if not seen during training
        token = token if token in self.vocab else self.UNK

        # Keep only last (n-1) context tokens and store as tuple so it can easily be used as dict key
        ctx_tokens = tuple(
            t if t in self.vocab else self.UNK
            for t in context_list[-(self.n - 1):]
        )

        key = (ctx_tokens, token)   # check cache to see if we computed this (context, token) in the past
        # Faster cache lookup (single dict access)
        cached = self._prob_cache.get(key)
        if cached is not None:
            return cached

        backoff_factor = 1.0    #keep track of beta multipliers

        # start from full order n down to bigram
        for k in range(self.n, 1, -1):
            need = k - 1    #how many context tokens are required
            if len(ctx_tokens) < need:  # if not enough context, skip this order
                continue

            ctx = ctx_tokens[-need:]
            gram = ctx + (token,)

            # check if this k-gram was seen in training
            num = self.counts_by_order[k].get(gram)
            if num:
                # if seen, compute max estimate
                den = self.context_counts_by_order[k].get(ctx, 0)
                if den > 0:
                    p = num / den
                    result = max(backoff_factor * p, 1e-12)
                    self._prob_cache[key] = result
                    return result
            # if unseen at this order, back off
            backoff_factor *= self.beta

        # Unigram fallback with add-alpha smoothing
        c = self.counts_by_order[1].get((token,), 0)
        num = c + self.unigram_alpha
        den = self.total_unigrams + self.unigram_alpha * self.vocab_size
        p1 = num / den if den > 0 else 0.0

        result = max(backoff_factor * p1, 1e-12)

        # cache unigram result
        self._prob_cache[key] = result

        return result


    def perplexity(self, eval_path: str) -> float:
        """
        Compute perplexity of the model on an evaluation file.
        Formula:
            PP = exp( - (1/N) * sum log P(w_t | context_t) )
        use the probability of the *ground-truth* next token.
        """
        log_sum = 0.0   # accumulate sum of log prob
        N = 0           # num of predicted tokens
        with open(eval_path, "rb") as f:    # read file line by line
            for line in f:
                toks = line.decode("utf-8", errors="ignore").strip().split()    #tokenize the line
                toks = [t if t in self.vocab else self.UNK for t in toks]       #replace oov with unk
                padded = self._pad(toks)

                for i in range(self.n - 1, len(padded)):        #loop over each position where we predict a next token
                    context = padded[i - (self.n - 1) : i]
                    gt = padded[i]  # ground-truth next token
                    p = self.prob(context, gt)  # P(gt | context)
                    log_sum += math.log(p)
                    N += 1
        # convert avg negative log probability back to normal space (normalization)
        return math.exp(-log_sum / N) if N > 0 else float("inf")
