from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple, Set


class NGramModel:
    """
    Count-based n-gram language model with add-alpha smoothing.
    Builds vocabulary from training set
    Maps unseen tokens to <UNK> at eval time
    """

    def __init__(self, n: int, alpha: float = 0.1):
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n
        self.alpha = alpha

        self.ngram_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.context_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.vocab: Set[str] = set()
        self.vocab_size: int = 0

        self.BOS = "<s>"
        self.EOS = "</s>"
        self.UNK = "<UNK>"

    def _pad(self, tokens: List[str]) -> List[str]:
        return [self.BOS] * (self.n - 1) + tokens + [self.EOS]

    def train_from_file(self, train_path: str) -> None:
        # first pass through data lets us build vocab from training
        with open(train_path, "rb") as f:
            for line in f:
                toks = line.decode("utf-8", errors="ignore").strip().split()
                for t in toks:
                    self.vocab.add(t)

        # adding special tokens to vocab
        self.vocab.update([self.BOS, self.EOS, self.UNK])
        self.vocab_size = len(self.vocab)

        # in the second pass count ngrams
        with open(train_path, "rb") as f:
            for line in f:
                toks = line.decode("utf-8", errors="ignore").strip().split()
                toks = [t if t in self.vocab else self.UNK for t in toks]
                toks = self._pad(toks)

                for i in range(self.n - 1, len(toks)):
                    context = tuple(toks[i - (self.n - 1): i])
                    ngram = context + (toks[i],)

                    self.ngram_counts[ngram] += 1
                    self.context_counts[context] += 1

    def prob(self, context: List[str], token: str) -> float:
        # map OOV to UNK
        token = token if token in self.vocab else self.UNK
        ctx = tuple([t if t in self.vocab else self.UNK for t in context])

        ngram = ctx + (token,)

        num = self.ngram_counts[ngram] + self.alpha
        den = self.context_counts[ctx] + self.alpha * self.vocab_size
        return num / den

    def perplexity(self, eval_path: str) -> float:
        """
        use probability of the ground-truth next token. :contentReference[oaicite:1]{index=1}
        """
        log_sum = 0.0
        N = 0

        with open(eval_path, "rb") as f:
            for line in f:
                toks = line.decode("utf-8", errors="ignore").strip().split()
                toks = [t if t in self.vocab else self.UNK for t in toks]
                toks = self._pad(toks)

                for i in range(self.n - 1, len(toks)):
                    context = toks[i - (self.n - 1): i]
                    gt = toks[i]        # ground truth next token
                    p = self.prob(context, gt)      # probability of ground truth token aka P(ground_truth_token | context)

                    log_sum += math.log(p)
                    N += 1

        return math.exp(-log_sum / N) if N > 0 else float("inf")
