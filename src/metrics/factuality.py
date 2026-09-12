"""Factuality: is each generated sentence supported by the entity's page?

The pipeline the proposal omitted. "Verify each sentence vs. the Wikipedia
page" is impossible as written — an NLI cross-encoder takes 512 tokens and a
biography is thousands, so feeding the page silently truncates it to the lead
section and every later claim scores unsupported. The missing step is evidence
retrieval (05 §4):

    page -> 3-sentence windows -> retrieve top-m -> NLI each -> max -> threshold

What this measures is "supported by this reference document", NOT "true".
A true fact absent from the page scores unsupported. The metric is therefore
biased downward, and the comparison rests on that bias being the same for
every arm — which the manual validation exists to check (05 §6).
"""
from __future__ import annotations

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.device import pick_device
from src.metrics.diversity import tokenize_words   # same word tokenizer as the
                                                   # diversity metrics, deliberately
from src.text import split_sentences

# DeBERTa-v3-large trained on MNLI + FEVER + ANLI + LingNLI + WANLI. FEVER is
# fact verification against Wikipedia — the closest available match to this task.
NLI_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
RETRIEVER_NAME = "sentence-transformers/all-mpnet-base-v2"

WINDOW = 3        # sentences per evidence chunk: biographical facts split across
                  # adjacent sentences ("She was born in London. Her father was...")
TOP_M = 5         # premises per claim, 3 from BM25 + 3 from dense, deduplicated
THETA = 0.5       # provisional; tuned on dev against manual labels (05 §6.5)
MAX_LEN = 512     # the cross-encoder's hard limit — the reason retrieval exists


# ------------------------------------------------------------------- loading

def load_nli(name: str = NLI_NAME):
    """Returns (tokenizer, model, entail_id).

    entail_id is read from config.id2label and never hard-coded. Measured
    across five MNLI checkpoints, entailment sits at index 0, 0, 2, 1 and 2 —
    so a hard-coded index gives you a different label's probability on most of
    them, with no error and no crash.
    """
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, dtype=torch.float16).to(pick_device()).eval()
    label2id = {v.lower(): k for k, v in model.config.id2label.items()}
    if "entailment" not in label2id:
        raise ValueError(f"no 'entailment' label in {model.config.id2label}")
    return tok, model, label2id["entailment"]


def load_retriever(name: str = RETRIEVER_NAME) -> SentenceTransformer:
    return SentenceTransformer(name, device=pick_device())


# ----------------------------------------------------------------- retrieval

def build_index(page_text: str, retriever, window: int = WINDOW) -> dict:
    """Chunk one Wikipedia page into overlapping windows and index it.

    Built ONCE per entity and cached: embedding 100 pages takes seconds,
    embedding them 8,100 times takes an afternoon.
    """
    sents = split_sentences(page_text)
    step = max(1, window - 1)                      # overlap by one sentence
    chunks = [" ".join(sents[i:i + window]) for i in range(0, len(sents), step)]
    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        return {"chunks": [], "bm25": None, "emb": None}
    return {
        "chunks": chunks,
        "bm25": BM25Okapi([tokenize_words(c) for c in chunks]),
        "emb": retriever.encode(chunks, normalize_embeddings=True,
                                show_progress_bar=False),
    }


def retrieve(claim: str, index: dict, retriever, m: int = TOP_M) -> list[str]:
    """Hybrid: union of BM25 top-k and dense top-k.

    Sparse wins on names, dates and numbers — most of a biography. Dense wins
    on paraphrase. The union is cheap and strictly more robust than either
    (05 §4.6, and the Retrieval lecture's hybrid slide).
    """
    chunks = index["chunks"]
    if not chunks:
        return []
    k = max(1, m // 2 + 1)

    lex = index["bm25"].get_scores(tokenize_words(claim))
    lex_top = list(np.argsort(lex)[::-1][:k])

    q = retriever.encode(claim, normalize_embeddings=True, show_progress_bar=False)
    dense_top = list(np.argsort(index["emb"] @ q)[::-1][:k])

    seen, out = set(), []
    for i in lex_top + dense_top:                  # dedup, preserve order
        if i not in seen:
            seen.add(i)
            out.append(chunks[i])
    return out[:m]


# ----------------------------------------------------------------------- NLI

@torch.no_grad()
def entail_probs(premises: list[str], hypotheses: list[str],
                 tok, model, entail_id: int, batch_size: int = 32,
                 cache: dict | None = None) -> list[float]:
    """P(entailment) per (premise, hypothesis) pair.

    Three things here are correctness, not style:
      - tok(premises, hypotheses) builds PAIRS. One concatenated string does
        not, and would silently produce nonsense.
      - truncation="only_first" cuts the PREMISE. Cutting the hypothesis would
        mean scoring a claim you never fully showed the model.
      - NLI is directional: premise=evidence, hypothesis=claim. Swapped, the
        question becomes "does this sentence prove Wikipedia?" — almost always
        neutral, and the metric collapses.
    """
    # Degenerate arms repeat the same sentence dozens of times, so the same
    # (premise, hypothesis) pair recurs constantly. Caching is the difference
    # between scoring greedy once and scoring it fifty times.
    todo = [i for i, (p_, h_) in enumerate(zip(premises, hypotheses))
            if cache is None or (p_, h_) not in cache]

    fresh: dict[int, float] = {}
    for i in range(0, len(todo), batch_size):
        idx = todo[i:i + batch_size]
        enc = tok([premises[j] for j in idx], [hypotheses[j] for j in idx],
                  return_tensors="pt", padding=True,
                  truncation="only_first", max_length=MAX_LEN).to(model.device)
        p = model(**enc).logits.float().softmax(-1)[:, entail_id].tolist()
        for j, v in zip(idx, p):
            fresh[j] = v
            if cache is not None:
                cache[(premises[j], hypotheses[j])] = v

    return [fresh[i] if i in fresh else cache[(premises[i], hypotheses[i])]
            for i in range(len(premises))]


# ------------------------------------------------------------------- scoring

def score_generation(text: str, entity: str, index: dict, retriever,
                     tok, model, entail_id: int,
                     theta: float = THETA, m: int = TOP_M,
                     cache: dict | None = None) -> dict:
    """Support rate for one generation, plus the detail needed for error analysis.

    Returns None-valued fields for degenerate generations (fewer than 2
    sentences), which are excluded from the denominator and counted per arm
    rather than scored 0 or 1 on a single label.
    """
    sents = split_sentences(text or "")
    if len(sents) < 2:
        return {"n_sentences": len(sents), "n_supported": None,
                "support_rate": None, "degenerate": True, "sentences": []}

    premises, hypotheses, spans = [], [], []
    for s in sents:
        prem = retrieve(s, index, retriever, m=m)
        # Prepend the entity: generations say "She was born...", the page says
        # "Ada Lovelace was born...". Cheap coreference mitigation, applied
        # identically to every arm (05 §2.8).
        hyp = f"{entity}: {s}"
        spans.append((len(premises), len(premises) + len(prem)))
        premises.extend(prem)
        hypotheses.extend([hyp] * len(prem))

    probs = entail_probs(premises, hypotheses, tok, model, entail_id,
                         cache=cache)

    details, n_sup = [], 0
    for s, (a, b) in zip(sents, spans):
        # MAX over premises, never mean: a claim is supported if ANY part of
        # the page supports it. The mean would punish claims supported by
        # exactly one sentence, which is the normal case.
        best = max(probs[a:b], default=0.0)
        sup = best >= theta
        n_sup += int(sup)
        details.append({"sentence": s, "p_entail": round(best, 4),
                        "supported": bool(sup),
                        "evidence": premises[a:b][int(np.argmax(probs[a:b]))]
                                    if b > a else None})

    return {"n_sentences": len(sents), "n_supported": n_sup,
            "support_rate": n_sup / len(sents), "degenerate": False,
            "sentences": details}