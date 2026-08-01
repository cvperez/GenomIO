# rag/dnabert_s.py
"""
DNABERT-S sequence embedder.

DNABERT-S (zhihan1996/DNABERT-S) is the DNA encoder selected by the embedder benchmark
in embedder_benchmark/. It is a 110M parameter BERT trained with a species aware
contrastive objective, which is the property that matters here: it pulls sequences from
the same organism together and pushes different organisms apart. Masked language models
of the same or much larger size do not do that, however good they are at predicting
nucleotides. See embedder_benchmark/experiment_2_results.md for the ranking.

The loading recipe below has three details that are easy to get wrong and produce either
a crash or silently wrong vectors:

  1. use_flash_attn must be set to False on the config before from_pretrained. The
     published remote code defaults to flash attention, which needs a CUDA build.
  2. low_cpu_mem_usage must be passed explicitly as False.
  3. The model returns a plain tuple, not a ModelOutput, so the last hidden state is
     out[0] rather than out.last_hidden_state.

Embeddings are mean pooled over the attention mask, not taken from CLS.
"""

import numpy as np
import torch

MODEL_NAME = "zhihan1996/DNABERT-S"
EMBEDDING_DIM = 768

# DNABERT-S declares model_max_length as the "unlimited" sentinel, so truncation=True on
# its own truncates nothing and a long record would run past max_position_embeddings.
# The limit has to be stated explicitly.
MAX_TOKENS = 512

_tokenizer = None
_model = None


def _load():
    """Load tokenizer and model once per process and keep them resident."""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    from transformers import AutoConfig, AutoModel, AutoTokenizer

    print(f"[INFO] Loading {MODEL_NAME} (first call downloads it from HuggingFace)")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.use_flash_attn = False
    _model = AutoModel.from_pretrained(
        MODEL_NAME,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    _model.eval()
    return _tokenizer, _model


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed(sequences, batch_size=16, progress_every=0):
    """
    Embed DNA sequences into a (len(sequences), 768) float32 array.

    Sequences longer than MAX_TOKENS worth of tokens are truncated, which for this
    tokenizer is roughly 2,000 bp. Most CDS records are well under that.

    progress_every: print a line every N sequences, 0 to stay quiet. Used by the index
    build, which runs for tens of minutes over the full corpus.
    """
    if not sequences:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    tokenizer, model = _load()
    cleaned = [str(s).upper() for s in sequences]

    # Every sequence in a batch is padded to the longest one in it, and the padding is
    # still pushed through the network. Corpus records run from under 300 to over 3,000 bp,
    # so batching them in input order wastes most of the compute on padding. Grouping
    # similar lengths together and restoring the original order afterwards costs nothing
    # and is a large speedup on the full corpus build.
    order = sorted(range(len(cleaned)), key=lambda i: len(cleaned[i]))
    embedded = np.zeros((len(cleaned), EMBEDDING_DIM), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            positions = order[start:start + batch_size]
            enc = tokenizer(
                [cleaned[i] for i in positions],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
            )
            result = model(**enc)
            hidden = result[0] if isinstance(result, tuple) else result.last_hidden_state
            embedded[positions] = _mean_pool(hidden, enc["attention_mask"]).cpu().numpy()

            done = min(start + batch_size, len(order))
            if progress_every and done % progress_every < batch_size:
                print(f"[INFO] DNABERT-S: {done}/{len(order)} embedded")

    return embedded


def embed_one(sequence):
    """Embed a single sequence into a (768,) float32 vector."""
    return embed([sequence])[0]
