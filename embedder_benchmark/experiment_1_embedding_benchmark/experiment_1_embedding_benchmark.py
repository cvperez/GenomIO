#!/usr/bin/env python3
"""
Experiment 1: Embedding Model Selection
Benchmarks 5 genomics-native embedding models on 5 bacterial CDS sequences
to determine which produces the most biologically meaningful vector space.

Run from repo root:
    python embedder_benchmark/experiment_1_embedding_benchmark/experiment_1_embedding_benchmark.py
"""
import os
import sys
import time
import logging
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_DIR = os.path.join(REPO_ROOT, "rag_corpus_uniform")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ── Sequence sources: (tag, filename, nth_qualifying_record, species_label) ─
# nth_qualifying_record: 1-based index among records with 300 <= len <= 900 bp
SEQUENCE_SOURCES = [
    ("seq1_TT", "GCF_000008125.1_ASM812v1_cds_from_genomic.fna",  1, 0),
    ("seq2_TT", "GCF_000008125.1_ASM812v1_cds_from_genomic.fna",  2, 0),
    ("seq3_CT", "GCF_000008725.1_ASM872v1_cds_from_genomic.fna",  1, 1),
    ("seq4_SA", "GCF_000009005.1_ASM900v1_cds_from_genomic.fna",  1, 2),
    ("seq5_DM", "GCF_000009025.1_ASM902v1_cds_from_genomic.fna",  1, 3),
]

AXIS_LABELS = [
    "T.thermophilus_1",
    "T.thermophilus_2",
    "C.trachomatis",
    "S.aureus",
    "D.mccartyi",
]

SPECIES_LABELS = [0, 0, 1, 2, 3]

SEQ_MIN_LEN = 300
SEQ_MAX_LEN = 900


# ── Sequence selection ──────────────────────────────────────────────────────

def select_sequences() -> list[dict]:
    """Parse rag_corpus_uniform FASTA files and return 5 CDS records."""
    results = []
    for tag, fname, nth, label in SEQUENCE_SOURCES:
        fpath = os.path.join(CORPUS_DIR, fname)
        count = 0
        found = None
        for rec in SeqIO.parse(fpath, "fasta"):
            seq = str(rec.seq).upper().strip()
            if SEQ_MIN_LEN <= len(seq) <= SEQ_MAX_LEN:
                count += 1
                if count == nth:
                    found = {
                        "tag":       tag,
                        "record_id": rec.id,
                        "sequence":  seq,
                        "length":    len(seq),
                        "label":     label,
                    }
                    break
        if found is None:
            raise RuntimeError(
                f"Could not find {nth}-th CDS in [{SEQ_MIN_LEN}, {SEQ_MAX_LEN}] bp "
                f"range in {fname}"
            )
        log.info(f"  {tag}: {found['record_id']}  ({found['length']} bp)")
        results.append(found)
    return results


# ── Shared mean pooling ─────────────────────────────────────────────────────

def mean_pool(last_hidden_state: torch.Tensor,
              attention_mask: torch.Tensor) -> np.ndarray:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    embedding = (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return embedding.squeeze(0).cpu().numpy()


# ── Embedding functions ─────────────────────────────────────────────────────

def embed_dnabert_s(sequences: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    model_name = "zhihan1996/DNABERT-S"
    log.info(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.use_flash_attn = False
    model = AutoModel.from_pretrained(
        model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False
    )
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            # DNABERT-S returns a plain tuple; index 0 is the last hidden state
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
    del model, tokenizer
    return np.stack(embeddings)


def embed_dnabert2(sequences: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    model_name = "zhihan1996/DNABERT-2-117M"
    log.info(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.use_flash_attn = False
    model = AutoModel.from_pretrained(
        model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False
    )
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
    del model, tokenizer
    return np.stack(embeddings)


def embed_nucleotide_transformer(sequences: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    model_name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    log.info(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True, output_hidden_states=True
    )
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
    del model, tokenizer
    return np.stack(embeddings)


def embed_nt_v2_100m(sequences: list[str]) -> np.ndarray:
    """NT v2 100M — fills the scale curve between 50M and 250M (all MLM-trained)."""
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    model_name = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
    log.info(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True, output_hidden_states=True
    )
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, output_hidden_states=True)
            embeddings.append(mean_pool(outputs.hidden_states[-1], inputs["attention_mask"]))
    del model, tokenizer
    return np.stack(embeddings)


def embed_nt_v2_250m(sequences: list[str]) -> np.ndarray:
    """NT v2 250M — third point on the NT v2 MLM scale curve."""
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    model_name = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"
    log.info(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True, output_hidden_states=True
    )
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, output_hidden_states=True)
            embeddings.append(mean_pool(outputs.hidden_states[-1], inputs["attention_mask"]))
    del model, tokenizer
    return np.stack(embeddings)


def embed_dnabert_1(sequences: list[str]) -> np.ndarray:
    """DNABERT original — 6-mer tokenisation, MLM. Predecessor of DNABERT-2 (BPE/MLM)
    and DNABERT-S (BPE/contrastive). Closes the historical arc of the DNABERT family."""
    from transformers import AutoTokenizer, AutoModel
    model_name = "armheb/DNA_bert_6"
    log.info(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            # DNABERT-1 expects space-separated 6-mers
            kmers = " ".join(seq[i:i+6] for i in range(len(seq) - 5))
            inputs = tokenizer(kmers, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
    del model, tokenizer
    return np.stack(embeddings)


def embed_evo2(sequences: list[str]) -> np.ndarray | None:
    try:
        from evo2 import Evo2
    except ImportError:
        log.warning("evo2 package not installed — skipping Evo2 7B.")
        return None
    log.info("Loading Evo2 7B ...")
    model = Evo2("evo2_7b")
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            output = model(seq, return_embeddings=True)
            hidden = output["embeddings"]["blocks.28.mlp.l3"]
            mask = torch.ones(1, hidden.shape[1])
            embeddings.append(mean_pool(hidden, mask))
    del model
    return np.stack(embeddings)


def _build_6mer_vocab() -> dict:
    """Build a {kmer: id} vocabulary with ACTG ordering matching MetaBERTa training."""
    special = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
    bases = ["A", "C", "T", "G"]
    kmers = [a+b+c+d+e+f for a in bases for b in bases for c in bases
             for d in bases for e in bases for f in bases]
    return {tok: i for i, tok in enumerate(special + kmers)}


def _encode_6mer(sequence: str, vocab: dict, kmer_size: int = 6,
                 max_length: int = 200) -> dict:
    """Overlapping k-mer tokenization returning input_ids and attention_mask tensors."""
    pad_id = vocab["[PAD]"]
    cls_id = vocab["[CLS]"]
    sep_id = vocab["[SEP]"]
    unk_id = vocab["[UNK]"]

    tokens = [cls_id]
    for i in range(len(sequence) - kmer_size + 1):
        tokens.append(vocab.get(sequence[i:i + kmer_size], unk_id))
    tokens.append(sep_id)

    if len(tokens) > max_length:
        tokens = tokens[:max_length - 1] + [sep_id]

    attention_mask = [1] * len(tokens)
    pad_len = max_length - len(tokens)
    tokens.extend([pad_id] * pad_len)
    attention_mask.extend([0] * pad_len)

    return {
        "input_ids":      torch.tensor([tokens]),
        "attention_mask": torch.tensor([attention_mask]),
    }


def embed_metaberta(sequences: list[str]) -> np.ndarray | None:
    vocab_file = os.path.join(REPO_ROOT, "MetaBERTa", "pipeline", "6mer_vocab.txt")
    if not os.path.isfile(vocab_file):
        log.warning(f"MetaBERTa/pipeline/6mer_vocab.txt not found — skipping MetaBERTa.")
        return None

    from transformers import AutoModel
    model_name = "MsAlEhR/MetaBerta-400-fragments-18k-genome"
    log.info(f"Loading {model_name} ...")
    vocab = _build_6mer_vocab()
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            encoded = _encode_6mer(seq, vocab, kmer_size=6, max_length=200)
            outputs = model(**encoded, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            embeddings.append(mean_pool(last_hidden, encoded["attention_mask"]))
    del model
    return np.stack(embeddings)


def _disable_caduceus_tie_weights():
    """Wrap (don't replace) tie_weights on any loaded Caduceus class to drop
    unsupported kwargs. Newer transformers calls model.tie_weights(missing_keys=...)
    but the cached Caduceus code defines tie_weights() with no kwargs. We still need
    the ORIGINAL tie_weights to run because Caduceus-PS shares weights between the
    forward and reverse strands — skipping it produces NaN at inference."""
    import sys
    patched = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or "caduceus" not in mod_name.lower():
            continue
        for name in dir(mod):
            obj = getattr(mod, name, None)
            if isinstance(obj, type) and "tie_weights" in obj.__dict__:
                orig = obj.tie_weights
                def _make(o):
                    def wrapped(self, *args, **kwargs):
                        return o(self)  # call original, drop unsupported kwargs
                    return wrapped
                obj.tie_weights = _make(orig)
                patched += 1
    if patched:
        log.info(f"  Wrapped tie_weights on {patched} Caduceus class(es)")
    return patched


def embed_caduceus(sequences: list[str]) -> np.ndarray | None:
    if not torch.cuda.is_available():
        log.warning("CUDA not available — skipping Caduceus.")
        return None
    log.info("  >>> CADUCEUS PATCH v6 ACTIVE <<<")
    from transformers import AutoTokenizer, AutoModel
    # Pre-load patch: PreTrainedModel base inherits the missing attribute
    try:
        import transformers.modeling_utils as _mu
        _mu.PreTrainedModel.all_tied_weights_keys = {}
    except Exception:
        pass
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    log.info(f"Loading {model_name} ...")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # First load attempt; on tie_weights signature mismatch, neutralise + retry
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    except TypeError as exc:
        msg = str(exc)
        if "tie_weights" in msg or "missing_keys" in msg:
            log.warning(f"  First load hit {msg}; disabling tie_weights and retrying")
            _disable_caduceus_tie_weights()
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        else:
            raise

    # Post-load layered patch — every level any code might look it up on.
    # 1) Walk class MRO and force-set on every base
    for cls in type(model).__mro__:
        try:
            cls.all_tied_weights_keys = {}
        except (TypeError, AttributeError):
            pass
    # 2) Every submodule instance gets it too
    for sub in model.modules():
        try:
            sub.all_tied_weights_keys = {}
        except Exception:
            pass
    # 3) Override __getattr__ on the Caduceus class so even if everything above
    #    somehow fails, the attribute lookup for this specific name returns {}.
    _cad_cls = type(model)
    _orig_getattr = _cad_cls.__getattr__ if hasattr(_cad_cls, "__getattr__") else None
    def _caduceus_getattr(self, name, _orig=_orig_getattr):
        if name == "all_tied_weights_keys":
            return {}
        if _orig is not None:
            return _orig(self, name)
        raise AttributeError(name)
    _cad_cls.__getattr__ = _caduceus_getattr

    log.info(f"  Post-patch hasattr check: {hasattr(model, 'all_tied_weights_keys')}")
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            mask = inputs.get("attention_mask",
                              torch.ones(inputs["input_ids"].shape, device=device))
            embeddings.append(mean_pool(last_hidden, mask))
    del model, tokenizer
    torch.cuda.empty_cache()
    stacked = np.stack(embeddings)
    # NaN sanitisation: if Caduceus produces NaN for some sequences, replace with 0
    n_nan = int(np.isnan(stacked).sum())
    if n_nan > 0:
        log.warning(f"  Caduceus output contained {n_nan} NaN values; replacing with 0")
        stacked = np.nan_to_num(stacked, nan=0.0)
    return stacked


def embed_hyenadna(sequences: list[str]) -> np.ndarray | None:
    if not torch.cuda.is_available():
        log.warning("CUDA not available — skipping HyenaDNA.")
        return None
    from transformers import AutoTokenizer, AutoModel
    model_name = "LongSafari/hyenadna-large-1m-seqlen-hf"
    log.info(f"Loading {model_name} ...")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True,
                              truncation=True, max_length=4096)
            if inputs["input_ids"].shape[1] >= 4096:
                log.warning("  HyenaDNA: sequence truncated to 4096 tokens")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            mask = inputs.get("attention_mask",
                              torch.ones(inputs["input_ids"].shape, device=device))
            embeddings.append(mean_pool(last_hidden, mask))
    del model, tokenizer
    torch.cuda.empty_cache()
    return np.stack(embeddings)


def embed_gena_lm(sequences: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel
    model_name = "AIRI-Institute/gena-lm-bigbird-base-t2t"
    log.info(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                       output_hidden_states=True)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, output_hidden_states=True)
            # CLS token from last layer
            cls = outputs.hidden_states[-1][:, 0, :]
            embeddings.append(cls.squeeze(0).cpu().numpy())
    del model, tokenizer
    return np.stack(embeddings)


def embed_nt_2500m(sequences: list[str]) -> np.ndarray | None:
    if not torch.cuda.is_available():
        log.warning("CUDA not available — skipping NT 2500M.")
        return None
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    model_name = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
    log.info(f"Loading {model_name} ...")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True, output_hidden_states=True,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            # Cast back to float32 for pooling stability
            last_hidden = outputs.hidden_states[-1].float()
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
    del model, tokenizer
    torch.cuda.empty_cache()
    return np.stack(embeddings)


def embed_nt_v2_50m(sequences: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    log.info(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True, output_hidden_states=True
    )
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
    del model, tokenizer
    return np.stack(embeddings)


def embed_germ(sequences: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel
    log.info("Loading GERM ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )
    germ_paths = [
        os.path.join(REPO_ROOT, "MetaBERTa", "GERM"),
        os.path.join(REPO_ROOT, "germ_checkpoint"),
        os.path.join(REPO_ROOT, "GERM"),
    ]
    germ_path = next((p for p in germ_paths if os.path.isdir(p)), None)
    if germ_path is None:
        log.warning("GERM checkpoint not found locally; falling back to DNABERT-2 base weights.")
        model = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
    else:
        log.info(f"  Using GERM checkpoint at {germ_path}")
        model = AutoModel.from_pretrained(germ_path, trust_remote_code=True)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
    del model, tokenizer
    return np.stack(embeddings)


def embed_splicebert(sequences: list[str]) -> np.ndarray | None:
    try:
        import multimolecule  # noqa: F401 — registers the SpliceBERT model + tokenizer
    except ImportError:
        log.warning("multimolecule not installed — skipping SpliceBERT. "
                    "Install with: pip install multimolecule")
        return None
    from transformers import AutoTokenizer, AutoModel
    model_name = "multimolecule/splicebert"
    log.info(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            # SpliceBERT's tokenizer expects nucleotides separated by spaces
            spaced = " ".join(seq)
            inputs = tokenizer(spaced, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
    del model, tokenizer
    return np.stack(embeddings)


def embed_grover(sequences: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel
    model_name = "PoetschLab/GROVER"
    log.info(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True,
                              truncation=True, max_length=1024)
            if inputs["input_ids"].shape[1] >= 1024:
                log.warning("  GROVER: sequence truncated to 1024 tokens")
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
    del model, tokenizer
    return np.stack(embeddings)


def embed_aido_dna_7b(sequences: list[str]) -> np.ndarray | None:
    """AIDO.DNA-7B from GenBio AI (replaces unavailable DNAGRINDER). GPU required."""
    if not torch.cuda.is_available():
        log.warning("CUDA not available — skipping AIDO.DNA-7B.")
        return None
    from transformers import AutoTokenizer, AutoModel
    model_name = "genbio-ai/AIDO.DNA-7B"
    log.info(f"Loading {model_name} ...")
    device = torch.device("cuda")
    # Robust tokenizer loading — log the full exception when all paths fail
    import traceback
    tokenizer = None
    for attempt_kwargs, label in [
        ({"trust_remote_code": True},                  "default"),
        ({"trust_remote_code": True, "use_fast": False}, "use_fast=False"),
        ({},                                              "no trust_remote_code"),
    ]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **attempt_kwargs)
            log.info(f"  Tokenizer loaded with {label}")
            break
        except Exception as exc:
            log.warning(f"  Tokenizer {label} failed:")
            log.warning("  " + traceback.format_exc().replace("\n", "\n  ")[:1500])
    if tokenizer is None:
        log.warning("  All tokenizer attempts failed — skipping AIDO.DNA-7B")
        return None
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            last_hidden = last_hidden.float()
            mask = inputs.get("attention_mask",
                              torch.ones(inputs["input_ids"].shape, device=device))
            embeddings.append(mean_pool(last_hidden, mask))
    del model, tokenizer
    torch.cuda.empty_cache()
    return np.stack(embeddings)


def embed_nt_v2_50m_3mer(sequences: list[str]) -> np.ndarray:
    """NT v2 50M 3mer (the 500M-3mer variant does not exist on HF Hub).
    Used to isolate the effect of 3mer tokenisation vs 6mer at matched scale."""
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species"
    log.info(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True, output_hidden_states=True
    )
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
    del model, tokenizer
    return np.stack(embeddings)


def embed_evo2_1b(sequences: list[str]) -> np.ndarray | None:
    if not torch.cuda.is_available():
        log.warning("CUDA not available — skipping Evo2 1B.")
        return None
    try:
        from evo2 import Evo2
    except ImportError:
        log.warning("evo2 package not installed — skipping Evo2 1B.")
        return None
    log.info("Loading Evo2 1B ...")
    device = torch.device("cuda")
    model = Evo2("evo2_1b")
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            if len(seq) > 4096:
                log.warning(f"  Evo2 1B: sequence truncated from {len(seq)} to 4096 chars")
                seq = seq[:4096]
            input_ids = torch.tensor(model.tokenizer.tokenize(seq),
                                     dtype=torch.int).unsqueeze(0).to(device)
            _, token_embs = model(input_ids, return_embeddings=True,
                                  layer_names=["blocks.20.mlp.l3"])
            emb = torch.mean(token_embs["blocks.20.mlp.l3"], dim=1).squeeze()
            embeddings.append(emb.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.stack(embeddings)


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(embeddings: np.ndarray, labels: list[int]) -> dict:
    sim_matrix = cosine_similarity(embeddings)

    intra_sim = float(sim_matrix[0, 1])

    cross_sims = [
        sim_matrix[i, j]
        for i in range(5)
        for j in range(i + 1, 5)
        if labels[i] != labels[j]
    ]
    inter_dist = float(1.0 - np.mean(cross_sims))

    sil = float(silhouette_score(embeddings, labels))

    return {
        "sim_matrix": sim_matrix,
        "intra_sim":  intra_sim,
        "inter_dist": inter_dist,
        "silhouette": sil,
    }


# ── Visualisation ───────────────────────────────────────────────────────────

def plot_heatmap(sim_matrix: np.ndarray, model_short_name: str,
                 axis_labels: list[str], output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(f"Cosine Similarity Matrix — {model_short_name}", fontsize=12, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"experiment_1_{model_short_name}_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Heatmap saved: {out_path}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="Comma-separated MODEL_REGISTRY keys to run (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run models even if they already have a CSV row")
    args = ap.parse_args()
    keep = set(args.only.split(",")) if args.only else None

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=== Selecting 5 CDS sequences from rag_corpus_uniform ===")
    seq_records = select_sequences()
    sequences = [r["sequence"]  for r in seq_records]
    labels    = [r["label"]     for r in seq_records]

    print("\nSequences selected:")
    for r in seq_records:
        print(f"  {r['tag']:12s}  {r['record_id']}  ({r['length']} bp)")

    MODEL_REGISTRY = [
        ("DNABERT_S",         embed_dnabert_s,                768),
        ("DNABERT_2",         embed_dnabert2,                 768),
        ("DNABERT_1",         embed_dnabert_1,                768),
        ("NT_v2_500M",        embed_nucleotide_transformer,  1024),
        ("NT_v2_250M",        embed_nt_v2_250m,              1024),
        ("NT_v2_100M",        embed_nt_v2_100m,               512),
        ("NT_v2_50M",         embed_nt_v2_50m,                512),
        ("NT_v2_50M_3mer",    embed_nt_v2_50m_3mer,           512),
        ("NT_2500M",          embed_nt_2500m,                2560),
        ("MetaBERTa",         embed_metaberta,                None),
        ("Caduceus",          embed_caduceus,                 256),
        ("HyenaDNA",          embed_hyenadna,                 256),
        ("GENA_LM",           embed_gena_lm,                  768),
        ("GERM",              embed_germ,                     768),
        ("SpliceBERT",        embed_splicebert,               512),
        ("GROVER",            embed_grover,                   768),
    ]

    registry = [t for t in MODEL_REGISTRY if (keep is None or t[0] in keep)]

    csv_path = os.path.join(OUTPUT_DIR, "experiment_1_results.csv")
    existing_models: set[str] = set()
    existing_rows: list[dict] = []
    if os.path.isfile(csv_path):
        existing_df = pd.read_csv(csv_path)
        existing_rows = existing_df.to_dict("records")
        log.info(f"Loaded {len(existing_df)} existing model rows from {csv_path}")
        if not args.force:
            existing_models = set(existing_df["model"].tolist())

    all_results = []

    for short_name, embed_fn, expected_dim in registry:
        if short_name in existing_models:
            log.info(f"  {short_name}: already has results, skipping (use --force to re-run)")
            continue
        print(f"\n{'='*60}")
        print(f"Model: {short_name}")
        print(f"{'='*60}")

        t_start = time.perf_counter()
        try:
            embeddings = embed_fn(sequences)
        except Exception as exc:
            log.warning(f"  {short_name} failed: {exc}")
            embeddings = None
        elapsed = time.perf_counter() - t_start

        if embeddings is None:
            print(f"  SKIPPED.")
            continue

        actual_dim   = embeddings.shape[1]
        time_per_seq = elapsed / len(sequences)

        metrics = compute_metrics(embeddings, labels)
        plot_heatmap(metrics["sim_matrix"], short_name, AXIS_LABELS, OUTPUT_DIR)

        row = {
            "model":                   short_name,
            "embedding_dim":           actual_dim,
            "intra_species_sim":       round(metrics["intra_sim"],  4),
            "inter_species_dist":      round(metrics["inter_dist"], 4),
            "silhouette_score":        round(metrics["silhouette"], 4),
            "inference_time_per_seq":  round(time_per_seq, 3),
            "total_inference_time_s":  round(elapsed, 3),
        }
        all_results.append(row)

        print(f"  embedding_dim          : {actual_dim}")
        print(f"  intra_species_sim      : {row['intra_species_sim']}")
        print(f"  inter_species_dist     : {row['inter_species_dist']}")
        print(f"  silhouette_score       : {row['silhouette_score']}")
        print(f"  inference_time_per_seq : {row['inference_time_per_seq']} s")

    if not all_results and not existing_rows:
        log.error("No models produced results.")
        return

    if not all_results:
        log.info("No new models ran; existing CSV unchanged.")
        return

    # Merge new results with existing rows; new rows win if --force re-ran a key
    df = pd.DataFrame(existing_rows + all_results)
    df = df.drop_duplicates(subset=["model"], keep="last")

    print("\n\n=== COMPARISON TABLE ===")
    print(df.to_string(index=False))

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
