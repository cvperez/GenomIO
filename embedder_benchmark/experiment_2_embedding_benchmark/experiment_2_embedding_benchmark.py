#!/usr/bin/env python3
"""
Experiment 2: Scaled Embedding Benchmark
Repeats Experiment 1 at scale: 50 CDS sequences × 20 species = 1,000 sequences.
Evaluates the same 4 embedding models with statistically meaningful sample sizes.

Run from repo root:
    python embedder_benchmark/experiment_2_embedding_benchmark/experiment_2_embedding_benchmark.py
"""
import glob
import os
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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_DIR = os.path.join(REPO_ROOT, "rag_corpus_uniform")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SEQS_PER_SPECIES = 50
SEQ_MIN_LEN = 300
SEQ_MAX_LEN = 900


# ── Sequence selection ──────────────────────────────────────────────────────

def select_sequences() -> tuple[list[dict], list[str]]:
    """Parse all .fna files in rag_corpus_uniform, return 50 records per species."""
    fna_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.fna")))
    if not fna_files:
        raise RuntimeError(f"No .fna files found in {CORPUS_DIR}")

    records = []
    species_tags = []

    for label, fpath in enumerate(fna_files):
        basename = os.path.basename(fpath)
        tag = basename.split("_cds")[0]
        species_tags.append(tag)
        count = 0
        for rec in SeqIO.parse(fpath, "fasta"):
            seq = str(rec.seq).upper().strip()
            if SEQ_MIN_LEN <= len(seq) <= SEQ_MAX_LEN:
                records.append({"sequence": seq, "label": label, "species": tag})
                count += 1
                if count == SEQS_PER_SPECIES:
                    break
        if count < SEQS_PER_SPECIES:
            log.warning(f"  {tag}: only {count} qualifying records (wanted {SEQS_PER_SPECIES})")
        else:
            log.info(f"  [{label:02d}] {tag}: {count} records selected")

    return records, species_tags


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
        for i, seq in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  DNABERT-S: {i+1}/{len(sequences)} sequences embedded")
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
        for i, seq in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  DNABERT-2: {i+1}/{len(sequences)} sequences embedded")
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
        for i, seq in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  NT v2 500M: {i+1}/{len(sequences)} sequences embedded")
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
        for i, seq in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, output_hidden_states=True)
            embeddings.append(mean_pool(outputs.hidden_states[-1], inputs["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  NT v2 100M: {i+1}/{len(sequences)} sequences embedded")
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
        for i, seq in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, output_hidden_states=True)
            embeddings.append(mean_pool(outputs.hidden_states[-1], inputs["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  NT v2 250M: {i+1}/{len(sequences)} sequences embedded")
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
        for i, seq in enumerate(sequences):
            kmers = " ".join(seq[j:j+6] for j in range(len(seq) - 5))
            inputs = tokenizer(kmers, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  DNABERT-1: {i+1}/{len(sequences)} sequences embedded")
    del model, tokenizer
    return np.stack(embeddings)


def _build_6mer_vocab() -> dict:
    special = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
    bases = ["A", "C", "T", "G"]
    kmers = [a+b+c+d+e+f for a in bases for b in bases for c in bases
             for d in bases for e in bases for f in bases]
    return {tok: i for i, tok in enumerate(special + kmers)}


def _encode_6mer(sequence: str, vocab: dict, kmer_size: int = 6,
                 max_length: int = 200) -> dict:
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
        log.warning("MetaBERTa/pipeline/6mer_vocab.txt not found — skipping MetaBERTa.")
        return None

    from transformers import AutoModel
    model_name = "MsAlEhR/MetaBerta-400-fragments-18k-genome"
    log.info(f"Loading {model_name} ...")
    vocab = _build_6mer_vocab()
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            encoded = _encode_6mer(seq, vocab, kmer_size=6, max_length=200)
            outputs = model(**encoded, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            embeddings.append(mean_pool(last_hidden, encoded["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  MetaBERTa: {i+1}/{len(sequences)} sequences embedded")
    del model
    return np.stack(embeddings)


def _disable_caduceus_tie_weights():
    """Wrap (don't replace) tie_weights — see experiment_1 for full rationale."""
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
                        return o(self)
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
    try:
        import transformers.modeling_utils as _mu
        _mu.PreTrainedModel.all_tied_weights_keys = {}
    except Exception:
        pass
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    log.info(f"Loading {model_name} ...")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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
    for cls in type(model).__mro__:
        try:
            cls.all_tied_weights_keys = {}
        except (TypeError, AttributeError):
            pass
    for sub in model.modules():
        try:
            sub.all_tied_weights_keys = {}
        except Exception:
            pass
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
        for i, seq in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            mask = inputs.get("attention_mask",
                              torch.ones(inputs["input_ids"].shape, device=device))
            embeddings.append(mean_pool(last_hidden, mask))
            if (i + 1) % 100 == 0:
                log.info(f"  Caduceus: {i+1}/{len(sequences)} sequences embedded")
    del model, tokenizer
    torch.cuda.empty_cache()
    stacked = np.stack(embeddings)
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
    n_truncated = 0
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt", padding=True,
                              truncation=True, max_length=4096)
            if inputs["input_ids"].shape[1] >= 4096:
                n_truncated += 1
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            mask = inputs.get("attention_mask",
                              torch.ones(inputs["input_ids"].shape, device=device))
            embeddings.append(mean_pool(last_hidden, mask))
            if (i + 1) % 100 == 0:
                log.info(f"  HyenaDNA: {i+1}/{len(sequences)} sequences embedded")
    if n_truncated > 0:
        log.warning(f"  HyenaDNA: {n_truncated}/{len(sequences)} sequences truncated to 4096 tokens")
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
        for i, seq in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, output_hidden_states=True)
            cls = outputs.hidden_states[-1][:, 0, :]
            embeddings.append(cls.squeeze(0).cpu().numpy())
            if (i + 1) % 100 == 0:
                log.info(f"  GENA-LM: {i+1}/{len(sequences)} sequences embedded")
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
        for i, seq in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1].float()
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  NT 2500M: {i+1}/{len(sequences)} sequences embedded")
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
        for i, seq in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  NT v2 50M: {i+1}/{len(sequences)} sequences embedded")
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
        for i, seq in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  GERM: {i+1}/{len(sequences)} sequences embedded")
    del model, tokenizer
    return np.stack(embeddings)


def embed_splicebert(sequences: list[str]) -> np.ndarray | None:
    try:
        import multimolecule  # noqa: F401 — registers SpliceBERT model + tokenizer
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
        for i, seq in enumerate(sequences):
            spaced = " ".join(seq)
            inputs = tokenizer(spaced, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  SpliceBERT: {i+1}/{len(sequences)} sequences embedded")
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
    n_truncated = 0
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt", padding=True,
                              truncation=True, max_length=1024)
            if inputs["input_ids"].shape[1] >= 1024:
                n_truncated += 1
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  GROVER: {i+1}/{len(sequences)} sequences embedded")
    if n_truncated > 0:
        log.warning(f"  GROVER: {n_truncated}/{len(sequences)} sequences truncated to 1024 tokens")
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
        for i, seq in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            last_hidden = last_hidden.float()
            mask = inputs.get("attention_mask",
                              torch.ones(inputs["input_ids"].shape, device=device))
            embeddings.append(mean_pool(last_hidden, mask))
            if (i + 1) % 100 == 0:
                log.info(f"  AIDO.DNA-7B: {i+1}/{len(sequences)} sequences embedded")
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
        for i, seq in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            embeddings.append(mean_pool(last_hidden, inputs["attention_mask"]))
            if (i + 1) % 100 == 0:
                log.info(f"  NT v2 50M 3mer: {i+1}/{len(sequences)} sequences embedded")
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
    n_truncated = 0
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            if len(seq) > 4096:
                n_truncated += 1
                seq = seq[:4096]
            input_ids = torch.tensor(model.tokenizer.tokenize(seq),
                                     dtype=torch.int).unsqueeze(0).to(device)
            _, token_embs = model(input_ids, return_embeddings=True,
                                  layer_names=["blocks.20.mlp.l3"])
            emb = torch.mean(token_embs["blocks.20.mlp.l3"], dim=1).squeeze()
            embeddings.append(emb.float().cpu().numpy())
            if (i + 1) % 100 == 0:
                log.info(f"  Evo2 1B: {i+1}/{len(sequences)} sequences embedded")
    if n_truncated > 0:
        log.warning(f"  Evo2 1B: {n_truncated}/{len(sequences)} sequences truncated to 4096 chars")
    del model
    torch.cuda.empty_cache()
    return np.stack(embeddings)


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(embeddings: np.ndarray, labels: list[int]) -> dict:
    N = len(labels)
    labels_arr = np.array(labels)

    log.info("  Computing cosine similarity matrix ...")
    sim_matrix = cosine_similarity(embeddings)

    # Upper-triangle pair indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    sim_vals = sim_matrix[i_idx, j_idx]
    intra_mask = labels_arr[i_idx] == labels_arr[j_idx]

    intra_sim  = float(sim_vals[intra_mask].mean())
    inter_dist = float(1.0 - sim_vals[~intra_mask].mean())

    log.info("  Computing silhouette score ...")
    sil = float(silhouette_score(embeddings, labels))

    # 20×20 species-mean similarity matrix
    n_species = len(set(labels))
    species_sim = np.zeros((n_species, n_species))
    for si in range(n_species):
        for sj in range(n_species):
            idx_i = np.where(labels_arr == si)[0]
            idx_j = np.where(labels_arr == sj)[0]
            species_sim[si, sj] = sim_matrix[np.ix_(idx_i, idx_j)].mean()

    return {
        "sim_matrix":  sim_matrix,
        "species_sim": species_sim,
        "intra_sim":   intra_sim,
        "inter_dist":  inter_dist,
        "silhouette":  sil,
    }


# ── Visualizations ──────────────────────────────────────────────────────────

def plot_species_heatmap(species_sim: np.ndarray, model_name: str,
                         species_tags: list[str], output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 10))
    short_tags = [t.split("_")[0] + "_" + t.split("_")[1] for t in species_tags]
    sns.heatmap(
        species_sim,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        xticklabels=short_tags,
        yticklabels=short_tags,
        ax=ax,
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_title(f"Species-Mean Cosine Similarity — {model_name}", fontsize=12, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"experiment_2_{model_name}_species_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Species heatmap saved: {out_path}")
    return out_path


def plot_pca(embeddings: np.ndarray, labels: list[int],
             species_tags: list[str], model_name: str, output_dir: str) -> str:
    coords = PCA(n_components=2).fit_transform(embeddings)
    labels_arr = np.array(labels)
    n_species = len(species_tags)
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(12, 8))
    short_tags = [t.split("_")[0] + "_" + t.split("_")[1] for t in species_tags]
    for sp in range(n_species):
        mask = labels_arr == sp
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=20, alpha=0.6,
            color=cmap(sp / n_species),
            label=short_tags[sp],
        )
    ax.set_title(f"PCA of Embeddings (2D) — {model_name}", fontsize=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="upper right", fontsize=6, ncol=2, markerscale=1.5)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"experiment_2_{model_name}_pca.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  PCA scatter saved: {out_path}")
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

    log.info("=== Selecting sequences from rag_corpus_uniform ===")
    seq_records, species_tags = select_sequences()
    sequences = [r["sequence"] for r in seq_records]
    labels    = [r["label"]    for r in seq_records]

    n_species = len(species_tags)
    print(f"\n{len(sequences)} sequences selected across {n_species} species "
          f"({SEQS_PER_SPECIES} per species, {SEQ_MIN_LEN}–{SEQ_MAX_LEN} bp)")

    # Save labels once per run (same for every model)
    labels_path = os.path.join(OUTPUT_DIR, "experiment_2_labels.npy")
    np.save(labels_path, np.array(labels, dtype=np.int32))
    log.info(f"Saved labels to {labels_path}")

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

    csv_path = os.path.join(OUTPUT_DIR, "experiment_2_results.csv")
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
            print("  SKIPPED.")
            continue

        actual_dim   = embeddings.shape[1]
        time_per_seq = elapsed / len(sequences)

        # Save embeddings for downstream UMAP plotting (PART 3)
        emb_path = os.path.join(OUTPUT_DIR, f"experiment_2_embeddings_{short_name}.npy")
        np.save(emb_path, embeddings.astype(np.float32))
        log.info(f"  Embeddings saved: {emb_path}")

        metrics = compute_metrics(embeddings, labels)
        plot_species_heatmap(metrics["species_sim"], short_name, species_tags, OUTPUT_DIR)
        plot_pca(embeddings, labels, species_tags, short_name, OUTPUT_DIR)

        row = {
            "model":                  short_name,
            "embedding_dim":          actual_dim,
            "intra_species_sim":      round(metrics["intra_sim"],  4),
            "inter_species_dist":     round(metrics["inter_dist"], 4),
            "silhouette_score":       round(metrics["silhouette"], 4),
            "inference_time_per_seq": round(time_per_seq, 3),
            "total_inference_time_s": round(elapsed, 3),
        }
        all_results.append(row)

        print(f"  embedding_dim          : {actual_dim}")
        print(f"  intra_species_sim      : {row['intra_species_sim']}")
        print(f"  inter_species_dist     : {row['inter_species_dist']}")
        print(f"  silhouette_score       : {row['silhouette_score']}")
        print(f"  inference_time_per_seq : {row['inference_time_per_seq']} s")
        print(f"  total_inference_time_s : {row['total_inference_time_s']} s")

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
