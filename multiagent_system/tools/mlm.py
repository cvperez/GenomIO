"""Masked-language-model gap filling.

A rewrite of ``predict_until_length`` from ``src/core/gap_filler_rag.py`` with four
changes and nothing else. The token sampling is deliberately identical, so any difference
in output is attributable to the four changes rather than to a quietly different sampler.

1. **The context goes in front.** The original built
   ``f"{left} [MASKS] {right}\\n\\n# Context: {context}"`` and then tokenized with
   ``truncation=True, max_length=4096``. Truncation removes from the *end*, so the
   retrieved context was the first thing deleted -- silently, with the run still recorded
   as RAG-enabled. Here the input is ``f"{context} {left} [MASKS] {right}"``, so whatever
   survives truncation includes the evidence.

   The ``# Context:`` label is gone too. It was English prose handed to a tokenizer that
   only understands nucleotides.

2. **Counters come back with the sequence.** The original computed attempts used, final
   mask count and per-attempt length error, printed all of it, and returned a bare
   string. A run that converged on the first attempt and one that exhausted all ten and
   returned its best guess were indistinguishable afterwards.

3. **``threshold`` is an argument.** It was hardcoded at 0.01 inside the sampling loop --
   the parameter most likely to need tuning per gap.

4. **The mask count is seeded from a measured ratio.** The original started at
   ``int(gap_length / 4)``, assuming four bases per token. GENA-LM uses BPE, and the real
   ratio measured on these contigs is about 5.9, so the old guess started roughly 45%
   too high and spent attempts walking back down. The seed now comes from tokenizing the
   actual flanks.

Also reported rather than hidden: ``truncated`` (whether the input overran the window at
all) and ``alphabet_violations`` (characters outside ACGTN that survived decoding).
Cleaning those silently would hide exactly the kind of failure this system exists to make
visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MASK_SLOT = "[MASKS]"
VALID_BASES = frozenset("ACGTN")


@dataclass
class FillResult:
    sequence: str = ""
    produced_length: int = 0
    target_length: int = 0
    attempts_used: int = 0
    seed_mask_count: int = 0
    final_mask_count: int = 0
    threshold: float = 0.01
    input_tokens: int = 0
    untruncated_tokens: int = 0
    max_length: int = 4096
    truncated: bool = False
    alphabet_violations: int = 0
    converged: bool = False
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "produced_length": self.produced_length,
            "target_length": self.target_length,
            "attempts_used": self.attempts_used,
            "seed_mask_count": self.seed_mask_count,
            "final_mask_count": self.final_mask_count,
            "threshold": self.threshold,
            "input_tokens": self.input_tokens,
            "untruncated_tokens": self.untruncated_tokens,
            "max_length": self.max_length,
            "truncated": self.truncated,
            "alphabet_violations": self.alphabet_violations,
            "converged": self.converged,
            "attempts": self.attempts,
        }


def build_model_input(context: str, left: str, right: str) -> str:
    """Context first, then the flanks around the mask slot.

    The single most consequential line in this module. See change 1 above.
    """
    head = f"{context.strip()} " if context.strip() else ""
    return f"{head}{left} {MASK_SLOT} {right}"


def count_tokens(tokenizer, text: str) -> int:
    """Token length with no truncation, so an overrun is measurable rather than silent."""
    return len(tokenizer(text, truncation=False)["input_ids"])


def measure_bases_per_token(tokenizer, *texts: str) -> float:
    """The tokenizer's real compression on *these* sequences.

    GENA-LM's BPE ratio varies with composition, so it is measured rather than assumed.
    Falls back to 4.0 -- the original's hardcoded guess -- only if there is nothing to
    measure from.
    """
    joined = "".join(t for t in texts if t)
    if not joined:
        return 4.0
    tokens = count_tokens(tokenizer, joined)
    # Discount the [CLS]/[SEP] pair, which is fixed overhead rather than compression.
    tokens = max(1, tokens - 2)
    return max(1.0, len(joined) / tokens)


def seed_mask_count(gap_length: int, bases_per_token: float) -> int:
    return max(1, round(gap_length / max(1.0, bases_per_token)))


def _decode(tokenizer, token_ids: list[int]) -> str:
    """Ids to nucleotides, dropping special tokens by id rather than by string match."""
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    kept = [i for i in token_ids if i not in special]
    tokens = tokenizer.convert_ids_to_tokens(kept)
    if isinstance(tokens, str):
        tokens = [tokens]
    return "".join(tokens).replace("▁", "").upper()


def _adjustment(diff: int, gap_length: int) -> int:
    """Unchanged from the original: how far to move the mask count after a miss."""
    relative = diff / max(1, gap_length)
    if relative > 0.2:
        return max(5, int(diff / 4))
    if relative > 0.1:
        return max(3, int(diff / 6))
    return max(1, int(diff / 8))


def fill_to_length(
    context: str,
    left: str,
    right: str,
    gap_length: int,
    tokenizer,
    model,
    *,
    threshold: float = 0.01,
    seed_masks: int | None = None,
    max_attempts: int = 10,
    max_length: int = 4096,
) -> FillResult:
    """Fill the gap between ``left`` and ``right`` to approximately ``gap_length`` bases."""
    import torch  # imported here so the module is testable without torch

    base = build_model_input(context, left, right)
    bases_per_token = measure_bases_per_token(tokenizer, left, right)
    masks = seed_masks if seed_masks and seed_masks > 0 else seed_mask_count(
        gap_length, bases_per_token
    )

    result = FillResult(
        target_length=gap_length,
        threshold=threshold,
        seed_mask_count=masks,
        max_length=max_length,
    )

    best_sequence, best_diff = "", float("inf")

    for attempt in range(1, max_attempts + 1):
        masked_input = base.replace(MASK_SLOT, " ".join([tokenizer.mask_token] * masks))

        untruncated = count_tokens(tokenizer, masked_input)
        inputs = tokenizer(
            masked_input, return_tensors="pt", truncation=True, max_length=max_length
        )
        used = int(inputs["input_ids"].shape[1])

        result.untruncated_tokens = untruncated
        result.input_tokens = used
        # Recorded, not corrected. If this is ever True the context may have been cut,
        # which is precisely the condition the acceptance gate exists to catch.
        result.truncated = untruncated > used

        with torch.no_grad():
            outputs = model(**inputs)

        positions = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(
            as_tuple=True
        )[0]

        predicted_ids: list[int] = []
        for index in positions:
            logits = outputs.logits[0, index]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            above = (probs >= threshold).nonzero(as_tuple=True)[0]
            if len(above) == 0:
                predicted_ids.append(int(torch.argmax(probs).item()))
            else:
                filtered = probs[above]
                filtered = filtered / filtered.sum()
                choice = int(torch.multinomial(filtered, num_samples=1).item())
                predicted_ids.append(int(above[choice].item()))

        sequence = _decode(tokenizer, predicted_ids)
        diff = abs(len(sequence) - gap_length)

        result.attempts_used = attempt
        result.final_mask_count = masks
        result.attempts.append(
            {
                "attempt": attempt,
                "masks": masks,
                "produced": len(sequence),
                "diff": diff,
                "masks_in_window": int(positions.shape[0]),
            }
        )

        if diff < best_diff:
            best_sequence, best_diff = sequence, diff

        if diff <= 3:
            result.converged = True
            best_sequence = sequence
            break

        step = _adjustment(diff, gap_length)
        masks = masks + step if len(sequence) < gap_length else max(1, masks - step)

    result.sequence = best_sequence
    result.produced_length = len(best_sequence)
    result.alphabet_violations = sum(1 for c in best_sequence if c not in VALID_BASES)
    return result
