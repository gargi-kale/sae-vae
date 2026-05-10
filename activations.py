"""
Extract GPT-2 small activations from OpenWebText and save to a memmap file.

Pipeline:
1. Stream OpenWebText from HuggingFace (no full download needed).
2. Tokenize into fixed-length sequences.
3. Run GPT-2 small with a forward hook on a middle layer's residual stream.
4. Drop the first BURN_IN tokens of each sequence (context-starved positions).
5. Shuffle within each batch and write to a float16 memmap file.

Output:
    activations.dat   : raw float16 array, shape (N_TOKENS, D_MODEL)
    activations.meta  : small text file with shape and dtype for re-loading

Usage:
    python extract_activations.py
"""

import os
import json
import numpy as np
import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm

# ----------------------------------------------------------------------------- 
# Config
# -----------------------------------------------------------------------------
MODEL_NAME      = "gpt2"               # GPT-2 small (124M)
HOOK_NAME       = "blocks.6.hook_resid_pre"  # middle layer residual stream
SEQ_LEN         = 128                  # tokens per sequence
BURN_IN         = 16                   # drop first N positions (low context)
BATCH_SIZE      = 32                   # sequences per forward pass
N_TOKENS_TARGET = 5_000_000            # total activation vectors to collect
D_MODEL         = 768                  # GPT-2 small hidden size

OUT_DIR         = "activations_data"
OUT_FILE        = os.path.join(OUT_DIR, "activations.dat")
META_FILE       = os.path.join(OUT_DIR, "activations.meta")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
def setup_model():
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    # Sanity check the hook name exists
    assert HOOK_NAME in model.hook_dict, (
        f"Hook {HOOK_NAME} not found. Available residual hooks: "
        f"{[k for k in model.hook_dict if 'resid' in k][:5]}..."
    )
    return model


def stream_text(min_chars=200):
    """Yield text strings from OpenWebText, filtering out very short docs."""
    ds = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    for example in ds:
        text = example["text"]
        if len(text) >= min_chars:
            yield text


def tokenize_and_batch(model, text_iter, seq_len, batch_size):
    """
    Tokenize streamed text into fixed-length sequences and yield batches.
    Concatenates documents with the EOS token between them, then chunks.
    """
    eos = model.tokenizer.eos_token_id
    buffer = []
    batch = []

    for text in text_iter:
        ids = model.tokenizer.encode(text)
        buffer.extend(ids)
        buffer.append(eos)

        # Drain the buffer into full-length sequences
        while len(buffer) >= seq_len:
            seq = buffer[:seq_len]
            buffer = buffer[seq_len:]
            batch.append(seq)

            if len(batch) == batch_size:
                yield torch.tensor(batch, dtype=torch.long)
                batch = []

    if batch:
        yield torch.tensor(batch, dtype=torch.long)


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------
@torch.no_grad()
def extract():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = setup_model()

    # Pre-allocate the output memmap. Float16 to halve disk usage.
    print(f"Allocating memmap: {N_TOKENS_TARGET:,} x {D_MODEL} float16 "
          f"(~{N_TOKENS_TARGET * D_MODEL * 2 / 1e9:.1f} GB)")
    acts_out = np.memmap(
        OUT_FILE,
        dtype="float16",
        mode="w+",
        shape=(N_TOKENS_TARGET, D_MODEL),
    )

    # Buffer for storing activations from a single forward pass before writing
    captured = {}

    def hook_fn(activation, hook):
        # activation shape: (batch, seq_len, d_model)
        captured["acts"] = activation.detach()

    text_iter = stream_text()
    batch_iter = tokenize_and_batch(model, text_iter, SEQ_LEN, BATCH_SIZE)

    write_idx = 0
    pbar = tqdm(total=N_TOKENS_TARGET, desc="Extracting", unit="tok")

    try:
        for token_batch in batch_iter:
            if write_idx >= N_TOKENS_TARGET:
                break

            token_batch = token_batch.to(DEVICE)

            # Run forward pass with hook
            model.run_with_hooks(
                token_batch,
                return_type=None,  # don't compute logits (saves time)
                fwd_hooks=[(HOOK_NAME, hook_fn)],
            )

            acts = captured["acts"]  # (B, T, D)

            # Drop the first BURN_IN positions per sequence
            acts = acts[:, BURN_IN:, :]
            # Flatten batch and time dims: (B * (T - BURN_IN), D)
            acts = acts.reshape(-1, D_MODEL)

            # Shuffle within this batch so the on-disk order isn't sorted
            # by document/position. Shuffling once here means we can later
            # read sequentially without positional bias.
            perm = torch.randperm(acts.shape[0], device=acts.device)
            acts = acts[perm]

            # Move to CPU, cast to float16, and write to memmap
            acts_np = acts.to(torch.float16).cpu().numpy()
            n = acts_np.shape[0]

            # Don't overshoot the target
            n = min(n, N_TOKENS_TARGET - write_idx)
            acts_out[write_idx : write_idx + n] = acts_np[:n]
            write_idx += n
            pbar.update(n)

    finally:
        pbar.close()
        acts_out.flush()
        del acts_out  # ensure file handle is released

    # Write metadata so we can reopen the memmap easily later
    meta = {
        "shape": [write_idx, D_MODEL],
        "dtype": "float16",
        "model": MODEL_NAME,
        "hook": HOOK_NAME,
        "seq_len": SEQ_LEN,
        "burn_in": BURN_IN,
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Wrote {write_idx:,} activation vectors to {OUT_FILE}")
    print(f"Metadata saved to {META_FILE}")


# -----------------------------------------------------------------------------
# Helper for downstream training: reopen the memmap
# -----------------------------------------------------------------------------
def load_activations(out_file=OUT_FILE, meta_file=META_FILE):
    """Reopen the memmap for read-only access during model training."""
    with open(meta_file) as f:
        meta = json.load(f)
    return np.memmap(
        out_file,
        dtype=meta["dtype"],
        mode="r",
        shape=tuple(meta["shape"]),
    )


if __name__ == "__main__":
    extract()