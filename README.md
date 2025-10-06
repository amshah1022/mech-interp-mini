# Mechanistic Peek into GPT‑2 Small with TransformerLens

This repo/notebook demonstrates two classic **mechanistic interpretability** tools on `gpt2-small` using [TransformerLens](https://github.com/neelnanda-io/TransformerLens):

1) **Logit Lens** – project intermediate residual streams to vocabulary logits to see where evidence for the target token accumulates across layers.  
2) **Causal Tracing (Activation Patching)** – identify which attention heads *cause* correct behavior by replacing their contributions with those from a clean run.

> **TL;DR**  
> - Plot a *logit‑lens curve* per question to visualize which layers build evidence for the correct answer.  
> - Run *per‑head activation patching* to surface top causal heads and visualize their attention patterns.

---

## Contents

- [Setup](#setup)
- [Quickstart](#quickstart)
- [What the Code Does](#what-the-code-does)
  - [`first_subtoken_id`](#first_subtoken_id)
  - [`logit_lens_curve`](#logit_lens_curve)
  - [`causal_head_scores`](#causal_head_scores)
- [Example: QA Probes](#example-qa-probes)
- [Outputs You Should See](#outputs-you-should-see)
- [Troubleshooting](#troubleshooting)
- [Extending This Notebook](#extending-this-notebook)
- [Citations & Further Reading](#citations--further-reading)
- [License](#license)

---

## Setup

Run in Google Colab or locally (Python ≥ 3.10). A GPU speeds things up but is optional.

```bash
pip install transformer-lens==1.15 einops torch --quiet
```

> **Note:** If you’re in Colab and see a warning about `HF_TOKEN`, you can ignore it for public models/datasets. To avoid the warning, create a token at `https://huggingface.co/settings/tokens` and add it to your Colab secrets as `HF_TOKEN`.

---

## Quickstart

```python
from transformer_lens import HookedTransformer
import torch, numpy as np, matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# Load GPT-2 small as a HookedTransformer
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
```

If you see:
- `The secret HF_TOKEN does not exist ...` → optional, safe to ignore for public artifacts.  
- `` `torch_dtype` is deprecated! Use `dtype` instead! `` → a deprecation warning inside dependencies; safe to ignore.

---

## What the Code Does

### `first_subtoken_id`

```python
def first_subtoken_id(token_str: str) -> int:
    ids = model.to_tokens(token_str, prepend_bos=False)[0]
    return int(ids[0].item())
```

Returns the **first GPT-2 token id** corresponding to a string (e.g., `" Jane"`). Many words begin with a leading space in BPE vocabularies; include it for accurate matching.

---

### `logit_lens_curve`

```python
def logit_lens_curve(prompt: str, target_token: str):
    toks = model.to_tokens(prompt, prepend_bos=True).to(device)
    # Cache residual streams across blocks
    _, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: ("hook_resid_pre" in n) or ("hook_resid_post" in n) or ("ln_final" in n)
    )
    tgt_id = first_subtoken_id(target_token)

    # Find existing layers dynamically
    layers = sorted({int(k.split(".")[1]) for k in cache.keys() if k.endswith(".hook_resid_pre")})

    vals = []
    for L in layers:
        resid_last = cache[f"blocks.{L}.hook_resid_pre"][:, -1:, :]   # [1,1,d_model]
        logits = model.unembed(model.ln_final(resid_last))            # [1,1,vocab]
        vals.append(float(logits[0, 0, tgt_id].item()))

    # Add a final point using last block's resid_post if present
    last_key = f"blocks.{layers[-1]}.hook_resid_post"
    if last_key in cache:
        resid_final = cache[last_key][:, -1:, :]
        logits_final = model.unembed(model.ln_final(resid_final))
        vals.append(float(logits_final[0, 0, tgt_id].item()))

    return vals
```

- Caches **residual streams** at each block.  
- Projects each residual (after `ln_final`) through the **unembedding** to get the logit for the **target token**.  
- Returns a list of logits—one per layer—optionally with a final point from `resid_post` of the last block.

Use it to **plot evidence accumulation** across layers.

---

### `causal_head_scores`

```python
def causal_head_scores(clean_prompt: str, corrupt_prompt: str, ans_token: str):
    t_clean = model.to_tokens(clean_prompt, prepend_bos=True).to(device)
    t_corr  = model.to_tokens(corrupt_prompt, prepend_bos=True).to(device)

    # Cache clean run attention outputs (z) to patch from
    _, cache_clean = model.run_with_cache(
        t_clean, names_filter=lambda n: ("attn.hook_z" in n)
    )

    ans_id = first_subtoken_id(ans_token)
    with torch.no_grad():
        base = model(t_corr)[0, -1, ans_id].item()

    scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))

    def patch_hook_factory(L, H):
        def hook(z, hook):  # z: [B, seq, n_heads, d_head]
            z[:, -1, H, :] = cache_clean[hook.name][:, -1, H, :]
            return z
        return hook

    for L in range(model.cfg.n_layers):
        for H in range(model.cfg.n_heads):
            with torch.no_grad():
                logits = model.run_with_hooks(
                    t_corr,
                    fwd_hooks=[(f"blocks.{L}.attn.hook_z", patch_hook_factory(L, H))]
                )
                scores[L, H] = logits[0, -1, ans_id].item() - base
    return scores
```

Per‑head **activation patching**:
- Run a **clean** prompt (correct answer implied) and a **corrupt** prompt (distractor added).  
- For each attention head `(L, H)`, **replace** its last‑position `z` vector in the corrupt run with the one from the clean run.  
- Measure the **increase in the target token logit** (`Δ logit`) at the final position.  
- Larger `Δ` ⇒ head is **more causally responsible** for the correct answer.

---

## Example: QA Probes

```python
qa = [
    {"q": "Who wrote Pride and Prejudice?", "a0": " Jane"},
    {"q": "Who wrote Emma?", "a0": " Jane"},
    {"q": "Who wrote Jane Eyre?", "a0": " Charlotte"},
    {"q": "What is the capital of France?", "a0": " Paris"},
    {"q": "What is 2+2?", "a0": " 4"},
]
for item in qa:
    plt.plot(logit_lens_curve(item["q"], item["a0"]), label=item["q"])
plt.xlabel("Layer"); plt.ylabel("Logit on first answer subtoken")
plt.title("Logit-lens evidence per layer"); plt.legend(fontsize=7); plt.show()
```

**Causal tracing** on a clean vs corrupt pair:

```python
clean = "Who wrote Pride and Prejudice?"
corrupt = "Who wrote Pride and Prejudice? Some say Emily Brontë."
scores = causal_head_scores(clean, corrupt, " Jane")

flat = [((L,H), scores[L,H]) for L in range(model.cfg.n_layers) for H in range(model.cfg.n_heads)]
top = sorted(flat, key=lambda x: x[1], reverse=True)[:10]

# Bar plot of top heads
labels = [f"L{L}H{H}" for (L,H),_ in top]
vals   = [v for _,v in top]
plt.bar(labels, vals)
plt.ylabel("Δ logit on ' Jane'")
plt.title("Top causal heads")
plt.show()
```

**Attention visualization** of the top head on the clean prompt:

```python
t_clean = model.to_tokens(clean, prepend_bos=True).to(device)
_, cache_clean = model.run_with_cache(t_clean, names_filter=lambda n: ("attn.hook_pattern" in n))

(L0, H0), _ = top[0]
attn = cache_clean[f"blocks.{L0}.attn.hook_pattern"][0]  # [seq_q, n_heads, seq_k]
attn_head = attn[:, H0, :].cpu().numpy()
plt.imshow(attn_head, aspect='auto')
plt.title(f"Attention L{L0}H{H0} (clean)")
plt.xlabel("Keys"); plt.ylabel("Queries")
plt.show()

print(model.to_string(t_clean[0]))
```

---

## Outputs You Should See

- **Logit‑lens line plot**: one line per question; later layers typically increase the correct token’s logit.  
- **Bar chart of top heads**: heads with largest `Δ logit` when patched (e.g., `L9H7` often pops up for name associations).  
- **Attention heatmap** for the best head on the clean prompt.  
- Printed tokenization of the clean prompt for inspection.

---

## Troubleshooting

- **`HF_TOKEN` warning in Colab**: optional; public models don’t require auth.  
- **`torch_dtype` deprecation**: emitted by internal deps; safe to ignore.  
- **CUDA OOM / slow**: switch to CPU (slower) or shorten prompts; Colab T4/A100 recommended.  
- **Token mismatch**: many answers require a **leading space** (e.g., `" Jane"`, `" Paris"`). Use `print(model.to_string(...))` and `model.to_tokens(...)` to verify.

---

## Extending This Notebook

- Swap `gpt2-small` for other HookedTransformer checkpoints (e.g., `attn-only` or larger GPT‑2 variants).  
- Patch **MLP outputs** (`hook_mlp_out`) or **residual streams** directly for circuit discovery.  
- Aggregate causal scores across **multiple prompts** per concept.  
- Save plots and scores to disk for reproducible experiments.

---

## Citations & Further Reading

- Nanda, N. et al. **TransformerLens** – practical hooks & tools for mechanistic interpretability.  
- Elhage, N. et al. (2021) **A Mathematical Framework for Transformer Circuits**.  
- Meng, K. et al. (2022) **Locating and Editing Factual Associations in GPT** (ROME).  
- Geva, M. et al. (2021, 2022) **Transformer Feed‑Forward Layers Are Key‑Value Memories**.

*(See original papers/repos for details.)*

---

## License

MIT — feel free to use and adapt with attribution.

