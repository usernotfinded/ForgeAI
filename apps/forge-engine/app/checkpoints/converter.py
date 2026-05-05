"""
ForgeAI — HuggingFace-to-ForgeAI Weight Converter
====================================================

Converts HuggingFace model weights (Llama / Mistral / Qwen / SmolLM
architecture family) into the ForgeAI native checkpoint format.

The converter:
  1. Reads the HF config.json and maps hyperparameters to GPTConfig.
  2. Loads weights from .safetensors (preferred) or pytorch .bin files.
  3. Re-maps state_dict keys from HF naming to ForgeAI naming.
  4. Saves the result as a ForgeAI checkpoint with metadata.json.

Supported source architectures (all share the same "LlamaForCausalLM" layout):
  - LLaMA / LLaMA-2 / LLaMA-3
  - Mistral / Mixtral (dense layers only)
  - Qwen2 / Qwen2.5
  - SmolLM / TinyLlama
  - Any model that follows the HF LlamaForCausalLM state_dict convention

Key mapping (HF → ForgeAI):
  model.embed_tokens.weight                      → token_emb.weight
  model.layers.{i}.self_attn.q_proj.weight       → blocks.{i}.attn.q_proj.weight
  model.layers.{i}.self_attn.k_proj.weight       → blocks.{i}.attn.k_proj.weight
  model.layers.{i}.self_attn.v_proj.weight       → blocks.{i}.attn.v_proj.weight
  model.layers.{i}.self_attn.o_proj.weight       → blocks.{i}.attn.o_proj.weight
  model.layers.{i}.mlp.gate_proj.weight          → blocks.{i}.ffn.gate.weight
  model.layers.{i}.mlp.up_proj.weight            → blocks.{i}.ffn.up.weight
  model.layers.{i}.mlp.down_proj.weight          → blocks.{i}.ffn.down.weight
  model.layers.{i}.input_layernorm.weight        → blocks.{i}.norm1.weight
  model.layers.{i}.post_attention_layernorm.weight → blocks.{i}.norm2.weight
  model.norm.weight                              → norm_f.weight
  lm_head.weight                                 → lm_head.weight
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import torch

from app.architectures.transformer import GPT, GPTConfig


# ── HF config.json → GPTConfig ──────────────────────────────────────────────


def _hf_config_to_gpt_config(hf_config: dict[str, Any]) -> GPTConfig:
    """
    Map HuggingFace config.json fields to ForgeAI GPTConfig.

    Works with Llama, Mistral, Qwen2, SmolLM, and TinyLlama config formats.
    """
    n_embd = hf_config["hidden_size"]
    n_layer = hf_config["num_hidden_layers"]
    n_head = hf_config["num_attention_heads"]

    # GQA: num_key_value_heads (Llama 2+, Mistral, Qwen2)
    # Falls back to n_head for older models without GQA
    n_kv_head = hf_config.get("num_key_value_heads", n_head)

    vocab_size = hf_config["vocab_size"]
    context_length = hf_config.get(
        "max_position_embeddings",
        hf_config.get("max_sequence_length", 2048),
    )

    # FFN: HF stores `intermediate_size` directly; ForgeAI uses ffn_multiplier.
    # For SwiGLU, intermediate_size ≈ (ffn_multiplier * n_embd * 2/3) rounded.
    # We reverse-engineer the multiplier, but store the exact intermediate_size
    # by finding the multiplier that reproduces it after rounding.
    intermediate_size = hf_config.get("intermediate_size", int(n_embd * 8 / 3))
    # Reverse: hidden = int(mult * n_embd * 2/3), rounded to 64
    # We pick the multiplier that gets closest after the rounding step
    ffn_multiplier = (intermediate_size * 3) / (2 * n_embd)

    rope_theta = float(hf_config.get("rope_theta", 10000.0))
    tie_embeddings = hf_config.get("tie_word_embeddings", True)

    return GPTConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        ffn_multiplier=ffn_multiplier,
        rope_theta=rope_theta,
        tie_embeddings=tie_embeddings,
        bias=False,
        dropout=0.0,
        attn_dropout=0.0,
    )


# ── State-dict key remapping ────────────────────────────────────────────────

# Regex patterns for HF key names → ForgeAI replacements
_KEY_MAP: list[tuple[re.Pattern[str], str]] = [
    # Embeddings
    (re.compile(r"^model\.embed_tokens\.weight$"), "token_emb.weight"),
    # Final norm
    (re.compile(r"^model\.norm\.weight$"), "norm_f.weight"),
    # Attention projections
    (re.compile(
        r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$"
    ), r"blocks.\1.attn.\2.weight"),
    (re.compile(
        r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.bias$"
    ), r"blocks.\1.attn.\2.bias"),
    # MLP (SwiGLU): gate / up / down
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$"), r"blocks.\1.ffn.gate.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$"), r"blocks.\1.ffn.up.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$"), r"blocks.\1.ffn.down.weight"),
    # Layer norms
    (re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.weight$"), r"blocks.\1.norm1.weight"),
    (re.compile(
        r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$"
    ), r"blocks.\1.norm2.weight"),
    # LM head
    (re.compile(r"^lm_head\.weight$"), "lm_head.weight"),
]


def _remap_key(hf_key: str) -> Optional[str]:
    """
    Map a single HuggingFace state_dict key to the ForgeAI equivalent.

    Returns None if the key should be skipped (e.g. rotary_emb, which ForgeAI
    computes on-the-fly via precompute_rope_freqs).
    """
    for pattern, replacement in _KEY_MAP:
        m = pattern.match(hf_key)
        if m:
            return pattern.sub(replacement, hf_key)

    # Keys we intentionally skip (recomputed by ForgeAI)
    skip_patterns = [
        "rotary_emb",           # RoPE: precomputed as a buffer
        "self_attn.rotary",     # alternate naming
        "inv_freq",             # RoPE inverse frequencies
    ]
    for pat in skip_patterns:
        if pat in hf_key:
            return None

    return None


# ── Weight loading ──────────────────────────────────────────────────────────


def _load_hf_weights(hf_dir: Path) -> dict[str, torch.Tensor]:
    """
    Load model weights from a HuggingFace model directory.

    Prefers .safetensors files (faster, safer).
    Falls back to pytorch_model*.bin if safetensors are not present.
    """
    safetensor_files = sorted(hf_dir.glob("*.safetensors"))
    bin_files = sorted(hf_dir.glob("pytorch_model*.bin"))
    single_bin = hf_dir / "pytorch_model.bin"

    if safetensor_files:
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "safetensors package required to load .safetensors files. "
                "Install with: pip install safetensors"
            )

        state_dict: dict[str, torch.Tensor] = {}
        for sf in safetensor_files:
            state_dict.update(load_file(str(sf), device="cpu"))
        return state_dict

    if bin_files or single_bin.exists():
        state_dict = {}
        files_to_load = bin_files if bin_files else [single_bin]
        for bf in files_to_load:
            shard = torch.load(str(bf), map_location="cpu", weights_only=True)
            state_dict.update(shard)
        return state_dict

    raise FileNotFoundError(
        f"No model weights found in {hf_dir}. "
        "Expected .safetensors or pytorch_model*.bin files."
    )


# ── Public API ──────────────────────────────────────────────────────────────


def convert_hf_to_forge(
    hf_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Convert a HuggingFace model to ForgeAI native checkpoint format.

    Args:
        hf_dir:     Path to HuggingFace model directory (with config.json + weights)
        output_dir: Path where the ForgeAI checkpoint will be saved

    Returns:
        Path to the output directory containing model.pt and metadata.json

    Raises:
        FileNotFoundError: If config.json or weight files are missing
        ValueError:        If the model architecture is not compatible
    """
    hf_dir = Path(hf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Read HF config and build GPTConfig ────────────────────────────────
    config_path = hf_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {hf_dir}")

    with open(config_path) as f:
        hf_config = json.load(f)

    # Validate architecture family
    arch_type = hf_config.get("model_type", "").lower()
    supported_types = {"llama", "mistral", "qwen2", "phi", "gemma", "stablelm"}
    if arch_type and arch_type not in supported_types:
        raise ValueError(
            f"Unsupported model_type '{arch_type}'. "
            f"Supported types: {sorted(supported_types)}. "
            "Only Llama-family architectures (Llama, Mistral, Qwen2, SmolLM, "
            "TinyLlama, Gemma) are currently supported."
        )

    gpt_config = _hf_config_to_gpt_config(hf_config)

    # Verify the FFN hidden dim matches what HF expects
    hf_intermediate = hf_config.get("intermediate_size")
    forge_intermediate = gpt_config.ffn_hidden_dim
    if hf_intermediate and forge_intermediate != hf_intermediate:
        # The rounding in GPTConfig may not match exactly — override
        # by adjusting ffn_multiplier until it does
        # Direct approach: use a custom ffn_multiplier that after the
        # round-to-64 step produces the exact HF intermediate_size
        # Since ffn_hidden_dim = round_to_64(int(mult * n_embd * 2/3)),
        # we just need the result to be hf_intermediate.
        # Set multiplier so the pre-rounding value is close enough:
        gpt_config.ffn_multiplier = (hf_intermediate * 3) / (2 * gpt_config.n_embd)
        # If still off after rounding, it's a minor shape mismatch we must handle
        forge_intermediate = gpt_config.ffn_hidden_dim

    # ── 2. Load HF weights ──────────────────────────────────────────────────
    hf_state_dict = _load_hf_weights(hf_dir)

    # ── 3. Remap keys ───────────────────────────────────────────────────────
    forge_state_dict: dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []

    for hf_key, tensor in hf_state_dict.items():
        forge_key = _remap_key(hf_key)
        if forge_key is None:
            skipped_keys.append(hf_key)
            continue
        forge_state_dict[forge_key] = tensor

    # Handle tied embeddings: if lm_head.weight is missing but tie_embeddings
    # is True, the head shares the embedding weight
    if "lm_head.weight" not in forge_state_dict and gpt_config.tie_embeddings:
        if "token_emb.weight" in forge_state_dict:
            # Weight tying — ForgeAI handles this in the model constructor,
            # but we still need the key present for load_state_dict
            forge_state_dict["lm_head.weight"] = forge_state_dict["token_emb.weight"]

    # ── 4. Validate against ForgeAI model ───────────────────────────────────
    model = GPT(gpt_config)
    expected_keys = set(model.state_dict().keys())
    actual_keys = set(forge_state_dict.keys())

    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys

    if missing:
        # Some keys like rope_freqs are buffers, not in state_dict from weights
        # Filter out non-persistent buffers
        real_missing = {k for k in missing if "rope_freqs" not in k}
        if real_missing:
            raise ValueError(
                f"Weight conversion incomplete. Missing keys in ForgeAI model:\n"
                f"  {sorted(real_missing)}\n"
                f"Skipped HF keys: {skipped_keys[:10]}{'...' if len(skipped_keys) > 10 else ''}"
            )

    if unexpected:
        # Remove extra keys that the model doesn't expect
        for k in unexpected:
            del forge_state_dict[k]

    # Shape validation
    for key in expected_keys & actual_keys:
        expected_shape = model.state_dict()[key].shape
        actual_shape = forge_state_dict[key].shape
        if expected_shape != actual_shape:
            raise ValueError(
                f"Shape mismatch for '{key}': "
                f"expected {expected_shape}, got {actual_shape}. "
                "The HF model may use a different FFN or attention configuration."
            )

    # ── 5. Save ForgeAI checkpoint ──────────────────────────────────────────
    torch.save(forge_state_dict, output_dir / "model.pt")

    # Save metadata
    metadata = {
        "step": 0,
        "epoch": 0,
        "loss": 0.0,
        "val_loss": None,
        "learning_rate": 0.0,
        "total_tokens_seen": 0,
        "model_config": {
            "vocab_size": gpt_config.vocab_size,
            "context_length": gpt_config.context_length,
            "n_embd": gpt_config.n_embd,
            "n_layer": gpt_config.n_layer,
            "n_head": gpt_config.n_head,
            "n_kv_head": gpt_config.n_kv_head,
            "ffn_multiplier": gpt_config.ffn_multiplier,
            "rope_theta": gpt_config.rope_theta,
            "tie_embeddings": gpt_config.tie_embeddings,
            "bias": gpt_config.bias,
        },
        "architecture": "transformer",
        "backend": "converted",
        "dtype": "float32",
        "source": "huggingface",
        "source_model_type": hf_config.get("model_type", "unknown"),
        "source_config": {
            "hidden_size": hf_config.get("hidden_size"),
            "num_hidden_layers": hf_config.get("num_hidden_layers"),
            "num_attention_heads": hf_config.get("num_attention_heads"),
            "num_key_value_heads": hf_config.get("num_key_value_heads"),
            "intermediate_size": hf_config.get("intermediate_size"),
            "vocab_size": hf_config.get("vocab_size"),
        },
        "conversion_info": {
            "skipped_keys": skipped_keys,
            "total_hf_keys": len(hf_state_dict),
            "total_forge_keys": len(forge_state_dict),
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create 'latest' symlink for compatibility with load_checkpoint
    if output_dir.parent != output_dir:
        # Only create symlink if we're inside a parent checkpoint dir
        pass
    # The output_dir itself IS the checkpoint dir, so no extra symlink needed

    return output_dir
