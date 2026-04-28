# Experimental MLX Backend Foundation

ForgeAI includes an optional experimental MLX backend foundation for future
Apple Silicon native training work.

This is not the final MLX-native trainer. The current stable ForgeAI v0.1
training path remains PyTorch-based for CUDA, MPS, and CPU. Existing tokenizer,
data preparation, training, checkpointing, evaluation, and chat behavior are not
changed by the experimental MLX smoke command.

## Status

- Experimental only.
- Not used by `forge train`.
- Not used for ForgeAI checkpoint serialization.
- Not a complete native MLX training backend.
- No pretrained models are downloaded or loaded.
- No Keras or KerasHub path is involved.

## Install

MLX is optional. It is not installed with the default ForgeAI engine package.

```bash
pip install -e "apps/forge-engine[mlx]"
```

## Smoke Test

Run:

```bash
forge experimental mlx-smoke
```

The command:

- checks whether MLX is importable
- prints platform and Apple Silicon detection
- prints the MLX default device when available
- creates a tiny MLX tensor
- runs a tiny numerical operation
- creates a tiny `mlx.nn.Linear` module
- runs one forward pass
- runs one tiny optimizer step by default
- avoids downloads and pretrained model loading

To skip the tiny train step:

```bash
forge experimental mlx-smoke --no-train-step
```

If MLX is not installed, ForgeAI prints a clear message with the optional
install command and exits without a Python traceback.

## Relationship To Existing Backends

The PyTorch core is the stable v0.1 path. It remains responsible for the current
ForgeAI CLI training workflow and checkpoint format.

The MLX backend is an experimental future path for Apple Silicon native work. It
currently provides availability checks and a tiny smoke test only. It does not
make ForgeAI training MLX-native yet.

