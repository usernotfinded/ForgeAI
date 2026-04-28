# Experimental Keras Integration

ForgeAI includes an optional, experimental Keras 3 integration for future model
prototyping and KerasHub exploration.

This integration does not replace the ForgeAI-native PyTorch core. The existing
CLI-first workflow for tokenization, data preparation, PyTorch training,
checkpointing, evaluation, smoke tests, and CI remains the stable v0.1 path.

## Status

- Experimental only.
- Not a stable training backend.
- Not used by `forge train`.
- Not used for ForgeAI checkpoint serialization.
- No pretrained KerasHub models are downloaded or loaded by the smoke command.

## Install

Keras is optional. It is not installed with the default ForgeAI engine package.

```bash
pip install -e "apps/forge-engine[keras]"
```

The optional group installs:

- `keras`
- `keras-hub`

## Smoke Test

Run:

```bash
forge experimental keras-smoke
```

The command:

- sets `KERAS_BACKEND` before importing Keras
- defaults to the Keras torch backend
- imports Keras if available
- prints the selected backend
- builds a tiny in-memory model
- runs one forward pass
- runs one tiny train step by default
- avoids downloads
- avoids KerasHub model loading

To skip the tiny train step:

```bash
forge experimental keras-smoke --no-train-step
```

If Keras is not installed, ForgeAI prints a clear message with the optional
install command and exits without a Python traceback.

## Limitations

Keras support is not stable. Treat it as an experimental surface for prototyping,
not as a supported replacement for ForgeAI-native PyTorch training.

Keras also does not solve native MLX training. ForgeAI remains Apple
Silicon/MLX-oriented as a product direction, but native MLX training is a
separate backend effort. The Keras integration defaults to Keras' torch backend
for the smoke path and does not make ForgeAI checkpointing or training MLX-native.
