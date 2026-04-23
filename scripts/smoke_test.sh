#!/usr/bin/env bash
# ForgeAI Minimal Smoke Test
# This script runs a very small training pipeline on toy data to ensure
# that the environment is set up correctly and no code paths are fundamentally broken.

set -e

echo "========================================"
echo " ForgeAI Minimal Smoke Test"
echo "========================================"

# Check if forge is installed
if ! command -v forge &> /dev/null; then
    echo "Warning: 'forge' command not found in PATH."
    echo "Falling back to 'python -m cli.main' (make sure you are in the project root)."
    export PYTHONPATH="$(pwd)/apps/forge-engine:$PYTHONPATH"
    FORGE_CMD="python -m cli.main"
else
    FORGE_CMD="forge"
fi

SMOKE_DIR="/tmp/forge_smoke_test"
mkdir -p "$SMOKE_DIR"
cd "$SMOKE_DIR"

echo "1. Creating toy dataset..."
mkdir -p data
cat << 'EOF' > data/toy_base.txt
The quick brown fox jumps over the lazy dog.
A wizard's job is to vex chumps quickly in fog.
Pack my box with five dozen liquor jugs.
How vexingly quick daft zebras jump!
EOF
# Repeat to get enough tokens for the default 1024/2048 context length
for i in {1..100}; do
    cat data/toy_base.txt >> data/toy.txt
done

echo "2. Training toy tokenizer..."
$FORGE_CMD tokenizer train --data ./data/ --vocab-size 100 --output ./tokenizer/

echo "3. Preparing data shards..."
$FORGE_CMD data prepare ./data/ --output ./data_processed/ --tokenizer ./tokenizer/

echo "4. Planning (dry run)..."
$FORGE_CMD plan --arch transformer --params 50M --data ./data_processed/

echo "5. Training (10 steps)..."
# We only do 10 steps just to verify the forward/backward/checkpoint loop runs without crashing
$FORGE_CMD train \
    --arch transformer \
    --params 50M \
    --data ./data_processed/ \
    --tokenizer ./tokenizer/ \
    --output ./checkpoints/ \
    --max-steps 10 \
    --save-every 10

echo "6. Chat inference via PyTorch fallback (no external engines needed for this check)..."
# Just verifying the model loads and can generate at least 1 token
$FORGE_CMD chat ./checkpoints/latest --tokenizer ./tokenizer/ --engine pytorch --max-tokens 5 <<< "Hello"

echo "========================================"
echo "✅ Smoke test completed successfully!"
echo "========================================"
