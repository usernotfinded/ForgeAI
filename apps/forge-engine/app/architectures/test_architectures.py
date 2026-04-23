"""
Quick sanity check — builds every preset and prints parameter counts.

Run from the forge-engine directory:
    python -m app.architectures.test_architectures

Expected output (approximate):
    forge-tiny   →  125.2M params
    forge-small  →  354.1M params
    forge-medium →  762.8M params
    forge-1b     →  1024.0M params
"""

def main() -> None:
    try:
        import torch
        from app.architectures import get_architecture, list_architectures
    except ModuleNotFoundError:
        print("Torch non disponibile: salto il sanity check delle architetture.")
        return

    print("=" * 55)
    print("ForgeAI — Architecture Sanity Check")
    print("=" * 55)

    for arch in list_architectures():
        arch_name = arch["name"]
        print(f"\n[{arch_name.upper()}]")
        for preset in arch["presets"]:
            preset_name = preset["name"]
            model = get_architecture(arch_name, preset=preset_name)
            params_m = model.num_parameters() / 1e6

            # Quick forward pass to verify the model works
            batch = torch.randint(0, model.config.vocab_size, (1, 16))
            logits, _ = model(batch)
            assert logits.shape == (1, 1, model.config.vocab_size)

            print(f"  {preset_name:<16} {params_m:>8.1f}M params  ✓")

    print("\nAll architectures OK.")

if __name__ == "__main__":
    main()
