#!/usr/bin/env -S uv run python
"""Diagnostic script to verify MPS (Apple Silicon GPU) setup.

Usage:
    uv run python scripts/check_mps.py            # basic checks
    uv run python scripts/check_mps.py --model     # also load a small model on MPS
"""

from __future__ import annotations

import argparse
import platform
import sys

import torch


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check_system_info() -> None:
    print_section("System Info")
    print(f"  Platform:       {platform.system()} {platform.machine()}")
    print(f"  macOS version:  {platform.mac_ver()[0] or 'N/A'}")
    print(f"  Python:         {sys.version.split()[0]}")
    print(f"  PyTorch:        {torch.__version__}")


def check_mps_availability() -> bool:
    print_section("MPS Backend")

    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()

    print(f"  MPS built:      {mps_built}")
    print(f"  MPS available:  {mps_available}")
    print(f"  CUDA available: {cuda_available}")

    if not mps_available:
        print("\n  [WARN] MPS is not available.")
        if not mps_built:
            print("         PyTorch was not built with MPS support.")
            print("         Reinstall with: pip install torch (arm64 build)")
        return False

    print("\n  [OK] MPS is available.")
    return True


def check_resolve_device() -> None:
    print_section("resolve_device()")
    from causalab.neural.pipeline import resolve_device

    result = resolve_device()
    print(f"  resolve_device()        -> {result!r}")
    print(f"  resolve_device('auto')  -> {resolve_device('auto')!r}")
    print(f"  resolve_device('cpu')   -> {resolve_device('cpu')!r}")

    expected = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if result == expected:
        print(f"\n  [OK] Detected expected device: {expected}")
    else:
        print(f"\n  [WARN] Expected {expected!r}, got {result!r}")


def check_tensor_ops() -> None:
    print_section("Tensor Operations on MPS")

    device = torch.device("mps")

    # float32 matmul
    a = torch.randn(256, 256, device=device, dtype=torch.float32)
    b = torch.randn(256, 256, device=device, dtype=torch.float32)
    c = a @ b
    torch.mps.synchronize()
    print(f"  float32 matmul (256x256):  OK  (result shape: {c.shape})")

    # float16 matmul
    a16 = a.half()
    b16 = b.half()
    c16 = a16 @ b16
    torch.mps.synchronize()
    print(f"  float16 matmul (256x256):  OK  (result shape: {c16.shape})")

    # bfloat16 — should NOT work on MPS
    try:
        abf = torch.randn(4, 4, device=device, dtype=torch.bfloat16)
        _ = abf @ abf
        torch.mps.synchronize()
        print("  bfloat16 matmul:           OK  (surprisingly supported!)")
    except (TypeError, RuntimeError) as e:
        print("  bfloat16 matmul:           Not supported (expected)")
        print(f"    -> {type(e).__name__}: {e}")

    # Memory info
    allocated = torch.mps.current_allocated_memory() / 1e6
    driver = torch.mps.driver_allocated_memory() / 1e6
    print(f"\n  MPS memory allocated: {allocated:.1f} MB")
    print(f"  MPS driver memory:    {driver:.1f} MB")


def check_model_load() -> None:
    print_section("Model Load on MPS")

    from causalab.neural.pipeline import LMPipeline, resolve_device

    device = resolve_device()
    print(f"  Loading gpt2 on {device}...")
    pipeline = LMPipeline("gpt2", max_new_tokens=5, device=device)
    model_device = next(pipeline.model.parameters()).device
    print(f"  Model device: {model_device}")

    # Quick forward pass
    tokens = pipeline.tokenizer("Hello world", return_tensors="pt")
    tokens = {k: v.to(model_device) for k, v in tokens.items()}
    with torch.no_grad():
        out = pipeline.model(**tokens)
    print(f"  Forward pass:  OK  (logits shape: {out.logits.shape})")

    if str(model_device).startswith("mps"):
        allocated = torch.mps.current_allocated_memory() / 1e6
        print(f"  MPS memory after load: {allocated:.1f} MB")

    del pipeline, tokens, out
    print("\n  [OK] Model loaded and ran successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MPS setup for causalab")
    parser.add_argument(
        "--model", action="store_true", help="Also test loading a model on MPS"
    )
    args = parser.parse_args()

    check_system_info()
    mps_ok = check_mps_availability()
    check_resolve_device()

    if mps_ok:
        check_tensor_ops()

    if args.model:
        check_model_load()

    print_section("Done")
    if mps_ok:
        print("  All MPS checks passed.")
    else:
        print("  MPS not available — resolve_device() will use cuda or cpu.")
    print()


if __name__ == "__main__":
    main()
