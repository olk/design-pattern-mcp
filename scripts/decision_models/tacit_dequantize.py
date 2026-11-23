#!/usr/bin/env python
"""Dequantize the trained Tacit-4B NF4 checkpoint to plain BF16 (runs in the image build).

The trainer (``tacit_self_distill.py::save_checkpoint``) writes the merged student still
NF4-wrapped (``config.json`` carries ``quantization_config.quant_method=bitsandbytes``),
and vLLM's CPU backend loads only AWQ / GPTQ / compressed-tensors INT8 — never
bitsandbytes. ``docker/Dockerfile.tacit-cpu`` runs this script in its ``dequant`` stage
so the CPU image serves the same trained weights as unquantized BF16.

Memory: the builder needs ~16 GB free RAM (3.2 GB NF4 input + ~8 GB BF16 output +
intermediates, all on CPU).

    python scripts/decision_models/tacit_dequantize.py --in-dir models/tacit-4b --out-dir /out/tacit-4b
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger("tacit-dequantize")

# Aux files copied from the input dir when present — the same list
# ``tacit_self_distill.py::save_checkpoint`` carries, plus the train provenance
# (the CPU publish gate reads ``self-distill.json`` from the baked image).
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
_EXTRA_FILES = ("tacit_decision.py", "generation_config.json", "self-distill.json")


def _copy_aux(in_dir: Path, out_dir: Path) -> None:
    """Copy tokenizer + provenance files (same shape as the trainer's checkpoint)."""
    if (in_dir / "tokenizer.json").exists():
        for name in _TOKENIZER_FILES:
            source = in_dir / name
            if source.exists():
                shutil.copy2(source, out_dir / name)
    for name in _EXTRA_FILES:
        source = in_dir / name
        if source.exists():
            shutil.copy2(source, out_dir / name)


def dequantize(in_dir: Path, out_dir: Path) -> None:
    """Write an unquantized BF16 copy of the ``in_dir`` checkpoint to ``out_dir``."""
    config = json.loads((in_dir / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config")
    if not quant:
        shutil.copytree(in_dir, out_dir, dirs_exist_ok=True)
        logger.info("already unquantized — copied %s -> %s verbatim", in_dir, out_dir)
        return
    # Heavy imports only on the quant path: the verbatim copy above is stdlib-only.
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if quant.get("quant_method") != "bitsandbytes":
        raise SystemExit(f"unsupported quant_method {quant.get('quant_method')!r}: expected 'bitsandbytes'")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_storage=getattr(torch, quant["bnb_4bit_quant_storage"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(in_dir),
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model = model.dequantize()
    model = model.to(torch.bfloat16)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Single shard like the trainer output (4B params x bf16 ≈ 7.7 GB < 10 GB): vLLM
    # loads a sharded layout too, but one model.safetensors keeps the baked layout
    # identical to the trained checkpoint and the Dockerfile size guard simple.
    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="10GB")
    # Load-bearing: vLLM CPU refuses any config still carrying a bitsandbytes block,
    # and the NF4 config's "dtype": "float32" no longer describes the baked weights.
    out_config_path = out_dir / "config.json"
    out_config = json.loads(out_config_path.read_text(encoding="utf-8"))
    out_config.pop("quantization_config", None)
    out_config["dtype"] = "bfloat16"
    out_config_path.write_text(json.dumps(out_config, indent=2) + "\n", encoding="utf-8")
    _copy_aux(in_dir, out_dir)
    in_size = (in_dir / "model.safetensors").stat().st_size
    out_size = (out_dir / "model.safetensors").stat().st_size
    logger.info("dequantized %s (%d bytes) -> %s (%d bytes)", in_dir, in_size, out_dir, out_size)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line for the dequantizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, required=True, help="trained NF4 checkpoint dir")
    parser.add_argument("--out-dir", type=Path, required=True, help="where to write the BF16 checkpoint")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Dequantize one checkpoint; non-zero when anything fails (fails the build loudly)."""
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    args = parse_args(argv)
    dequantize(args.in_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
