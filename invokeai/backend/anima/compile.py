"""Scoped compilation of Anima's repeated DiT blocks."""

from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def compiled_anima_blocks(transformer: torch.nn.Module) -> Iterator[None]:
    """Compile block forwards while preserving module names and restoring them on exit.

    Keep eager BF16/FP16 rounding between fused operators. Compiling the forward
    instead of replacing modules also keeps model-cache and LoRA parameter paths
    intact. Compilation errors propagate; there is no eager fallback.
    """
    original_forwards = []
    try:
        for block in transformer.blocks:
            compiled_forward = torch.compile(
                block.forward,
                backend="inductor",
                fullgraph=True,
                options={"emulate_precision_casts": True},
            )
            original_forwards.append((block, "forward" in block.__dict__, block.forward))
            block.forward = compiled_forward
        yield
    finally:
        for block, had_instance_forward, original_forward in reversed(original_forwards):
            if had_instance_forward:
                block.forward = original_forward
            else:
                del block.forward
