# Anima 2.9B compiled inference benchmark

This measures the same 40-block Anima checkpoint on the same branch with
`anima_compile_blocks: false` and `true`. Both modes require full transformer
residency. Checkpoint weights, numerical dtype, sampler, steps and CFG are
unchanged. The default configuration leaves compilation disabled.

## Enabling compilation

Select an Anima model, open **Generate → Advanced**, and enable
**Compile Anima (faster inference)**. The toggle is off by default, is saved
with generation parameters, and sends an explicit `compile_blocks` boolean to
the denoiser. Switching it off overrides the server compilation default for that
job. Restart the backend and rebuild the frontend when updating from a version
without this control.

Workflows/API graphs may set `anima_denoise.compile_blocks` to `true` or `false`.
Omitting it (or setting `null`) preserves the `anima_compile_blocks` server
setting. Compiled jobs request full model loading and whole-model eviction
independently of the server's partial-loading configuration. An eager job can
use partial loading again once the previous job has released the transformer.
The first run and new shapes can take longer due to compilation.

## Environment

- RTX 5060, 8151 MiB VRAM; NVIDIA 610.57.04.
- PyTorch 2.7.1+cu128, Diffusers 0.40.0, Transformers 5.5.4,
  Tokenizers 0.22.2, Hugging Face Hub 1.28.0.
- `novaAnimeAM_v5029B.safetensors`, 40 blocks, BF16, 5572 MiB of weights.
- Upstream base: `e927a2ebf5`; support and optimization cherry-picked on top.

## Procedure

Two separate InvokeAI runtime roots used copied databases and fresh queues.
Both ran the same exported graph, with invocation caching disabled on every
node. Each mode generated three images with seed 42, resolution 992×1320,
Euler, 24 steps and CFG 5. Text encoding, denoising, preview callbacks and VAE
decoding all used the actual InvokeAI queue.

Shared configuration:

```yaml
device_working_mem_gb: 0.75
enable_partial_loading: true
anima_require_full_vram: true
```

Only `anima_compile_blocks` changed between modes. CUDA event profiling was
enabled for both with `INVOKEAI_ANIMA_PROFILE_STEPS=1`. Inductor's disk cache
was already warm; these results do not measure compilation with an empty cache.
New resolutions or other graph changes may require additional compilation.

## Results

| Run | Compilation | Denoising | VAE decode | Whole graph |
| --- | --- | ---: | ---: | ---: |
| 1 | Disabled | 65.716 s | 1.877 s | 68.864 s |
| 2 | Disabled | 66.792 s | 1.526 s | 69.007 s |
| 3 | Disabled | 60.697 s | 1.549 s | 62.920 s |
| 1 | Enabled | 48.402 s | 1.767 s | 51.345 s |
| 2 | Enabled | 49.247 s | 1.672 s | 51.601 s |
| 3 | Enabled | 49.153 s | 1.629 s | 51.567 s |

Median whole-graph time: **68.864 s → 51.567 s**, a **25.1% reduction**
(1.34× throughput). All six generations completed, with zero
transformer weights in RAM and no decode OOM. The baseline range shows
background-load variability; these are three samples per mode on one GPU.

Repeated images matched exactly within each mode. Between modes, landscape
image MAE was 11.882/255, RMSE 26.092/255 and PSNR 19.80 dB. Visual inspection
showed consistent composition with changed distant buildings, trees and bridge.


## Correctness and limitations

Compilation preserves checkpoint weights and requests eager BF16/FP16 casts
between fused operators, but compiled numerics can still change the final image.
It is an opt-in acceleration, not a promise of identical output for a given seed.
Earlier same-seed landscape and portrait comparisons retained detailed renders,
with visible differences in background objects. Image differences are not a
perceptual quality score. Actual LoRA generation was not benchmarked.

When full residency is required, only the Anima transformer uses whole-model
loading and eviction. Other models keep their existing partial-loading policy.
This avoids streaming DiT weights during denoising and lets the cache evict the
entire transformer before VAE decoding. An explicit fixed VRAM cache cap was
rejected after a repeated decode OOM and is not part of the measured recipe.

Compilation requires CUDA and Triton. Regional prompting and LLLite fail
explicitly when compilation is enabled. Compilation and residency failures
propagate without an eager fallback. Block module identities, state-dict paths
and hooks are preserved; original forwards are restored on success and errors.

## Tests

Using the same pinned dependency overlay in separate checkouts:

- Clean upstream: 553 passed, 89 skipped.
- Anima branch after the generation toggle: 569 passed, 89 skipped.

The shared suite covers model loading/cache, Anima transformer adapters,
denoising, scheduler dispatch, VAE and text encoding. The sixteen added tests cover
40-block checkpoint depth, missing blocks, scoped loading policy, parameter and
hook preservation, restoration after inference/compilation failures, per-job
compilation overrides, and full-residency policy transitions.
The frontend build, all lint checks, and 2343 frontend tests also passed; graph
tests verify that both toggle positions reach the Anima denoiser.
Ruff and `git diff --check` passed for the changed Python files.

A separate runtime smoke test used both server flags set to `false` and partial
loading enabled. Three 992×1320, four-step jobs completed in the sequence
compiled → eager → compiled, including both toggle positions queued from the
Generate UI. This verified per-job overrides, warm cache policy changes, and VAE
decode without OOM. The UI toggle also survived a page reload. These smoke runs
are separate from the 24-step performance measurements above.

The existing development environment had Diffusers 0.41.0.dev0 and Transformers
5.17.0, which caused the same five unrelated dependency-contract failures on both
branches. The results above use upstream's pinned versions instead; the user's
original environment was left unchanged.
