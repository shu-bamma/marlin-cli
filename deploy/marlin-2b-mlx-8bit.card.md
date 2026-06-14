<!-- This is the model card to publish at NemoStation/Marlin-2B-MLX-8bit.
     Gate the repo the same way as the base (Settings → enable gated access,
     mirror the base model's access form). License: Apache-2.0. -->
---
base_model: NemoStation/Marlin-2B
base_model_relation: quantized
library_name: mlx
pipeline_tag: video-text-to-text
license: apache-2.0
language:
  - en
tags:
  - mlx
  - video
  - multimodal
  - video-captioning
  - temporal-grounding
  - quantized
  - 8-bit precision
extra_gated_prompt: "Access to Marlin-2B-MLX-8bit uses the same form as the base model. Tell us who you are and what you're building so we can support your use case."
---

# Marlin-2B — MLX 8-bit (Apple Silicon)

**Original release:** [NemoStation/Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B)

8-bit [MLX](https://github.com/ml-explore/mlx) conversion of
[**NemoStation/Marlin-2B**](https://huggingface.co/NemoStation/Marlin-2B) for fast,
local, private inference on Apple Silicon. Same weights, same behavior — see the
[**base model card**](https://huggingface.co/NemoStation/Marlin-2B) for benchmarks,
architecture, training, and intended use.

| | |
|---|---|
| Base model | [NemoStation/Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B) (2B video VLM — dense captioning + temporal grounding) |
| Format | MLX, 8-bit · **~2.5 GB** (base BF16 ~5.1 GB) |
| Runs on | Apple Silicon (M-series), via SGLang-MLX |
| License | Apache-2.0 (inherited from base) |

## Use it (mlx-vlm)

```bash
pip install mlx-vlm
python -m mlx_vlm.generate \
  --model NemoStation/Marlin-2B-MLX-8bit \
  --video clip.mp4 --fps 2 \
  --prompt "Describe the video."
```

> Temporal grounding ("From `<start>` to `<end>`") needs the timestamp-aware
> serving path — use the `marlin` CLI (SGLang-MLX) for grounding; mlx-vlm's
> one-shot path is best for dense captioning.

## Conversion recipe

```bash
python -m mlx_vlm.convert \
  --hf-path NemoStation/Marlin-2B \
  --mlx-path ./Marlin-2B-MLX-8bit \
  -q --q-bits 8
```

## Access

Gated with the same access form as the base model — request access above. Apache-2.0.
