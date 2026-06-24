# Marlin

**Video understanding on your Mac.** `marlin` is the command-line tool for
[**Marlin-2B**](https://huggingface.co/NemoStation/Marlin-2B) — a 2B video VLM
for the two questions you actually ask a video: **what** is happening, and
**when**. Runs free and local on Apple Silicon — no API key, no Hugging Face account.

<p>
  <a href="https://vlm.nemostation.com/"><img src="https://img.shields.io/badge/▶_Try_it_live-Gradio_demo-FF6B35?style=for-the-badge" alt="Try it live"/></a>
  <a href="https://huggingface.co/NemoStation/Marlin-2B"><img src="https://img.shields.io/badge/🤗_Model-Marlin--2B-FFD21E?style=for-the-badge" alt="Hugging Face"/></a>
  <a href="https://pypi.org/project/nemostation/"><img src="https://img.shields.io/pypi/v/nemostation?style=for-the-badge&color=7DD3FC&label=pip%20install" alt="PyPI"/></a>
</p>

- **`marlin caption`** → a Scene description + a `<start>–<end>` event timeline
- **`marlin find`** → the single `start → end` span where your query happens

## Install

> **Apple Silicon (M-series Mac) only for now.** NVIDIA / other platforms are
> coming as a separate optimized build.

```bash
uv tool install nemostation      # or: pipx install nemostation
marlin setup                     # one-time: sign in, build the local engine
marlin caption clip.mp4          # describe what's in a video
marlin find clip.mp4 "a deer crossing"   # locate when it happens → start → end
```

First run opens your browser for a quick sign-in (two questions, then Google),
detects your Mac, and builds the local MLX engine. The 8-bit weights are
**public** — nothing gated. Add `--json` to any command for parseable output.

## What it produces

**`marlin caption "video.mp4"`** — *what's in it*

![Marlin caption example](https://huggingface.co/datasets/NemoStation/marlin-assets/resolve/main/caption_example.jpg)

**`marlin find "video.mp4", "gunfight"`** — *when it happens*

![Marlin find example](https://huggingface.co/datasets/NemoStation/marlin-assets/resolve/main/find_example.jpg)

Each call runs one model pass on one bounded clip (~2 min at 2 fps) — the same
contract as the inference server. For longer videos, window with `ffmpeg` and loop.

## Why Marlin

At 2B params it's the strongest open model in its weight class on dense
captioning (DREAM-1K, CaReBench) and natural-language temporal grounding
(TimeLens-Bench) — competitive with Gemini-2.5 at a fraction of the cost. See
the [benchmarks on the model card](https://huggingface.co/NemoStation/Marlin-2B).

## Use it from an agent

```bash
marlin skills install        # → .claude/skills/ + .agents/skills/
```

Installs the `video-understanding` skill so Claude Code / Codex use `marlin` as
"eyes on a video" — clip-length and single-find limits baked in. Every verb
honors `--json` (stdout parseable, progress on stderr).

## Links

- **Try it live** → [vlm.nemostation.com](https://vlm.nemostation.com/)
- **Model card + benchmarks** → [huggingface.co/NemoStation/Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B)
- **Team / custom fine-tuning** → [nemostation.com](https://nemostation.com/)
