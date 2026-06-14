# marlin

Find moments in your videos. `marlin` is the agent-first CLI for
[Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B) — a 2B video VLM
for dense captioning + temporal grounding. Runs **free on your Mac (Apple
Silicon) or NVIDIA GPU** (MLX / vLLM), or against **NemoStation hosted**
inference with one env var.

```bash
uv tool install marlin-cli            # or: pipx install marlin-cli
marlin setup                          # auto-detects Apple Silicon / NVIDIA / hosted
marlin engine install                 # local only — builds the engine for your machine
marlin index ./footage                # caption + embed (resume-safe)
marlin find "deer crossing the road"  # exact clip, model-verified timestamps
```

## Why timestamps are right here

Search is two-stage: coarse retrieval over timestamped dense captions, then
Marlin temporal grounding *inside* the winning 30s chunks. Grounding short
chunks matches the model's training distribution (and sidesteps vLLM's
long-video timestamp bug), so spans land where the event actually is.

## Agents

```bash
marlin skills install        # → .claude/skills/ + .agents/skills/
```

Every verb honors `--json` (auto when piped), long indexes run with
`--async` + `marlin status <job_id>`. See `skills/video-search/SKILL.md`.

## How it runs (auto-detected)

| | Apple Silicon | NVIDIA | hosted |
|---|---|---|---|
| engine | SGLang-MLX | vLLM | Modal (vLLM), scale-to-zero |
| serve | `marlin serve` — auto-starts on first `find` | `marlin serve` | `deploy/modal_app.py` |
| auth | none (weights gated: 1-click form) | none | `MARLIN_API_KEY` |
| config | `marlin setup --local` | `marlin setup --local` | `marlin setup --hosted --base-url … --api-key …` |

Non-interactive/agent path: set `MARLIN_BASE_URL` (+ `MARLIN_API_KEY` if
hosted) — env always beats `~/.marlin/config.json`.

## Speech

`marlin index --stt` adds faster-whisper speech rows to the same index
(`pip install 'marlin-cli[stt]'`) — search meetings by what was *said* and
what *happened* in one query.

## Status

v0.1 — base + `video-search` skill. Verified end-to-end against a live
endpoint; more skills (footage-catalog, dashcam-event-finder, clip-scorer)
ride the same verbs next.
