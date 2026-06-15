# marlin

Find moments in your videos. `marlin` is the agent-first CLI for
[Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B) — a 2B video VLM
for dense captioning + temporal grounding. Runs **free and local** on Apple
Silicon (MLX) or NVIDIA (vLLM) — no API key, no network for inference.

```bash
uv tool install nemostation                          # or: pipx install nemostation
marlin                                               # first run: detect Apple Silicon/NVIDIA + build the engine
marlin find "deer crossing the road" --in ./footage  # auto-indexes, then model-verified timestamps
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

## How it runs (auto-detected, local)

| | Apple Silicon | NVIDIA |
|---|---|---|
| engine | SGLang-MLX | vLLM |
| serve | auto-starts on first `find` (or `marlin serve`) | same |
| weights | gated — 1-click access form | gated — 1-click form |

No API key — inference is local. Agents use the same verbs with `--json`
(clean stdout, progress on stderr). A hosted `base_url` swap lives in
`deploy/` for a future skill; it's not surfaced in the CLI yet.

## Speech

`marlin index --stt` adds faster-whisper speech rows to the same index
(`pip install 'nemostation[stt]'`) — search meetings by what was *said* and
what *happened* in one query.

## Status

v0.1 — base + `video-search` skill. Verified end-to-end against a live
endpoint; more skills (footage-catalog, dashcam-event-finder, clip-scorer)
ride the same verbs next.
