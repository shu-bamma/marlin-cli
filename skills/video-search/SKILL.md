---
name: video-search
description: Find moments in videos by describing them. Use whenever the user asks to find when something happens in a video, search video footage, locate a scene or clip, get timestamps for an event, or ask questions like "when did X happen" / "show me every time Y occurs" across video files, folders of footage, or YouTube links. Powered by Marlin-2B — runs locally on Apple Silicon (MLX) or NVIDIA (vLLM), free, no API key.
license: MIT
metadata:
  requires:
    bins: ["marlin", "ffmpeg"]
---

# video-search — find the moment in any footage

Search hours of video by describing what happens. Two-stage: coarse retrieval
over Marlin-2B dense captions, then precise temporal grounding inside the
winning clips — timestamps come from the model that was trained for them.

## Verb table (use these exact shapes)

| Job | Command |
|---|---|
| First run | nothing to configure — the first `marlin` command onboards (or `marlin setup --non-interactive`) |
| Install local engine | `marlin engine install` (Apple Silicon → SGLang-MLX; NVIDIA → vLLM); also auto-builds on first index/find |
| Start local server | `marlin serve` (or `--detach`; also auto-starts on first index/find) |
| Index footage | `marlin index ./footage --json` |
| Index big folder (async) | `marlin index ./footage --async --json` → `{"job_id": "..."}` |
| Index with speech | `marlin index ./footage --stt --json` (meetings/lectures/body-cam) |
| Index a YouTube link | `marlin index "https://youtube.com/watch?v=..." --json` |
| Check job | `marlin status <job_id> --json` |
| **Find a moment** | `marlin find "deer crossing the road" --json` |
| Find scoped + auto-index | `marlin find "goal celebration" --in ./matches --json` |
| Find + cut clips | `marlin find "person at the door" --clip --json` |

## Rules

1. Always pass `--json` (stdout is parseable; progress goes to stderr).
2. New footage: prefer `find --in <path>` — it indexes anything missing, then searches scoped.
3. Folders over ~30 min of video: use `index --async`, report the `job_id`, poll `status`.
4. Speech matters (meetings, lectures, interviews, body-cam)? add `--stt` at index time. Visual-only footage (dashcam, CCTV, sports, b-roll) doesn't need it.
5. Quote timestamps from the output verbatim; `"grounded": true` means model-verified span, `false` means index-span fallback — say which.
6. Local-only for now (Apple Silicon / NVIDIA) — no API key, no network for inference.

## First-run onboarding (automatic)

There's no separate setup step. The first `marlin` command (e.g. `marlin find …`)
auto-onboards: it detects the machine (Apple Silicon → MLX, NVIDIA → vLLM),
builds the local engine once (a few minutes), then runs. On Apple Silicon the
MLX weights are **gated** — if a 403 is reported, tell the user to open the
printed access-form link and approve it (1-click), then retry. The engine
**auto-starts on the first `index`/`find`** (first call warms ~40s).

Marlin runs **locally only** for now (Apple Silicon or NVIDIA). A hosted API
lives in the repo as a drop-in `base_url` swap but is not surfaced yet.

## Output contracts

`marlin find "query" --json`:
```json
{
  "query": "deer crossing the road",
  "results": [
    {
      "video": "/footage/2026-04-12_morning.mp4",
      "start": 75.2, "end": 81.0,
      "text": "a deer enters the road from the left shoulder...",
      "kind": "event", "score": 0.0321,
      "grounded": true, "tier": "to_pair",
      "clip": "./marlin_clips/2026-04-12_morning_01m15s-01m21s.mp4"
    }
  ]
}
```
`marlin index --json` → `{"videos": 3, "chunks": 41, "events": 187, "errors": []}`.
`marlin index --async --json` → `{"job_id": "a1b2c3d4"}`; `marlin status <id> --json` → `{"state": "running|done", "chunks": 12, ...}`.

## Errors → fixes

| stderr contains | Fix |
|---|---|
| `not configured` | Run the first-run onboarding above. |
| `ffmpeg/ffprobe not found` | `brew install ffmpeg` (macOS) / `apt install ffmpeg`. |
| `Connection refused` / `APIConnectionError` | Local server not running → `marlin serve` (keep it running), or re-run `marlin setup`. |
| `403` / gated repo | MLX weights need access — open the printed access-form link, approve (1-click), retry. |
| `index is empty` | `marlin index <path>` first, or use `find --in <path>`. |
| `no videos found` | Check the path; supported: mp4 mov mkv webm avi m4v, or a YouTube URL. |
| `speech indexing needs faster-whisper` | `pip install 'nemostation[stt]'`. |

## When NOT to use

- Pure speech transcription/subtitles with no "find the moment" need → use a Whisper tool.
- Generating or editing video content → this skill only *understands* footage.
- Live streams → not supported yet; bounded files and URLs only.
