---
name: video-understanding
description: Use Marlin-2B as eyes on a video — describe what's in a clip (caption) or locate when something happens (find). Use whenever the user asks what's in a video, to describe or summarize a clip, to find when/where an event occurs, or to get a timestamp for a moment. Runs locally on Apple Silicon (MLX) or NVIDIA (vLLM), free, no API key. Bounded clips only (~2 min) — window longer videos.
license: MIT
metadata:
  requires:
    bins: ["marlin", "ffmpeg"]
---

# video-understanding — Marlin-2B as eyes on a video

Two verbs, each on a **single clip**, each `--json`:

| Job | Command |
|---|---|
| Describe a video | `marlin caption <video> --json` |
| Describe (one paragraph) | `marlin caption <video> --detail --json` |
| Find when X happens | `marlin find <video> "<query>" --json` |

`caption` = *what's in it* (a scene description + a `<start>–<end>` event timeline).
`find` = *when one thing happens* (a single `start → end` span).

## Hard limits — read before using

1. **One call sees ~2 minutes.** Marlin samples ~2 fps up to ~240 frames, so a single `caption`/`find` only "sees" a bounded clip. On longer videos the timestamps drift and detail is lost.
2. **Longer than ~2 min → window it.** Cut overlapping windows with ffmpeg, run per window, then **offset the returned timestamps by the window's start**:
   ```bash
   ffmpeg -i long.mp4 -ss 0   -t 120 -c copy w0.mp4   # 0–120s
   ffmpeg -i long.mp4 -ss 110 -t 120 -c copy w1.mp4   # 110–230s (10s overlap)
   marlin caption w0.mp4 --json ; marlin caption w1.mp4 --json
   ```
   Every timestamp from `w1` must have `110` added back.
3. **`find` returns ONE span — there is no multi-find.** It gives the single best match. To find **every** occurrence, window the video (as above) and run `find` per window, then collect the hits. Never assume one `find` call surfaces all matches.
4. **Visual only.** These modes don't read audio/speech. For "what was said," use a transcription tool.
5. **No library search.** Searching across a whole folder of videos is not in these verbs (that's the experimental `index`/`search`, not shipped — see the repo roadmap).

## Output contracts

`marlin caption <video> --json`:
```json
{ "video": "clip.mp4",
  "scene": "A wide shot of a quiet road at dusk…",
  "events": [ {"start": 2.0, "end": 5.4, "text": "a deer steps onto the road from the left"} ] }
```
`marlin caption <video> --detail --json` → `{ "video": "clip.mp4", "caption": "one paragraph…" }`

`marlin find <video> "query" --json`:
```json
{ "video": "clip.mp4", "query": "a deer crossing", "start": 2.0, "end": 5.4, "found": true, "tier": "from_pair" }
```
`"found": false` (tier `no_match`) means the query wasn't located in that clip.

## Rules

1. Always pass `--json`; the spinner goes to stderr, JSON to stdout.
2. Quote timestamps verbatim from the output — don't round or invent.
3. One video per call. For many videos, loop over them.
4. Long video? window it (limit 2) — don't send a 20-minute file and trust the timestamps.
5. `find` is single-answer (limit 3) — window + loop for "every time X happens."

## First run

Install and first-run in one command:
```bash
uv tool install nemostation && marlin
```
`marlin` auto-onboards on first run: Google sign-in (opens the browser), hardware detection, and local engine build (a few minutes). The weights are public — no Hugging Face account needed.

## Errors → fixes

| stderr contains | fix |
|---|---|
| `not configured` | run `marlin` once (it onboards), or `marlin setup`. |
| `not a file` | pass a path to one video file. |
| `ffmpeg/ffprobe not found` | `brew install ffmpeg` (macOS) / `apt install ffmpeg`. |
| `Connection refused` / `APIConnectionError` | local engine not up → `marlin serve` (it also auto-starts on the first call). |
| `sign-in required` | run `marlin login` (opens the browser) to sign in with Google. |

## When NOT to use

- Searching a whole library of videos → not yet (experimental `index`/`search`).
- Transcribing speech / making subtitles → use a Whisper tool.
- Generating or editing video → this only *understands* footage.
