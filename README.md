# marlin

Understand any video from the terminal. `marlin` is the agent-first CLI for
[Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B) — a 2B video VLM that
**describes** a clip (dense captioning) and **locates** moments in it (temporal
grounding). Runs **free and local** on Apple Silicon (MLX) or NVIDIA (vLLM) —
no API key, no network for inference.

```bash
uv tool install nemostation              # or: pipx install nemostation
marlin                                   # first run: sign in with Google, detect Apple Silicon/NVIDIA, build the engine
marlin caption clip.mp4                  # describe what's in a video
marlin find clip.mp4 "a deer crossing"   # locate when it happens → start → end
```

Add `--json` to any verb for clean, parseable output (auto when piped).

## Two modes, one clip

| Verb | What it does |
|---|---|
| `marlin caption <video>` | scene description + a `<start>–<end>` event timeline |
| `marlin caption <video> --detail` | one free-form paragraph |
| `marlin find <video> "<query>"` | the single `start → end` span where the query happens |

Both run **one model call on one bounded clip** (~2 min at 2 fps) — the same
thing the inference server does, matching Marlin's training distribution. For
longer videos, cut overlapping windows with `ffmpeg` and run per window;
`find` returns one span (no multi-find), so window + loop for every occurrence.

## Agents

```bash
marlin skills install        # → .claude/skills/ + .agents/skills/
```

Installs the `video-understanding` skill so Claude Code / Codex use marlin as
"eyes on a video" — with the limits (clip length, single-find) baked in. Every
verb honors `--json` (stdout parseable, progress on stderr). See
`skills/video-understanding/SKILL.md`.

## How it runs (auto-detected, local)

| | Apple Silicon | NVIDIA |
|---|---|---|
| engine | SGLang-MLX | vLLM |
| serve | auto-starts on first `caption`/`find` (or `marlin serve`) | same |
| weights | public — `Marlin-2B-MLX-8bit` | public |

No API key, no Hugging Face account — inference is local and the weights are
public. First run does one **Google sign-in** (so we can send you updates). A
hosted `base_url` swap lives in `deploy/` for a future skill; not surfaced yet.

## Roadmap

Shipping now: local `caption` + `find` on single clips, Apple Silicon and
NVIDIA. Next, once the storage + ranking design lands (present in the CLI today
as hidden/experimental verbs, not finalized):

- **`index` / `search`** — caption + embed a whole folder into a local index,
  then semantic search across your library (two-stage retrieval). Database and
  ranking are still being decided.
- **Speech** — fold faster-whisper transcripts into the index, to search by what
  was *said* as well as what *happened*.
- **More skills** — social-media analysis, footage catalog, clip scoring — each
  a `SKILL.md` riding the same verbs.
