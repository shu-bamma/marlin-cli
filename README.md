# marlin

Understand any video from the terminal. `marlin` is the agent-first CLI for
[Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B) — a 2B video VLM that
**describes** a clip (dense captioning) and **locates** moments in it (temporal
grounding). Runs **free and local** on Apple Silicon (MLX) or NVIDIA (vLLM) —
no API key, no network for inference.

```bash
uv tool install nemostation   # install
marlin setup                  # first-run: sign in with Google, detect hardware, build engine
marlin caption clip.mp4       # describe what's in a video
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

## How it runs (Apple Silicon, local)

| | Apple Silicon (Mac) |
|---|---|
| engine | SGLang-MLX |
| serve | auto-starts on first `caption`/`find` (or `marlin serve`) |
| weights | public — `Marlin-2B-MLX-8bit` (8-bit, no Hugging Face account) |

**Apple-Silicon only for now.** The CLI ships the Metal (MLX) build — the
validated, public, 8-bit path. NVIDIA/other machines get a clear "coming
soon" message; an optimized NVIDIA build will ship as a separate release.

No API key, no Hugging Face account — inference is local and the weights are
public. First run opens your browser for a one-time **sign-in**: two quick
questions (affiliation + what you'll use Marlin for), then Google. A hosted
`base_url` swap lives in `deploy/` for a future skill; not surfaced yet.

## Roadmap

Shipping now: local `caption` + `find` on single clips, Apple Silicon. Next,
once the storage + ranking design lands (present in the CLI today as
hidden/experimental verbs, not finalized):

- **NVIDIA build** — an optimized non-Apple-Silicon release (separate from the
  MLX 8-bit build that ships today).

- **`index` / `search`** — caption + embed a whole folder into a local index,
  then semantic search across your library (two-stage retrieval). Database and
  ranking are still being decided.
- **Speech** — fold faster-whisper transcripts into the index, to search by what
  was *said* as well as what *happened*.
- **More skills** — social-media analysis, footage catalog, clip scoring — each
  a `SKILL.md` riding the same verbs.
