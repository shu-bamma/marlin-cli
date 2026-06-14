"""Hosted Marlin-2B: vLLM OpenAI-compatible server on Modal, scale-to-zero.

Deploy:   modal deploy deploy/modal_app.py
Secrets:  modal secret create huggingface HF_TOKEN=...      (Marlin-2B is gated)
          modal secret create marlin-server MARLIN_SERVER_KEY=...  (shared trial key)

Resulting URL (shown by `modal deploy`) is the `--base-url` for
`marlin setup --hosted` — e.g. https://<org>--marlin-vllm-serve.modal.run/v1

Cost shape: L4 @ ~$0.80/hr billed per second, scaledown to zero after 5 min
idle. Set min_containers=1 (~$19/mo) before a launch so the first click
doesn't eat a cold start.
"""

import modal

MODEL = "NemoStation/Marlin-2B"
PORT = 8000
MINUTES = 60

hf_cache = modal.Volume.from_name("marlin-hf-cache", create_if_missing=True)

# CUDA *devel* base (not debian_slim): vLLM's flashinfer JIT-compiles the
# Qwen3.5 GDN linear-attention kernel at startup and needs nvcc on disk.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm>=0.10.0", "huggingface_hub[hf_transfer]>=0.34")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("marlin-vllm")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("marlin-server"),
    ],
    timeout=20 * MINUTES,
    scaledown_window=5 * MINUTES,
    min_containers=0,  # raise to 1 for launch day
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=PORT, startup_timeout=15 * MINUTES)
def serve():
    import os
    import subprocess

    # config.json brands the architecture as MarlinForConditionalGeneration
    # (a pure subclass of Qwen3_5ForConditionalGeneration in remote code);
    # map it to vLLM's native implementation.
    cmd = [
        "vllm", "serve", MODEL, "--host", "0.0.0.0", "--port", str(PORT),
        "--max-num-seqs", "8",
        # 30s/480p chunks need a few k tokens; capping far below the 262k
        # qwen3_5 default frees VRAM for KV cache and speeds startup.
        "--max-model-len", "32768",
        "--api-key", os.environ["MARLIN_SERVER_KEY"],
        "--hf-overrides", '{"architectures": ["Qwen3_5ForConditionalGeneration"]}',
    ]
    subprocess.Popen(cmd, env=os.environ.copy())
