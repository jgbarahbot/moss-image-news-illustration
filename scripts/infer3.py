#!/usr/bin/env python3
"""MOSS-Image-SFT batch inference: N illustrations in ONE process.

Loads text encoder once (encode all prompts), DiT once (resident), VAE once.
Then runs the 50-step diffusion loop per prompt and saves one PNG each.
Avoids re-loading the 16 GB encoder + 10 GB DiT N times (the expensive part).

Reads /prompts.json: [{"name": "...", "prompt": "...",
                        "negative": "...",  # optional; empty = no negative branch
                        "seed": 42}, ...]
Env: MOSS_H / MOSS_W / MOSS_STEPS (defaults 1024 / 1024 / 50).
Output: /out/<name>-<H>x<W>.png  (one per prompt).
"""
import gc
import importlib.util
import json
import os
import time

import torch

t0 = time.time()


def log(*a):
    print(f"[moss {time.strftime('%H:%M:%S')}]", *a, flush=True)


def vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def flush_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

from diffusers import AutoencoderKLFlux2, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoModel, AutoTokenizer

MODEL = "/model"
OUT = "/out"
DEVM = "cuda" if torch.cuda.is_available() else "cpu"

log(f"backend: {DEVM}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    log(f"gpu: {p.name}, {p.total_memory / 1e9:.1f} GB VRAM")

with open("/prompts.json", encoding="utf-8") as _f:
    JOBS = json.load(_f)
log(f"jobs: {[j['name'] for j in JOBS]}")

NAME_SEED = {j["name"]: int(j.get("seed", 42 + i)) for i, j in enumerate(JOBS)}

# --- 0. custom classes + scheduler (no weights) ---
spec = importlib.util.spec_from_file_location("moss_pipe", f"{MODEL}/pipeline_moss_image.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sched_cfg = json.load(open(f"{MODEL}/scheduler/scheduler_config.json"))
scheduler = mod.MOSSImageScheduler(
    num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
    shift=sched_cfg.get("shift", 5.0),
    training_shift=sched_cfg.get("training_shift", 1.0),
)
image_processor = VaeImageProcessor(vae_scale_factor=16)


def encode_dev(te, tok, text):
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    inputs = tok(formatted, return_tensors="pt", truncation=True,
                 max_length=4096).to(DEVM)
    out = te(**inputs, use_cache=False, return_dict=True)
    row = out.last_hidden_state[0].unsqueeze(0)  # [1, S, C]
    mask = torch.ones(1, row.shape[1], dtype=torch.bool, device=DEVM)
    return row, mask


# --- 1. text encoder: load once, encode all prompts, delete ---
log("loading text encoder (Qwen3.5-9B, 16 GB) + tokenizer (once)")
t1 = time.time()
tok = AutoTokenizer.from_pretrained(f"{MODEL}/tokenizer", local_files_only=True)
te = AutoModel.from_pretrained(
    f"{MODEL}/text_encoder", torch_dtype=torch.bfloat16,
    local_files_only=True).eval()
log(f"encoder loaded in {time.time() - t0:.0f}s")
te.to(DEVM)
cuda_sync()
EMBS = []
with torch.no_grad():
    for job in JOBS:
        pos_e, pos_m = encode_dev(te, tok, job["prompt"])
        neg_text = job.get("negative", "")
        if neg_text:
            neg_e, neg_m = encode_dev(te, tok, neg_text)
        else:
            neg_e, neg_m = encode_dev(te, tok, "")
        if neg_m.sum().item() == 0:
            neg_e = neg_e.new_zeros(1, 1, neg_e.shape[-1])
            neg_m = torch.zeros(1, 1, dtype=torch.bool, device=DEVM)
            neg_m[0, 0] = True
        EMBS.append((job, pos_e, pos_m, neg_e, neg_m))
        log(f"  encoded: {job['name']} (pos {pos_e.shape[1]} tokens, neg {neg_m.sum().item()} tokens)")
te.to("cpu")
del te, tok
gc.collect()
flush_cache()
log(f"all encoded; encoder freed. peak VRAM {vram_gb():.2f} GB")

# --- 2. DiT: load once, resident ---
log("loading DiT (5B, 10 GB) (once, resident)")
t1 = time.time()
transformer = mod.MOSSImageTransformer.from_pretrained(
    f"{MODEL}/transformer", torch_dtype=torch.bfloat16,
    local_files_only=True).eval()
transformer.to(DEVM)
cuda_sync()
log(f"DiT on {DEVM} in {time.time() - t1:.0f}s")

# --- 3. VAE: load once, coexists with DiT (tiny) ---
log("loading VAE (0.2 GB) (once)")
vae = AutoencoderKLFlux2.from_pretrained(
    f"{MODEL}/vae", torch_dtype=torch.float32,
    local_files_only=True).eval()
vae.to(DEVM)
cuda_sync()

H = int(os.environ.get("MOSS_H", "1024"))
W = int(os.environ.get("MOSS_W", "1024"))
STEPS = int(os.environ.get("MOSS_STEPS", "50"))
CFG = 3.0
SKIP = [22]

os.makedirs(OUT, exist_ok=True)
saved = []
for job, pos_e, pos_m, neg_e, neg_m in EMBS:
    name = job["name"]
    log(f"=== generating {name}  ({W}x{H}, {STEPS} steps, seed {NAME_SEED[name]}) ===")
    t1 = time.time()
    log("  prompt:", job["prompt"][:100])
    latents = randn_tensor(
        (1, transformer.config.latent_channels, H // 16, W // 16),
        generator=torch.Generator("cpu").manual_seed(NAME_SEED[name]),
        device=DEVM, dtype=torch.float32,
    )
    scheduler.set_timesteps(STEPS, device=DEVM)
    nsteps = len(scheduler.timesteps)
    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            pos = transformer(latents, t, pos_e.to(DEVM), pos_m.to(DEVM)).sample.float()
            neg = transformer(
                latents, t, neg_e.to(DEVM), neg_m.to(DEVM), skip_layers=SKIP
            ).sample.float()
            pred = neg + CFG * (pos - neg)
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
    log(f"  diffusion done in {time.time() - t1:.0f}s; peak VRAM {vram_gb():.2f} GB")
    t1 = time.time()
    z = latents.to(dtype=vae.dtype)
    bn = vae.bn
    eps = vae.config.batch_norm_eps
    z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) \
        + bn.running_mean.to(z).view(1, -1, 1, 1)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3) \
        .reshape(b, c // 4, 2 * h, 2 * w)
    with torch.no_grad():
        decoded = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    image = image_processor.postprocess(decoded, output_type="pil")[0]
    assert image.size == (W, H), f"output {image.size} != {(W, H)}"
    path = os.path.join(OUT, f"{name}-{H}x{W}.png")
    image.save(path)
    saved.append(path)
    log(f"  saved: {path}  (gen {time.time() - t1:.0f}s)")
    del latents, z
    gc.collect()
    flush_cache()

log(f"ALL DONE in {time.time() - t0:.0f}s. saved {len(saved)} images")
for s in saved:
    print(f"MOK {s}", flush=True)
