---
name: moss-image-news-illustration
version: 1.0.0
author: Hermes Agent (curator)
license: MIT
description: "Use when illustrating a news article with MOSS-Image."
metadata:
  hermes:
    tags: [moss-image, news-illustration, gpu-swap, photorealistic]
---

# MOSS-Image: news article illustrations (no layout, no in-image text)

## When to Use
The user asks to **illustrate a news article/story** (e.g. "ilustra el
artículo principal") — one or several photorealistic 1024×1024 images,
NO layout JSON, NO in-image text. This is a different task from the meeting
infographic (layout JSON + real text) — see `meeting-infographic` for that.

Model, GPU-swap mechanics, and batch pipeline are shared:
- Model: MOSS-Image-SFT (same as infographics). Swap mechanics: the
  `hyperqwen-gpu-swap` skill (OFF→target→ON, systemd-run, KillMode=process,
  flock, per-engine timeout).
- Batch script: this skill's `scripts/infer3.py` — copy it to
  `~/.hermes/gpu-swaps/infer3.py` on first use (the cache dir
  `~/.hermes/cache/moss-image/` is pruned after 24h idle). It reads a
  PROMPTS JSON array of `{"name", "prompt", "negative"?, "seed"?}`, loads
  encoder/DiT/VAE once, 50 steps at 1024², CFG 3.0, saves
  `~/.hermes/gpu-swaps/moss-out/<name>-1024x1024.png`. Target/swap scripts:
  copy this skill's `scripts/moss-news-target.sh` and
  `scripts/moss-news-swap.sh` to `~/.hermes/gpu-swaps/` (all paths are env
  driven with `~/.hermes` defaults; per round set `LOG`/`PROMPTS` in the
  target script, or pass them in the systemd-run Environment). New systemd
  unit name per batch.

## The prompt recipe (user-validated through the 2026-10-25/26 A/B ladder)
- **Positive prompt = "Generate a photorealistic image of: " + the image
  description + " No text." + ONE short style expression.**
- **Style expression** (one, short): user-validated best (2026-10-26, judged
  from 16-image batch): "Shallow depth of field, soft bokeh." and
  "Atmospheric, moody, dramatic light."; similar: "Soft focus, dreamy
  background blur.", "Volumetric lighting, cinematic contrast.",
  "Cinematic." (5-variant A/B: the bokeh + moody ones beat cinematic/natural
  light/documentary for text-free output).
- **Negative prompt (the heavy anti-text work goes here, NOT in the
  positive):** `text, words, nsfw, worst quality, bad anatomy, watermark,
  jpeg artifact` — `text, words` FIRST (word order matters per the user; the
  earlier order with them last was weaker). Do NOT pad the positive with
  long anti-text prose (a "Clean, minimalist ... zero lettering." tail
  RESTYLES the image, it doesn't suppress text).
- **Variants = same prompt, different seeds.** Batch all N in ONE swap
  (model loading is ~2.5 min, ~1.5–2 min per image): 10 images ≈ 16 min vs
  ~4.5 min each in separate swaps.

## Content design (the article-driven mode — user-directed, current default)
Read the FULL article body (not just the summary), then design one concrete
scene per key article concept. Two hard rules from the user's corrections:

1. **Fidelity review phase (do this BEFORE generating, every round):** after
   drafting each prompt, re-read the article and check every concrete claim
   in the prompt against it. Two observed failure modes to catch:
   - **Wrong magnitude/unit:** the article says "1 million TOKENS of context"
     → a prompt describing "a million PAGES of text" is a fabrication
     (a million tokens ≈ a few hundred pages at most). If a scene needs a
     visual for an abstract number, describe the *concept* (e.g. "a chip
     surrounded by a towering library of books") and do not attach a count
     to it that the article does not make.
   - **Wrong mechanism:** "bilingual (English and Chinese, trained on both)"
     → a prompt of "two people of different cultures in a conversation"
     implies a translator mediating, which the article does not say. The
     model being trained on both languages is a property of the model, not
     a scene with two speakers. Prefer a single concrete object that carries
     the property (e.g. "a chip with two interlocking gear systems, one
     shaped by Latin letterforms and one by Chinese character shapes" —
     visual glyph-shapes, not rendered text — and still confirm the user
     reads the scene as intended).
   Fix the prompt before the swap; if a concept has no honest visual, skip
   it rather than inventing a scene that misrepresents the article.
2. **Be concrete; never use vague hedges like "suggesting ...".** The model
   fills "suggesting X" with its own (often weird) interpretation. Describe
   the image in as much detail as needed: objects, their positions, sizes,
   lighting, mood, and any non-text visual encodings of the concept.
   A prompt of 60–120 well-chosen words beats a 20-word one with hedges.

The ladder history (what was tried, in order, for the Spark-X2.6-4B lead):
summary-only positive → title-only text images → no-text + summary (ES then
EN) → negative-prompt iterations (order of text/words) → style-expression
A/B (5 styles × 3 seeds; 2 winners) → 16-image batch (4 styles × 4 seeds)
→ **article-driven concrete scenes (current mode; 5 scenes × 2 seeds),
with the fidelity-review correction applied to this skill.**

## Procedure
1. Read the FULL article body of the target story (editions JSON: `lead.body`
   or the relevant note's `body`).
2. Draft 3–5 scene prompts per the recipe above; run the fidelity review
   against the article; fix; show the user the exact prompts before the swap
   (the user reviews prompt text, not just images).
3. Write the batch JSON to `~/.hermes/gpu-swaps/<topic>_<round>.json` (new
   name per round so rounds stay comparable); 2–4 seeds per prompt.
4. Pre-flight: `json.load` the JSON; check `/tmp/gpu-swap.lock` empty and
   hyperqwen serving (`curl /v1/models` :8081); `bash -n` both target and
   swap scripts.
5. Tell the user the expected duration (~16 min for 10) and that you cannot
   answer meanwhile (the agent runs on the model being swapped out).
6. Launch: `systemd-run --user --unit=gpu-swap-<round> --property=TimeoutStartSec=0 --property=KillMode=process bash ~/.hermes/gpu-swaps/moss-<round>-swap.sh`.
7. Wait for the unit to go inactive; verify the swap log shows
   "SWAP COMPLETO: target rc=0, on rc=0" and vLLM serving again; count PNGs
   in moss-out (expect one per job).
8. Deliver all images as MEDIA: attachments grouped by prompt (prompt text
   above each pair/quad). **No vision verification possible** (the local LLM
   has no vision; `vision_analyze` returns 400 on these) — the user judges
   text-freeness and fidelity. On a user correction, fix ONLY that prompt,
   keep the round JSON under a new name, and re-batch just the corrected
   prompts.
