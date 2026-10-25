"""BLIP-2 multi-path generation and confidence scoring.

Written from notebooks/02_generate_paths.ipynb; the output cache is read by notebooks 03-05.

For each question we store:
- greedy:  the standard single BLIP-2 answer (baseline)
- paths:   k reasoning paths = sampled evidence sentence -> answer, each with
           answer confidence (token log-probabilities) and
           grounding confidence (BLIP-2 image-text matching on the evidence)
"""

import json
import os
import re
import time
import zlib

import torch
from PIL import Image

EVIDENCE_PROMPTS = [
    "Question: {q} Describe the objects in the image that are needed to answer this question.",
    "Question: {q} Briefly describe the part of the image this question is about.",
]
ANSWER_PROMPT = "Context: {e} Question: {q} Short answer:"
GREEDY_PROMPT = "Question: {q} Short answer:"   # BLIP-2 paper's zero-shot VQA prompt


# ---------------------------------------------------------------- models
def _dtype_kw(dtype):
    """transformers >= 4.56 uses `dtype`, older versions `torch_dtype`."""
    import transformers
    from packaging.version import Version
    new = Version(transformers.__version__) >= Version("4.56")
    return {"dtype": dtype} if new else {"torch_dtype": dtype}


def load_models(vqa_id: str, itm_id: str, device: str = "cuda"):
    """BLIP-2 FlanT5-XL for generation (bf16: avoids T5 fp16 overflow) and the ITM model (fp16)."""
    from transformers import AutoProcessor, Blip2ForConditionalGeneration, Blip2ForImageTextRetrieval
    vqa_proc = AutoProcessor.from_pretrained(vqa_id)
    vqa = Blip2ForConditionalGeneration.from_pretrained(vqa_id, **_dtype_kw(torch.bfloat16)).to(device).eval()
    itm_proc = AutoProcessor.from_pretrained(itm_id)
    itm = Blip2ForImageTextRetrieval.from_pretrained(itm_id, **_dtype_kw(torch.float16)).to(device).eval()
    return vqa_proc, vqa, itm_proc, itm


# ---------------------------------------------------------------- generation
@torch.no_grad()
def generate(proc, model, image, prompts, n_per_prompt=1, sample=False,
             max_new_tokens=10, temperature=0.7, top_p=0.9):
    """Generate text for each prompt on one image; return text + token log-probabilities.

    Log-probabilities come from the raw logits (before temperature / top-p),
    so they measure the model's own confidence, not the sampler's.
    """
    inputs = proc(images=[image] * len(prompts), text=prompts, return_tensors="pt", padding=True).to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    kw = dict(max_new_tokens=max_new_tokens, num_return_sequences=n_per_prompt,
              return_dict_in_generate=True, output_logits=True)
    kw.update(dict(do_sample=True, temperature=temperature, top_p=top_p) if sample else dict(do_sample=False))
    out = model.generate(**inputs, **kw)

    seqs = out.sequences[:, 1:]                                   # drop T5 decoder-start token
    logits = torch.stack(out.logits, dim=1).float()               # (batch, steps, vocab)
    lp = logits.log_softmax(-1).gather(-1, seqs.unsqueeze(-1)).squeeze(-1)
    eos = model.generation_config.eos_token_id
    is_eos = torch.isin(seqs, torch.tensor(eos if isinstance(eos, list) else [eos], device=seqs.device))
    mask = (is_eos.cumsum(1) - is_eos.long()) == 0                # tokens up to and including the first end token
    n_tok = mask.sum(1).clamp(min=1)
    lp_sum = (lp * mask).sum(1)
    texts = proc.batch_decode(seqs, skip_special_tokens=True)
    return [
        {"text": t.strip(), "lp_sum": s.item(), "lp_mean": (s / n).item(), "n_tok": int(n)}
        for t, s, n in zip(texts, lp_sum, n_tok)
    ]


@torch.no_grad()
def grounding(proc, model, image, texts):
    """Grounding confidence of each text against the image.

    itm: P(match) from BLIP-2's image-text matching head (main signal)
    itc: image-text cosine similarity (cheaper alternative, kept for ablation)
    """
    texts = [t if t.strip() else "nothing" for t in texts]
    # tokenizer and image processor called separately: the ITM text encoder (BERT) takes
    # plain text, so no image placeholder tokens are inserted
    inputs = dict(proc.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt"))
    inputs["pixel_values"] = proc.image_processor([image] * len(texts), return_tensors="pt")["pixel_values"]
    inputs = {k: v.to(model.device) for k, v in inputs.items() if k in ("input_ids", "attention_mask", "pixel_values")}
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    itm = model(**inputs, use_image_text_matching_head=True).logits_per_image.float().softmax(-1)[:, 1]
    itc = model(**inputs, use_image_text_matching_head=False).logits_per_image.float()
    itc = itc.diagonal() if itc.dim() == 2 else itc
    return itm.tolist(), itc.tolist()


def process_question(models, image, question, k_per_prompt=5, temperature=0.7, top_p=0.9, ev_tokens=40):
    """Greedy answer + k reasoning paths with both confidence signals, for one question."""
    vqa_proc, vqa, itm_proc, itm = models
    greedy = generate(vqa_proc, vqa, image, [GREEDY_PROMPT.format(q=question)])[0]

    ev = generate(vqa_proc, vqa, image, [p.format(q=question) for p in EVIDENCE_PROMPTS],
                  n_per_prompt=k_per_prompt, sample=True, max_new_tokens=ev_tokens,
                  temperature=temperature, top_p=top_p)
    ans = generate(vqa_proc, vqa, image, [ANSWER_PROMPT.format(e=e["text"], q=question) for e in ev])
    itm_s, itc_s = grounding(itm_proc, itm, image, [e["text"] for e in ev])

    paths = [{
        "template": j // k_per_prompt,
        "evidence": e["text"], "ev_lp_mean": e["lp_mean"],
        "answer": a["text"], "lp_mean": a["lp_mean"], "lp_sum": a["lp_sum"], "n_tok": a["n_tok"],
        "itm": g, "itc": c,
    } for j, (e, a, g, c) in enumerate(zip(ev, ans, itm_s, itc_s))]
    return {"greedy": {"answer": greedy["text"], "lp_mean": greedy["lp_mean"], "lp_sum": greedy["lp_sum"]},
            "paths": paths}


# ---------------------------------------------------------------- cache
def read_cache(path: str) -> list[dict]:
    """Read a JSONL cache, skipping a half-written last line after a crash."""
    if not os.path.exists(path):
        return []
    recs = []
    with open(path) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs


def run(df, image_dir: str, out_path: str, models, k: int = 10, temperature=0.7, top_p=0.9,
        seed: int = 42, ev_tokens: int = 40, log_every: int = 25, limit: int | None = None):
    """Process every question in df, appending one JSON line each to out_path.

    Resumable: questions already in out_path are skipped, so a disconnect loses nothing.
    """
    done = {r["id"] for r in read_cache(out_path)}
    todo = df[~df["id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"{len(done)} done, {len(todo)} to go -> {out_path}")
    k_per_prompt = k // len(EVIDENCE_PROMPTS)

    t0 = time.time()
    with open(out_path, "a") as f:
        for i, r in enumerate(todo.itertuples(index=False), 1):
            torch.manual_seed(seed + zlib.crc32(r.id.encode()))   # same paths on every re-run
            image = Image.open(f"{image_dir}/{r.imageId}.jpg").convert("RGB")
            rec = {"id": r.id, **process_question(models, image, r.question, k_per_prompt,
                                                  temperature, top_p, ev_tokens)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % log_every == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)}  {rate:.2f} s/question  ETA {rate * (len(todo) - i) / 60:.0f} min")


# ---------------------------------------------------------------- answers
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")


def normalize(ans: str) -> str:
    """Lower-case, drop punctuation and articles, squeeze spaces (GQA uses exact match)."""
    ans = _PUNCT.sub(" ", ans.lower())
    ans = _ARTICLES.sub(" ", ans)
    return " ".join(ans.split())
