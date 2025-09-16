import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from finvader import finvader
from huggingface_hub import hf_hub_download
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import deque

from exorde.models import (
    Classification,
    LanguageScore,
    Sentiment,
    Embedding,
    TextType,
    Emotion,
    Irony,
    Age,
    Gender,
    Analysis,
)

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)

# -------------------- Tunables -------------------
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "25"))  # Naikkan ke 64 pada GPU
USE_FP16_DEFAULT = bool(int(os.getenv("USE_FP16_DEFAULT", "0"))) # Efektif di GPU (1=True/0=False)
MAX_THREADS_PY = max(4, (os.cpu_count() or 8) - 1)

# Panjang maksimum (bisa override via ENV)
HF_MAX_LEN = int(os.getenv("HF_MAX_LEN", "128"))                # untuk semua pipeline HuggingFace
SEN_TRANSFORMER_MAX_LEN = int(os.getenv("SEN_TRANSFORMER_MAX_LEN", "144"))  # SentenceTransformer

# Hybrid mixing params (bisa override via ENV)
HYBRID_ZS = os.getenv("HYBRID_ZS", "1") not in {"0", "false", "no"}
COS_TOP_K = int(os.getenv("COS_TOP_K", "1"))               # rekomendasi cepat-aman: 2
COS_ACCEPT_THRESHOLD = float(os.getenv("COS_ACCEPT_THRESHOLD", "0.55"))  # dari 0.84 → 0.72
COMBINE_ALPHA = float(os.getenv("COMBINE_ALPHA", "0.40"))  # bobot ZS vs Cosine
# Early-accept margin (gap top1-top2 cosine, *raw* sebelum kalibrasi); 0.06–0.10 lazim
COS_MARGIN_RAW = float(os.getenv("COS_MARGIN_RAW", "0.06"))
# Kalibrasi cosine → [0,1] (eksponen bisa dilunakkan via ENV)
CALIB_EXP = float(os.getenv("CALIB_EXP", "0.95"))  # 1.0=linear, 1.05–1.10 sedikit lebih ketat

# -------------------- Telemetry ------------------
ENABLE_TELEMETRY = True
if os.getenv("ENABLE_TELEMETRY", "").lower() in {"0", "false", "no"}:
    ENABLE_TELEMETRY = False

LOG_TELEMETRY_BATCH = True
if os.getenv("LOG_TELEMETRY_BATCH", "").lower() in {"0", "false", "no"}:
    LOG_TELEMETRY_BATCH = False

METRICS_PATH = os.getenv("METRICS_PATH", "")              # contoh: /tmp/tagger_metrics.prom
MA_WINDOW = int(os.getenv("TELEM_MA_WINDOW", "10"))       # moving-average window (batch)

# -------------------- FAST HEADS (point 2 & 3 sebelumnya) --------------------
# ENV flags khusus heads
FAST_HEADS = os.getenv("FAST_HEADS", "1") not in {"0", "false", "no"}
HEADS_MAX_LEN = int(os.getenv("HEADS_MAX_LEN", "128")) #ASLI 98 TAPI TIDAK SAMA DENGAN 128
HEADS_BATCH_SIZE = int(os.getenv("HEADS_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
USE_INT8_HEADS = os.getenv("USE_INT8_HEADS", "1") not in {"0", "false", "no"}

# -------------------- [POINT 2] Batch size khusus SENTIMENT --------------------
SENTIMENT_BATCH_SIZE = int(os.getenv("SENTIMENT_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))

# -------------------- [POINT 3] INT8 untuk SENTIMENT di CPU -------------------
USE_INT8_SENTIMENT = os.getenv("USE_INT8_SENTIMENT", "1") not in {"0","false","no"}

# -------------------- [POINT 1] FAST mode untuk SENTIMENT ---------------------
FAST_SENTIMENT = os.getenv("FAST_SENTIMENT", "1") not in {"0","false","no"}

# -------------------- [POINT 4] Early-exit SENTIMENT -------------------------
SENTIMENT_EARLY_EXIT = os.getenv("SENTIMENT_EARLY_EXIT", "1") not in {"0","false","no"}
EARLY_EXIT_ABS = float(os.getenv("EARLY_EXIT_ABS", "0.70"))  # |fin_compound| >= 0.70 → kurangi ketergantungan GDB

def _now():
    return time.time()

class _StageTimer:
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled
        self.t0 = None
        self.duration = 0.0
    def __enter__(self):
        if self.enabled:
            self.t0 = _now()
        return self
    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.t0 is not None:
            self.duration = _now() - self.t0
            if LOG_TELEMETRY_BATCH:
                logging.info(f"[TIMER] {self.name}: {self.duration:.3f}s")

class _TelemetryWindow:
    """Moving-average untuk ZS rate & durasi stages."""
    def __init__(self, window=10):
        self.window = window
        self._zs_rate = deque(maxlen=window)
        self._batch_total = deque(maxlen=window)
        self._docs = deque(maxlen=window)
        self._zs_per_doc = deque(maxlen=window)
        self._emb = deque(maxlen=window)
        self._hybrid = deque(maxlen=window)
        self._heads = deque(maxlen=window)
        self._sent = deque(maxlen=window)
        self._build = deque(maxlen=window)

    def update(self, *, zs_rate, batch_total, docs, zs_time_per_doc,
               emb, hybrid, heads, sent, build):
        self._zs_rate.append(zs_rate)
        self._batch_total.append(batch_total)
        self._docs.append(docs)
        self._zs_per_doc.append(zs_time_per_doc)
        self._emb.append(emb)
        self._hybrid.append(hybrid)
        self._heads.append(heads)
        self._sent.append(sent)
        self._build.append(build)

    @staticmethod
    def _avg(q):
        return (sum(q) / len(q)) if q else 0.0

    def snapshot(self):
        return {
            "zs_rate": self._avg(self._zs_rate),
            "batch_total": self._avg(self._batch_total),
            "docs": self._avg(self._docs),
            "zs_per_doc": self._avg(self._zs_per_doc),
            "emb": self._avg(self._emb),
            "hybrid": self._avg(self._hybrid),
            "heads": self._avg(self._heads),
            "sent": self._avg(self._sent),
            "build": self._avg(self._build),
            "window": len(self._batch_total),
        }

_TELEM = _TelemetryWindow(window=MA_WINDOW)

def _emit_prometheus_metrics(*, docs, total_s, zs_rate, zs_time_per_doc,
                             emb_s, hybrid_s, heads_s, sent_s, build_s,
                             avg_snapshot=None):
    lines = []
    lines.append(f'exorde_tagging_docs_total {docs:.0f}')
    lines.append(f'exorde_tagging_batch_duration_seconds{{stage="total"}} {total_s:.6f}')
    lines.append(f'exorde_tagging_batch_duration_seconds{{stage="embeddings"}} {emb_s:.6f}')
    lines.append(f'exorde_tagging_batch_duration_seconds{{stage="hybrid"}} {hybrid_s:.6f}')
    lines.append(f'exorde_tagging_batch_duration_seconds{{stage="heads"}} {heads_s:.6f}')
    lines.append(f'exorde_tagging_batch_duration_seconds{{stage="sentiment"}} {sent_s:.6f}')
    lines.append(f'exorde_tagging_batch_duration_seconds{{stage="build"}} {build_s:.6f}')
    lines.append(f'exorde_tagging_zs_rate_ratio {zs_rate:.6f}')
    lines.append(f'exorde_tagging_zs_time_per_doc_seconds {zs_time_per_doc:.6f}')
    dps = (docs / total_s) if total_s > 0 else 0.0
    lines.append(f'exorde_tagging_docs_per_second {dps:.6f}')
    lines.append(
        f'exorde_tagging_params{{cos_top_k="{COS_TOP_K}", cos_accept_threshold="{COS_ACCEPT_THRESHOLD}", combine_alpha="{COMBINE_ALPHA}", '
        f'batch_size="{DEFAULT_BATCH_SIZE}", hf_max_len="{HF_MAX_LEN}", st_max_len="{SEN_TRANSFORMER_MAX_LEN}", '
        f'cos_margin_raw="{COS_MARGIN_RAW}", calib_exp="{CALIB_EXP}"}} 1'
    )
    if avg_snapshot:
        lines.append(f'exorde_tagging_ma_window_batches {avg_snapshot["window"]}')
        lines.append(f'exorde_tagging_ma_batch_duration_seconds {avg_snapshot["batch_total"]:.6f}')
        lines.append(f'exorde_tagging_ma_zs_rate_ratio {avg_snapshot["zs_rate"]:.6f}')
        lines.append(f'exorde_tagging_ma_zs_time_per_doc_seconds {avg_snapshot["zs_per_doc"]:.6f}')
        lines.append(f'exorde_tagging_ma_docs_per_batch {avg_snapshot["docs"]:.6f}')
        lines.append(f'exorde_tagging_ma_stage_seconds{{stage="embeddings"}} {avg_snapshot["emb"]:.6f}')
        lines.append(f'exorde_tagging_ma_stage_seconds{{stage="hybrid"}} {avg_snapshot["hybrid"]:.6f}')
        lines.append(f'exorde_tagging_ma_stage_seconds{{stage="heads"}} {avg_snapshot["heads"]:.6f}')
        lines.append(f'exorde_tagging_ma_stage_seconds{{stage="sent"}} {avg_snapshot["sent"]:.6f}')
        lines.append(f'exorde_tagging_ma_stage_seconds{{stage="build"}} {avg_snapshot["build"]:.6f}')
    blob = "\n".join(lines)
    if LOG_TELEMETRY_BATCH:
        logging.info("[METRICS]\n%s", blob)
    if METRICS_PATH:
        try:
            with open(METRICS_PATH, "w") as f:
                f.write(blob + "\n")
        except Exception as e:
            logging.warning("[METRICS] Failed to write metrics to %s: %s", METRICS_PATH, e)

def _maybe_quantize_dynamic(model):
    """Dynamic INT8 quantization untuk CPU (Linear layers)."""
    if USE_INT8_HEADS and not torch.cuda.is_available():
        try:
            import torch.nn as nn
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            logging.info("[HEADS] Dynamic int8 quantization applied.")
        except Exception as e:
            logging.warning("[HEADS] Quantization failed: %s", e)
    return model

# [REUSABLE] quantize untuk SENTIMENT (POINT 3)
def _maybe_quantize_dynamic_sent(model):
    if USE_INT8_SENTIMENT and not torch.cuda.is_available():
        try:
            import torch.nn as nn
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            logging.info("[SENT] Dynamic int8 quantization applied.")
        except Exception as e:
            logging.warning("[SENT] Quantization failed: %s", e)
    return model

def _run_head_fast(model, tok, texts, batch_size, max_len, device_id):
    """Tokenize→forward batched; return list of list[(label, score)] mirip output pipeline."""
    dev = torch.device(f"cuda:{device_id}") if device_id != -1 and torch.cuda.is_available() else torch.device("cpu")
    model = model.to(dev).eval()

    out_all = []
    for i in range(0, len(texts), batch_size):
        sub = texts[i:i+batch_size]
        enc = tok(
            sub,
            return_tensors="pt",
            truncation=True,
            padding=True,      # longest per sub-batch
            max_length=max_len
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        labels = [model.config.id2label[j] for j in range(probs.shape[1])]
        for row in probs:
            out_all.append(list(zip(labels, row.astype(float).tolist())))
    return out_all

# -------------------- Pipelines ------------------
def _pipeline_kwargs(device, use_fp16):
    kwargs = dict(
        device=device,
        truncation=True,
        padding=True,
        max_length=HF_MAX_LEN,
        batch_size=DEFAULT_BATCH_SIZE,
    )
    if device != -1 and torch.cuda.is_available() and use_fp16:
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return kwargs

def initialize_models(device: int = -1, use_fp16: bool = USE_FP16_DEFAULT, sentence_batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Initializer cepat + siapkan pipeline untuk hybrid Cosine + Zero-shot.
    (Termasuk perubahan point 2: heads sebagai (model, tokenizer) + dynamic int8 CPU)
    (Tambahan: POINT 1, 3, 4 pada SENTIMENT)
    """
    logging.info("[TAGGING] Initializing models for batched inference...")
    models = {}

    # Global torch hints
    torch.set_grad_enabled(False)
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    if device != -1 and not torch.cuda.is_available():
        logging.warning("[TAGGING] device!=-1 diminta, tapi CUDA tidak tersedia → fallback ke CPU")
        device = -1

    # Embeddings (shared for docs + labels)
    logging.info("[TAGGING] Loading: sentence-transformers/all-MiniLM-L6-v2")
    st_device = "cuda" if device != -1 else "cpu"
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=st_device)
    st_model.max_seq_length = SEN_TRANSFORMER_MAX_LEN
    try:
        st_model.tokenizer.model_max_length = SEN_TRANSFORMER_MAX_LEN
    except Exception:
        pass
    logging.info(f"[TAGGING] ST max_seq_length set to {st_model.max_seq_length}")

    models["sentence_transformer"] = st_model
    models["sentence_batch_size"] = sentence_batch_size
    models["label_cache"] = {}
    models["device_id"] = device

    # Zero-shot (untuk hybrid)
    if HYBRID_ZS:
        logging.info("[TAGGING] Loading (ZS for hybrid): MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33")
        models["zs_pipe"] = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
            hypothesis_template="This text is about {}.",
            multi_label=False,
            **_pipeline_kwargs(device, use_fp16),
        )

    # -------------------- POINT 2 HEADS: (model, tokenizer) --------------------
    text_classification_models = [
        ("Emotion", "SamLowe/roberta-base-go_emotions"),
        ("Irony", "cardiffnlp/twitter-roberta-base-irony"),
        ("TextType", "marieke93/MiniLM-evidence-types"),
    ]
    for col_name, model_name in text_classification_models:
        logging.info(f"[TAGGING][HEADS] Loading: {model_name}")
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tok.model_max_length = HEADS_MAX_LEN  # khusus heads

        mdl = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=(torch.float16 if device != -1 and torch.cuda.is_available() and use_fp16 else None)
        )
        mdl.eval()

        if device == -1:
            mdl = _maybe_quantize_dynamic(mdl)

        # Simpan tuple untuk FAST mode
        models[col_name] = (mdl, tok)

        # Jika ingin legacy pipeline (FAST_HEADS=0), siapkan pipeline juga
        if not FAST_HEADS:
            models[col_name] = pipeline(
                "text-classification",
                model=mdl,
                tokenizer=tok,
                return_all_scores=True,
                **_pipeline_kwargs(device, use_fp16),
            )

    # -------------------- SENTIMENT (POINT 1, 3) sebagai (model, tokenizer) ----
    logging.info("[TAGGING] Loading: VADER + emoji/Loughran dictionaries")
    models["sentiment_analyzer"] = SentimentIntensityAnalyzer()
    try:
        emoji_lexicon = hf_hub_download("ExordeLabs/SentimentDetection", "emoji_unic_lexicon.json")
        loughran_dict = hf_hub_download("ExordeLabs/SentimentDetection", "loughran_dict.json")
        with open(emoji_lexicon) as f:
            models["sentiment_analyzer"].lexicon.update(json.load(f))
        with open(loughran_dict) as f:
            models["sentiment_analyzer"].lexicon.update(json.load(f))
    except Exception:
        logging.info("[TAGGING] Could not extend VADER lexicon; continuing.")

    def _load_sent(model_name: str):
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tok.model_max_length = HF_MAX_LEN
        mdl = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=(torch.float16 if models.get("device_id",-1) != -1 and torch.cuda.is_available() and USE_FP16_DEFAULT else None)
        ).eval()
        if models.get("device_id", -1) == -1:
            mdl = _maybe_quantize_dynamic_sent(mdl)  # INT8 dinamis di CPU
        return (mdl, tok)

    logging.info("[TAGGING] Loading: lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    models["gdb_sent"] = _load_sent("lxyuan/distilbert-base-multilingual-cased-sentiments-student")

    logging.info("[TAGGING] Loading: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    models["fdb_sent"] = _load_sent("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

    # Flag & batch size untuk sentiment FAST
    models["fast_sentiment"] = FAST_SENTIMENT
    models["sentiment_batch_size"] = SENTIMENT_BATCH_SIZE

    logging.info("[TAGGING] Models loaded.")
    return models

# -------------------- Utils ----------------------
def _threaded(map_fn, items, max_workers=MAX_THREADS_PY):
    out = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(map_fn, idx, item): idx for idx, item in enumerate(items)}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
    return out

def _embed_texts(st_model: SentenceTransformer, texts, batch_size, normalize=True):
    return st_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=normalize,
    ).astype(np.float32)

def _get_label_embeddings(models, labeldict):
    key = tuple(labeldict.keys())
    cache = models["label_cache"]
    if key in cache:
        return cache[key]
    st_model = models["sentence_transformer"]
    label_texts = list(key)
    label_embs = _embed_texts(st_model, label_texts, batch_size=max(8, models.get("sentence_batch_size", 32)))
    cache[key] = (label_texts, label_embs)
    return cache[key]

def _cosine_matrix(doc_embs: np.ndarray, label_embs: np.ndarray):
    sims = doc_embs @ label_embs.T
    top_idx = np.argmax(sims, axis=1)
    top_scores = sims[np.arange(sims.shape[0]), top_idx]
    return sims, top_idx, top_scores

def _calibrate_cos_score(x: np.ndarray) -> np.ndarray:
    y = (x + 1.0) / 2.0
    if CALIB_EXP != 1.0:
        y = y ** CALIB_EXP
    return np.clip(y, 0.0, 1.0)

# -------------------- Hybrid Labeling ------------
def _hybrid_label_classify(documents, doc_embs, label_texts, label_embs, models,
                           top_k=COS_TOP_K, accept_t=COS_ACCEPT_THRESHOLD, alpha=COMBINE_ALPHA):
    sims, top_idx, top_scores = _cosine_matrix(doc_embs, label_embs)
    if sims.shape[1] >= 2:
        second_best = np.partition(sims, -2, axis=1)[:, -2]
        margin_raw = top_scores - second_best
        margin_mask = margin_raw >= COS_MARGIN_RAW
    else:
        margin_mask = np.zeros_like(top_scores, dtype=bool)

    cos_scores_01 = _calibrate_cos_score(top_scores)
    need_zs_mask = ~((cos_scores_01 >= accept_t) | margin_mask)
    need_zs_indices = np.nonzero(need_zs_mask)[0].tolist()
    zs_count = len(need_zs_indices)

    N = len(documents)
    out = [None] * N

    accept_idx = np.nonzero(~need_zs_mask)[0]
    for i in accept_idx:
        lbl = label_texts[top_idx[i]]
        sc = float(np.round(cos_scores_01[i], 4))
        out[i] = Classification(label=lbl, score=sc)

    if zs_count == N and accept_t > 0.70:
        accept_t2 = max(0.70, accept_t - 0.08)
        need_zs_mask2 = ~((cos_scores_01 >= accept_t2) | margin_mask)
        recovered_idx = np.nonzero(~need_zs_mask2)[0]
        if len(recovered_idx) > 0 and LOG_TELEMETRY_BATCH:
            logging.info("[HYBRID][RESCUE] lower thr once: %.2f -> %.2f; rescued=%d/%d",
                         accept_t, accept_t2, len(recovered_idx), N)
        for i in recovered_idx:
            if out[i] is None:
                lbl = label_texts[top_idx[i]]
                sc = float(np.round(cos_scores_01[i], 4))
                out[i] = Classification(label=lbl, score=sc)
        need_zs_mask = need_zs_mask2
        need_zs_indices = np.nonzero(need_zs_mask)[0].tolist()
        zs_count = len(need_zs_indices)

    zs_pipe = models.get("zs_pipe")
    zs_start = _now()
    if zs_pipe is not None and zs_count > 0:
        groups = {}
        for i in need_zs_indices:
            k_eff = min(top_k, sims.shape[1]) if sims.shape[1] > 1 else 1
            cand_idx = np.argpartition(-sims[i], kth=max(0, k_eff - 1))[:k_eff]
            cand_labels = [label_texts[j] for j in cand_idx]
            key = tuple(sorted(cand_labels))
            groups.setdefault(key, []).append((i, documents[i], cand_idx))

        for key, items in groups.items():
            cand_labels_sorted = list(key)
            batch_texts = [t for (_, t, _) in items]
            zs_outputs = zs_pipe(batch_texts, candidate_labels=cand_labels_sorted)
            if isinstance(zs_outputs, dict):
                zs_outputs = [zs_outputs]

            for (i, _text, cand_idx), zs in zip(items, zs_outputs):
                zs_label = zs["labels"][0]
                zs_score = float(zs["scores"][0])
                try:
                    j_global = label_texts.index(zs_label)
                except ValueError:
                    j_global = int(np.argmax(sims[i]))
                cos_for_zs_label = float(_calibrate_cos_score(np.array([sims[i, j_global]]))[0])
                final_score = alpha * zs_score + (1.0 - alpha) * cos_for_zs_label
                out[i] = Classification(label=zs_label, score=float(np.round(final_score, 4)))
    else:
        for i in need_zs_indices:
            lbl = label_texts[top_idx[i]]
            sc = float(np.round(cos_scores_01[i], 4))
            out[i] = Classification(label=lbl, score=sc)

    zs_dur = (_now() - zs_start) if zs_count > 0 and zs_pipe is not None else 0.0
    per_doc_zs = (zs_dur / zs_count) if zs_count > 0 else 0.0

    if ENABLE_TELEMETRY and LOG_TELEMETRY_BATCH:
        avg_cos = float(np.mean(cos_scores_01)) if N > 0 else 0.0
        zs_rate = (zs_count / N) if N > 0 else 0.0
        logging.info(
            "[HYBRID] ZS_rate=%.1f%%  N=%d  K=%d  thr=%.2f  alpha=%.2f  margin=%.2f  avg_cos=%.3f  ZS_time=%.3fs (%.3fs/doc)",
            zs_rate * 100.0, N, top_k, accept_t, alpha, COS_MARGIN_RAW, avg_cos, zs_dur, per_doc_zs
        )

    avg_cos_top1 = float(np.mean(cos_scores_01)) if N > 0 else 0.0
    return out, zs_count, per_doc_zs, avg_cos_top1

def _fast_label_classify(doc_embs: np.ndarray, label_embs: np.ndarray, label_texts: list[str]):
    sims = doc_embs @ label_embs.T
    top_idx = np.argmax(sims, axis=1)
    top_sim = sims[np.arange(sims.shape[0]), top_idx]
    scores = _calibrate_cos_score(top_sim)
    return [Classification(label=label_texts[i], score=float(np.round(scores[j], 4)))
            for j, i in enumerate(top_idx)]

# -------------------- Sentiment ------------------
# [POINT 1,2,3] FAST + batch khusus + INT8
def batch_sentiment_analysis(documents: list[str], models: dict):
    sentiment_analyzer = models["sentiment_analyzer"]
    device_id = models.get("device_id", -1)
    bs = models.get("sentiment_batch_size", SENTIMENT_BATCH_SIZE)
    fast = models.get("fast_sentiment", True)

    # --- HF sentiment models dengan _run_head_fast ---
    if fast:
        gdb_mdl, gdb_tok = models["gdb_sent"]
        fdb_mdl, fdb_tok = models["fdb_sent"]

        gdb_raw = _run_head_fast(gdb_mdl, gdb_tok, documents, batch_size=bs, max_len=HF_MAX_LEN, device_id=device_id)
        fdb_raw = _run_head_fast(fdb_mdl, fdb_tok, documents, batch_size=bs, max_len=HF_MAX_LEN, device_id=device_id)

        def _pos_minus_neg_from_fast(raw_list):
            res = np.empty(len(raw_list), dtype=np.float32)
            for i, scores in enumerate(raw_list):
                d = {lbl.lower(): float(sc) for (lbl, sc) in scores}
                res[i] = d.get("positive", 0.0) - d.get("negative", 0.0)
            return res

        gdb_scores = _pos_minus_neg_from_fast(gdb_raw)
        fdb_scores = _pos_minus_neg_from_fast(fdb_raw)

    else:
        # fallback ke pipeline (tidak direkomendasikan)
        gdb_pipe = models["gdb_pipe"]
        fdb_pipe = models["fdb_pipe"]

        def _pos_minus_neg_from_scores(scores_list):
            res = np.empty(len(scores_list), dtype=np.float32)
            for i, scores in enumerate(scores_list):
                d = {s["label"].lower(): float(s["score"]) for s in scores}
                res[i] = d.get("positive", 0.0) - d.get("negative", 0.0)
            return res

        gdb_scores = _pos_minus_neg_from_scores(gdb_pipe(documents))
        fdb_scores = _pos_minus_neg_from_scores(fdb_pipe(documents))

    # --- VADER & FinVADER di thread ---
    def _vader_job(i, text):
        return float(sentiment_analyzer.polarity_scores(text)["compound"])
    vader_scores = np.array(_threaded(_vader_job, documents), dtype=np.float32)

    def _fin_job(i, text):
        return float(finvader(text, use_sentibignomics=True, use_henry=True, indicator="compound"))
    fin_vader_scores = np.array(_threaded(_fin_job, documents), dtype=np.float32)

    # [POINT 4] Hitung mask early-exit (berdasar gabungan finansial)
    fin_compound_tmp = np.round(0.70 * fdb_scores + 0.30 * fin_vader_scores, 2)
    skip_gdb_mask = (np.abs(fin_compound_tmp) >= EARLY_EXIT_ABS) if SENTIMENT_EARLY_EXIT else None

    return vader_scores, fin_vader_scores, fdb_scores, gdb_scores, skip_gdb_mask

# [POINT 4] kombiner menerima skip_gdb_mask (opsional) untuk kurangi bobot GDB saat finansial sangat yakin
def compute_compound_sentiments(
    vader_scores: np.ndarray,
    fin_vader_scores: np.ndarray,
    fdb_scores: np.ndarray,
    gdb_scores: np.ndarray,
    skip_gdb_mask: np.ndarray | None = None,
):
    # Blend finansial ringkas
    fin_compound = np.round(0.70 * fdb_scores + 0.30 * fin_vader_scores, 2)

    compound = np.empty_like(fin_compound)
    abs_fin = np.abs(fin_compound)

    # default buckets
    mask_hi  = abs_fin >= 0.6
    mask_mid = (abs_fin >= 0.4) & ~mask_hi
    mask_low = (abs_fin >= 0.1) & ~mask_hi & ~mask_mid
    mask_min = ~mask_hi & ~mask_mid & ~mask_low

    # Early-exit: tambahkan area hi jika skip_gdb_mask aktif
    if skip_gdb_mask is not None:
        mask_hi = mask_hi | skip_gdb_mask

    compound[mask_hi]  = 0.30 * gdb_scores[mask_hi] + 0.10 * vader_scores[mask_hi] + 0.60 * fin_compound[mask_hi]
    compound[mask_mid] = 0.40 * gdb_scores[mask_mid] + 0.20 * vader_scores[mask_mid] + 0.40 * fin_compound[mask_mid]
    compound[mask_low] = 0.60 * gdb_scores[mask_low] + 0.25 * vader_scores[mask_low] + 0.15 * fin_compound[mask_low]
    compound[mask_min] = 0.60 * gdb_scores[mask_min] + 0.40 * vader_scores[mask_min]

    compound = np.round(compound, 2)
    return compound, fin_compound

# -------------------- Main Tag -------------------
@torch.inference_mode()
def tag(documents: list[str], lab_configuration):
    """Optimized batch tagging function (hybrid Cosine + Zero-shot untuk labels)."""
    if not documents:
        return []
    assert all(isinstance(doc, str) for doc in documents)

    if LOG_TELEMETRY_BATCH:
        logging.info(f"[TAGGING] Starting optimized pipeline for {len(documents)} docs...")

    total_t = _StageTimer("BATCH_TOTAL", ENABLE_TELEMETRY); total_t.__enter__()

    # 1) Embeddings
    emb_t = _StageTimer("Embeddings", ENABLE_TELEMETRY); emb_t.__enter__()
    models = lab_configuration["models"]
    st = models["sentence_transformer"]
    st_batch_size = models.get("sentence_batch_size", DEFAULT_BATCH_SIZE)
    doc_embs = _embed_texts(st, documents, batch_size=st_batch_size, normalize=True)
    embeddings_list = doc_embs.astype(np.float32).tolist()
    emb_t.__exit__(None, None, None)

    # 2) Label classification — HYBRID
    hyb_t = _StageTimer("Labeling(HYBRID)", ENABLE_TELEMETRY); hyb_t.__enter__()
    labeldict = lab_configuration["labeldict"]
    label_texts, label_embs = _get_label_embeddings(models, labeldict)
    if HYBRID_ZS and ("zs_pipe" in models):
        classifications, zs_count, per_doc_zs, avg_cos = _hybrid_label_classify(
            documents, doc_embs, label_texts, label_embs, models,
            top_k=COS_TOP_K, accept_t=COS_ACCEPT_THRESHOLD, alpha=COMBINE_ALPHA
        )
    else:
        classifications = _fast_label_classify(doc_embs, label_embs, label_texts)
        zs_count, per_doc_zs, avg_cos = 0, 0.0, 0.0
    hyb_t.__exit__(None, None, None)

    # 3) Heads cepat
    heads_t = _StageTimer("Heads(Emotion/Irony/TextType)", ENABLE_TELEMETRY); heads_t.__enter__()
    tc_results = {}
    if FAST_HEADS:
        device_id = models.get("device_id", -1)
        for col_name in ("Emotion", "Irony", "TextType"):
            head = models[col_name]
            if isinstance(head, tuple):  # (model, tokenizer)
                mdl, tok = head
                preds = _run_head_fast(
                    mdl, tok, documents,
                    batch_size=HEADS_BATCH_SIZE,
                    max_len=HEADS_MAX_LEN,
                    device_id=device_id
                )
                tc_results[col_name] = [[(lbl, float(sc)) for (lbl, sc) in pred] for pred in preds]
            else:
                preds = head(documents)
                tc_results[col_name] = [[(y["label"], float(y["score"])) for y in pred] for pred in preds]
    else:
        for col_name in ("Emotion", "Irony", "TextType"):
            pipe = models[col_name]
            preds = pipe(documents)  # return_all_scores=True
            tc_results[col_name] = [[(y["label"], float(y["score"])) for y in pred] for pred in preds]
    heads_t.__exit__(None, None, None)

    # 4) Sentiment (FAST + INT8 + Early-exit)
    sent_t = _StageTimer("Sentiment", ENABLE_TELEMETRY); sent_t.__enter__()
    vader_scores, fin_vader_scores, fdb_scores, gdb_scores, skip_gdb_mask = batch_sentiment_analysis(documents, models)
    compound_sentiments, compound_financial_sentiments = compute_compound_sentiments(
        vader_scores, fin_vader_scores, fdb_scores, gdb_scores, skip_gdb_mask=skip_gdb_mask
    )
    sent_t.__exit__(None, None, None)

    # 5) Build results
    build_t = _StageTimer("BuildResults", ENABLE_TELEMETRY); build_t.__enter__()
    results = []
    for i, _ in enumerate(documents):
        sentiment = Sentiment(float(compound_sentiments[i]))
        embedding = Embedding(embeddings_list[i])
        classification = classifications[i]

        gender = Gender(male=0.5, female=0.5)

        tt = {lbl: sc for (lbl, sc) in tc_results["TextType"][i]}
        text_type = TextType(
            assumption=tt.get("Assumption", 0.0),
            anecdote=tt.get("Anecdote", 0.0),
            none=tt.get("None", 0.0),
            definition=tt.get("Definition", 0.0),
            testimony=tt.get("Testimony", 0.0),
            other=tt.get("Other", 0.0),
            study=tt.get("Statistics/Study", 0.0),
        )

        em = {k: round(v, 4) for k, v in tc_results["Emotion"][i]}
        emotion = Emotion(
            love=em.get("love", 0.0),
            admiration=em.get("admiration", 0.0),
            joy=em.get("joy", 0.0),
            approval=em.get("approval", 0.0),
            caring=em.get("caring", 0.0),
            excitement=em.get("excitement", 0.0),
            gratitude=em.get("gratitude", 0.0),
            desire=em.get("desire", 0.0),
            anger=em.get("anger", 0.0),
            optimism=em.get("optimism", 0.0),
            disapproval=em.get("disapproval", 0.0),
            grief=em.get("grief", 0.0),
            annoyance=em.get("annoyance", 0.0),
            pride=em.get("pride", 0.0),
            curiosity=em.get("curiosity", 0.0),
            neutral=em.get("neutral", 0.0),
            disgust=em.get("disgust", 0.0),
            disappointment=em.get("disappointment", 0.0),
            realization=em.get("realization", 0.0),
            fear=em.get("fear", 0.0),
            relief=em.get("relief", 0.0),
            confusion=em.get("confusion", 0.0),
            remorse=em.get("remorse", 0.0),
            embarrassment=em.get("embarrassment", 0.0),
            surprise=em.get("surprise", 0.0),
            sadness=em.get("sadness", 0.0),
            nervousness=em.get("nervousness", 0.0),
        )

        ir = {lbl: sc for (lbl, sc) in tc_results["Irony"][i]}
        irony = Irony(irony=ir.get("irony", 0.0), non_irony=ir.get("non_irony", 0.0))

        age = Age(below_twenty=0.0, twenty_thirty=0.0, thirty_forty=0.0, forty_more=0.0)
        language_score = LanguageScore(1.0)

        results.append(Analysis(
            classification=classification,
            language_score=language_score,
            sentiment=sentiment,
            embedding=embedding,
            gender=gender,
            text_type=text_type,
            emotion=emotion,
            irony=irony,
            age=age,
        ))
    build_t.__exit__(None, None, None)
    total_t.__exit__(None, None, None)

    if ENABLE_TELEMETRY:
        docs = len(documents)
        zs_rate = (zs_count / docs) if docs > 0 else 0.0
        _TELEM.update(
            zs_rate=zs_rate,
            batch_total=total_t.duration,
            docs=docs,
            zs_time_per_doc=per_doc_zs,
            emb=emb_t.duration,
            hybrid=hyb_t.duration,
            heads=heads_t.duration,
            sent=sent_t.duration,
            build=build_t.duration,
        )
        snap = _TELEM.snapshot()

        if LOG_TELEMETRY_BATCH:
            dps = (docs/total_t.duration) if total_t.duration > 0 else 0.0
            logging.info(
                "[BATCH] docs=%d  TOTAL=%.3fs  ZS=%.1f%%  DPS=%.2f  params: K=%d thr=%.2f alpha=%.2f batch=%d HF_MAX=%d ST_MAX=%d HEADS_MAX=%d FAST_HEADS=%s SENT_BATCH=%d EARLY_EXIT=%s",
                docs, total_t.duration, zs_rate*100.0, dps,
                COS_TOP_K, COS_ACCEPT_THRESHOLD, COMBINE_ALPHA,
                DEFAULT_BATCH_SIZE, HF_MAX_LEN, SEN_TRANSFORMER_MAX_LEN, HEADS_MAX_LEN, str(FAST_HEADS),
                SENTIMENT_BATCH_SIZE, str(SENTIMENT_EARLY_EXIT)
            )
    logging.info(f"[TAGGING] Update Tagging Bluesky v1.3.2")
    logging.info(f"[TAGGING] SIMPLE (margin+rescue+batchedZS+calib_exp+FAST_HEADS+FAST_SENT+INT8+EARLY_EXIT) tag_margin_rescue_batchedZS_calib_exp_fast_head_fast_sentiment.py")
    return results
