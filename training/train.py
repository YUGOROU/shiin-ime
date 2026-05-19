# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tqdm>=4.60",
#   "numpy>=1.24",
#   "huggingface_hub>=0.23",
#   "hf_transfer>=0.1",
#   "trackio[gpu]>=0.25",
# ]
# ///
"""
train.py — 子音のみIME Transformer+GRU Seq2Seq 訓練 (Phase 2)

入力: 子音列 (a-z, SRC_VSZ=29)
出力: ひらがな+カタカナ読み (TGT_VSZ≈176)
"""

import os
import json
import csv
import math
import random
import time
import argparse
from pathlib import Path

try:
    import torch
except ImportError:
    raise SystemExit("torch not found. PyTorch を先にインストールしてください: https://pytorch.org")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# ── 語彙 ─────────────────────────────────────────────────────────────────
PAD, SOS, EOS = 0, 1, 2

# 入力語彙: 子音列 (a-z + 特殊トークン)
SRC_VOCAB = ["<pad>", "<sos>", "<eos>"] + list("abcdefghijklmnopqrstuvwxyz")
SRC_C2I   = {c: i for i, c in enumerate(SRC_VOCAB)}
SRC_VSZ   = len(SRC_VOCAB)  # 29

# 出力語彙: ひらがな + カタカナ + ー
_HIRAGANA = [chr(c) for c in range(0x3041, 0x3097)]  # ぁ-ゖ (86文字)
_KATAKANA = [chr(c) for c in range(0x30A1, 0x30F7)]  # ァ-ヶ (86文字)
TGT_VOCAB = ["<pad>", "<sos>", "<eos>"] + _HIRAGANA + _KATAKANA + ["ー"]
TGT_C2I   = {c: i for i, c in enumerate(TGT_VOCAB)}
TGT_I2C   = {i: c for i, c in enumerate(TGT_VOCAB)}
TGT_VSZ   = len(TGT_VOCAB)  # 176

TRACKIO_PROJECT = "shiin-ime"


def _enc_src(s: str) -> list[int]:
    return [SOS] + [SRC_C2I[c] for c in s if c in SRC_C2I] + [EOS]


def _enc_tgt(s: str) -> list[int]:
    return [SOS] + [TGT_C2I[c] for c in s if c in TGT_C2I] + [EOS]


def _dec_tgt(ids) -> str:
    return "".join(TGT_I2C[i] for i in ids if i not in (PAD, SOS, EOS) and i in TGT_I2C)


# ── StreamDataset ────────────────────────────────────────────────────────
MAX_SRC, MAX_TGT = 80, 150

class StreamDataset(IterableDataset):
    """
    JSONL を逐次読み込み + シャッフルバッファ。
    sentence_ratio: 訓練バッチに占める sentence ペアの目標比率 (0.0-1.0)。
    オンライン受理サンプリングで比率を維持する。
    """
    def __init__(self, paths: list[Path], split="train",
                 val_ratio=0.05, buf_size=200_000, seed=42,
                 epoch_size: int | None = None,
                 sentence_ratio: float = 0.7):
        self.paths          = [str(p) for p in sorted(paths)]
        self.split          = split
        self.val_every      = max(1, round(1.0 / val_ratio))
        self.buf_size       = buf_size
        self.seed           = seed
        self.epoch_size     = epoch_size
        self.sentence_ratio = sentence_ratio
        self._epoch         = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wid, nw = (worker_info.id, worker_info.num_workers) if worker_info else (0, 1)
        my_paths = [p for i, p in enumerate(self.paths) if i % nw == wid]

        rng = np.random.default_rng(self.seed + self._epoch * 997 + wid)
        buf: list = []

        n_files = len(self.paths)
        if self.epoch_size and n_files > 0:
            per_file = self.epoch_size // n_files
            skip     = self._epoch * per_file
        else:
            per_file = None
            skip     = 0

        # オンライン受理サンプリング用カウンタ
        sent_seen = word_seen = 0

        for path in my_paths:
            line_no = file_train = file_collected = 0
            with open(path, "rb") as f:
                for raw in f:
                    is_val = (line_no % self.val_every == 0)
                    line_no += 1
                    if (self.split == "val") != is_val:
                        continue
                    file_train += 1
                    if file_train <= skip:
                        continue
                    if per_file and file_collected >= per_file:
                        break
                    try:
                        d = json.loads(raw)
                        src = d.get("consonants", "")
                        tgt = d.get("reading", "")
                        typ = d.get("type", "word")
                    except Exception:
                        continue
                    if not tgt or len(src) > MAX_SRC or len(tgt) > MAX_TGT:
                        continue

                    # ── sentence_ratio に基づく受理サンプリング ──────────
                    total = sent_seen + word_seen
                    if typ == "sentence":
                        cur_sent_ratio = sent_seen / total if total > 0 else 0.0
                        if cur_sent_ratio > self.sentence_ratio and total > 1000:
                            # sentence が目標を超えたら確率的にスキップ
                            if rng.random() > self.sentence_ratio:
                                continue
                        sent_seen += 1
                    else:
                        cur_word_ratio = word_seen / total if total > 0 else 1.0
                        if cur_word_ratio > (1.0 - self.sentence_ratio) and total > 1000:
                            if rng.random() > (1.0 - self.sentence_ratio):
                                continue
                        word_seen += 1

                    buf.append((_enc_src(src), _enc_tgt(tgt)))
                    file_collected += 1
                    if len(buf) >= self.buf_size:
                        perm = rng.permutation(len(buf))
                        for idx in perm:
                            yield buf[idx]
                        buf.clear()

        if buf:
            perm = rng.permutation(len(buf))
            for idx in perm:
                yield buf[idx]


def collate(batch):
    srcs, tgts = zip(*batch)
    sl = [len(s) for s in srcs]
    tl = [len(t) for t in tgts]
    S = np.zeros((len(batch), max(sl)), dtype=np.int64)
    T = np.zeros((len(batch), max(tl)), dtype=np.int64)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        S[i, :len(s)] = s
        T[i, :len(t)] = t
    return (torch.from_numpy(S), torch.from_numpy(T),
            torch.tensor(sl, dtype=torch.long), torch.tensor(tl, dtype=torch.long))


# ── モデル ────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


class Encoder(nn.Module):
    def __init__(self, vsz: int, emb: int, hid: int, layers: int,
                 nhead: int, drop: float):
        super().__init__()
        self.embed = nn.Embedding(vsz, emb, padding_idx=PAD)
        self.pos   = PositionalEncoding(emb, max_len=MAX_SRC + 4, dropout=drop)
        enc_layer  = nn.TransformerEncoderLayer(
            emb, nhead, dim_feedforward=hid * 2,
            dropout=drop, batch_first=True, norm_first=True,
            enable_nested_tensor=False,
        )
        self.tf   = nn.TransformerEncoder(enc_layer, layers)
        self.proj = nn.Linear(emb, hid)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        B, T = x.shape
        # src_key_padding_mask: True = padding (ignore)
        pad_mask = torch.arange(T, device=x.device).unsqueeze(0) >= lens.to(x.device).unsqueeze(1)
        e       = self.pos(self.embed(x))
        enc_out = self.proj(self.tf(e, src_key_padding_mask=pad_mask))  # (B, T, hid)

        # Decoder 初期 hidden: 有効位置の平均プーリング → (B, hid)
        valid = (~pad_mask).unsqueeze(-1).float()
        h = (enc_out * valid).sum(1) / valid.sum(1).clamp(min=1)
        return enc_out, h, pad_mask  # pad_mask を Attention でも使う


class Attention(nn.Module):
    def __init__(self, hid: int):
        super().__init__()
        self.W = nn.Linear(hid * 2, hid)
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, h_top: torch.Tensor,
                enc_out: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        T      = enc_out.size(1)
        energy = self.v(torch.tanh(
            self.W(torch.cat([h_top.unsqueeze(1).expand(-1, T, -1), enc_out], dim=2))
        )).squeeze(2)
        return F.softmax(energy.masked_fill(pad_mask, -1e4), dim=1)


class Decoder(nn.Module):
    def __init__(self, emb: int, hid: int, layers: int, drop: float):
        super().__init__()
        self.embed = nn.Embedding(TGT_VSZ, emb, padding_idx=PAD)
        self.attn  = Attention(hid)
        self.rnn   = nn.GRU(emb + hid, hid, layers, batch_first=True,
                            dropout=drop if layers > 1 else 0.0)
        self.fc    = nn.Linear(hid * 2 + emb, TGT_VSZ)
        self.drop  = nn.Dropout(drop)

    def step(self, tok: torch.Tensor, h: torch.Tensor,
             enc_out: torch.Tensor, pad_mask: torch.Tensor):
        e   = self.drop(self.embed(tok.unsqueeze(1)))        # (B, 1, emb)
        a   = self.attn(h[-1], enc_out, pad_mask)            # (B, T)
        ctx = torch.bmm(a.unsqueeze(1), enc_out)             # (B, 1, hid)
        _, h2 = self.rnn(torch.cat([e, ctx], dim=2), h)
        out = self.fc(torch.cat([h2[-1], ctx.squeeze(1), e.squeeze(1)], dim=1))
        return out, h2


class Seq2Seq(nn.Module):
    def __init__(self, hid: int = 256, emb: int = 64,
                 enc_layers: int = 3, dec_layers: int = 2,
                 nhead: int = 4, drop: float = 0.2):
        super().__init__()
        self.enc = Encoder(SRC_VSZ, emb, hid, enc_layers, nhead, drop)
        self.dec = Decoder(emb, hid, dec_layers, drop)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_lens: torch.Tensor, tf: float = 0.5):
        B, TL      = tgt.shape
        enc_out, h, pad_mask = self.enc(src, src_lens)
        h = h.unsqueeze(0).expand(self.dec.rnn.num_layers, -1, -1).contiguous()
        logits = torch.zeros(B, TL, TGT_VSZ, device=src.device)
        tok    = tgt[:, 0]
        for t in range(1, TL):
            logit, h = self.dec.step(tok, h, enc_out, pad_mask)
            logits[:, t] = logit
            tok = tgt[:, t] if random.random() < tf else logit.argmax(1)
        return logits

    @torch.no_grad()
    def predict_top3(self, src_str: str, beam: int = 10,
                     max_len: int = 150, device: str = "cpu") -> list[tuple[str, float]]:
        src = torch.tensor(_enc_src(src_str), dtype=torch.long, device=device).unsqueeze(0)
        sl  = torch.tensor([src.size(1)], device=device)
        enc_out, h0, pad_mask = self.enc(src, sl)
        h = h0.unsqueeze(0).expand(self.dec.rnn.num_layers, -1, -1).contiguous()
        beams, done = [(0.0, [SOS], h)], []
        for _ in range(max_len):
            cands = []
            for score, toks, bh in beams:
                if toks[-1] == EOS:
                    done.append((score / max(len(toks), 1), toks)); continue
                tok   = torch.tensor([toks[-1]], device=device)
                logit, bh2 = self.dec.step(tok, bh, enc_out, pad_mask)
                lp    = F.log_softmax(logit, dim=1).squeeze(0)
                for v, idx in zip(*lp.topk(beam)):
                    cands.append((score + v.item(), toks + [idx.item()], bh2))
            cands.sort(key=lambda x: x[0], reverse=True)
            beams = cands[:beam]
            if len(done) >= 3: break
        for score, toks, _ in beams:
            done.append((score / max(len(toks), 1), toks))
        done.sort(key=lambda x: x[0], reverse=True)
        return [(_dec_tgt(t[1:]), round(s, 4)) for s, t in done[:3]]


# ── CER ──────────────────────────────────────────────────────────────────
def _cer(pred: str, ref: str) -> float:
    if not ref: return 0.0
    n, m = len(pred), len(ref)
    dp   = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            dp[j] = min(prev[j]+1, dp[j-1]+1, prev[j-1]+(pred[i-1]!=ref[j-1]))
    return dp[m] / m

@torch.no_grad()
def evaluate(model, loader, device, n: int = 4000) -> float:
    model.eval()
    total, count = 0.0, 0
    for src, tgt, sl, _ in loader:
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=False):
            logits = model(src, tgt, sl.to(device), tf=0.0)
        preds = logits.argmax(-1)
        for i in range(src.size(0)):
            total += _cer(_dec_tgt(preds[i].tolist()), _dec_tgt(tgt[i].tolist()))
            count += 1
            if count >= n: break
        if count >= n: break
    model.train()
    return total / max(count, 1)


# ── HuggingFace データセットダウンロード ─────────────────────────────────
def _download_dataset_from_hf(repo_id: str, data_dir: Path):
    from huggingface_hub import hf_hub_download, list_repo_files
    token = os.environ.get("HF_TOKEN")

    api_repo = repo_id if "/" in repo_id else f"{repo_id}"
    try:
        files = sorted(list_repo_files(api_repo, repo_type="dataset", token=token))
    except Exception as e:
        raise RuntimeError(f"HF dataset listing failed for {api_repo}: {e}")

    jsonl_files = [f for f in files if f.endswith(".jsonl")]
    if not jsonl_files:
        raise RuntimeError(f"No .jsonl files found in {api_repo}")

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(jsonl_files)} JSONL files from {api_repo}...")
    for fname in jsonl_files:
        hf_hub_download(
            repo_id=api_repo, filename=fname,
            repo_type="dataset", token=token,
            local_dir=str(data_dir),
        )
        print(f"  Downloaded: {fname}")


# ── HuggingFace モデルアップロード ────────────────────────────────────────
def _upload_model_to_hf(out_dir: Path, args):
    token = os.environ.get("HF_TOKEN")
    if not token: return

    from huggingface_hub import HfApi, ModelCard

    api       = HfApi(token=token)
    username  = api.whoami()["name"]
    repo_name = args.hf_model_repo or "shiin-ime-gru"
    repo_id   = repo_name if "/" in repo_name else f"{username}/{repo_name}"

    print(f"Uploading model to {repo_id} (private)...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

    for fname in ["model_best.pt", "model_final.pt", "vocab.json", "training_log.csv"]:
        fpath = out_dir / fname
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  Uploaded: {fname}")

    card = f"""\
---
license: other
language:
- ja
tags:
- transformer
- gru
- seq2seq
- ime
- japanese
private: true
---

# shiin-ime (Phase 2)

子音のみ日本語IME — Transformer Encoder + GRU Decoder Seq2Seq。

## Architecture
- Transformer Encoder (layers={args.enc_layers}, d={args.hidden}, heads={args.nhead})
- GRU Decoder (layers={args.dec_layers}, hidden={args.hidden})
- Input vocab: a-z consonants (SRC_VSZ=29)
- Output vocab: hiragana + katakana (TGT_VSZ≈176)
"""
    ModelCard(card).push_to_hub(repo_id, token=token)
    print(f"Model uploaded: https://huggingface.co/models/{repo_id}")


# ── Trackio 初期化 ─────────────────────────────────────────────────────────
def _init_trackio(args) -> bool:
    token = os.environ.get("HF_TOKEN")
    try:
        import trackio
        kwargs: dict = dict(
            project=TRACKIO_PROJECT,
            name=f"run_{int(time.time())}",
            config=vars(args),
            auto_log_gpu=True,
            gpu_log_interval=30.0,
            embed=False,
        )
        if token and args.trackio_space:
            from huggingface_hub import HfApi
            username   = HfApi(token=token).whoami()["name"]
            space_name = args.trackio_space
            space_id   = space_name if "/" in space_name else f"{username}/{space_name}"
            kwargs["space_id"] = space_id
            kwargs["private"]  = True
            print(f"Trackio → HF Space: https://huggingface.co/spaces/{space_id}")
        else:
            print("Trackio → local")
        trackio.init(**kwargs)
        return True
    except Exception as e:
        print(f"Trackio init failed: {e}. Training continues without tracking.")
        return False


# ── 訓練ループ ────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {vram:.1f}GB")
    print(f"Device: {device}")

    # ── HF からデータをDL ────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    if args.hf_dataset_repo:
        _download_dataset_from_hf(args.hf_dataset_repo, data_dir)

    # ── Trackio ──────────────────────────────────────────────────────────
    try:
        import trackio as _trackio  # noqa: F401
        use_trackio = _init_trackio(args)
    except ImportError:
        use_trackio = False

    # ── Dataset ──────────────────────────────────────────────────────────
    # data/ サブディレクトリ (hf_hub_download が作る) も検索
    jsonl_paths  = sorted(data_dir.glob("*.jsonl"))
    jsonl_paths += sorted((data_dir / "data").glob("*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No .jsonl in {data_dir}")

    n_dl = min(len(jsonl_paths), max(2, args.workers // 4))
    epoch_size = args.max_train_steps * args.batch if args.max_train_steps else None
    tr_ds = StreamDataset(jsonl_paths, "train", args.val_ratio, seed=42,
                          epoch_size=epoch_size, sentence_ratio=args.sentence_ratio)
    va_ds = StreamDataset(jsonl_paths, "val",   args.val_ratio, seed=42,
                          sentence_ratio=args.sentence_ratio)
    print(f"Files: {len(jsonl_paths)}  Workers: {n_dl}  sentence_ratio: {args.sentence_ratio}"
          + (f"  EpochSize: {epoch_size//1_000_000:.0f}M" if epoch_size else ""))

    dl_kw = dict(batch_size=args.batch, collate_fn=collate, num_workers=n_dl,
                 pin_memory=(device.type == "cuda"), persistent_workers=True, prefetch_factor=4)
    tr_dl = DataLoader(tr_ds, **dl_kw)
    va_dl = DataLoader(va_ds, **dl_kw)

    # ── モデル ───────────────────────────────────────────────────────────
    model = Seq2Seq(
        hid=args.hidden, emb=args.embed,
        enc_layers=args.enc_layers, dec_layers=args.dec_layers,
        nhead=args.nhead, drop=args.dropout,
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, dynamic=True)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    use_amp = (device.type == "cuda") and args.amp
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP: {use_amp}")

    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # vocab.json を保存 (coreml_convert.py と Swift 側が参照)
    with (out / "vocab.json").open("w") as f:
        json.dump({
            "src_vocab": SRC_VOCAB,
            "src_c2i":   SRC_C2I,
            "tgt_vocab": TGT_VOCAB,
            "tgt_c2i":   TGT_C2I,
            "tgt_i2c":   {str(k): v for k, v in TGT_I2C.items()},
            "PAD": PAD, "SOS": SOS, "EOS": EOS,
        }, f, ensure_ascii=False, indent=2)

    log_path    = out / "training_log.csv"
    best_cer    = float("inf")
    start_epoch = 1
    log_mode    = "w"

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd   = ckpt["model"]
        try:
            model.load_state_dict(sd)
        except RuntimeError:
            if hasattr(model, "_orig_mod"):
                model._orig_mod.load_state_dict(sd)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_cer    = ckpt.get("cer", float("inf"))
        print(f"Resumed from {args.resume}  epoch={start_epoch-1}  CER={best_cer:.4f}")
        log_mode = "a"
    else:
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_cer", "lr", "elapsed_min"])

    # ── 訓練 ─────────────────────────────────────────────────────────────
    t0 = time.time()
    epoch_bar = tqdm(range(start_epoch, args.epochs + 1),
                     desc="全体進捗", unit="epoch", position=0,
                     dynamic_ncols=True, leave=True)

    for epoch in epoch_bar:
        tr_ds.set_epoch(epoch)
        va_ds.set_epoch(epoch)
        model.train()
        steps = 0
        tf    = max(0.0, 0.5 - 0.04 * epoch)

        total_loss_t = torch.zeros(1, device=device)
        step_bar = tqdm(tr_dl,
                        total=args.max_train_steps or None,
                        desc=f"  Epoch {epoch:2d}/{args.epochs}",
                        unit="step", position=1, leave=False,
                        dynamic_ncols=True)

        for src, tgt, sl, _ in step_bar:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(src, tgt, sl.to(device), tf=tf)
                loss   = criterion(logits[:, 1:].reshape(-1, TGT_VSZ), tgt[:, 1:].reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total_loss_t += loss.detach()
            steps += 1
            if steps % 100 == 0:
                step_bar.set_postfix(loss=f"{total_loss_t.item()/steps:.4f}", refresh=False)
            if args.max_train_steps and steps >= args.max_train_steps:
                break

        step_bar.close()

        avg_loss = total_loss_t.item() / steps
        val_cer  = evaluate(model, va_dl, device)
        lr_now   = opt.param_groups[0]["lr"]
        elapsed  = (time.time() - t0) / 60
        scheduler.step(val_cer)

        vram_used = (torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0)

        epoch_bar.set_postfix(
            CER=f"{val_cer:.4f}", best=f"{best_cer:.4f}",
            loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}", refresh=True,
        )
        tqdm.write(
            f"Epoch {epoch:3d} | loss {avg_loss:.4f} | CER {val_cer:.4f}"
            f" | lr {lr_now:.2e} | {elapsed:.1f}m | VRAM {vram_used:.1f}GB"
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        if use_trackio:
            try:
                import trackio
                trackio.log({
                    "epoch": epoch, "train_loss": avg_loss, "val_cer": val_cer,
                    "lr": lr_now, "elapsed_min": elapsed, "vram_gb": vram_used,
                }, step=epoch)
                trackio.log_gpu()
            except Exception:
                pass

        with log_path.open(log_mode, newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_loss:.6f}", f"{val_cer:.6f}",
                                    f"{lr_now:.2e}", f"{elapsed:.1f}"])
        log_mode = "a"

        raw_sd = (model._orig_mod.state_dict()
                  if hasattr(model, "_orig_mod") else model.state_dict())
        torch.save({"epoch": epoch, "model": raw_sd,
                    "opt": opt.state_dict(), "cer": val_cer, "args": vars(args)},
                   out / "model_final.pt")
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save({"epoch": epoch, "model": raw_sd, "cer": val_cer},
                       out / "model_best.pt")
            print(f"  ★ Best CER: {best_cer:.4f}")

    print(f"\nFinished. Best val CER: {best_cer:.4f}")

    if use_trackio:
        try:
            import trackio
            trackio.finish()
        except Exception:
            pass

    # ── デモ推論 ─────────────────────────────────────────────────────────
    demo = Seq2Seq(hid=args.hidden, emb=args.embed,
                   enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                   nhead=args.nhead, drop=0.0).to(device)
    demo.load_state_dict(torch.load(out / "model_best.pt", map_location=device)["model"])
    demo.eval()
    for s in ["wtsh", "kywhrds", "bnt", "tpng", "gths"]:
        print(f"  {s!r:8} → {demo.predict_top3(s, device=str(device))}")

    _upload_model_to_hf(out, args)


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",       default="outputs/preprocess_cache")
    ap.add_argument("--out-dir",        default="outputs")
    ap.add_argument("--epochs",         type=int,   default=10)
    ap.add_argument("--batch",          type=int,   default=2048)
    ap.add_argument("--workers",        type=int,   default=8)
    ap.add_argument("--hidden",         type=int,   default=256)
    ap.add_argument("--embed",          type=int,   default=64)
    ap.add_argument("--enc-layers",     type=int,   default=3,
                    help="Transformer Encoder のレイヤー数")
    ap.add_argument("--dec-layers",     type=int,   default=2,
                    help="GRU Decoder のレイヤー数")
    ap.add_argument("--nhead",          type=int,   default=4,
                    help="Transformer Attention ヘッド数")
    ap.add_argument("--dropout",        type=float, default=0.2)
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--val-ratio",      type=float, default=0.05)
    ap.add_argument("--sentence-ratio", type=float, default=0.7,
                    help="訓練バッチに占める sentence ペアの目標比率 (0.0-1.0)")
    ap.add_argument("--max-train-steps",type=int,   default=0,
                    help="0=全量")
    ap.add_argument("--amp",            action="store_true",  default=True)
    ap.add_argument("--no-amp",         dest="amp",   action="store_false")
    ap.add_argument("--compile",        action="store_true",  default=True)
    ap.add_argument("--no-compile",     dest="compile", action="store_false")
    ap.add_argument("--resume",         default="")
    ap.add_argument("--hf-dataset-repo", default="",
                    help="訓練前にHFからJSONLをDL。空=スキップ")
    ap.add_argument("--hf-model-repo",  default="",
                    help="訓練後にHFへモデルをアップロード。空=スキップ")
    ap.add_argument("--trackio-space",  default="")
    train(ap.parse_args())
