"""
train.py — 子音のみIME BiGRU Seq2Seq 訓練

- HF_TOKEN が設定されている場合:
    - Trackio を HF Spaces (private) にデプロイしてトレーニングを監視
    - 訓練完了後にモデルを HuggingFace Model (private) としてアップロード
"""

import os
import json
import csv
import random
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm

# ── 語彙 ─────────────────────────────────────────────────────────────────
PAD, SOS, EOS = 0, 1, 2
VOCAB  = ["<pad>", "<sos>", "<eos>"] + list("abcdefghijklmnopqrstuvwxyz")
C2I    = {c: i for i, c in enumerate(VOCAB)}
I2C    = {i: c for i, c in enumerate(VOCAB)}
VSZ    = len(VOCAB)  # 29

TRACKIO_PROJECT  = "shiin-ime"

def _enc(s: str) -> list[int]:
    return [SOS] + [C2I[c] for c in s if c in C2I] + [EOS]

def _dec(ids) -> str:
    return "".join(I2C[i] for i in ids if i not in (PAD, SOS, EOS) and i in I2C)


# ── StreamDataset ────────────────────────────────────────────────────────
MAX_SRC, MAX_TGT = 48, 72

class StreamDataset(IterableDataset):
    """シーケンシャルJSONL読み込み＋シャッフルバッファ。
    ランダムシーク不要でNVMe帯域(~1900MB/s)を最大活用。"""
    def __init__(self, paths: list[Path], split="train",
                 val_ratio=0.05, buf_size=200_000, seed=42,
                 epoch_size: int | None = None):
        self.paths      = [str(p) for p in sorted(paths)]
        self.split      = split
        self.val_every  = max(1, round(1.0 / val_ratio))
        self.buf_size   = buf_size
        self.seed       = seed
        self.epoch_size = epoch_size  # Noneなら全量, 指定するとエポックごとに窓をスライド
        self._epoch     = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wid, nw = (worker_info.id, worker_info.num_workers) if worker_info else (0, 1)
        my_paths = [p for i, p in enumerate(self.paths) if i % nw == wid]

        rng = np.random.default_rng(self.seed + self._epoch * 997 + wid)
        buf: list = []

        # スライディングウィンドウ: epoch_size指定時にエポックごとに異なる区間を読む
        # ファイル毎の区間サイズ/スキップ量を計算 (全ファイル均等分割)
        n_files = len(self.paths)
        if self.epoch_size and n_files > 0:
            per_file = self.epoch_size // n_files
            skip     = self._epoch * per_file  # このエポックでスキップするtrain行数/ファイル
        else:
            per_file = None
            skip     = 0

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
                        continue  # JSON非パースで高速スキップ
                    if per_file and file_collected >= per_file:
                        break
                    try:
                        d = json.loads(raw)
                        src, tgt = d.get("consonants", ""), d.get("romaji", "")
                    except Exception:
                        continue
                    if not tgt or len(src) > MAX_SRC or len(tgt) > MAX_TGT:
                        continue
                    buf.append((_enc(src), _enc(tgt)))
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


# ── モデル ─────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, emb, hid, layers, drop):
        super().__init__()
        self.embed = nn.Embedding(VSZ, emb, padding_idx=PAD)
        self.rnn   = nn.GRU(emb, hid, layers, batch_first=True,
                            bidirectional=True, dropout=drop if layers > 1 else 0.0)
        self.proj  = nn.Linear(hid * 2, hid)

    def forward(self, x, lens):
        e      = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        h_top  = torch.tanh(self.proj(torch.cat([h[-2], h[-1]], dim=1)))
        return out, h_top.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)


class Attention(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.W = nn.Linear(hid + hid * 2, hid)
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, h_top, enc_out, mask):
        T      = enc_out.size(1)
        energy = self.v(torch.tanh(
            self.W(torch.cat([h_top.unsqueeze(1).expand(-1, T, -1), enc_out], dim=2))
        )).squeeze(2)
        return F.softmax(energy.masked_fill(~mask, -1e4), dim=1)


class Decoder(nn.Module):
    def __init__(self, emb, hid, layers, drop):
        super().__init__()
        self.embed = nn.Embedding(VSZ, emb, padding_idx=PAD)
        self.attn  = Attention(hid)
        self.rnn   = nn.GRU(emb + hid * 2, hid, layers, batch_first=True,
                            dropout=drop if layers > 1 else 0.0)
        self.fc    = nn.Linear(hid + hid * 2 + emb, VSZ)
        self.drop  = nn.Dropout(drop)

    def step(self, tok, h, enc_out, mask):
        e   = self.drop(self.embed(tok.unsqueeze(1)))
        a   = self.attn(h[-1], enc_out, mask)
        ctx = torch.bmm(a.unsqueeze(1), enc_out)
        _, h2 = self.rnn(torch.cat([e, ctx], dim=2), h)
        return self.fc(torch.cat([h2[-1], ctx.squeeze(1), e.squeeze(1)], dim=1)), h2


class Seq2Seq(nn.Module):
    def __init__(self, hid=256, emb=64, layers=2, drop=0.2):
        super().__init__()
        self.enc = Encoder(emb, hid, layers, drop)
        self.dec = Decoder(emb, hid, layers, drop)

    def _mask(self, src, lens):
        T = src.shape[1]
        idx = torch.arange(T, device=src.device).unsqueeze(0)
        return idx < lens.to(src.device).unsqueeze(1)

    def forward(self, src, tgt, src_lens, tf=0.5):
        B, TL      = tgt.shape
        enc_out, h = self.enc(src, src_lens)
        mask       = self._mask(src, src_lens)
        logits     = torch.zeros(B, TL, VSZ, device=src.device)
        tok        = tgt[:, 0]
        for t in range(1, TL):
            logit, h = self.dec.step(tok, h, enc_out, mask)
            logits[:, t] = logit
            tok = tgt[:, t] if random.random() < tf else logit.argmax(1)
        return logits

    @torch.no_grad()
    def predict_top3(self, src_str, beam=10, max_len=72, device="cpu"):
        src = torch.tensor(_enc(src_str), dtype=torch.long, device=device).unsqueeze(0)
        sl  = torch.tensor([src.size(1)], device=device)
        enc_out, h = self.enc(src, sl)
        mask = self._mask(src, sl)
        beams, done = [(0.0, [SOS], h)], []
        for _ in range(max_len):
            cands = []
            for score, toks, bh in beams:
                if toks[-1] == EOS:
                    done.append((score / max(len(toks), 1), toks)); continue
                tok   = torch.tensor([toks[-1]], device=device)
                logit, bh2 = self.dec.step(tok, bh, enc_out, mask)
                lp    = F.log_softmax(logit, dim=1).squeeze(0)
                for v, idx in zip(*lp.topk(beam)):
                    cands.append((score + v.item(), toks + [idx.item()], bh2))
            cands.sort(key=lambda x: x[0], reverse=True)
            beams = cands[:beam]
            if len(done) >= 3: break
        for score, toks, _ in beams:
            done.append((score / max(len(toks), 1), toks))
        done.sort(key=lambda x: x[0], reverse=True)
        return [(_dec(t[1:]), round(s, 4)) for s, t in done[:3]]


# ── CER ──────────────────────────────────────────────────────────────────
def _cer(pred, ref):
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
def evaluate(model, loader, device, n=4000):
    model.eval()
    total, count = 0.0, 0
    for src, tgt, sl, _ in loader:
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=False):
            logits = model(src, tgt, sl.to(device), tf=0.0)
        preds = logits.argmax(-1)
        for i in range(src.size(0)):
            total += _cer(_dec(preds[i].tolist()), _dec(tgt[i].tolist()))
            count += 1
            if count >= n: break
        if count >= n: break
    model.train()
    return total / max(count, 1)


# ── HuggingFace モデルアップロード ────────────────────────────────────────
def _upload_model_to_hf(out_dir: Path, args):
    token = os.environ.get("HF_TOKEN")
    if not token:
        return

    from huggingface_hub import HfApi, ModelCard

    api      = HfApi(token=token)
    username = api.whoami()["name"]
    repo_name = args.hf_model_repo or "shiin-ime-gru"
    repo_id  = repo_name if "/" in repo_name else f"{username}/{repo_name}"

    print(f"Uploading model to {repo_id} (private)...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

    upload_files = ["model_best.pt", "model_final.pt", "vocab.json", "training_log.csv"]
    for fname in upload_files:
        fpath = out_dir / fname
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  Uploaded: {fname}")

    card_content = f"""\
---
license: other
language:
- ja
tags:
- gru
- seq2seq
- ime
- japanese
private: true
---

# shiin-ime-gru

子音のみ日本語IME — Character-level BiGRU Seq2Seq モデル。README は後で整理予定。

## Architecture
- BiGRU Encoder (hidden={args.hidden}, layers={args.layers})
- GRU Decoder with Luong Attention
- Vocab: a-z (29 tokens)
- Params: ~1.4M

## Usage
```python
import torch
from train import Seq2Seq, _dec

ckpt  = torch.load("model_best.pt", map_location="cpu")
model = Seq2Seq(hid={args.hidden}, emb={args.embed}, layers={args.layers})
model.load_state_dict(ckpt["model"])
model.eval()
print(model.predict_top3("tbt"))
```
"""
    ModelCard(card_content).push_to_hub(repo_id, token=token)
    print(f"Model uploaded: https://huggingface.co/models/{repo_id}")


# ── Trackio 初期化 ─────────────────────────────────────────────────────────
def _init_trackio(args) -> bool:
    """Trackio を初期化。HF_TOKEN があれば HF Spaces (private) にデプロイ。"""
    token = os.environ.get("HF_TOKEN")
    try:
        import trackio

        kwargs: dict = dict(
            project = TRACKIO_PROJECT,
            name    = f"run_{int(time.time())}",
            config  = vars(args),
            auto_log_gpu    = True,
            gpu_log_interval= 30.0,
            embed   = False,
        )

        if token and args.trackio_space:
            from huggingface_hub import HfApi
            username     = HfApi(token=token).whoami()["name"]
            space_name   = args.trackio_space
            space_id     = space_name if "/" in space_name else f"{username}/{space_name}"
            kwargs["space_id"] = space_id
            kwargs["private"]  = True
            print(f"Trackio → HF Space: https://huggingface.co/spaces/{space_id}")
        else:
            print("Trackio → local (HF_TOKEN unset)")

        trackio.init(**kwargs)
        return True
    except Exception as e:
        print(f"Trackio init failed: {e}. Training continues without tracking.")
        return False


# ── 訓練ループ ────────────────────────────────────────────────────────────
def train(args):
    # ── デバイス設定 ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark       = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {vram:.1f}GB")
    print(f"Device: {device}")

    # ── Trackio ──────────────────────────────────────────────────────────
    try:
        import trackio as _trackio
        use_trackio = _init_trackio(args)
    except ImportError:
        use_trackio = False
        print("trackio not installed. Skipping.")

    # ── Dataset ──────────────────────────────────────────────────────────
    jsonl_paths = sorted(Path(args.data_dir).glob("*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No .jsonl in {args.data_dir}")
    cache_dir = Path(args.data_dir)

    n_dl  = min(len(jsonl_paths), max(2, args.workers // 4))
    epoch_size = args.max_train_steps * args.batch if args.max_train_steps else None
    tr_ds = StreamDataset(jsonl_paths, "train", args.val_ratio, seed=42,
                          epoch_size=epoch_size)
    va_ds = StreamDataset(jsonl_paths, "val",   args.val_ratio, seed=42)
    print(f"Files: {len(jsonl_paths)}  Workers: {n_dl}"
          + (f"  Epoch size: {epoch_size//1_000_000:.0f}M" if epoch_size else ""))

    dl_kw = dict(batch_size=args.batch, collate_fn=collate, num_workers=n_dl,
                 pin_memory=(device.type=="cuda"), persistent_workers=True, prefetch_factor=4)
    tr_dl = DataLoader(tr_ds, **dl_kw)
    va_dl = DataLoader(va_ds, **dl_kw)

    # ── モデル ───────────────────────────────────────────────────────────
    model = Seq2Seq(hid=args.hidden, emb=args.embed,
                    layers=args.layers, drop=args.dropout).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, dynamic=True)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    # ── AMP ──────────────────────────────────────────────────────────────
    use_amp = (device.type == "cuda") and args.amp
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP: {use_amp}")

    # ── Optimizer ────────────────────────────────────────────────────────
    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "vocab.json").open("w") as f:
        json.dump({"vocab": VOCAB, "C2I": C2I,
                   "I2C": {str(k): v for k, v in I2C.items()},
                   "PAD": PAD, "SOS": SOS, "EOS": EOS}, f, ensure_ascii=False, indent=2)

    log_path = out / "training_log.csv"

    # ── チェックポイント再開 ──────────────────────────────────────────────
    best_cer    = float("inf")
    start_epoch = 1
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
        log_mode = "a"  # 既存ログに追記
    else:
        log_mode = "w"
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
            # sl はCPUのまま渡す (Encoder内pack_padded_sequenceがCPUを要求)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(src, tgt, sl, tf=tf)
                loss   = criterion(logits[:, 1:].reshape(-1, VSZ), tgt[:, 1:].reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total_loss_t += loss.detach()
            steps += 1
            if steps % 100 == 0:
                step_bar.set_postfix(
                    loss=f"{total_loss_t.item()/steps:.4f}",
                    refresh=False,
                )
            if args.max_train_steps and steps >= args.max_train_steps:
                break

        step_bar.close()

        avg_loss = total_loss_t.item() / steps
        val_cer  = evaluate(model, va_dl, device)
        lr_now   = opt.param_groups[0]["lr"]
        elapsed  = (time.time() - t0) / 60
        scheduler.step(val_cer)

        vram_used = (torch.cuda.max_memory_allocated() / 1e9
                     if device.type == "cuda" else 0.0)

        epoch_bar.set_postfix(
            CER=f"{val_cer:.4f}",
            best=f"{best_cer:.4f}",
            loss=f"{avg_loss:.4f}",
            lr=f"{lr_now:.2e}",
            refresh=True,
        )
        tqdm.write(
            f"Epoch {epoch:3d} | loss {avg_loss:.4f} | CER {val_cer:.4f}"
            f" | lr {lr_now:.2e} | {elapsed:.1f}m | VRAM {vram_used:.1f}GB"
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Trackio ログ
        if use_trackio:
            try:
                import trackio
                trackio.log({
                    "epoch":      epoch,
                    "train_loss": avg_loss,
                    "val_cer":    val_cer,
                    "lr":         lr_now,
                    "elapsed_min": elapsed,
                    "vram_gb":    vram_used,
                }, step=epoch)
                trackio.log_gpu()
            except Exception:
                pass

        with log_path.open(log_mode, newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_loss:.6f}", f"{val_cer:.6f}",
                                    f"{lr_now:.2e}", f"{elapsed:.1f}"])
        log_mode = "a"  # 2epoch目以降は追記

        # state_dict は compile wrap を外して保存
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
                   layers=args.layers, drop=args.dropout).to(device)
    demo.load_state_dict(torch.load(out / "model_best.pt", map_location=device)["model"])
    demo.eval()
    for s in ["tbt", "wtsh", "gks", "nhn", "yb"]:
        print(f"  {s!r:6} → {demo.predict_top3(s, device=str(device))}")

    # ── HF モデルアップロード ──────────────────────────────────────────
    _upload_model_to_hf(out, args)


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",  default="outputs/preprocess_cache")
    ap.add_argument("--out-dir",   default="outputs")
    ap.add_argument("--epochs",    type=int,   default=10)
    ap.add_argument("--batch",     type=int,   default=2048)
    ap.add_argument("--workers",   type=int,   default=8)
    ap.add_argument("--hidden",    type=int,   default=256)
    ap.add_argument("--embed",     type=int,   default=64)
    ap.add_argument("--layers",    type=int,   default=2)
    ap.add_argument("--dropout",   type=float, default=0.2)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--val-ratio",       type=float, default=0.05)
    ap.add_argument("--max-train-steps", type=int,   default=0,
                    help="0=全量。指定するとepochあたりのstep数を制限")
    ap.add_argument("--amp",       action="store_true",  default=True)
    ap.add_argument("--no-amp",    dest="amp",   action="store_false")
    ap.add_argument("--compile",   action="store_true",  default=True)
    ap.add_argument("--no-compile",dest="compile", action="store_false")
    ap.add_argument("--resume",    default="",
                    help="再開するチェックポイント .pt (例: outputs/model_final.pt)")
    ap.add_argument("--hf-model-repo", default="",
                    help="HFモデルリポジトリ名 (例: my-repo または username/my-repo)。空=アップロードしない")
    ap.add_argument("--trackio-space", default="",
                    help="Trackio HF Space名 (例: my-space または username/my-space)。空=ローカルのみ")
    train(ap.parse_args())
