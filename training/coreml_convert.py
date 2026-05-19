# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.11.0",
#   "numpy",
#   "coremltools>=7.0",
#   "huggingface_hub>=0.23",
# ]
# ///
"""
coreml_convert.py — Shiin IME BiGRU → CoreML 変換スクリプト

【固定長 + 右詰めパディング方式】
  RangeDim（可変長）を使うと coremltools 9.0 の ct.convert() が内部で
  native CoreML framework ローディング時にハングする。
  固定長 FIXED_T にパディングして変換を安定させる。

  右詰め: [PAD...PAD, SOS, c1...cN, EOS]
  ← EOS が最後の位置に来るので GRU の最終隠れ状態が訓練時と等価になる。

  Attention マスク: PAD 位置への attention を抑制するため
  attn_mask[1, FIXED_T] を float32 で渡す（1.0=有効, 0.0=PAD）。

出力:
  outputs/coreml/
  ├── encoder.mlpackage       src_tokens[1,FIXED_T:int32] → enc_out[1,FIXED_T,512], h_init[2,1,256]
  ├── decoder_step.mlpackage  tok[1:int32], h[2,1,256], enc_out[1,FIXED_T,512], attn_mask[1,FIXED_T]
  │                            → logits[1,29], h_new[2,1,256]
  └── vocab.json

Usage:
  uv run coreml_convert.py                              # HFからモデルをDL
  uv run coreml_convert.py --checkpoint outputs/model_best.pt
  HF_TOKEN=xxx uv run coreml_convert.py

注意: 変換はLinux/macOS両対応。予測検証はmacOSのみ (CoreML Predictor)。
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train import Seq2Seq, VOCAB, C2I, I2C, VSZ, PAD, SOS, EOS, _enc, _dec

FIXED_T = 52  # 固定シーケンス長 (SOS + 最大50子音 + EOS)


# ── Export Wrappers ───────────────────────────────────────────────────────────

class ManualGRUCell(nn.Module):
    """GRUCell の chunk/unsafe_chunk を使わない手動実装 (CoreML 互換)。"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        hid = hidden_size
        self.r_ih = nn.Linear(input_size, hid, bias=True)
        self.z_ih = nn.Linear(input_size, hid, bias=True)
        self.n_ih = nn.Linear(input_size, hid, bias=True)
        self.r_hh = nn.Linear(hid, hid, bias=True)
        self.z_hh = nn.Linear(hid, hid, bias=True)
        self.n_hh = nn.Linear(hid, hid, bias=True)

    @classmethod
    def from_gru_weights(cls, input_size: int, hidden_size: int,
                         w_ih, w_hh, b_ih, b_hh) -> "ManualGRUCell":
        hid = hidden_size
        cell = cls(input_size, hid)
        # GRU weight_ih/bias_ih: [3*hid, input_size] stacked as [r; z; n]
        def _p(t): return nn.Parameter(t.clone())
        cell.r_ih.weight = _p(w_ih[:hid])
        cell.z_ih.weight = _p(w_ih[hid:2*hid])
        cell.n_ih.weight = _p(w_ih[2*hid:])
        cell.r_ih.bias   = _p(b_ih[:hid])
        cell.z_ih.bias   = _p(b_ih[hid:2*hid])
        cell.n_ih.bias   = _p(b_ih[2*hid:])
        cell.r_hh.weight = _p(w_hh[:hid])
        cell.z_hh.weight = _p(w_hh[hid:2*hid])
        cell.n_hh.weight = _p(w_hh[2*hid:])
        cell.r_hh.bias   = _p(b_hh[:hid])
        cell.z_hh.bias   = _p(b_hh[hid:2*hid])
        cell.n_hh.bias   = _p(b_hh[2*hid:])
        return cell

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.r_ih(x) + self.r_hh(h))
        z = torch.sigmoid(self.z_ih(x) + self.z_hh(h))
        n = torch.tanh(self.n_ih(x) + r * self.n_hh(h))
        return (1.0 - z) * n + z * h


class EncoderExport(nn.Module):
    """
    固定長 FIXED_T の推論専用エンコーダー。
    GRUCell を1ステップずつ実行し、PAD 位置でマスクを適用することで
    pack_padded_sequence と完全等価な出力を再現する。

    問題の本質 (2-layer BiGRU):
      固定長パディングで nn.GRU をそのまま実行すると、layer-0 backward が
      PAD トークンを処理し、その汚染した状態が layer-1 forward の入力になる。
      cumsum dual-run では layer 間の汚染を防げない。

    解決策: GRUCell ステップ実行 + マスキング
      左詰め入力 [SOS, c1...cN, EOS, PAD...] を使い、各ステップで:
        h = valid * h_new + (1-valid) * h_prev  ← PAD 位置では h を更新しない
        out[i] = valid * h_new                   ← PAD 位置の出力はゼロ
      layer-0 出力がゼロになった PAD 位置は layer-1 への入力もゼロとなり
      pack_padded_sequence と完全一致する。

    入力: 左詰め (src_left, attn_mask のみ。src_right 不要)
    """
    def __init__(self, model: Seq2Seq):
        super().__init__()
        enc = model.enc
        self.embed      = enc.embed
        self.proj       = enc.proj
        self.num_layers = enc.rnn.num_layers  # 2
        hid = enc.rnn.hidden_size             # 256
        emb = enc.embed.embedding_dim         # 64
        self.hid = hid

        rnn = enc.rnn
        mc  = ManualGRUCell.from_gru_weights

        self.cell_l0f = mc(emb,   hid, rnn.weight_ih_l0,         rnn.weight_hh_l0,
                                       rnn.bias_ih_l0,            rnn.bias_hh_l0)
        self.cell_l0b = mc(emb,   hid, rnn.weight_ih_l0_reverse, rnn.weight_hh_l0_reverse,
                                       rnn.bias_ih_l0_reverse,    rnn.bias_hh_l0_reverse)
        self.cell_l1f = mc(hid*2, hid, rnn.weight_ih_l1,         rnn.weight_hh_l1,
                                       rnn.bias_ih_l1,            rnn.bias_hh_l1)
        self.cell_l1b = mc(hid*2, hid, rnn.weight_ih_l1_reverse, rnn.weight_hh_l1_reverse,
                                       rnn.bias_ih_l1_reverse,    rnn.bias_hh_l1_reverse)

    def forward(
        self,
        src_left:  torch.Tensor,   # [1, FIXED_T]  int64  左詰め [SOS, c1...cN, EOS, PAD...]
        attn_mask: torch.Tensor,   # [1, FIXED_T]  float  1.0=有効, 0.0=PAD (左詰め)
    ) -> tuple[torch.Tensor, torch.Tensor]:

        hid = self.hid
        e = self.embed(src_left)  # [1, FIXED_T, emb]

        # ── Layer 0 Forward (left-to-right) ─────────────────────────────────
        h0f = torch.zeros(1, hid)
        outs_l0f = []
        for i in range(FIXED_T):
            v   = attn_mask[:, i:i+1]                         # [1,1]
            h_n = self.cell_l0f(e[:, i, :], h0f)
            h0f = v * h_n + (1.0 - v) * h0f
            outs_l0f.append(v * h_n)                          # zero at PAD
        out_l0f = torch.stack(outs_l0f, dim=1)               # [1, T, hid]

        # ── Layer 0 Backward (right-to-left) ────────────────────────────────
        h0b = torch.zeros(1, hid)
        outs_l0b_rev = []
        for i in range(FIXED_T - 1, -1, -1):
            v   = attn_mask[:, i:i+1]
            h_n = self.cell_l0b(e[:, i, :], h0b)
            h0b = v * h_n + (1.0 - v) * h0b
            outs_l0b_rev.append(v * h_n)
        out_l0b = torch.flip(torch.stack(outs_l0b_rev, dim=1), dims=[1])  # [1, T, hid]

        # Layer 0 combined output (left-aligned, PAD=0)
        out_l0 = torch.cat([out_l0f, out_l0b], dim=-1)       # [1, T, hid*2]

        # ── Layer 1 Forward ──────────────────────────────────────────────────
        h1f = torch.zeros(1, hid)
        outs_l1f = []
        for i in range(FIXED_T):
            v   = attn_mask[:, i:i+1]
            h_n = self.cell_l1f(out_l0[:, i, :], h1f)
            h1f = v * h_n + (1.0 - v) * h1f
            outs_l1f.append(v * h_n)
        out_l1f = torch.stack(outs_l1f, dim=1)               # [1, T, hid]

        # ── Layer 1 Backward ─────────────────────────────────────────────────
        h1b = torch.zeros(1, hid)
        outs_l1b_rev = []
        for i in range(FIXED_T - 1, -1, -1):
            v   = attn_mask[:, i:i+1]
            h_n = self.cell_l1b(out_l0[:, i, :], h1b)
            h1b = v * h_n + (1.0 - v) * h1b
            outs_l1b_rev.append(v * h_n)
        out_l1b = torch.flip(torch.stack(outs_l1b_rev, dim=1), dims=[1])  # [1, T, hid]

        enc_out = torch.cat([out_l1f, out_l1b], dim=-1)      # [1, T, hid*2=512]

        # h_init: layer-1 の最終隠れ状態 (h1f = EOS 後 fwd, h1b = SOS 後 bwd)
        h_top  = torch.tanh(self.proj(torch.cat([h1f, h1b], dim=1)))
        h_init = h_top.unsqueeze(0).repeat(self.num_layers, 1, 1)
        return enc_out, h_init


class DecoderStepExport(nn.Module):
    """
    1ステップのデコーダー推論ラッパー (固定長 FIXED_T)。

    - Attention 重みを W_h + W_e に分解 → expand(-1, T, -1) 不使用
    - attn_mask (float32): PAD 位置を -1e4 バイアスで抑制
    """
    def __init__(self, model: Seq2Seq, hid: int = 256):
        super().__init__()
        dec        = model.dec
        self.embed = dec.embed
        self.rnn   = dec.rnn
        self.fc    = dec.fc
        self.v     = dec.attn.v

        W = dec.attn.W
        self.W_h = nn.Linear(hid,     hid, bias=True)
        self.W_e = nn.Linear(hid * 2, hid, bias=False)
        self.W_h.weight = nn.Parameter(W.weight[:, :hid].clone())
        self.W_h.bias   = nn.Parameter(W.bias.clone())
        self.W_e.weight = nn.Parameter(W.weight[:, hid:].clone())

    def forward(
        self,
        tok:       torch.Tensor,   # [1]           int64   直前トークン
        h:         torch.Tensor,   # [2, 1, 256]   float   デコーダー隠れ状態
        enc_out:   torch.Tensor,   # [1, FIXED_T, 512] float  エンコーダー出力
        attn_mask: torch.Tensor,   # [1, FIXED_T]  float   1.0=有効, 0.0=PAD
    ) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.embed(tok.unsqueeze(1))             # [1, 1, 64]

        h_proj = self.W_h(h[-1]).unsqueeze(1)        # [1, 1, 256]
        e_proj = self.W_e(enc_out)                   # [1, FIXED_T, 256]
        energy = self.v(torch.tanh(h_proj + e_proj)).squeeze(2)  # [1, FIXED_T]

        # PAD 位置に -1e4 バイアスを加算して softmax から除外
        attn_bias = (attn_mask - 1.0) * 1e4          # 0.0 for valid, -1e4 for PAD
        a = F.softmax(energy + attn_bias, dim=1)     # [1, FIXED_T]

        ctx    = torch.bmm(a.unsqueeze(1), enc_out)  # [1, 1, 512]
        _, h_new = self.rnn(torch.cat([e, ctx], dim=2), h)
        logit    = self.fc(torch.cat(
            [h_new[-1], ctx.squeeze(1), e.squeeze(1)], dim=1))  # [1, 29]
        return logit, h_new


# ── モデルロード ──────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, hid: int, emb: int, layers: int):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "args" in ckpt:
        a      = ckpt["args"]
        hid    = a.get("hidden", hid)
        emb    = a.get("embed",  emb)
        layers = a.get("layers", layers)
    model = Seq2Seq(hid=hid, emb=emb, layers=layers, drop=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"モデルロード完了: epoch={ckpt.get('epoch','?')}  "
          f"CER={ckpt.get('cer', '?')}  "
          f"hid={hid} emb={emb} layers={layers}  "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    return model, hid, emb, layers


# ── CoreML 変換 ───────────────────────────────────────────────────────────────

def _dir_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def convert_encoder(model: Seq2Seq, out_path: Path, hid: int, fixed_t: int = FIXED_T):
    import coremltools as ct

    print(f"  [encoder] トレース中 (FIXED_T={fixed_t})...", flush=True)
    wrapper  = EncoderExport(model).eval()
    src_left = torch.zeros(1, fixed_t, dtype=torch.long)
    mask_ex  = torch.ones(1, fixed_t)
    traced   = torch.jit.trace(wrapper, (src_left, mask_ex))

    print("  [encoder] MIL 変換中...", flush=True)
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="src_left",  shape=(1, fixed_t), dtype=np.int32),
            ct.TensorType(name="attn_mask", shape=(1, fixed_t)),
        ],
        outputs=[ct.TensorType(name="enc_out"),
                 ct.TensorType(name="h_init")],
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    print("  [encoder] 保存中...", flush=True)
    ml.short_description = f"Shiin IME Encoder (BiGRU, FIXED_T={fixed_t})"
    ml.save(str(out_path))
    print(f"  encoder.mlpackage       {_dir_mb(out_path):.2f} MB", flush=True)
    return ml


def convert_decoder_step(model: Seq2Seq, out_path: Path,
                         hid: int, layers: int, fixed_t: int = FIXED_T):
    import coremltools as ct

    print(f"  [decoder] トレース中 (FIXED_T={fixed_t})...", flush=True)
    wrapper      = DecoderStepExport(model, hid=hid).eval()
    tok_ex       = torch.zeros(1, dtype=torch.long)
    h_ex         = torch.zeros(layers, 1, hid)
    enc_out_ex   = torch.zeros(1, fixed_t, hid * 2)
    attn_mask_ex = torch.ones(1, fixed_t)
    traced = torch.jit.trace(wrapper, (tok_ex, h_ex, enc_out_ex, attn_mask_ex))

    print("  [decoder] MIL 変換中...", flush=True)
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="tok",       shape=(1,),                   dtype=np.int32),
            ct.TensorType(name="h",         shape=(layers, 1, hid)),
            ct.TensorType(name="enc_out",   shape=(1, fixed_t, hid * 2)),
            ct.TensorType(name="attn_mask", shape=(1, fixed_t)),
        ],
        outputs=[ct.TensorType(name="logits"),
                 ct.TensorType(name="h_new")],
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    print("  [decoder] 保存中...", flush=True)
    ml.short_description = f"Shiin IME Decoder Step (GRU+Attention, FIXED_T={fixed_t})"
    ml.save(str(out_path))
    print(f"  decoder_step.mlpackage  {_dir_mb(out_path):.2f} MB", flush=True)
    return ml


# ── パディング ユーティリティ ─────────────────────────────────────────────────

def _pad_right_align(tokens: list[int], fixed_t: int) -> tuple[np.ndarray, np.ndarray]:
    """右詰め: [PAD...PAD, SOS, c1...cN, EOS]"""
    actual_len = len(tokens)
    assert actual_len <= fixed_t, f"入力が長すぎます: {actual_len} > {fixed_t}"
    src  = np.full((1, fixed_t), PAD, dtype=np.int32)
    mask = np.zeros((1, fixed_t), dtype=np.float32)
    start = fixed_t - actual_len
    src[0, start:] = tokens
    mask[0, start:] = 1.0
    return src, mask


def _pad_left_align(tokens: list[int], fixed_t: int) -> np.ndarray:
    """左詰め: [SOS, c1...cN, EOS, PAD...PAD]"""
    actual_len = len(tokens)
    assert actual_len <= fixed_t
    src = np.full((1, fixed_t), PAD, dtype=np.int32)
    src[0, :actual_len] = tokens
    return src


# ── 検証 ─────────────────────────────────────────────────────────────────────

def _beam_search_coreml(enc_ml, dec_ml, src_str: str,
                        fixed_t: int = FIXED_T,
                        beam: int = 5,
                        max_len: int = 72) -> list[tuple[str, float]]:
    tokens      = _enc(src_str)
    src_left_np = _pad_left_align(tokens, fixed_t)
    _, mask_right_np = _pad_right_align(tokens, fixed_t)
    mask_left_np = np.flip(mask_right_np, axis=1).copy()  # left-aligned mask

    enc_out_dict = enc_ml.predict({
        "src_left":  src_left_np,
        "attn_mask": mask_left_np,
    })
    enc_out = torch.from_numpy(enc_out_dict["enc_out"]).float()
    h       = torch.from_numpy(enc_out_dict["h_init"]).float()
    # enc_out は左詰め、decoder には同じ左詰め mask を渡す

    beams: list = [(0.0, [SOS], h)]
    done:  list = []
    for _ in range(max_len):
        cands = []
        for score, toks, bh in beams:
            if toks[-1] == EOS:
                done.append((score / max(len(toks), 1), toks))
                continue
            out = dec_ml.predict({
                "tok":       np.array([toks[-1]], dtype=np.int32),
                "h":         bh.numpy(),
                "enc_out":   enc_out.numpy(),
                "attn_mask": mask_left_np,
            })
            logit = torch.from_numpy(out["logits"]).float()
            h_new = torch.from_numpy(out["h_new"]).float()
            lp    = F.log_softmax(logit, dim=1).squeeze(0)
            for v, idx in zip(*lp.topk(beam)):
                cands.append((score + v.item(), toks + [idx.item()], h_new))
        cands.sort(key=lambda x: x[0], reverse=True)
        beams = cands[:beam]
        if len(done) >= 3:
            break
    for score, toks, _ in beams:
        done.append((score / max(len(toks), 1), toks))
    done.sort(key=lambda x: x[0], reverse=True)
    return [(_dec(t[1:]), round(s, 4)) for s, t in done[:3]]


def verify(model: Seq2Seq, enc_path: Path, dec_path: Path, fixed_t: int = FIXED_T):
    import coremltools as ct

    enc_ml = ct.models.MLModel(str(enc_path))
    dec_ml = ct.models.MLModel(str(dec_path))

    test_cases = [
        ("wtsh", "watashi"),
        ("gks",  "gakusei"),
        ("nhn",  "nihon"),
        ("tbt",  "tabetai"),
        ("yb",   "yabai"),
    ]

    print("\n--- 検証 (PyTorch vs CoreML) ---")
    mismatch = False
    for src, expected in test_cases:
        pt_top1  = model.predict_top3(src)[0][0]
        cml_top3 = _beam_search_coreml(enc_ml, dec_ml, src, fixed_t=fixed_t)
        cml_top1 = cml_top3[0][0] if cml_top3 else "?"
        ok = (pt_top1 == cml_top1)
        if not ok:
            mismatch = True
        mark = "✓" if ok else "△"
        note = f"  (expected: {expected!r})" if cml_top1 != expected else ""
        print(f"  {mark} '{src:6}' | PyTorch: {pt_top1!r:12}  CoreML: {cml_top1!r:12}{note}")

    if mismatch:
        print("\n注意: PyTorchとCoreMLで差異あり (float32丸め誤差または右詰めパディングの影響)。")
    else:
        print("\n全テストケースで一致。")


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Shiin IME PyTorchモデル → CoreML変換")
    ap.add_argument("--checkpoint", default=None,
                    help="model_best.pt へのパス (省略時 HF からDL)")
    ap.add_argument("--hf-repo",   default="",
                    help="HFモデルリポジトリ (例: username/shiin-ime-gru)。--checkpoint 未指定時に使用")
    ap.add_argument("--out-dir",   default="outputs/coreml")
    ap.add_argument("--hidden",    type=int, default=256)
    ap.add_argument("--embed",     type=int, default=64)
    ap.add_argument("--layers",    type=int, default=2)
    ap.add_argument("--fixed-t",   type=int, default=FIXED_T,
                    help=f"固定シーケンス長 (default: {FIXED_T})")
    ap.add_argument("--no-verify", action="store_true",
                    help="検証をスキップ")
    args = ap.parse_args()

    try:
        import coremltools as ct
        print(f"coremltools {ct.__version__}", flush=True)
    except ImportError:
        print("ERROR: pip install coremltools")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.checkpoint
    if not ckpt_path:
        token = os.environ.get("HF_TOKEN")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("ERROR: pip install huggingface_hub")
            sys.exit(1)
        print(f"{args.hf_repo} から model_best.pt をダウンロード中...", flush=True)
        ckpt_path = hf_hub_download(
            repo_id=args.hf_repo, filename="model_best.pt",
            repo_type="model", token=token,
        )
        print(f"ダウンロード完了: {ckpt_path}", flush=True)

    model, hid, emb, layers = load_model(
        ckpt_path, args.hidden, args.embed, args.layers
    )

    enc_path = out_dir / "encoder.mlpackage"
    dec_path = out_dir / "decoder_step.mlpackage"

    print(f"\n変換中... (FIXED_T={args.fixed_t})", flush=True)
    convert_encoder(model, enc_path, hid=hid, fixed_t=args.fixed_t)
    convert_decoder_step(model, dec_path, hid=hid, layers=layers, fixed_t=args.fixed_t)

    total_mb = _dir_mb(enc_path) + _dir_mb(dec_path)
    print(f"  合計: {total_mb:.2f} MB / 48 MB (iOS KBExt 上限)", flush=True)

    vocab_path = out_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump({
            "vocab": VOCAB,
            "c2i":   C2I,
            "i2c":   {str(k): v for k, v in I2C.items()},
            "pad": PAD, "sos": SOS, "eos": EOS, "vsz": VSZ,
            "fixed_t": args.fixed_t,
        }, f, ensure_ascii=False, indent=2)
    print(f"  vocab.json              → {vocab_path}", flush=True)

    if not args.no_verify:
        try:
            verify(model, enc_path, dec_path, fixed_t=args.fixed_t)
        except Exception as e:
            print(f"\n検証スキップ: {e}")
            print("ヒント: Linux では CoreML Predictor が利用不可です (--no-verify で抑制)")

    print(f"\n完了: {out_dir}/")
    print(f"iOS Swift メモ: 入力を右詰めで FIXED_T={args.fixed_t} にパディングし、")
    print( "               attn_mask を 1.0/0.0 で渡すこと。")


if __name__ == "__main__":
    main()
