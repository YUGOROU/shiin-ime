#!/usr/bin/env python3
"""
test_model.py — model_best.pt の推論テスト

使い方:
    python test_model.py
    python test_model.py --model ../outputs/model_best.pt
"""

import argparse
from pathlib import Path
import torch
from train import Seq2Seq, SRC_C2I, TGT_C2I, TGT_I2C, PAD, SOS, EOS, _dec_tgt

# ── テストケース (子音列, 期待する読み) ──────────────────────────────────
# 子音はローマ字から母音[aeiou]を除いたもの
TEST_CASES = [
    # 単語 (短)
    ("wtsh",    "わたし",           "私"),
    ("kyw",     "きょう",           "今日"),
    ("tky",     "とうきょう",       "東京"),
    ("nhn",     "にほん",           "日本"),
    ("rng",     "りんご",           "リンゴ"),
    ("bnk",     "べんきょう",       "勉強"),
    ("smr",     "さむらい",         "侍"),
    ("mgr",     "まがる",           "曲がる"),
    ("krs",     "くらす",           "暮らす"),
    ("hyr",     "はやい",           "早い"),
    # 単語 (長め)
    ("tpng",    "たいぴんぐ",       "タイピング"),
    ("bnt",     "べんとう",         "弁当"),
    ("snk",     "さんかく",         "三角"),
    ("shgk",    "しょうがっこう",   "小学校"),
    # 文・フレーズ
    ("kywhrds", "きょうははれです", "今日は晴れです"),
]


def load_model(model_path: Path, device: torch.device) -> Seq2Seq:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = Seq2Seq(
        hid       = args.get("hidden",     256),
        emb       = args.get("embed",       64),
        enc_layers= args.get("enc_layers",   3),
        dec_layers= args.get("dec_layers",   2),
        nhead     = args.get("nhead",        4),
        drop      = 0.0,
    ).to(device)
    sd = ckpt["model"]
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        model._orig_mod.load_state_dict(sd)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../outputs/model_best.pt")
    ap.add_argument("--beam",  type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(model_path, device)
    ckpt  = torch.load(model_path, map_location="cpu", weights_only=False)
    print(f"Loaded: {model_path}  epoch={ckpt.get('epoch','?')}  CER={ckpt.get('cer', '?')}\n")

    # ── ヘッダー ──────────────────────────────────────────────────────────
    print(f"{'Input':<12} {'Expected':<18} {'Top-1':^18} {'Top-2':^18} {'Top-3':^18}  OK?")
    print("─" * 95)

    correct = 0
    for cons, expected, kanji in TEST_CASES:
        preds = model.predict_top3(cons, beam=args.beam, device=str(device))
        tops  = [p for p, _ in preds]

        top1    = tops[0] if len(tops) > 0 else ""
        top2    = tops[1] if len(tops) > 1 else ""
        top3    = tops[2] if len(tops) > 2 else ""
        ok      = "✓" if top1 == expected else "✗"
        if top1 == expected:
            correct += 1

        print(f"{cons:<12} {expected:<18} {top1:^18} {top2:^18} {top3:^18}  {ok} ({kanji})")

    print("─" * 95)
    print(f"Top-1 exact match: {correct}/{len(TEST_CASES)} ({correct/len(TEST_CASES)*100:.0f}%)\n")


if __name__ == "__main__":
    main()
