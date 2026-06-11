# Shiin IME（子音のみIME）

子音だけを入力すると日本語の変換候補を提示する、iOS用カスタムキーボード。

> 例: `wtsh` → わたし / `tbt` → 食べた / `gks` → 学生

母音を省いて子音のみで入力することで、日本語の音韻を直感的に把握しているユーザーの打鍵数を減らすことを狙ったプロジェクトです。すべての変換はオンデバイス（CoreML）で完結します。

## リポジトリ構成

```
training/   モデル訓練・前処理・CoreML変換コード (Python)
ios/        iOSキーボードエクステンション本体 (Swift / XcodeGen)
design/     デザインブリーフ・UIモックアップ (React/jsx プロトタイプ)
```

## モデル

| 項目 | 内容 |
|------|------|
| アーキテクチャ | Transformer Encoder + GRU Decoder（attention付き seq2seq, 文字レベル） |
| パラメータ | 約1.3M |
| 推論 | CoreML オンデバイス（encoder + decoder_step, 合計 約7MB） |
| 入出力 | 子音列（例: `wtsh`）→ 読み候補（例: `watashi` / わたし） |
| HuggingFace | [YUGOROU/shiin-ime-gru](https://huggingface.co/YUGOROU/shiin-ime-gru) |

## ビルド

### iOSアプリ

[XcodeGen](https://github.com/yonaskolb/XcodeGen) が必要です。

```bash
cd ios
xcodegen generate
open ShiinIME.xcodeproj
```

### CoreML変換（訓練済みモデルから）

[uv](https://github.com/astral-sh/uv) を使用します。

```bash
cd training
uv run coreml_convert.py
```

訓練・前処理パイプラインの詳細は `training/` 内の各スクリプトを参照してください。

## ライセンス

[MIT License](LICENSE)
