# Shiin IME

子音のみを入力するだけで日本語変換候補を提示するiOS用カスタムキーボード。

例: `wtsh` → わたし / `tbt` → 食べた / `gks` → 学生

## 構成

```
training/   BiGRU訓練コード (Python)
ios/        iOSキーボードエクステンション (Swift)
```

## モデル

- アーキテクチャ: 文字レベル双方向GRU (~3M params)
- 推論: CoreML on-device (encoder + decoder, 合計 ~7MB)
- HuggingFace: [YUGOROU/shiin-ime-gru](https://huggingface.co/YUGOROU/shiin-ime-gru)

## ビルド

```bash
# iOSアプリ
cd ios
xcodegen generate
open ShiinIME.xcodeproj

# CoreML変換
cd training
uv run coreml_convert.py
```
