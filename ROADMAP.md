# Shiin IME — 開発ロードマップ

> 最終更新: 2026-05-18  
> このドキュメントはコンテキスト圧縮後も作業を継続できるよう設計されている

---

## プロジェクト概要

子音のみ入力から読み（ローマ字）候補への変換を行うiOS用カスタムIME。

| 項目 | 内容 |
|------|------|
| モデル | Transformer Encoder + GRU Decoder (hidden=256), ~1.3M params |
| 入出力 | 子音列 (例: `wtsh`) → ローマ字候補 (例: `watashi`) |
| 推論 | CoreML on-device, iOS KBExtension |
| サイズ制約 | encoder+decoder ≦ 48MB (現在 7.02MB) |
| HF model | `YUGOROU/shiin-ime-gru` (private) |
| HF data | `YUGOROU/shiin-ime-preprocess` (private) |

---

## 現在の状態 (2026-05-18)

- **v2モデル**: best CER **0.2160** (epoch 3/10, lr=1e-3)
  - 訓練: wiki(55GB) + cc100(6.8GB) + livedoor(155MB), 1B samples, 10 epochs
  - `model_best.pt` → HF `YUGOROU/shiin-ime-gru` にアップロード済み
- **iOS app**: modelVersion `"v5"` 反映済み、mlpackage更新済み
- **CoreML**: encoder 4.47MB + decoder 2.55MB = 7.02MB

### 既知の変換課題 (ime-data-2026-05-17.csv より)

| カテゴリ | 件数 | 具体例 |
|---------|------|--------|
| カタカナ語 → ひらがな語に引っ張られる | 8件 | `bnt`→お弁当(×イベント), `kmr`→金る(×カメラ) |
| 固有名詞が全く出ない | 7件 | `gtthb`→???(×GitHub), `krsms`→クラスます(×クリスマス) |
| 高頻度語偏り / 同一子音列競合 | 4件 | `knrn`→関連(×訓練), `mrr`→生まれる(×もらえる) |
| 長モーラ崩壊 | 3件 | `kspt`(6mora)→クスっpっt, `tpng`(5mora)→問いピング |

根本原因:
- 訓練コーパスの99%がひらがな/漢字 → カタカナ語を学習できていない
- 5mora以上でGRUの記憶が崩壊するアーキ的限界
- ビームサーチが同一prefixの候補に収束する（Diverse Beam Search未実装）

---

## Phase 1: 即効性のある改善（再訓練不要）

### ✅ 1.1 Diverse Beam Search (DBS) 実装

**ファイル**: `ios/ShiinIMEKeyboard/Sources/ShiinInferenceEngine.swift`

**アルゴリズム**:
- G=3グループ、beamPerGroup=3、diversity λ=0.8
- 各ステップで、グループg > 0 は前グループが選んだトークンにペナルティを付与
- これにより3候補が同一prefixに収束することを防ぐ

**パラメータチューニング余地**:
- `diversity` (λ): 0.5〜1.5 を試す。大きすぎると低確率語が混入する
- `numGroups`: 3（候補数と一致させる）
- `beamPerGroup`: 2〜4

**検証方法**:
```bash
cd training
uv run test_coreml_beam.py  # 標準 vs DBS の比較
```

期待される改善:
- `bnt` → [イベント, お弁当, ベント] のように多様な候補が出る
- `tpng` → 少なくとも先頭文字が異なる3候補

---

## Phase 2: アーキテクチャ改善 + 再訓練

### 2.1 アーキテクチャ改善（設計中）

**候補アプローチ**:

| 案 | 内容 | メリット | デメリット |
|----|------|---------|-----------|
| A | Encoder に Attention Head 追加 | 長moraの崩壊を解決 | CoreML変換の複雑さ増加 |
| B | Transformer Encoder + GRU Decoder | 文脈長制限を根本解決 | パラメータ増加、再訓練必須 |
| C | CTC head を追加 (greedy decode) | 高速推論 | ビーム出力の質が下がる可能性 |

**重要制約**: アーキ変更がinterface変更（入出力テンソル形状）を伴う場合、既存 `model_best.pt` からの継続訓練は不可能。全コーパスでのスクラッチ再訓練が必要。

### 2.2 Kakologコーパス前処理

**ガイド**: `training/kakolog_loading_guide.md` 参照

```bash
# 必須: datasets==2.21.0 (3.x系では動作しない)
pip install datasets==2.21.0

python -c "
from datasets import load_dataset
ds = load_dataset(
    'KakologArchives/KakologArchives',
    'sample',
    number_of_files=10,
    trust_remote_code=True,
)
"
```

**データ特性**:
- コメント中央値5文字、口語/俗語が多い
- フィルタ通過率 ~51%（6文字以上、JP文字50%以上）
- `'all'` configは190GB超なので `'sample'` か `number_of_files` で制限

**前処理後の出力先**: `outputs/preprocess_cache/data/data/kakolog.jsonl`

### 2.3 継続訓練（アーキ変更なしの場合）

```bash
# Vast.ai で実行
HF_TOKEN=xxx TORCHDYNAMO_DISABLE=1 python train.py \
  --data-dir outputs/preprocess_cache/data/data \
  --out-dir outputs \
  --resume outputs/model_best.pt \  # v2ベストから継続
  --epochs 5 \
  --batch 2048 \
  --lr 3e-4 \  # 継続訓練は低LRで
  ...
```

混合比率の目安（Kakolog追加時）:
- wiki: 50%, cc100: 20%, kakolog: 20%, livedoor: 10%

### 2.4 カタカナ/固有名詞コーパス追加（Kakologで不十分な場合）

- `allenai/c4` の ja config（数十GB）
- Wikipedia記事タイトルリスト（読み仮名つき）
- `ime-data-2026-05-17.csv` の失敗例を正例として Fine-tune

---

## Phase 3: 公開準備

### 3.1 UIブラッシュアップ
- Claude Designによる暫定デザイン決定
- キーボードレイアウト、候補表示の改善

### 3.2 GitHub公開
- リポジトリ: `YUGOROU/shiin-ime` (現在 private)
- 構成: `training/`, `ios/`, `README.md`
- 内部Claudeドキュメント (`ROADMAP.md`, `*_HANDOFF.md`) は公開時に整理
- Hugging Face: model (`YUGOROU/shiin-ime-gru`) と dataset を public に変更

---

## 環境・インフラ メモ

### Vast.ai
- GPU推奨: RTX 5090 ($0.37/hr) → PyTorch 2.7.0+cu128 必須 (`sm_120` Blackwell)
- RTX 4090 でも可（2.5.1で動作）
- SSH: `uvx vastai ssh-url <id>` で正確なIPを取得（プロキシ経由は不安定）
- 訓練コマンドに `TORCHDYNAMO_DISABLE=1` 必須（RTX 5090）

### HuggingFace
- `model_best.pt` のアップロードは正常（12.3MB）
- 大ファイル（36MB+）は `hf upload` コマンドでも20MB付近でハングする事例あり
  → `upload_file` を再起動すると解決することが多い

### iOS ビルド
- `modelVersion` を bump すると compiled model キャッシュが無効化される
- 現在: `"v5"`
- XcodeGen: `cd ios && xcodegen generate` でプロジェクト再生成

---

## 参照ファイル

| ファイル | 用途 |
|---------|------|
| `training/train.py` | 訓練スクリプト本体 |
| `training/coreml_convert.py` | CoreML変換（HFからDL→変換） |
| `training/test_coreml_beam.py` | Python側ビームサーチ検証 |
| `training/kakolog_loading_guide.md` | Kakologコーパス読み込み手順 |
| `ios/ShiinIMEKeyboard/Sources/ShiinInferenceEngine.swift` | ビームサーチ実装 |
| `ios/project.yml` | XcodeGenプロジェクト定義 |
