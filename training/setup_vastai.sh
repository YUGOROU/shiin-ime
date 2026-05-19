#!/usr/bin/env bash
# setup_vastai.sh — 子音のみIME GPU環境セットアップ & 実行スクリプト
#
# 想定環境: PyTorch公式Docker イメージ (CUDA対応GPUサーバー)
#
# 使い方:
#   bash setup_vastai.sh                               # 前処理 → 訓練
#   bash setup_vastai.sh --preprocess                  # 前処理のみ
#   bash setup_vastai.sh --train                       # 訓練のみ
#   bash setup_vastai.sh --attach                      # tmuxにアタッチ
#
# 環境変数:
#   HF_TOKEN          HuggingFace アクセストークン (アップロードに必要)
#   HF_MODEL_REPO     モデルのアップロード先 (例: username/shiin-ime-gru)
#   HF_DATASET_REPO   データセットのアップロード先 (例: username/shiin-ime-preprocess)

set -euo pipefail

WORK_DIR="${HOME}/shiin-ime"
SESSION="shiin"
LOG_FILE="${WORK_DIR}/setup.log"

MODE="all"
case "${1:-}" in
  --preprocess) MODE="preprocess" ;;
  --train)      MODE="train"      ;;
  --attach)     tmux attach -t "$SESSION"; exit 0 ;;
esac

# ── ディレクトリ確認 ────────────────────────────────────────────────────────
echo "[setup] Working directory: $WORK_DIR"
if [[ ! -f "$WORK_DIR/train.py" ]]; then
  echo "[setup] ERROR: train.py not found in $WORK_DIR"
  echo "        Clone or copy the repository first:"
  echo "        git clone https://github.com/YOUR_USERNAME/shiin-ime $WORK_DIR"
  exit 1
fi
mkdir -p "$WORK_DIR/outputs/preprocess_cache"
cd "$WORK_DIR"

# uv のインストール (未インストールの場合)
if ! command -v uv &>/dev/null; then
  echo "[setup] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv $(uv --version)"

echo "[setup] Installing Python packages via uv..."
uv pip install --system -q \
    "datasets>=2.18" \
    "cutlet>=0.4" \
    "unidic-lite>=1.0" \
    "fugashi>=1.3" \
    "huggingface_hub>=0.23" \
    "hf_transfer>=0.1" \
    "trackio[gpu]>=0.25"

# fugashi のlibmecabフォールバック
if ! python -c "import fugashi" 2>/dev/null; then
  apt-get update -qq && apt-get install -y -qq libmecab-dev mecab mecab-ipadic-utf8
  uv pip install --system -q mecab-python3
fi

# HF_TOKEN が設定されていれば HuggingFace にログイン
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[setup] Logging in to HuggingFace..."
  python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"
  export HF_TOKEN="$HF_TOKEN"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  echo "  HF login OK"
fi

# ── tmux セッション ────────────────────────────────────────────────────────
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

WORKERS=$(nproc)

# tmux 内で環境変数を引き継ぐ
if [[ -n "${HF_TOKEN:-}" ]]; then
  tmux send-keys -t "$SESSION" "export HF_TOKEN='$HF_TOKEN'" Enter
  tmux send-keys -t "$SESSION" "export HF_HUB_ENABLE_HF_TRANSFER=1" Enter
fi
if [[ -n "${HF_MODEL_REPO:-}" ]]; then
  tmux send-keys -t "$SESSION" "export HF_MODEL_REPO='$HF_MODEL_REPO'" Enter
fi
if [[ -n "${HF_DATASET_REPO:-}" ]]; then
  tmux send-keys -t "$SESSION" "export HF_DATASET_REPO='$HF_DATASET_REPO'" Enter
fi
tmux send-keys -t "$SESSION" "cd $WORK_DIR" Enter

# ── 実行コマンド ───────────────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "preprocess" ]]; then
  PREPROCESS_CMD="python preprocess.py --workers $WORKERS"
  [[ -n "${HF_DATASET_REPO:-}" ]] && PREPROCESS_CMD+=" --hf-dataset-repo '$HF_DATASET_REPO'"
  tmux send-keys -t "$SESSION" \
    "echo '[IME] Preprocessing...' && $PREPROCESS_CMD 2>&1 | tee -a $LOG_FILE" \
    Enter
fi

if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
  TRAIN_CMD="python train.py \
    --data-dir outputs/preprocess_cache \
    --out-dir  outputs \
    --epochs   10 \
    --batch    2048 \
    --workers  $WORKERS \
    --hidden   256 \
    --layers   2 \
    --dropout  0.2 \
    --lr       1e-3"
  [[ -n "${HF_MODEL_REPO:-}" ]] && TRAIN_CMD+=" --hf-model-repo '$HF_MODEL_REPO'"
  tmux send-keys -t "$SESSION" \
    "echo '[IME] Training...' && $TRAIN_CMD 2>&1 | tee -a $LOG_FILE && echo '[IME] Done!' || echo '[IME] FAILED'" \
    Enter
fi

cat <<EOF

[setup] Running in tmux '$SESSION'

  Attach  : tmux attach -t $SESSION
  Detach  : Ctrl-b d
  Log     : tail -f $LOG_FILE

  HF_TOKEN set     : $( [[ -n "${HF_TOKEN:-}" ]] && echo "YES (uploads enabled)" || echo "NO (local only)" )
  HF_MODEL_REPO    : ${HF_MODEL_REPO:-"(not set)"}
  HF_DATASET_REPO  : ${HF_DATASET_REPO:-"(not set)"}

  Expected outputs:
    outputs/model_best.pt
    outputs/model_final.pt
    outputs/vocab.json
    outputs/training_log.csv

EOF
