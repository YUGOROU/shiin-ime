#!/usr/bin/env bash
# setup_preprocess.sh — 前処理インスタンス用セットアップ & 実行
#
# 想定環境: CPU 多コア + RAM 大容量 + Disk 200GB 以上
#           (GPU 不要 / PyTorch 不要)
#
# 使い方:
#   bash setup_preprocess.sh                   # 全コーパス前処理 → HF アップロード
#   bash setup_preprocess.sh --preprocess-only # アップロードしない
#   bash setup_preprocess.sh --attach          # tmux にアタッチ
#
# 必須環境変数:
#   HF_TOKEN          HuggingFace アクセストークン
#   HF_DATASET_REPO   データセットのアップロード先 (例: username/shiin-ime-preprocess)

set -euo pipefail

WORK_DIR="${HOME}/shiin-ime"
SESSION="preprocess"
LOG_FILE="${WORK_DIR}/preprocess.log"

MODE="all"
case "${1:-}" in
  --preprocess-only) MODE="preprocess" ;;
  --attach)          tmux attach -t "$SESSION"; exit 0 ;;
esac

# ── ディレクトリ確認 ────────────────────────────────────────────────────
echo "[setup] Working directory: $WORK_DIR"
for f in preprocess.py preprocess_kakolog.py; do
  if [[ ! -f "$WORK_DIR/training/$f" ]]; then
    echo "[setup] ERROR: training/$f not found in $WORK_DIR"
    echo "        git clone https://github.com/YUGOROU/shiin-ime $WORK_DIR"
    exit 1
  fi
done

mkdir -p "$WORK_DIR/outputs/preprocess_cache"
cd "$WORK_DIR/training"

# ── uv インストール ────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "[setup] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv $(uv --version)"

# ── unidic 辞書ダウンロード (初回のみ) ────────────────────────────────
if ! python3 -c "import unidic; import os; assert os.path.exists(unidic.DICDIR)" 2>/dev/null; then
  echo "[setup] Downloading unidic dictionary..."
  uv run --with unidic-lite python -m unidic download || true
fi

# ── HF ログイン ────────────────────────────────────────────────────────
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[setup] Logging in to HuggingFace..."
  uv run --with huggingface_hub python -c \
    "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi

# ── tmux セッション ─────────────────────────────────────────────────────
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

WORKERS=$(nproc)

# tmux 内で環境変数を引き継ぐ
[[ -n "${HF_TOKEN:-}" ]]        && tmux send-keys -t "$SESSION" "export HF_TOKEN='$HF_TOKEN'" Enter
[[ -n "${HF_DATASET_REPO:-}" ]] && tmux send-keys -t "$SESSION" "export HF_DATASET_REPO='$HF_DATASET_REPO'" Enter
tmux send-keys -t "$SESSION" "export HF_HUB_ENABLE_HF_TRANSFER=1" Enter
tmux send-keys -t "$SESSION" "cd $WORK_DIR/training" Enter

# ── コマンド構築 ────────────────────────────────────────────────────────
OUT_DIR="$WORK_DIR/outputs/preprocess_cache"

HF_REPO_ARG=""
if [[ "$MODE" == "all" && -n "${HF_DATASET_REPO:-}" ]]; then
  HF_REPO_ARG="--hf-dataset-repo '$HF_DATASET_REPO'"
fi

# 1. 通常コーパス (wiki, cc100, livedoor)
PREPROCESS_CMD="uv run preprocess.py \
  --out-dir $OUT_DIR \
  --workers $WORKERS \
  $HF_REPO_ARG"

# 2. Kakolog (datasets==2.21.0 が inline deps で自動インストールされる)
KAKOLOG_CMD="uv run preprocess_kakolog.py \
  --out-dir $OUT_DIR \
  --workers $WORKERS \
  --number-of-files 20 \
  $HF_REPO_ARG"

tmux send-keys -t "$SESSION" \
  "echo '[IME] Preprocessing wiki/cc100/livedoor...' && $PREPROCESS_CMD 2>&1 | tee -a $LOG_FILE" \
  Enter
tmux send-keys -t "$SESSION" \
  "echo '[IME] Preprocessing Kakolog...' && $KAKOLOG_CMD 2>&1 | tee -a $LOG_FILE && echo '[IME] All done!' || echo '[IME] FAILED'" \
  Enter

cat <<EOF

[setup] Running in tmux '$SESSION'

  Attach  : tmux attach -t $SESSION
  Detach  : Ctrl-b d
  Log     : tail -f $LOG_FILE

  HF_TOKEN set      : $( [[ -n "${HF_TOKEN:-}" ]] && echo "YES" || echo "NO (local only)" )
  HF_DATASET_REPO   : ${HF_DATASET_REPO:-"(not set — local only)"}

  Expected outputs:
    $OUT_DIR/wiki.jsonl
    $OUT_DIR/cc100.jsonl
    $OUT_DIR/livedoor.jsonl
    $OUT_DIR/kakolog.jsonl

EOF
