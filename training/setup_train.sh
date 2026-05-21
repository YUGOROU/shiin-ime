#!/usr/bin/env bash
# setup_train.sh — 訓練インスタンス用セットアップ & 実行
#
# 想定環境: RTX 5090 (Blackwell, CUDA 12.8+), Disk 100GB 以上
#           PyTorch 公式 Docker イメージ (pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime 推奨)
#
# 使い方:
#   bash setup_train.sh            # HF からデータ DL → 訓練 → HF アップロード
#   bash setup_train.sh --attach   # tmux にアタッチ
#
# 必須環境変数:
#   HF_TOKEN          HuggingFace アクセストークン
#   HF_DATASET_REPO   前処理済みデータセット (例: username/shiin-ime-preprocess)
#   HF_MODEL_REPO     モデルのアップロード先  (例: username/shiin-ime-gru)
#
# オプション環境変数:
#   TRACKIO_SPACE     Trackio HF Space 名 (例: username/shiin-ime-training)
#   TRAIN_EPOCHS      エポック数 (デフォルト: 10)
#   TRAIN_BATCH       バッチサイズ (デフォルト: 2048)

set -euo pipefail

WORK_DIR="${HOME}/shiin-ime"
SESSION="train"
LOG_FILE="${WORK_DIR}/train.log"
DATA_DIR="${WORK_DIR}/outputs/preprocess_cache"

case "${1:-}" in
  --attach) tmux attach -t "$SESSION"; exit 0 ;;
esac

# ── ディレクトリ確認 ────────────────────────────────────────────────────
echo "[setup] Working directory: $WORK_DIR"
if [[ ! -f "$WORK_DIR/training/train.py" ]]; then
  echo "[setup] ERROR: training/train.py not found in $WORK_DIR"
  echo "        git clone https://github.com/YUGOROU/shiin-ime $WORK_DIR"
  exit 1
fi

mkdir -p "$DATA_DIR" "$WORK_DIR/outputs"
cd "$WORK_DIR/training"

# ── GPU 確認・互換 PyTorch 自動インストール ────────────────────────────
BLACKWELL=0
python3 - <<'PYCHECK'
import sys, subprocess
import torch
cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0)
print(f"PyTorch: {torch.__version__}  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}  CC: {cc[0]}.{cc[1]}")

# sm_70 (V100) 以下は PyTorch 2.5+ で削除済み → 2.4.0 に下げる
if cc[0] < 8 and cc != (0,0):
    import re
    major = int(re.match(r'(\d+)', torch.__version__).group(1))
    minor = int(re.split(r'[.\+]', torch.__version__)[1])
    if major > 2 or (major == 2 and minor >= 5):
        print(f"[setup] CC {cc[0]}.{cc[1]} は PyTorch 2.5+ 非対応 → 2.4.0+cu121 をインストール中...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
            "torch==2.4.0+cu121", "--index-url",
            "https://download.pytorch.org/whl/cu121", "-q"])
        print("[setup] PyTorch 2.4.0 インストール完了")

# Blackwell (sm_12+) は torch.compile 無効
if cc[0] >= 12:
    sys.exit(1)
PYCHECK
BLACKWELL=$?

# ── uv インストール ────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "[setup] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv $(uv --version)"

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
EPOCHS="${TRAIN_EPOCHS:-10}"
BATCH="${TRAIN_BATCH:-2048}"

# tmux 内で環境変数を引き継ぐ
[[ -n "${HF_TOKEN:-}" ]]        && tmux send-keys -t "$SESSION" "export HF_TOKEN='$HF_TOKEN'" Enter
[[ -n "${HF_DATASET_REPO:-}" ]] && tmux send-keys -t "$SESSION" "export HF_DATASET_REPO='$HF_DATASET_REPO'" Enter
[[ -n "${HF_MODEL_REPO:-}" ]]   && tmux send-keys -t "$SESSION" "export HF_MODEL_REPO='$HF_MODEL_REPO'" Enter
tmux send-keys -t "$SESSION" "export HF_HUB_ENABLE_HF_TRANSFER=1" Enter
[[ "$BLACKWELL" == "1" ]] && tmux send-keys -t "$SESSION" "export TORCHDYNAMO_DISABLE=1" Enter
tmux send-keys -t "$SESSION" "cd $WORK_DIR/training" Enter

# ── 訓練コマンド構築 ────────────────────────────────────────────────────
HF_DATASET_ARG=""
[[ -n "${HF_DATASET_REPO:-}" ]] && HF_DATASET_ARG="--hf-dataset-repo '$HF_DATASET_REPO'"

HF_MODEL_ARG=""
[[ -n "${HF_MODEL_REPO:-}" ]] && HF_MODEL_ARG="--hf-model-repo '$HF_MODEL_REPO'"

TRACKIO_ARG=""
[[ -n "${TRACKIO_SPACE:-}" ]] && TRACKIO_ARG="--trackio-space '$TRACKIO_SPACE'"

NO_COMPILE_ARG=""
[[ "$BLACKWELL" == "1" ]] && NO_COMPILE_ARG="--no-compile"

TORCHDYNAMO_PREFIX=""
[[ "$BLACKWELL" == "1" ]] && TORCHDYNAMO_PREFIX="TORCHDYNAMO_DISABLE=1"

# uv run は独立 venv を作るためシステムの PyTorch が見えない。
# 非 torch 依存だけ pip でシステム環境に入れ、python で直接実行する。
tmux send-keys -t "$SESSION" \
  "pip install tqdm numpy 'huggingface_hub>=0.23' hf_transfer trackio --quiet" Enter

TRAIN_CMD="$TORCHDYNAMO_PREFIX python train.py \
  --data-dir $DATA_DIR \
  --out-dir  $WORK_DIR/outputs \
  --epochs   $EPOCHS \
  --batch    $BATCH \
  --workers  $WORKERS \
  --hidden   256 \
  --embed    64 \
  --enc-layers 3 \
  --dec-layers 2 \
  --nhead    4 \
  --dropout  0.2 \
  --lr       1e-3 \
  --sentence-ratio 0.7 \
  $NO_COMPILE_ARG \
  $HF_DATASET_ARG \
  $HF_MODEL_ARG \
  $TRACKIO_ARG"

tmux send-keys -t "$SESSION" \
  "echo '[IME] Training...' && $TRAIN_CMD 2>&1 | tee -a $LOG_FILE && echo '[IME] Done!' || echo '[IME] FAILED'" \
  Enter

cat <<EOF

[setup] Running in tmux '$SESSION'

  Attach  : tmux attach -t $SESSION
  Detach  : Ctrl-b d
  Log     : tail -f $LOG_FILE

  HF_TOKEN set      : $( [[ -n "${HF_TOKEN:-}" ]] && echo "YES" || echo "NO" )
  HF_DATASET_REPO   : ${HF_DATASET_REPO:-"(not set — local data only)"}
  HF_MODEL_REPO     : ${HF_MODEL_REPO:-"(not set — no upload)"}
  TRACKIO_SPACE     : ${TRACKIO_SPACE:-"(not set — local only)"}

  Architecture: Transformer Encoder (3L, d=256, h=4) + GRU Decoder (2L)
  Epochs: $EPOCHS  Batch: $BATCH

  Expected outputs:
    $WORK_DIR/outputs/model_best.pt
    $WORK_DIR/outputs/model_final.pt
    $WORK_DIR/outputs/vocab.json
    $WORK_DIR/outputs/training_log.csv

EOF
