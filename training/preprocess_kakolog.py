# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets==2.21.0",
#   "cutlet>=0.4",
#   "fugashi>=1.3",
#   "unidic-lite>=1.0",
#   "huggingface_hub>=0.23",
#   "hf_transfer>=0.1",
# ]
# ///
"""
preprocess_kakolog.py — KakologArchives 専用前処理スクリプト

datasets==2.21.0 が必須 (3.x 系では trust_remote_code が動作しない)。
同じ出力形式で preprocess.py と同じ --out-dir に書き込むことで混合訓練に使用する。

出力形式 (JSONL):
  {"consonants":"wtsh","reading":"わたし","source":"kakolog","type":"word"}
  {"consonants":"kywhrds","reading":"きょうははれです","source":"kakolog","type":"sentence"}
"""

import os
import re
import json
import logging
import multiprocessing as mp
import threading
import argparse
from pathlib import Path
from queue import Queue, Empty

import cutlet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
for _noisy in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

VOWELS   = re.compile(r"[aeiou]")
_JP_RE   = re.compile(r"[぀-ヿ一-鿿]")
_HTML_RE = re.compile(r"<[^>]+>")


def kata_to_hira(s: str) -> str:
    return "".join(
        chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c
        for c in s
    )


def is_valid_kakolog(text: str) -> bool:
    """Kakolog向けフィルタ: 6文字以上、JP文字50%以上、HTML・スパムなし"""
    text = text.strip()
    if len(text) < 6:
        return False
    if _HTML_RE.search(text):
        return False
    if re.fullmatch(r"[wWｗWＷ\s！？。、…・ｗ]+", text):
        return False
    jp_chars = sum(1 for c in text if _JP_RE.match(c))
    if jp_chars / len(text) < 0.50:
        return False
    return True


# ── Worker ────────────────────────────────────────────────────────────────
def _proc_chunk(args: tuple[list[str], str]) -> list[bytes]:
    texts, source = args
    import sys
    try:
        katsu = cutlet.Cutlet()
        import fugashi as _fugashi
        tagger = _fugashi.Tagger()
    except Exception as e:
        print(f"[_proc_chunk] init failed: {e}", file=sys.stderr, flush=True)
        raise

    _test = re.sub(r"[^a-z\s]", "", katsu.romaji("東京").lower())
    if not _test:
        raise RuntimeError("cutlet/MeCab not working. Run: python -m unidic download")

    out: list[bytes] = []
    source_b = source.encode()
    err_count = 0

    for raw in texts:
        text = re.sub(r"[\U00010000-\U0010ffff]", "", raw)
        # Kakolog は短い投稿が多いため、改行で区切るのみ（句読点は少ない）
        for sent in re.split(r"[\n。！？]+", text):
            sent = sent.strip()
            if not is_valid_kakolog(sent):
                continue
            try:
                words_parsed = list(tagger(sent))

                # ── 文レベルペア ──────────────────────────────────────────
                sent_hira = kata_to_hira("".join(
                    w.feature.kana or "" for w in words_parsed
                ))
                if not sent_hira:
                    continue

                sent_rom  = re.sub(r"[^a-z\s]", "", katsu.romaji(sent).lower())
                sent_cons = VOWELS.sub("", sent_rom.replace(" ", ""))

                if 2 <= len(sent_cons) <= 80 and 2 <= len(sent_hira) <= 150:
                    hira_b = sent_hira.encode("utf-8")
                    out.append(
                        b'{"consonants":"' + sent_cons.encode() +
                        b'","reading":"'   + hira_b +
                        b'","source":"'    + source_b + b'","type":"sentence"}\n'
                    )

                # ── 単語レベルペア ────────────────────────────────────────
                for w in words_parsed:
                    kana = w.feature.kana or ""
                    if not kana or len(kana) < 2:
                        continue
                    word_hira = kata_to_hira(kana)
                    word_rom  = re.sub(r"[^a-z\s]", "", katsu.romaji(w.surface).lower())
                    word_cons = VOWELS.sub("", word_rom.replace(" ", ""))
                    if 1 <= len(word_cons) <= 20 and 1 <= len(word_hira) <= 30:
                        hira_b = word_hira.encode("utf-8")
                        out.append(
                            b'{"consonants":"' + word_cons.encode() +
                            b'","reading":"'   + hira_b +
                            b'","source":"'    + source_b + b'","type":"word"}\n'
                        )

            except Exception as e:
                err_count += 1
                if err_count <= 3:
                    print(f"[_proc_chunk] error ({err_count}): {e} | {sent[:40]!r}",
                          file=sys.stderr, flush=True)
                continue
    return out


def _chunk_iter(it, size: int):
    buf: list = []
    for item in it:
        buf.append(item)
        if len(buf) >= size:
            yield buf; buf = []
    if buf:
        yield buf


def _writer(q: Queue, out_path: Path):
    written = 0
    with out_path.open("wb") as f:
        while True:
            try:
                batch = q.get(timeout=60)
            except Empty:
                continue
            if batch is None:
                break
            for row in batch:
                f.write(row)
                written += 1
    log.info(f"Writer done: {written:,} pairs → {out_path}")


# ── Kakolog ローダー ───────────────────────────────────────────────────────
def _iter_kakolog(max_docs: int | None, number_of_files: int, token: str | None):
    """
    datasets==2.21.0 + trust_remote_code=True が必須。
    'sample' config を使用。'all' (190GB) は使用しない。
    """
    from datasets import load_dataset

    log.info(f"Loading KakologArchives/KakologArchives (sample, files={number_of_files})")
    try:
        ds = load_dataset(
            "KakologArchives/KakologArchives",
            "sample",
            number_of_files=number_of_files,
            trust_remote_code=True,
            token=token,
        )
    except Exception as e:
        log.error(f"Kakolog load failed: {e}")
        log.error("datasets==2.21.0 が必要です。pip install datasets==2.21.0 を実行してください。")
        return

    # load_dataset は DatasetDict を返す場合がある
    if hasattr(ds, "keys"):
        split = "train" if "train" in ds else list(ds.keys())[0]
        ds = ds[split]

    log.info(f"Kakolog loaded: {len(ds):,} rows")

    # テキスト列を探す
    text_col = None
    for c in ("text", "content", "body", "sentence"):
        if c in ds.column_names:
            text_col = c
            break
    if text_col is None:
        log.error(f"No text column found. Columns: {ds.column_names}")
        return

    count = 0
    for row in ds:
        text = row.get(text_col) or ""
        if text:
            yield str(text)
            count += 1
        if max_docs and count >= max_docs:
            break
    log.info(f"Kakolog: {count:,} texts yielded")


# ── 前処理メイン ──────────────────────────────────────────────────────────
def preprocess_kakolog(out_path: Path, n_workers: int, chunk_size: int,
                        max_docs: int | None, number_of_files: int):
    token = os.environ.get("HF_TOKEN")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_q: Queue = Queue(maxsize=n_workers * 4)
    writer = threading.Thread(target=_writer, args=(write_q, out_path), daemon=True)
    writer.start()

    task_iter = (
        (chunk, "kakolog")
        for chunk in _chunk_iter(_iter_kakolog(max_docs, number_of_files, token), chunk_size)
    )
    processed = 0
    with mp.Pool(n_workers) as pool:
        for batch in pool.imap_unordered(_proc_chunk, task_iter, chunksize=1):
            write_q.put(batch)
            processed += len(batch)
            if processed > 0 and processed % 200_000 == 0:
                log.info(f"[kakolog] {processed:,} pairs queued...")

    write_q.put(None)
    writer.join()
    log.info(f"[kakolog] complete: {processed:,} pairs → {out_path}")


# ── HuggingFace Dataset アップロード（既存ファイルに追加） ───────────────
def _upload_to_hf(out_file: Path, args):
    token = os.environ.get("HF_TOKEN")
    if not token or not args.hf_dataset_repo:
        return

    from huggingface_hub import HfApi

    api      = HfApi(token=token)
    username = api.whoami()["name"]
    repo_name = args.hf_dataset_repo
    repo_id  = repo_name if "/" in repo_name else f"{username}/{repo_name}"

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(out_file),
        path_in_repo=f"data/{out_file.name}",
        repo_id=repo_id,
        repo_type="dataset",
    )
    log.info(f"Uploaded: {out_file.name} → {repo_id}")


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Kakolog専用前処理。datasets==2.21.0 が必要。"
    )
    ap.add_argument("--out-dir",         default="outputs/preprocess_cache")
    ap.add_argument("--workers",         type=int, default=mp.cpu_count())
    ap.add_argument("--chunk-size",      type=int, default=4000,
                    help="Kakologは短文が多いため小さめに設定")
    ap.add_argument("--max-docs",        type=int, default=0,
                    help="0=上限なし")
    ap.add_argument("--number-of-files", type=int, default=20,
                    help="Kakolog sample config のファイル数 (デフォルト20 ≈ 数百万件)")
    ap.add_argument("--hf-dataset-repo", default="",
                    help="HFデータセットリポジトリ名。空=アップロードしない")
    args = ap.parse_args()

    log.info(f"CPU cores: {mp.cpu_count()}")
    out_path = Path(args.out_dir) / "kakolog.jsonl"
    preprocess_kakolog(
        out_path    = out_path,
        n_workers   = args.workers,
        chunk_size  = args.chunk_size,
        max_docs    = args.max_docs or None,
        number_of_files = args.number_of_files,
    )

    if args.hf_dataset_repo:
        _upload_to_hf(out_path, args)

    log.info("Kakolog preprocessing complete.")
