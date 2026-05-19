# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "cutlet>=0.4",
#   "fugashi>=1.3",
#   "unidic-lite>=1.0",
#   "huggingface_hub>=0.23",
#   "hf_transfer>=0.1",
#   "pyarrow>=15",
# ]
# ///
"""
preprocess.py — 子音のみIME 前処理パイプライン (Phase 2)

Kakolog は専用スクリプト preprocess_kakolog.py で処理。

出力形式 (JSONL):
  {"consonants":"wtsh","reading":"わたし","source":"wiki","type":"word"}
  {"consonants":"kywhrds","reading":"きょうははれです","source":"wiki","type":"sentence"}
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
_KANA_RE = re.compile(r"[぀-ヿ一-鿿]")
_HTML_RE = re.compile(r"<[^>]+>")

# Kakolog は preprocess_kakolog.py で処理
DATASETS: dict[str, tuple] = {
    "wiki":     ("hpprc/jawiki-paragraphs",       "train", "text",     None),
    "cc100":    ("range3/cc100-ja",                "train", "text",     5_000_000),
    "livedoor": ("shunk031/livedoor-news-corpus",  "train", "sentence", None),
}


def kata_to_hira(s: str) -> str:
    """カタカナ → ひらがな。ー など対応するひらがながない文字はそのまま残す。"""
    return "".join(
        chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c
        for c in s
    )


# ── テキストフィルタ ──────────────────────────────────────────────────────
def is_valid(text: str) -> bool:
    if len(text) < 6:                                            return False
    if _HTML_RE.search(text):                                    return False
    ascii_alnum = sum(c.isascii() and c.isalnum() for c in text)
    if ascii_alnum / len(text) > 0.70:                          return False
    if not _KANA_RE.search(text):                               return False
    if re.fullmatch(r"[wWｗWＷ\s！？。、…・]+", text):           return False
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
        raise RuntimeError(
            "cutlet returned no ASCII — MeCab/unidic not working. "
            "Run: python -m unidic download"
        )

    out: list[bytes] = []
    source_b = source.encode()
    err_count = 0

    for raw in texts:
        text = re.sub(r"[\U00010000-\U0010ffff]", "", raw)
        for sent in re.split(r"[。！？\n]+", text):
            sent = sent.strip()
            if not is_valid(sent):
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


# ── HF シャードダウンロード ────────────────────────────────────────────────
def _find_text_col(col_names: list[str]) -> str | None:
    for c in ("text", "sentence", "content", "body", "paragraph"):
        if c in col_names:
            return c
    return col_names[0] if col_names else None


def _iter_texts(hf_id: str, split: str, col: str, max_docs, token: str | None):
    from huggingface_hub import list_repo_files

    def _find_shards(revision=None):
        try:
            files = sorted(list_repo_files(
                hf_id, repo_type="dataset", token=token, revision=revision,
            ))
        except Exception as e:
            log.warning(f"list_repo_files({hf_id}, rev={revision}): {e}")
            return [], revision
        shards = [f for f in files if f.endswith(".parquet") and split in f]
        if not shards:
            shards = [f for f in files if f.endswith(".parquet")]
        return shards, revision

    shards, rev = _find_shards(revision=None)
    if not shards:
        shards, rev = _find_shards(revision="refs/convert/parquet")
        if shards:
            log.info(f"{hf_id}: using refs/convert/parquet branch ({len(shards)} shards)")

    if shards:
        yield from _iter_parquet(hf_id, shards, col, max_docs, token, revision=rev)
    else:
        log.info(f"{hf_id}: no parquet shards → falling back to load_dataset(streaming=False)")
        yield from _iter_datasets_lib(hf_id, split, col, max_docs, token)


def _iter_parquet(hf_id: str, shards: list[str], col: str, max_docs, token,
                  revision: str | None = None):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    log.info(f"{hf_id}: {len(shards)} parquet shards")
    count = 0
    for shard in shards:
        try:
            log.info(f"  Downloading: {shard}")
            local_path = hf_hub_download(
                repo_id=hf_id, filename=shard,
                repo_type="dataset", token=token, revision=revision,
            )
            table   = pq.read_table(local_path)
            use_col = col if col in table.schema.names else _find_text_col(table.schema.names)
            if not use_col:
                log.warning(f"  No usable column in {shard} (cols: {table.schema.names})")
                continue
            for val in table.column(use_col).to_pylist():
                if val:
                    yield str(val)
                    count += 1
                if max_docs and count >= max_docs:
                    return
        except Exception as e:
            log.warning(f"  shard {shard}: {e}. Skipping.")
            continue
        if max_docs and count >= max_docs:
            break
    log.info(f"{hf_id}: {count:,} texts from parquet")


def _iter_datasets_lib(hf_id: str, split: str, col: str, max_docs, token):
    from datasets import load_dataset
    try:
        ds = load_dataset(hf_id, split=split, token=token)
        if max_docs and len(ds) > max_docs:
            ds = ds.select(range(max_docs))
        use_col = col if col in ds.column_names else _find_text_col(ds.column_names)
        if not use_col:
            log.warning(f"{hf_id}: no usable text column (cols: {ds.column_names})")
            return
        log.info(f"{hf_id}: {len(ds):,} rows, col='{use_col}'")
        count = 0
        for row in ds:
            text = row.get(use_col) or ""
            if text:
                yield str(text)
                count += 1
        log.info(f"{hf_id}: {count:,} texts from load_dataset fallback")
    except Exception as e:
        log.warning(f"load_dataset({hf_id}): {e}. Skipping.")


# ── 前処理メイン ──────────────────────────────────────────────────────────
def preprocess_source(name: str, out_path: Path, n_workers: int, chunk_size: int):
    hf_id, split, col, max_docs = DATASETS[name]
    token = os.environ.get("HF_TOKEN")
    log.info(f"[{name}] {hf_id}  workers={n_workers}  chunk={chunk_size}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_q: Queue = Queue(maxsize=n_workers * 4)
    writer = threading.Thread(target=_writer, args=(write_q, out_path), daemon=True)
    writer.start()

    task_iter = (
        (chunk, name)
        for chunk in _chunk_iter(_iter_texts(hf_id, split, col, max_docs, token), chunk_size)
    )
    processed = 0
    with mp.Pool(n_workers) as pool:
        for batch in pool.imap_unordered(_proc_chunk, task_iter, chunksize=1):
            write_q.put(batch)
            processed += len(batch)
            if processed > 0 and processed % 500_000 == 0:
                log.info(f"[{name}] {processed:,} pairs queued...")

    write_q.put(None)
    writer.join()
    log.info(f"[{name}] complete: {processed:,} pairs")


# ── HuggingFace Dataset アップロード ────────────────────────────────────
def _upload_to_hf(cache_dir: Path, args):
    token = os.environ.get("HF_TOKEN")
    if not token or not args.hf_dataset_repo:
        return

    from huggingface_hub import HfApi, DatasetCard

    api      = HfApi(token=token)
    username = api.whoami()["name"]
    repo_name = args.hf_dataset_repo
    repo_id  = repo_name if "/" in repo_name else f"{username}/{repo_name}"

    log.info(f"Uploading dataset to {repo_id} (private)...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

    jsonl_files = list(cache_dir.glob("*.jsonl"))
    for jf in jsonl_files:
        api.upload_file(
            path_or_fileobj=str(jf),
            path_in_repo=f"data/{jf.name}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        log.info(f"  Uploaded: {jf.name}")

    card_content = f"""\
---
license: other
task_categories:
- other
language:
- ja
private: true
---

# shiin-ime-preprocess

子音のみ日本語IME 訓練データセット。

## Files
{chr(10).join(f'- `data/{jf.name}`' for jf in jsonl_files)}
"""
    DatasetCard(card_content).push_to_hub(repo_id, token=token)
    log.info(f"Dataset uploaded: https://huggingface.co/datasets/{repo_id}")


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",    default="outputs/preprocess_cache")
    ap.add_argument("--workers",    type=int, default=mp.cpu_count())
    ap.add_argument("--chunk-size", type=int, default=8000)
    ap.add_argument("--datasets",   nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--hf-dataset-repo", default="",
                    help="HFデータセットリポジトリ名。空=アップロードしない")
    args = ap.parse_args()

    log.info(f"CPU cores: {mp.cpu_count()}")
    out_dir = Path(args.out_dir)

    for name in args.datasets:
        if name not in DATASETS:
            log.error(f"Unknown: {name}. Options: {list(DATASETS)}")
            continue
        preprocess_source(name, out_dir / f"{name}.jsonl", args.workers, args.chunk_size)

    log.info("All preprocessing complete.")
    _upload_to_hf(out_dir, args)
