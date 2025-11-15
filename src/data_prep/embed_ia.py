#embed_ia.py

# pip install pandas pyarrow numpy scikit-learn sentence-transformers tqdm joblib


# Note - for my computer ipython works much faster because it uses the gpu (MPS)
# I didn't want to fix this, I just ran with ipython

from pathlib import Path
import os, sys
import json
import math
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm
from joblib import dump
from sklearn.decomposition import PCA

import torch
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

from src.utils import paths

DEFAULT_SPLIT_INTO = 6


def write_table_with_optional_split(
    table: pa.Table, out_parquet: Path, split_into: int
) -> None:
    if split_into < 1:
        raise ValueError("split_into must be >= 1")

    if split_into <= 1 or table.num_rows == 0:
        pq.write_table(table, out_parquet)
        print("Wrote embeddings to ", out_parquet)
        return

    rows = table.num_rows
    chunk_rows = max(1, math.ceil(rows / split_into))
    stem = out_parquet.stem
    manifest_entries = []
    written_parts = 0

    for i in range(split_into):
        start = i * chunk_rows
        if start >= rows:
            break
        length = min(chunk_rows, rows - start)
        part_path = out_parquet.with_name(
            f"{stem}.part{i + 1:02d}of{split_into:02d}.parquet"
        )
        pq.write_table(table.slice(start, length), part_path)
        manifest_entries.append({"file": part_path.name, "rows": length})
        written_parts += 1

    if written_parts == 0:
        pq.write_table(table, out_parquet)
        print("Wrote embeddings →", out_parquet)
        return

    manifest_path = out_parquet.with_suffix(".parts.json")
    manifest_path.write_text(
        json.dumps(
            {
                "total_rows": rows,
                "parts": manifest_entries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {written_parts} parquet chunks to {manifest_path}")

CONFIG_OVERRIDE = os.getenv("TRCIRCUS_CONFIG")
if CONFIG_OVERRIDE:
    paths.configure(Path(CONFIG_OVERRIDE))


ROOT = paths.repo_root()
DATA_RAW = paths.raw_data_dir()
DATA_PROCESSED = paths.processed_data_dir(ensure=True)
ARTIFACTS = paths.artifacts_dir(ensure=True)

IN_PARQUET = paths.truth_posts_parquet()
OUT_PARQUET = paths.post_embeddings_parquet()
WHITENER = ARTIFACTS / "whitener.pkl"

paths.ensure_directory(DATA_PROCESSED)
paths.ensure_directory(ARTIFACTS)

assert IN_PARQUET.exists(), f"Missing input file: {IN_PARQUET}"
print("ROOT:", ROOT)
print("Input:", IN_PARQUET)
print("Output:", OUT_PARQUET)
print("Whitener:", WHITENER)


# Device selection (CUDA -> MPS -> CPU) and model config
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    # mps
    try:
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        pass
    return "cpu"

DEVICE = pick_device()
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# You can tweak this; 256 is safe, 512 may be fine on GPU, lower on CPU.
BATCH = 256

SPLIT_INTO = DEFAULT_SPLIT_INTO

print(f"torch: {torch.__version__} | device picked: {DEVICE}")
print(f"model: {MODEL_NAME} | batch: {BATCH}")


if DEVICE == "cpu":
    torch.set_num_threads(min(8, os.cpu_count() or 8))


model = SentenceTransformer(MODEL_NAME, device=DEVICE)



# Load posts, clean if needed
use_cols = ["id", "created_at", "content", "content_clean", "url", "day"]
df = pd.read_parquet(IN_PARQUET, columns=[c for c in use_cols if c in pd.read_parquet(IN_PARQUET).columns])

# Ensure UTC tz-aware and a clean text column
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
text_col = "content_clean" if "content_clean" in df.columns else "content"
if text_col not in df.columns:
    raise ValueError("Need either 'content_clean' or 'content' in the Parquet.")

df[text_col] = df[text_col].fillna("").astype(str)

# If we only have 'content', strip html to produce a clean column
if text_col == "content":
    def strip_html(s: str) -> str:
        try:
            return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
        except Exception:
            return s
    df["content_clean"] = df["content"].map(strip_html)
    text_col = "content_clean"

# Drop empty / bad timestamps / duplicate ids
df = df[df[text_col].str.strip().ne("")]
df = df.dropna(subset=["created_at"])
df = df.drop_duplicates(subset="id").copy()

# Derive day in UTC; we will store it as date only
df["day"] = df["created_at"].dt.tz_convert("UTC").dt.floor("D")

print("Rows to embed:", len(df))
df.head(2)



# Embed in batches
texts = df[text_col].tolist()
embs = []
for i in tqdm(range(0, len(texts), BATCH), desc=f"Embedding on {DEVICE}"):
    batch = texts[i: i+BATCH]
    E = model.encode(
        batch,
        batch_size=BATCH,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    embs.append(np.asarray(E, dtype=np.float32))

emb = np.vstack(embs)
n, d = emb.shape
print(f"Embedded {n} posts → dim {d}")


# PCA whitening (global)
pca = PCA(whiten=True, svd_solver="auto", random_state=0)
emb_white = pca.fit_transform(emb).astype(np.float32)
dump(
    {
        "mean_": pca.mean_.astype(np.float32),
        "components_": pca.components_.astype(np.float32),
        "explained_variance_": pca.explained_variance_.astype(np.float32),
        "model": MODEL_NAME,
    },
    WHITENER,
)
print("Saved whitener →", WHITENER)


def to_fixed_size_list_array(mat: np.ndarray) -> pa.FixedSizeListArray:
    n, d = mat.shape
    flat = pa.array(mat.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, d)

id_arrow = pa.array(df["id"].astype(str))
created_at_arrow = pa.Array.from_pandas(df["created_at"], type=pa.timestamp("ns", tz="UTC"))
day_arrow = pa.Array.from_pandas(df["day"].dt.date, type=pa.date32())
content_arrow = pa.array(df["content_clean"].astype(str))
url_arrow = pa.array(df["url"].astype(str)) if "url" in df.columns else pa.array([""] * len(df))

emb_raw_arrow = to_fixed_size_list_array(emb)
emb_white_arrow = to_fixed_size_list_array(emb_white)

table = pa.table(
    {
        "id": id_arrow,
        "created_at": created_at_arrow,  # tz-aware UTC
        "day": day_arrow,                # date only
        "content_clean": content_arrow,
        "url": url_arrow,
        "emb_raw": emb_raw_arrow,
        "emb_white": emb_white_arrow,
    }
)

write_table_with_optional_split(table, OUT_PARQUET, SPLIT_INTO)
