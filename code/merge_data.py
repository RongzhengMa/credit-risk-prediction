import polars as pl
from pathlib import Path
from glob import glob
import gc
import os
from tqdm import tqdm

def make_agg_exprs(df):
    exprs = []
    for col in df.columns:
        if col == "case_id":
            continue
        if col.endswith(("P", "A")):
            exprs.extend([
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).max().alias(f"{col}_max"),
                pl.col(col).min().alias(f"{col}_min"),
                pl.col(col).std().alias(f"{col}_std"),
            ])
        elif col.endswith("D"):
            exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
        elif col.endswith("M"):
            exprs.extend([
                pl.col(col).first().alias(f"{col}_first"),
                pl.col(col).last().alias(f"{col}_last"),
                pl.col(col).n_unique().alias(f"{col}_nunique")
            ])
        elif col.endswith(("L", "T")):
            exprs.extend([
                pl.col(col).sum().alias(f"{col}_sum"),
                pl.col(col).max().alias(f"{col}_max"),
                pl.col(col).min().alias(f"{col}_min"),
            ])
        elif "num_group" in col:
            exprs.extend([
                pl.col(col).max().alias(f"{col}_max"),
                pl.col(col).n_unique().alias(f"{col}_nunique")
            ])
    exprs.append(pl.len().alias("record_count"))
    return exprs

def process_depthX(depth_files, depth_level, case_id_list):
    temp_files = []
    all_columns = set()

    for idx, (item, concat_mode) in enumerate(depth_files):
        file_list = item if isinstance(item, list) else [train_dir / item]
        for fidx, f in enumerate(tqdm(file_list, desc=f"Processing depth{depth_level} files")):
            df = pl.read_parquet(f)

            df = df.filter(pl.col("case_id").is_in(case_id_list))

            for col in df.columns:
                if col.endswith("D"):
                    df = df.with_columns(pl.col(col).cast(pl.Date))
            if depth_level == 1:
                df = df.sort(["case_id", "num_group1"])

            df_agg = df.group_by("case_id").agg(make_agg_exprs(df))
            all_columns.update(df_agg.columns)

            temp_path = temp_dir / f"depth{depth_level}_agg_temp_{idx}_{fidx}.parquet"
            df_agg.write_parquet(temp_path)
            temp_files.append(temp_path)

            del df, df_agg
            gc.collect()

    all_columns = list(all_columns)
    if "case_id" in all_columns:
        all_columns.remove("case_id")

    return temp_files, all_columns

def merge_depthX(temp_files, all_columns, depth_level, batch_size=5):
    temp_save_dir = temp_dir / f"depth{depth_level}_temp_batches"
    temp_save_dir.mkdir(parents=True, exist_ok=True)

    batch_files = [temp_files[i:i + batch_size] for i in range(0, len(temp_files), batch_size)]

    print(f"Total {len(batch_files)} batches to process for depth{depth_level}.")

    for idx, batch in enumerate(tqdm(batch_files, desc=f"Processing batches for depth{depth_level}")):
        dfs = []
        for fpath in batch:
            df = pl.read_parquet(fpath)

            missing_cols = [col for col in all_columns if col not in df.columns]
            if missing_cols:
                df_missing = pl.DataFrame({col: [None] * df.height for col in missing_cols})
                df = df.with_columns(df_missing)

            df = df.select(["case_id"] + all_columns)
            dfs.append(df)
            del df
            gc.collect()

        df_batch = pl.concat(dfs, how="vertical_relaxed")
        del dfs
        gc.collect()

        agg_exprs = []
        for col in df_batch.columns:
            if col == "case_id":
                continue
            if col.endswith(("_sum", "_count", "_nunique")):
                agg_exprs.append(pl.col(col).sum().alias(col))
            else:
                agg_exprs.append(pl.col(col).mean().alias(col))

        df_chunk = df_batch.group_by("case_id").agg(agg_exprs)
        chunk_save_path = temp_save_dir / f"chunk_{idx}.parquet"
        df_chunk.write_parquet(chunk_save_path)

        del df_batch, df_chunk
        gc.collect()

    print(f"All batches saved to {temp_save_dir}")


def finalize_depthX_merge(depth_level):
    batch_dir = temp_dir / f"depth{depth_level}_temp_batches"
    batch_files = sorted(batch_dir.glob("chunk_*.parquet"))

    print(f"Final merging {len(batch_files)} batch files for depth{depth_level}...")

    merged = None

    for idx, fpath in enumerate(tqdm(batch_files, desc=f"Final merging depth{depth_level}")):
        df = pl.read_parquet(fpath)

        if merged is None:
            merged = df
        else:
            merged = pl.concat([merged, df], how="vertical_relaxed")
            merged = merged.group_by("case_id").agg([
                pl.col(col).mean().alias(col) for col in merged.columns if col != "case_id"
            ])

        del df
        gc.collect()

    temp_save_path = temp_dir / f"depth{depth_level}_aggregated.parquet"
    merged.write_parquet(temp_save_path)

    print(f"Depth{depth_level} saved to {temp_save_path}")

    del merged
    gc.collect()


train_dir = Path("/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/train")
temp_dir = Path("/kaggle/working/temp_save")
temp_dir.mkdir(parents=True, exist_ok=True)

df_base_full = pl.read_parquet(train_dir / "train_base.parquet")

df_base = df_base_full.sample(n=500_000, with_replacement=False, shuffle=True, seed=42)
case_id_list = df_base["case_id"].to_list()

del df_base_full
gc.collect()

subtables = [
    ("train_static_cb_0.parquet", None),
    (glob(str(train_dir / "train_static_0_*.parquet")), "vertical"),
    (glob(str(train_dir / "train_applprev_1_*.parquet")), "vertical"),
    ("train_tax_registry_a_1.parquet", None),
    ("train_tax_registry_b_1.parquet", None),
    ("train_tax_registry_c_1.parquet", None),
    (glob(str(train_dir / "train_credit_bureau_a_1_*.parquet")), "vertical"),
    ("train_credit_bureau_b_1.parquet", None),
    ("train_other_1.parquet", None),
    ("train_person_1.parquet", None),
    ("train_deposit_1.parquet", None),
    ("train_debitcard_1.parquet", None),
    ("train_credit_bureau_b_2.parquet", None),
    (glob(str(train_dir / "train_credit_bureau_a_2_*.parquet")), "vertical"),
    ("train_applprev_2.parquet", None),
    ("train_person_2.parquet", None)
]

depth0=subtables[:2]
depth1=subtables[2:12]
depth2=subtables[12:]

depth0_temp_dir = Path("/kaggle/working/temp_save/depth0")
depth0_temp_dir.mkdir(parents=True, exist_ok=True)

for item, concat_mode in depth0:
    file_list = item if isinstance(item, list) else [train_dir / item]
    for f in tqdm(file_list, desc="Saving depth0 temp"):
        df = pl.read_parquet(f)
        save_path = depth0_temp_dir / Path(f).name
        df.write_parquet(save_path)
        del df
        gc.collect()


depth0_files = sorted(depth0_temp_dir.glob("*.parquet"))

static_files = []
static_cb_files = []

for f in depth0_files:
    if "cb" in f.name:
        static_cb_files.append(f)
    else:
        static_files.append(f)

depth0_static_tables = []
for f in tqdm(static_files, desc="Reading static (vertical merge)"):
    df = pl.read_parquet(f)
    depth0_static_tables.append(df)
    gc.collect()

df_static = pl.concat(depth0_static_tables, how="vertical")

depth0_cb_tables = []
for f in tqdm(static_cb_files, desc="Reading static_cb (horizontal merge)"):
    df = pl.read_parquet(f)
    depth0_cb_tables.append(df)
    gc.collect()

df_cb = pl.concat(depth0_cb_tables, how="vertical") if len(depth0_cb_tables) > 1 else depth0_cb_tables[0]

df_cb = df_cb.unique(subset=["case_id"], keep="first")

df_depth0 = df_static.join(df_cb, on="case_id", how="left")

common_cols = set(df_base.columns) & set(df_depth0.columns)
common_cols.discard('case_id')
if common_cols:
    df_depth0 = df_depth0.drop(list(common_cols))

df_base = df_base.join(df_depth0, on="case_id", how="left")
del depth0_static_tables, depth0_cb_tables, df_cb, df_depth0
gc.collect()

temp_files, all_columns = process_depthX(depth1, 1,case_id_list)
merge_depthX(temp_files, all_columns, 1, batch_size=5)
finalize_depthX_merge(1)

temp_files, all_columns = process_depthX(depth2, 2,case_id_list)
merge_depthX(temp_files, all_columns, 2, batch_size=5)
finalize_depthX_merge(2)

df_depth1 = pl.read_parquet(temp_dir / "depth1_aggregated.parquet")
df_depth2 = pl.read_parquet(temp_dir / "depth2_aggregated.parquet")

df_base = df_base.join(df_depth1, on="case_id", how="left")
df_base = df_base.join(df_depth2, on="case_id", how="left")

df_base.write_parquet(temp_dir / "base_data.parquet")