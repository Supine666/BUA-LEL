# utils/preprocess_fold.py
import os
from typing import Dict, Optional, List, Tuple, Set

import numpy as np
import pandas as pd
from torch.utils.data import Subset

from datasets import DualTaskDataset  # 你已经拆成 datasets 了建议这样引

def norm_pid(x) -> str:
    return str(x).strip()

def scan_image_pids(image_dir: str) -> List[str]:
    exts = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
    pids = []
    for fn in os.listdir(image_dir):
        if fn.lower().endswith(exts):
            pids.append(norm_pid(os.path.splitext(fn)[0]))
    return pids

def read_excel_df(clinical_excel: str) -> pd.DataFrame:
    df = pd.read_excel(clinical_excel, engine="openpyxl")
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).apply(norm_pid)
    return df

def convert_labels(raw_labels: np.ndarray) -> np.ndarray:
    s = pd.Series(raw_labels)
    if s.isna().any():
        raise ValueError("标签列存在 NaN")
    try:
        labels = s.astype(float).astype(int).to_numpy()
    except Exception:
        s_str = s.astype(str).str.strip().str.upper()
        mapping = {"LN0": 0, "LN1-3": 1, "LN1–3": 1, "LN1—3": 1, "LN4+": 2, "LN4 +": 2}
        mapped = s_str.map(mapping)
        if mapped.isna().any():
            bad = s_str[mapped.isna()].value_counts().head(20)
            raise ValueError("无法识别的标签: \n" + str(bad))
        labels = mapped.astype(int).to_numpy()

    if not np.all(np.isin(np.unique(labels), [0, 1, 2])):
        raise ValueError(f"标签必须为0/1/2，发现: {np.unique(labels)}")
    return labels

def build_pid_and_labels(image_dir: str, clinical_excel: str) -> Tuple[List[str], np.ndarray]:
    df = read_excel_df(clinical_excel)
    pid_col = df.columns[0]
    label_col = df.columns[-1]

    pid2y = {}
    y_arr = convert_labels(df[label_col].to_numpy())
    for pid, y in zip(df[pid_col].to_numpy(), y_arr):
        pid2y[str(pid).strip()] = int(y)

    img_pids = scan_image_pids(image_dir)
    valid_pids = [pid for pid in img_pids if pid in pid2y]
    y_all = np.array([pid2y[pid] for pid in valid_pids], dtype=np.int64)
    return valid_pids, y_all

def fit_fold_cat_maps(clinical_excel: str, train_pids: Set[str]) -> Dict[str, Dict[str, int]]:
    df = read_excel_df(clinical_excel)
    pid_col = df.columns[0]
    df_tr = df[df[pid_col].isin(train_pids)].copy()

    feats = df_tr.iloc[:, 1:-1]
    cat_df = feats.select_dtypes(exclude=[np.number])

    cat_maps: Dict[str, Dict[str, int]] = {}
    for col in cat_df.columns:
        obs = cat_df[col][~cat_df[col].isna()].astype(str)
        cats = sorted(obs.unique().tolist())
        cat_maps[col] = {c: i for i, c in enumerate(cats)}
    return cat_maps

def fit_fold_num_scaler(clinical_excel: str, train_pids: Set[str]) -> Optional[Dict[str, np.ndarray]]:
    df = read_excel_df(clinical_excel)
    pid_col = df.columns[0]
    df_tr = df[df[pid_col].isin(train_pids)].copy()

    feats = df_tr.iloc[:, 1:-1]
    num_df = feats.select_dtypes(include=[np.number])

    if num_df.shape[1] == 0:
        return None

    means, stds = [], []
    for col in num_df.columns:
        vals = num_df[col].to_numpy(dtype=np.float32)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            mu, sd = 0.0, 1.0
        else:
            mu = float(vals.mean())
            sd = float(vals.std(ddof=0))
            if sd < 1e-6:
                sd = 1.0
        means.append(mu)
        stds.append(sd)
    return {"mean": np.array(means, dtype=np.float32), "std": np.array(stds, dtype=np.float32)}

def subset_by_pid_set(ds: DualTaskDataset, pid_set: Set[str]) -> Subset:
    idxs = [i for i, pid in enumerate(ds.valid_pids) if pid in pid_set]
    if len(idxs) == 0:
        raise RuntimeError("subset_by_pid_set 得到空集合：请检查 pid 交集是否正确")
    return Subset(ds, idxs)