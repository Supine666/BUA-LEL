# scripts/run_cv.py
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# Add project root to Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import DualTaskDataset
from models.sgmtf import SGMTFModel

from utils.seed import seed_everything
from utils.preprocess_fold import (
    build_pid_and_labels,
    fit_fold_cat_maps,
    fit_fold_num_scaler,
    subset_by_pid_set,
)
from utils.optim import (
    freeze_bn_running_stats,
    build_optimizer_param_groups,
)
from utils.roc import plot_multiclass_roc

from engines.losses import make_nomissing_loss_fn
from engines.train_eval import train_one_epoch, evaluate


# =========================
# Config
# =========================
@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    image_dir: str = str(PROJECT_ROOT / "data" / "images")
    mask_dir: str = str(PROJECT_ROOT / "data" / "masks")
    clinical_excel: str = str(PROJECT_ROOT / "data" / "clinical.xlsx")

    save_dir: str = "./checkpoints_bul"
    num_workers: int = 4

    folds: int = 5
    epochs: int = 50
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    early_stop_patience: int = 12

    use_pca: bool = False
    pca_dim: int = 100
    clinical_embed_dim: int = 128
    detach_roi_in_cls: bool = True
    detach_segfeat_in_cls: bool = False

    # 无缺失：训练侧强制 m=1，且关闭一致性项
    lambda_cons: float = 0.0

    # Boundary graph refinement / anchor constraint
    use_boundary_refiner: bool = True
    boundary_num_nodes: int = 64
    boundary_hidden_dim: int = 128
    boundary_gnn_layers: int = 3
    boundary_max_offset: float = 0.08
    lambda_anchor: float = 0.0
    boundary_token_gate_init: float = -2.0
    use_uncertainty_weighting: bool = False

    force_full_observed_mask: bool = True

    # 数据侧：避免“未知类别=缺失”打破无缺失设定
    unknown_cat_as_missing: bool = False

    # MedSAM backbone
    # 与 models/sgmtf.py 中 SGMTFModel 的新增参数保持一致
    medsam_checkpoint_path: Optional[str] = str(PROJECT_ROOT / "work_dir" / "MedSAM" / "medsam_vit_b.pth")
    medsam_model_type: str = "vit_b"
    freeze_medsam: bool = True
    unfreeze_medsam_last_n: int = 0
    image_size: int = 1024

    # scheduler
    scheduler: str = "cosine"
    min_lr: float = 1e-6

    # cls
    use_class_weight: bool = True
    num_classes: int = 3
    label_smoothing: float = 0.05

    # seg
    seg_thr: float = 0.5
    seg_bce_weight: float = 0.5

    # score
    score_w_seg: float = 0.3
    score_w_cls: float = 0.7

    # stability
    freeze_bn: bool = True
    grad_accum_steps: int = 1

    # cls/fusion higher lr
    cls_lr_mult: float = 2.0
    cls_name_keywords: Tuple[str, ...] = (
        "cls_head", "classifier", "logit", "fc",
        "gating", "clin_module", "phi_clin", "img_gate", "clin_gate",
        # 新增：MedSAM 中间层分类分支相关模块
        # encoder.cls_fusion 位于 MedSAMMultiScaleEncoder 内部
        # clsfeat_to_imgdim 是 SGMTFModel 中新增的分类特征投影层
        "cls_fusion", "clsfeat_to_imgdim", "feature_fusion",
        "hetero_graph", "clin_graph"
    )

    # io
    save_fold_metrics_csv: bool = True
    summary_table_name: str = "paper_table_mean_std.csv"
    save_final_oof_roc: bool = True
    final_roc_name: str = "OOF_ROC.png"
    save_oof_npz: bool = True


def composite_score(stats: Dict[str, float], w_seg=0.5, w_cls=0.5) -> float:
    return w_seg * float(stats.get("dice", 0.0)) + w_cls * float(stats.get("macro_f1", 0.0))


# =========================
# Main
# =========================
def main(cfg: TrainConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)

    all_pids, y_all = build_pid_and_labels(cfg.image_dir, cfg.clinical_excel)
    print(f"[Split] total matched (image ∩ clinical) = {len(all_pids)}")

    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)

    best_rows = []
    oof_true_all: List[np.ndarray] = []
    oof_proba_all: List[np.ndarray] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all), start=1):
        print(f"\n================= Fold {fold}/{cfg.folds} =================")

        train_pids = {all_pids[i] for i in train_idx}
        val_pids = {all_pids[i] for i in val_idx}

        cat_maps = fit_fold_cat_maps(cfg.clinical_excel, train_pids)
        num_scaler = fit_fold_num_scaler(cfg.clinical_excel, train_pids)

        fold_ds = DualTaskDataset(
            image_dir=cfg.image_dir,
            mask_dir=cfg.mask_dir,
            clinical_excel=cfg.clinical_excel,
            mode="seg",
            allow_missing_mask=True,
            return_pid=False,
            transform=None,
            num_scaler=num_scaler,
            cat_maps=cat_maps,
            unknown_cat_as_missing=cfg.unknown_cat_as_missing,
            return_cat_targets=True,
        )

        train_set = subset_by_pid_set(fold_ds, train_pids)
        val_set = subset_by_pid_set(fold_ds, val_pids)

        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        clinical_dim = fold_ds.get_feature_dim()

        model = SGMTFModel(
            clinical_dim=clinical_dim,
            numeric_slice=fold_ds.numeric_slice,
            onehot_slices_dict=fold_ds.onehot_slices,
            num_classes=cfg.num_classes,
            use_pca=cfg.use_pca,
            pca_dim=cfg.pca_dim,
            clin_embed_dim=cfg.clinical_embed_dim,
            detach_roi_in_cls=cfg.detach_roi_in_cls,
            detach_segfeat_in_cls=cfg.detach_segfeat_in_cls,
            lambda_cons=cfg.lambda_cons,

            # Boundary graph refinement parameters
            use_boundary_refiner=cfg.use_boundary_refiner,
            boundary_num_nodes=cfg.boundary_num_nodes,
            boundary_hidden_dim=cfg.boundary_hidden_dim,
            boundary_gnn_layers=cfg.boundary_gnn_layers,
            boundary_max_offset=cfg.boundary_max_offset,
            lambda_anchor=cfg.lambda_anchor,
            boundary_token_gate_init=cfg.boundary_token_gate_init,

            # 新增：传入 MedSAM 编码器参数
            medsam_checkpoint_path=cfg.medsam_checkpoint_path,
            medsam_model_type=cfg.medsam_model_type,
            freeze_medsam=cfg.freeze_medsam,
            unfreeze_medsam_last_n=cfg.unfreeze_medsam_last_n,
            image_size=cfg.image_size,
        ).to(device)

        # 打印可训练参数，确认 MedSAM 冻结策略是否生效
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] total params = {total_params:,}")
        print(f"[Model] trainable params = {trainable_params:,}")
        print(
            f"[MedSAM] checkpoint={cfg.medsam_checkpoint_path}, "
            f"freeze={cfg.freeze_medsam}, "
            f"unfreeze_last_n={cfg.unfreeze_medsam_last_n}, "
            f"image_size={cfg.image_size}"
        )
        print(
            f"[Boundary] use={cfg.use_boundary_refiner}, "
            f"nodes={cfg.boundary_num_nodes}, hidden={cfg.boundary_hidden_dim}, "
            f"layers={cfg.boundary_gnn_layers}, max_offset={cfg.boundary_max_offset}, "
            f"lambda_anchor={cfg.lambda_anchor}, "
            f"gate_init={cfg.boundary_token_gate_init}, "
            f"uncertainty_weighting={cfg.use_uncertainty_weighting}"
        )

        # cls criterion
        if cfg.use_class_weight:
            train_labels = y_all[train_idx]
            counts = np.bincount(train_labels, minlength=cfg.num_classes).astype(np.float32)
            weights = (counts.sum() / (counts + 1e-6))
            weights = weights / weights.mean()
            class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
            cls_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
            print("class_weights:", weights)
        else:
            cls_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        # optimizer param groups
        groups, (n_other, n_cls) = build_optimizer_param_groups(
            model=model,
            base_lr=cfg.lr,
            cls_lr_mult=cfg.cls_lr_mult,
            weight_decay=cfg.weight_decay,
            cls_name_keywords=tuple(k.lower() for k in cfg.cls_name_keywords),
        )
        optimizer = torch.optim.AdamW(groups)
        print(f"optimizer groups: other={n_other} params, cls/fusion={n_cls} params, cls_lr_mult={cfg.cls_lr_mult}")

        # scheduler
        if cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
        elif cfg.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        else:
            scheduler = None

        scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        fold_dir = os.path.join(cfg.save_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        epoch_metrics: List[Dict[str, float]] = []
        best_score = -1e9
        best_epoch = -1
        best_path = os.path.join(cfg.save_dir, f"best_fold{fold}.pth")
        patience = 0

        loss_fn = make_nomissing_loss_fn(
            cls_criterion=cls_criterion,
            seg_bce_weight=cfg.seg_bce_weight,
            force_full_observed_mask=cfg.force_full_observed_mask,
            use_uncertainty_weighting=cfg.use_uncertainty_weighting,
            lambda_anchor=cfg.lambda_anchor,
        )

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            if cfg.freeze_bn:
                freeze_bn_running_stats(model)

            lr_group0 = optimizer.param_groups[0]["lr"]
            lr_group1 = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr_group0

            train_stats = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler_amp=scaler_amp,
                loss_fn=loss_fn,
                max_grad_norm=cfg.max_grad_norm,
                grad_accum_steps=max(1, cfg.grad_accum_steps),
            )

            val_stats = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                num_classes=cfg.num_classes,
                seg_thr=cfg.seg_thr,
                cls_criterion=cls_criterion,
                seg_bce_weight=cfg.seg_bce_weight,
                force_full_observed_mask=cfg.force_full_observed_mask,
                return_preds=False,
            )

            score = composite_score(val_stats, w_seg=cfg.score_w_seg, w_cls=cfg.score_w_cls)

            if scheduler is not None:
                scheduler.step()

            dt = time.time() - t0
            print(
                f"Epoch {epoch:03d} | lr(backbone) {lr_group0:.2e} lr(cls/fusion) {lr_group1:.2e} | "
                f"train: total {train_stats['total_loss']:.4f} seg {train_stats['seg_loss']:.4f} cls {train_stats['cls_loss']:.4f} | "
                f"val: dice {val_stats['dice']:.4f} miou {val_stats['miou']:.4f} "
                f"segP {val_stats['seg_precision']:.4f} segR {val_stats['seg_recall']:.4f} | "
                f"acc {val_stats['acc']:.4f} f1 {val_stats['macro_f1']:.4f} auc {val_stats['auc_macro_ovr']:.4f} | "
                f"sens {val_stats['sens_macro']:.4f} spec {val_stats['spec_macro']:.4f} | "
                f"score {score:.4f} | time {dt:.1f}s"
            )

            row = {"fold": fold, "epoch": epoch, "lr_backbone": lr_group0, "lr_cls": lr_group1, "score": score}
            row.update(train_stats)
            row.update(val_stats)
            epoch_metrics.append(row)

            if score > best_score:
                best_score = score
                best_epoch = epoch
                patience = 0

                torch.save(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_score": best_score,
                        "cfg": cfg.__dict__,
                        "cat_maps": cat_maps,
                        "num_scaler": num_scaler,
                        "feature_names": getattr(fold_ds, "feature_names", None),
                        "numeric_slice": getattr(fold_ds, "numeric_slice", None),
                        "onehot_slices": getattr(fold_ds, "onehot_slices", None),
                    },
                    best_path,
                )
                print(f"✅ Save best to {best_path} (score={best_score:.4f})")

                best_eval = evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    num_classes=cfg.num_classes,
                    seg_thr=cfg.seg_thr,
                    cls_criterion=cls_criterion,
                    seg_bce_weight=cfg.seg_bce_weight,
                    force_full_observed_mask=cfg.force_full_observed_mask,
                    return_preds=True,
                )
                oof_true_all.append(best_eval["_y_true"])
                oof_proba_all.append(best_eval["_y_proba"])
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    print(f"🛑 Early stopping: patience={cfg.early_stop_patience}")
                    break

        if cfg.save_fold_metrics_csv:
            df_fold = pd.DataFrame(epoch_metrics)
            csv_path = os.path.join(fold_dir, f"fold{fold}_metrics.csv")
            df_fold.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"📄 Saved fold metrics csv -> {csv_path}")

        df_fold = pd.DataFrame(epoch_metrics)
        best_row = df_fold.loc[df_fold["score"].idxmax()].to_dict()
        best_row["best_epoch"] = best_epoch
        best_rows.append(best_row)
        print(f"Fold {fold} done. Best score={best_score:.4f} at epoch {best_epoch}")

    # ----- final OOF ROC -----
    if cfg.save_final_oof_roc and len(oof_true_all) > 0:
        y_true_oof = np.concatenate(oof_true_all, axis=0)
        y_proba_oof = np.concatenate(oof_proba_all, axis=0)

        final_roc_path = os.path.join(cfg.save_dir, cfg.final_roc_name)
        plot_multiclass_roc(
            y_true=y_true_oof,
            y_proba=y_proba_oof,
            num_classes=cfg.num_classes,
            save_path=final_roc_path,
            title="ROC (5-fold Cross-Validation)"
        )
        print(f"📈 Saved ONE final OOF ROC -> {final_roc_path}")

        if cfg.save_oof_npz:
            npz_path = os.path.join(cfg.save_dir, "oof_predictions.npz")
            np.savez(npz_path, y_true=y_true_oof, y_proba=y_proba_oof)
            print(f"💾 Saved OOF predictions -> {npz_path}")

    # ----- summary table -----
    df_best = pd.DataFrame(best_rows)

    paper_cols = [
        "dice", "miou", "seg_precision", "seg_recall",
        "acc", "macro_f1", "auc_macro_ovr",
        "sens_macro", "spec_macro", "score"
    ]
    for c in range(cfg.num_classes):
        paper_cols += [f"sens_c{c}", f"spec_c{c}"]
    paper_cols = [c for c in paper_cols if c in df_best.columns]

    mean_vals = df_best[paper_cols].mean(numeric_only=True)
    std_vals = df_best[paper_cols].std(numeric_only=True, ddof=1)

    summary = []
    for k in paper_cols:
        mval = mean_vals[k]
        sval = std_vals[k]
        if np.isnan(mval):
            summary.append({"metric": k, "mean": np.nan, "std": np.nan, "mean±std": "NaN"})
        else:
            summary.append({"metric": k, "mean": float(mval), "std": float(sval), "mean±std": f"{mval:.4f} ± {sval:.4f}"})

    df_summary = pd.DataFrame(summary)
    summary_path = os.path.join(cfg.save_dir, cfg.summary_table_name)
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n📌 Saved paper summary table (mean±std) -> {summary_path}")

    df_best_path = os.path.join(cfg.save_dir, "best_per_fold_metrics.csv")
    df_best.to_csv(df_best_path, index=False, encoding="utf-8-sig")
    print(f"📌 Saved best-per-fold metrics -> {df_best_path}")

    print("\n✅ All folds finished.")


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)
