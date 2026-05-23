# -*- coding: utf-8 -*-
# D:\pythonpro\SG-MTF-main\models\sgmtf.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import SGMTFPerformanceDecoder, EnhancedClassifier
from .roi.multi_region_pooling import MultiRegionROIPooling
from .backbones.medsam_encoder import MedSAMMultiScaleEncoder
from .clinical_graph import ClinicalVariableGraphEncoder
from .morph_clin_hetero_graph import MorphClinicalHeteroGraph
from .boundary import ReliabilityAnchoredBoundaryGraphRefinement


class SGMTFModel(nn.Module):
    """
    SG-MTF / BUA-LEL with diagnosis-oriented reliability-anchored
    boundary graph refinement.

    Main pipeline:
      1) MedSAM encoder extracts segmentation features and multi-level cls_feat.
      2) Segmentation decoder predicts a coarse lesion probability map.
      3) ReliabilityAnchoredBoundaryGraphRefinement refines the low-resolution
         lesion map and produces boundary geometry / uncertainty evidence tokens.
      4) MultiRegionROIPooling decomposes the refined lesion prior into:
            core / boundary / peritumor morphology nodes.
      5) ClinicalVariableGraphEncoder encodes clinical variables into clinical nodes.
      6) MorphClinicalHeteroGraph performs structured interaction among:
            core, boundary, peritumor, z_geo, z_unc, and clinical nodes.
      7) EnhancedClassifier predicts molecular subtype.

    Forward interface is kept:
        seg_logits, cls_logits, aux = model(x_img, c_obs, m, task)
    """

    def __init__(
        self,
        clinical_dim: int,
        numeric_slice=None,
        onehot_slices_dict=None,
        num_classes: int = 3,
        use_pca: bool = False,
        pca_dim: int = 100,
        clin_embed_dim: int = 128,
        cls_hidden_dims=(512, 256),
        cls_dropout: float = 0.3,
        detach_roi_in_cls: bool = True,
        detach_segfeat_in_cls: bool = False,
        lambda_cons: float = 0.0,
        drop_path_rate: float = 0.1,
        se_ratio: float = 0.25,
        gn_groups: int = 32,

        # MedSAM options
        medsam_checkpoint_path=None,
        medsam_model_type: str = "vit_b",
        freeze_medsam: bool = True,
        unfreeze_medsam_last_n: int = 0,
        image_size: int = 1024,

        # Clinical graph options
        clinical_graph_dim: int = 128,
        clinical_graph_hidden_dim: int = 128,
        clinical_graph_dropout: float = 0.1,
        clinical_graph_layers: int = 3,
        clinical_graph_pooling: str = "mean",
        use_causal_graph: bool = True,
        add_reverse_edges: bool = True,
        add_self_loops: bool = True,

        # Morphology-clinical heterogeneous graph options
        hetero_hidden_dim: int = 256,
        hetero_out_dim: int = 256,
        hetero_layers: int = 2,

        # Boundary graph refinement options
        use_boundary_refiner: bool = True,
        boundary_num_nodes: int = 64,
        boundary_hidden_dim: int = 128,
        boundary_gnn_layers: int = 3,
        boundary_max_offset: float = 0.08,
        lambda_anchor: float = 0.01,
        boundary_token_gate_init: float = -2.0,
    ):
        super().__init__()

        self.clinical_dim = int(clinical_dim)
        self.num_classes = int(num_classes)
        self.detach_roi_in_cls = bool(detach_roi_in_cls)
        self.detach_segfeat_in_cls = bool(detach_segfeat_in_cls)
        # Kept only for compatibility with old training scripts.
        # Consistency loss is disabled and not used in total loss.
        self.lambda_cons = 0.0
        self.use_boundary_refiner = bool(use_boundary_refiner)
        self.lambda_anchor = float(lambda_anchor)

        # Boundary evidence token gate.
        # sigmoid(-2.0) ≈ 0.119, so z_geo / z_unc are introduced
        # as weak residual evidence at early training instead of
        # dominating the morphology-clinical heterogeneous graph.
        self.boundary_token_gate = nn.Parameter(
            torch.tensor(float(boundary_token_gate_init), dtype=torch.float32)
        )

        # Kept for checkpoint compatibility / debugging.
        self.numeric_slice = numeric_slice
        self.onehot_slices_dict = onehot_slices_dict
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        self.clin_embed_dim = clin_embed_dim
        self.drop_path_rate = drop_path_rate
        self.se_ratio = se_ratio

        # ============================================================
        # 1) MedSAM encoder
        #    return_dict=True gives:
        #      - seg_feats: f1, f2, f3, f4
        #      - cls_feat: fused block_5/block_8/block_11 feature [B,256,64,64]
        # ============================================================
        self.encoder = MedSAMMultiScaleEncoder(
            checkpoint_path=medsam_checkpoint_path,
            model_type=medsam_model_type,
            freeze_medsam=freeze_medsam,
            unfreeze_last_n=unfreeze_medsam_last_n,
            out_channels=(120, 240, 480, 960),
            gn_groups=gn_groups,
            force_img_size=image_size,
            out_indices=(4, 7, 10),
            return_cls_feat=True,
            cls_out_dim=256,
        )

        # ============================================================
        # 2) Segmentation decoder and coarse segmentation head
        # ============================================================
        self.decoder = SGMTFPerformanceDecoder(gn_groups=gn_groups)
        self.seg_head = nn.Conv2d(120, 1, kernel_size=1, bias=True)

        # ============================================================
        # 3) Diagnosis-oriented reliability-anchored boundary graph refinement
        #    Input:  feat_seg [B,120,H/4,W/4], coarse logits [B,1,H/4,W/4],
        #            cls_feat [B,256,64,64]
        #    Output: refined logits/prob + z_geo + z_unc + anchor_loss
        # ============================================================
        if self.use_boundary_refiner:
            self.boundary_refiner = ReliabilityAnchoredBoundaryGraphRefinement(
                seg_channels=120,
                cls_channels=256,
                token_dim=256,
                num_nodes=boundary_num_nodes,
                hidden_dim=boundary_hidden_dim,
                num_gnn_layers=boundary_gnn_layers,
                max_offset=boundary_max_offset,
                dropout=cls_dropout,
            )
        else:
            self.boundary_refiner = None

        # ============================================================
        # 4) Multi-region morphology evidence extraction
        #    cls_feat [B,256,64,64] + refined roi_prob ->:
        #      v_img:       [B,256]
        #      region_feats:[B,3,256] = core / boundary / peritumor
        # ============================================================
        self.roi_pool = MultiRegionROIPooling(
            in_channels=256,
            boundary_kernel=7,
            peritumor_kernel=17,
            use_region_attention=True,
            dropout=cls_dropout,
            binarize_roi=False,
            detach_roi=detach_roi_in_cls,
            less_overlap_peritumor=True,
            use_adaptive_kernel=True,
            boundary_radius_ratio=0.15,
            peritumor_radius_ratio=0.40,
            min_boundary_radius=2,
            max_boundary_radius=4,
            min_peritumor_radius=4,
            max_peritumor_radius=8,
            use_small_lesion_gate=True,
            small_lesion_area_threshold=500.0,
            small_lesion_gate_temperature=150.0,
            hard_disable_peritumor_for_small=False,
            suppress_peritumor_attention=True,
        )

        # ============================================================
        # 5) Clinical graph encoder
        #    clinical_nodes:  [B,12,clinical_graph_dim]
        #    clinical_global: [B,clinical_graph_dim]
        # ============================================================
        self.clin_graph = ClinicalVariableGraphEncoder(
            clinical_dim=self.clinical_dim,
            numeric_slice=numeric_slice,
            onehot_slices_dict=onehot_slices_dict,
            graph_dim=clinical_graph_dim,
            hidden_dim=clinical_graph_hidden_dim,
            dropout=clinical_graph_dropout,
            use_causal_graph=use_causal_graph,
            add_reverse_edges=add_reverse_edges,
            add_self_loops=add_self_loops,
            causal_graph_layers=clinical_graph_layers,
            pooling=clinical_graph_pooling,
        )

        # ============================================================
        # 6) Morphology-clinical heterogeneous graph fusion
        #    If boundary_refiner is enabled:
        #       5 morphology nodes = core / boundary / peritumor / z_geo / z_unc
        #    Otherwise:
        #       3 morphology nodes = core / boundary / peritumor
        # ============================================================
        self.num_morph_nodes = 5 if self.use_boundary_refiner else 3

        self.hetero_graph = MorphClinicalHeteroGraph(
            morph_dim=256,
            clinical_dim=clinical_graph_dim,
            hidden_dim=hetero_hidden_dim,
            out_dim=hetero_out_dim,
            num_layers=hetero_layers,
            dropout=cls_dropout,
            use_residual_global=True,
            num_morph_nodes=self.num_morph_nodes,
        )

        # ============================================================
        # 7) Classifier
        # ============================================================
        self.cls_head = EnhancedClassifier(
            feature_dim=hetero_out_dim,
            output_size=self.num_classes,
            hidden_dims=cls_hidden_dims,
            dropout_rate=cls_dropout,
        )

        # ============================================================
        # 8) Uncertainty weights
        #    log_var_imp is kept only for compatibility.
        # ============================================================
        self.log_var_seg = nn.Parameter(torch.tensor(0.0))
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))
        self.log_var_imp = nn.Parameter(torch.tensor(0.0))

        # Important: do not re-initialize pretrained MedSAM image_encoder.
        self._init_task_weights()

    def forward(self, x_img, c_obs=None, m=None, task="both"):
        """
        Args:
            x_img: [B,C,H,W]
            c_obs: [B,clinical_dim], required for cls/both
            m: kept for compatibility
            task: "seg", "cls", or "both"

        Returns:
            seg_logits: [B,1,H,W] or None
            cls_logits: [B,num_classes] or None
            aux: dict
        """
        if x_img.dim() != 4:
            raise ValueError(f"x_img must be [B,C,H,W], got {tuple(x_img.shape)}")

        B, _, H, W = x_img.shape

        need_seg = task in ["seg", "both"]
        need_cls = task in ["cls", "both"]

        # ============================================================
        # MedSAM encoder
        # ============================================================
        enc_out = self.encoder(x_img, return_dict=True)

        f1, f2, f3, f4 = enc_out["seg_feats"]
        cls_feat = enc_out["cls_feat"]  # [B,256,64,64]

        # ============================================================
        # Segmentation branch
        # ============================================================
        feat_seg = self.decoder(f1, f2, f3, f4)            # [B,120,H/4,W/4]

        seg_logits_low_raw = self.seg_head(feat_seg)       # coarse logits [B,1,H/4,W/4]

        if self.use_boundary_refiner:
            boundary_out = self.boundary_refiner(
                f_seg=feat_seg,
                seg_logits=seg_logits_low_raw,
                f_cls=cls_feat,
            )
            seg_logits_low = boundary_out["refined_logits"]
            seg_prob_low = boundary_out["refined_prob"]
        else:
            boundary_out = None
            seg_logits_low = seg_logits_low_raw
            seg_prob_low = torch.sigmoid(seg_logits_low)

        roi_prob = seg_prob_low

        seg_logits = None
        if need_seg:
            seg_logits = F.interpolate(
                seg_logits_low,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        cls_logits = None

        aux = {
            "roi_prob": roi_prob,
            "seg_logits_low": seg_logits_low,
            "seg_logits_low_raw": seg_logits_low_raw,
            "seg_prob_low": seg_prob_low,
            "cls_feat": cls_feat,
            "feat_seg": feat_seg,
        }

        if boundary_out is not None:
            aux.update({
                "boundary_out": boundary_out,
                "boundary_anchor_loss": boundary_out["anchor_loss"],
                "z_geo": boundary_out["z_geo"],
                "z_unc": boundary_out["z_unc"],
                "boundary_uncertainty": boundary_out["boundary_uncertainty"],
                "boundary_score": boundary_out["boundary_score"],
            })
        else:
            zero = seg_logits_low.sum() * 0.0
            aux.update({
                "boundary_out": None,
                "boundary_anchor_loss": zero,
                "z_geo": None,
                "z_unc": None,
                "boundary_uncertainty": None,
                "boundary_score": None,
            })

        # ============================================================
        # Classification branch:
        #   MedSAM cls_feat
        #   -> core/boundary/peritumor morphology nodes
        #   -> boundary geometry / uncertainty evidence nodes
        #   -> clinical variable graph nodes
        #   -> morphology-clinical heterogeneous graph
        #   -> classifier
        # ============================================================
        if need_cls:
            if c_obs is None:
                raise ValueError("Classification requires c_obs.")

            if c_obs.dim() != 2 or c_obs.shape[0] != B or c_obs.shape[1] != self.clinical_dim:
                raise ValueError(
                    f"c_obs must be [B,{self.clinical_dim}] and match batch size, "
                    f"but got {tuple(c_obs.shape)}."
                )

            # m is no longer used for imputation, but keep a light check.
            if m is not None:
                if m.shape[0] != B or m.shape[1] != self.clinical_dim:
                    raise ValueError(
                        f"m must be [B,{self.clinical_dim}] when provided, "
                        f"but got {tuple(m.shape)}."
                    )

            roi_for_cls = roi_prob.detach() if self.detach_roi_in_cls else roi_prob
            feat_for_cls = cls_feat.detach() if self.detach_segfeat_in_cls else cls_feat

            # --------------------------------------------------------
            # 1) Morphology evidence nodes from refined lesion prior
            # --------------------------------------------------------
            v_img, roi_info = self.roi_pool(
                feat_for_cls,
                roi_for_cls,
            )  # v_img: [B,256]

            morph_nodes_3 = roi_info["region_feats"]  # [B,3,256]

            if self.use_boundary_refiner:
                z_geo = boundary_out["z_geo"]
                z_unc = boundary_out["z_unc"]

                # If the user wants to detach image-side features for classification,
                # also detach boundary evidence tokens.
                if self.detach_segfeat_in_cls:
                    z_geo = z_geo.detach()
                    z_unc = z_unc.detach()

                # --------------------------------------------------------
                # Boundary evidence token gate
                # --------------------------------------------------------
                # z_geo / z_unc are useful diagnostic evidence, but they are
                # randomly initialized and may be noisy at early training.
                # A learnable scalar gate introduces them as weak residual
                # evidence first, then lets the model increase their influence
                # if they help subtype classification.
                boundary_gate = torch.sigmoid(self.boundary_token_gate)
                z_geo_gated = boundary_gate * z_geo
                z_unc_gated = boundary_gate * z_unc

                boundary_evidence_nodes = torch.stack(
                    [z_geo_gated, z_unc_gated],
                    dim=1,
                )  # [B,2,256]

                morph_nodes = torch.cat(
                    [morph_nodes_3, boundary_evidence_nodes],
                    dim=1,
                )  # [B,5,256]
            else:
                z_geo = None
                z_unc = None
                z_geo_gated = None
                z_unc_gated = None
                boundary_gate = None
                boundary_evidence_nodes = None
                morph_nodes = morph_nodes_3  # [B,3,256]

            # --------------------------------------------------------
            # 2) Clinical variable graph nodes
            # --------------------------------------------------------
            graph_out = self.clin_graph(c_obs.float(), m)

            clinical_nodes = graph_out["clinical_nodes"]      # [B,12,clinical_graph_dim]
            v_clin = graph_out["clinical_global"]             # [B,clinical_graph_dim]

            # --------------------------------------------------------
            # 3) Morphology-clinical heterogeneous graph fusion
            # --------------------------------------------------------
            hetero_out = self.hetero_graph(
                morph_nodes=morph_nodes,
                clinical_nodes=clinical_nodes,
                morph_global=v_img,
                clinical_global=v_clin,
            )

            vfused = hetero_out["hetero_global"]              # [B,hetero_out_dim]
            cls_logits = self.cls_head(vfused)

            aux.update({
                # Image/morphology branch
                "v_img": v_img,
                "morph_nodes": morph_nodes,
                "morph_nodes_3": morph_nodes_3,
                "boundary_evidence_nodes": boundary_evidence_nodes,
                "roi_info": roi_info,
                "region_feats": morph_nodes_3,
                "region_feats_with_boundary_evidence": morph_nodes,
                "region_weights": roi_info.get("region_weights", None),
                "z_geo_cls": z_geo,
                "z_unc_cls": z_unc,
                "z_geo_gated": z_geo_gated,
                "z_unc_gated": z_unc_gated,
                "boundary_token_gate": (
                    torch.sigmoid(self.boundary_token_gate).detach()
                    if self.use_boundary_refiner else None
                ),

                # Clinical branch
                "v_clin": v_clin,
                "v_clin_star": v_clin,       # compatibility with old logging code
                "clinical_nodes": clinical_nodes,

                # Heterogeneous graph fusion
                "vfused": vfused,
                "hetero_global": vfused,
                "hetero_nodes": hetero_out["hetero_nodes"],
                "hetero_node_attn": hetero_out["hetero_node_attn"],
                "hetero_cross_attn": hetero_out["hetero_cross_attn"],
                "hetero_morph_node_attn": hetero_out.get("hetero_morph_node_attn", None),
                "hetero_clinical_node_attn": hetero_out.get("hetero_clinical_node_attn", None),

                **graph_out,
            })

        if task == "seg":
            return seg_logits, None, aux

        if task == "cls":
            return None, cls_logits, aux

        return seg_logits, cls_logits, aux

    def forward_cls_with_region_feats(self, x_img, c_obs, m, region_feats_override):
        """
        前向分类分支，用于局部扰动分析，允许替换 region_feats
        """
        B = x_img.shape[0]
        enc_out = self.encoder(x_img, return_dict=True)
        cls_feat = enc_out["cls_feat"]

        roi_prob = torch.zeros((B, 1, cls_feat.shape[2], cls_feat.shape[3]), device=x_img.device)
        v_img, roi_info = self.roi_pool(cls_feat, roi_prob)
        morph_nodes_3 = region_feats_override  # 使用传入的扰动特征

        if self.use_boundary_refiner:
            boundary_out = self.boundary_refiner(f_seg=cls_feat, seg_logits=roi_prob, f_cls=cls_feat)
            z_geo_gated = torch.sigmoid(self.boundary_token_gate) * boundary_out["z_geo"]
            z_unc_gated = torch.sigmoid(self.boundary_token_gate) * boundary_out["z_unc"]
            morph_nodes = torch.cat([morph_nodes_3, torch.stack([z_geo_gated, z_unc_gated], dim=1)], dim=1)
        else:
            morph_nodes = morph_nodes_3

        graph_out = self.clin_graph(c_obs.float(), m)
        clinical_nodes = graph_out["clinical_nodes"]
        v_clin = graph_out["clinical_global"]

        hetero_out = self.hetero_graph(
            morph_nodes=morph_nodes,
            clinical_nodes=clinical_nodes,
            morph_global=v_img,
            clinical_global=v_clin
        )

        vfused = hetero_out["hetero_global"]
        cls_logits = self.cls_head(vfused)
        return cls_logits

    # -------------------------- multi-task loss without consistency interference --------------------------
    def get_total_loss(
        self,
        Lseg,
        Lcls,
        Limp=0.0,
        Lcons=0.0,
        Lanchor=0.0,
        lambda_anchor=None,
        clamp=(-5.0, 5.0),
    ):
        """
        Multi-task loss without consistency interference loss.

        Final objective:
            total_loss = segmentation loss + classification loss + boundary anchor loss

        Notes:
            - Lcons is kept only for compatibility with old training code.
            - Limp is kept only for compatibility and should normally be 0.
            - Consistency loss is completely ignored.
        """
        lo, hi = clamp

        lv_seg = self.log_var_seg.clamp(lo, hi)
        lv_cls = self.log_var_cls.clamp(lo, hi)

        if not torch.is_tensor(Limp):
            ref = Lcls if torch.is_tensor(Lcls) else Lseg
            Limp = torch.zeros((), device=ref.device, dtype=ref.dtype)

        if not torch.is_tensor(Lanchor):
            ref = Lcls if torch.is_tensor(Lcls) else Lseg
            Lanchor = torch.zeros((), device=ref.device, dtype=ref.dtype)

        if lambda_anchor is None:
            lambda_anchor = self.lambda_anchor

        loss = (
            0.5 * torch.exp(-lv_seg) * Lseg + 0.5 * lv_seg
            + 0.5 * torch.exp(-lv_cls) * Lcls + 0.5 * lv_cls
            + float(lambda_anchor) * Lanchor
        )

        weights = {
            "w_seg": float((0.5 * torch.exp(-lv_seg)).detach().cpu()),
            "w_cls": float((0.5 * torch.exp(-lv_cls)).detach().cpu()),
            "w_imp": 0.0,
            "lambda_cons": 0.0,
            "lambda_anchor": float(lambda_anchor),
            "Lanchor": float(Lanchor.detach().cpu()),
        }

        return loss, weights

    def _init_task_weights(self):
        """
        Initialize only task-specific modules.

        Skip:
            encoder.image_encoder

        because it is the pretrained MedSAM ViT image encoder loaded from
        medsam_checkpoint_path.
        """
        skip_prefixes = (
            "encoder.image_encoder",
        )

        for name, mm in self.named_modules():
            if any(name == p or name.startswith(p + ".") for p in skip_prefixes):
                continue

            if isinstance(mm, nn.Conv2d):
                nn.init.kaiming_normal_(mm.weight, mode="fan_out", nonlinearity="relu")
                if mm.bias is not None:
                    nn.init.zeros_(mm.bias)

            elif isinstance(mm, nn.Linear):
                nn.init.xavier_uniform_(mm.weight)
                if mm.bias is not None:
                    nn.init.zeros_(mm.bias)

            elif isinstance(mm, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
                if hasattr(mm, "weight") and mm.weight is not None:
                    nn.init.constant_(mm.weight, 1.0)
                if hasattr(mm, "bias") and mm.bias is not None:
                    nn.init.constant_(mm.bias, 0.0)

    # Backward-compatible alias
    def _init_basic_weights(self):
        self._init_task_weights()
