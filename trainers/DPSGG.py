# trainers/dpsgg.py
"""
Dual-Prompt SGG (DPSGG) trainer for Visual Genome-style datasets.

MVP supports PredCls/SGCls with CLIP ViT backbone + VPT-Deep (image-side prompts)
+ relation PromptLearner (text-side prompts). Geometry cues and union-ROI are
concatenated to subject/object features. A lightweight CAVPT-style top-K class
selector prunes predicate classes per image.

This file is designed to drop into the existing Dassl.pytorch trainers folder.
It follows patterns from CoOp/CoCoOp/DPT in this repo.

Assumptions about the dataset batch (modify parse_batch_* if you use different keys):
- image: Tensor[B, 3, H, W]
- boxes: list[Tensor[N_i, 4]] in xyxy, 0-1 normalized or absolute pixels (either is ok — we autodetect)
- obj_labels: list[Tensor[N_i]] integers in [0, num_obj_classes)
- pair_idx: list[Tensor[M_i, 2]] indices into boxes for subject/object per pair
- rel_labels: list[Tensor[M_i]] integers in [0, num_pred_classes) (for single-label training)

If you prefer multi-label (e.g., PU/BCE), set cfg.TRAINER.DPSGG.MULTILABEL=True and
return rel_multi_hot: list[Tensor[M_i, num_pred_classes]].
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from trainers.coop import load_clip_to_cpu, TextEncoder  # 
# ^ load_clip_to_cpu/TextEncoder are available in your baseline (see coop.py)
# They give us the CLIP model and a thin text encoder wrapper.

# ------------------------------
# Utilities
# ------------------------------

def _is_normed_xyxy(boxes: torch.Tensor) -> bool:
    """Heuristic: if max coord <= 2.0 it's probably normalized (0~1 or -1~1).
    """
    return float(boxes.max()) <= 2.0


def _boxes_to_int_xyxy(boxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert possibly-normalized xyxy to integer pixel xyxy in [0, W/H]."""
    if _is_normed_xyxy(boxes):
        b = boxes.clone()
        b[:, [0, 2]] = (b[:, [0, 2]] * W).clamp(0, W - 1)
        b[:, [1, 3]] = (b[:, [1, 3]] * H).clamp(0, H - 1)
        return b.round()
    return boxes.clone()


def _crop_and_resize(images: torch.Tensor, boxes: torch.Tensor, out_size: int) -> torch.Tensor:
    """Vectorized crop-resize using grid_sample.
    images: [B,3,H,W], boxes: [N,4] absolute xyxy in pixels but belonging to a single image.
    Returns: [N,3,out_size,out_size].
    """
    B, C, H, W = images.shape
    device = images.device
    # Assume boxes belong to image 0 (we call per image). Build normalized grids.
    x1, y1, x2, y2 = [boxes[:, i] for i in range(4)]
    # Prevent degenerate boxes
    x2 = torch.clamp(x2, min=(x1 + 1))
    y2 = torch.clamp(y2, min=(y1 + 1))
    # Create sampling grid per ROI
    N = boxes.shape[0]
    # grid in [-1,1]
    xs = torch.linspace(-1, 1, out_size, device=device)
    ys = torch.linspace(-1, 1, out_size, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [S,S]
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [S,S,2]
    grid = grid.view(1, out_size, out_size, 2).repeat(N, 1, 1, 1)  # [N,S,S,2]
    # Map grid to image coords
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    hw = (x2 - x1) / 2.0
    hh = (y2 - y1) / 2.0
    grid[..., 0] = grid[..., 0] * (hw / (W / 2.0)).unsqueeze(-1).unsqueeze(-1) + (cx / (W / 2.0)).unsqueeze(-1).unsqueeze(-1)
    grid[..., 1] = grid[..., 1] * (hh / (H / 2.0)).unsqueeze(-1).unsqueeze(-1) + (cy / (H / 2.0)).unsqueeze(-1).unsqueeze(-1)
    # grid_sample needs NCHW input — tile image N times
    imgs = images[0:1].expand(N, -1, -1, -1)
    crops = F.grid_sample(imgs, grid, mode="bilinear", align_corners=True)
    return crops


class PairSpatialEncoder(nn.Module):
    """Encodes geometry for a subject-object pair (Δ, IoU, area ratios, etc.)."""
    def __init__(self, dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, 128), nn.ReLU(inplace=True),
            nn.Linear(128, dim), nn.ReLU(inplace=True)
        )

    def forward(self, sb: torch.Tensor, ob: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # boxes in absolute xyxy
        sx1, sy1, sx2, sy2 = sb.unbind(-1)
        ox1, oy1, ox2, oy2 = ob.unbind(-1)
        sw = (sx2 - sx1).clamp(min=1); sh = (sy2 - sy1).clamp(min=1)
        ow = (ox2 - ox1).clamp(min=1); oh = (oy2 - oy1).clamp(min=1)
        scx = sx1 + sw / 2; scy = sy1 + sh / 2
        ocx = ox1 + ow / 2; ocy = oy1 + oh / 2
        # deltas normalized by image size
        dx = (ocx - scx) / W; dy = (ocy - scy) / H
        dw = torch.log(ow / sw); dh = torch.log(oh / sh)
        # IoU
        ix1 = torch.maximum(sx1, ox1); iy1 = torch.maximum(sy1, oy1)
        ix2 = torch.minimum(sx2, ox2); iy2 = torch.minimum(sy2, oy2)
        iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
        inter = iw * ih
        union = sw * sh + ow * oh - inter
        iou = inter / union.clamp(min=1)
        feat = torch.stack([dx, dy, dw, dh, sw/W, sh/H, ow/W, oh/H, iou], dim=-1)
        return self.mlp(feat)


class RelPromptLearner(nn.Module):
    """Prompt learner for predicate vocabulary (class names = predicate strings).
    Follows CoOp-style context tokens around the class token. Role markers are
    modeled as learnable tokens <SUBJ>, <OBJ> that are inserted as additional
    context tokens (not literal words), preserving CLIP tokenization.
    """
    def __init__(self, cfg, pred_names: List[str], clip_model):
        super().__init__()
        n_cls = len(pred_names)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        n_ctx = cfg.TRAINER.DPSGG.N_CTX_PRED
        ctx_init = cfg.TRAINER.DPSGG.CTX_INIT_PRED
        class_token_position = cfg.TRAINER.DPSGG.CLASS_TOKEN_POSITION_PRED

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1+n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        # role tokens
        self.subj_tok = nn.Parameter(torch.empty(1, ctx_dim, dtype=dtype))
        self.obj_tok  = nn.Parameter(torch.empty(1, ctx_dim, dtype=dtype))
        nn.init.normal_(self.subj_tok, std=0.02)
        nn.init.normal_(self.obj_tok,  std=0.02)

        pred_names = [p.replace("_", " ") for p in pred_names]
        name_lens = [len(clip.simple_tokenizer.SimpleTokenizer().encode(p)) for p in pred_names]
        # template: "<SOS> [CTX...] <SUBJ> [CLASS] <OBJ> . <EOS>"
        prompts = [f"{prompt_prefix} {p}." for p in pred_names]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = class_token_position
        self.ctx = nn.Parameter(ctx_vectors)

    def _assemble(self, prefix, suffix, ctx, subj_tok, obj_tok):
        if self.class_token_position == "end":
            # <SOS>, CTX, <SUBJ>, <OBJ>, CLASS, rest
            prompts = torch.cat([prefix, ctx, subj_tok, obj_tok, suffix], dim=1)
        elif self.class_token_position == "front":
            # <SOS>, CLASS, CTX, <SUBJ>, <OBJ>, rest
            class_i = suffix[:, :1, :]
            rest    = suffix[:, 1:, :]
            prompts = torch.cat([prefix, class_i, ctx, subj_tok, obj_tok, rest], dim=1)
        else:
            # default: middle => split ctx
            half = self.n_ctx // 2
            class_i = suffix[:, :1, :]
            rest    = suffix[:, 1:, :]
            prompts = torch.cat([prefix, ctx[:, :half], class_i, ctx[:, half:], subj_tok, obj_tok, rest], dim=1)
        return prompts

    def forward(self, subj_ctx: Optional[torch.Tensor] = None, obj_ctx: Optional[torch.Tensor] = None):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        subj_tok = self.subj_tok.expand(self.n_cls, -1, -1) if subj_ctx is None else subj_ctx
        obj_tok  = self.obj_tok.expand(self.n_cls, -1, -1)  if obj_ctx  is None else obj_ctx
        prompts = self._assemble(prefix, suffix, ctx, subj_tok, obj_tok)
        return prompts


class SimpleTopKSelector(nn.Module):
    """CAVPT-like top-K predicate selector using a frozen zeroshot branch.
    Given a global CLIP image feature, rank predicate text features and return
    indices of top-K candidates.
    """
    def __init__(self, K: int):
        super().__init__()
        self.K = K

    @torch.no_grad()
    def forward(self, img_feat: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        # img_feat: [B, C] (normalized), text_feats: [P, C] (normalized)
        logits = img_feat @ text_feats.t()  # [B,P]
        topk = logits.topk(min(self.K, text_feats.size(0)), dim=1).indices  # [B,K]
        return topk


class DPSGGModel(nn.Module):
    def __init__(self, cfg, pred_names: List[str], clip_model, crop_size: int = 224):
        super().__init__()
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual  # frozen except VPT if enabled externally
        self.text_encoder  = TextEncoder(clip_model)
        self.logit_scale   = clip_model.logit_scale
        self.prompt_learner = RelPromptLearner(cfg, pred_names, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.spatial = PairSpatialEncoder(dim=128)
        self.selector = SimpleTopKSelector(cfg.TRAINER.DPSGG.TOPK)
        self.crop_size = crop_size
        # fusion
        vis_dim = clip_model.visual.output_dim
        self.fuse = nn.Sequential(
            nn.Linear(vis_dim * 3 + 128, vis_dim), nn.ReLU(inplace=True),
            nn.Linear(vis_dim, vis_dim)
        )

    def _pair_visual(self, image: torch.Tensor, boxes: torch.Tensor, pairs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (h_s, h_o, h_u, geom) for all pairs.
        image: [1,3,H,W] (we call per-image), boxes: [N,4] int xyxy, pairs: [M,2]
        """
        device = image.device
        H, W = image.shape[-2:]
        s_idx, o_idx = pairs[:,0], pairs[:,1]
        sb = boxes[s_idx]; ob = boxes[o_idx]
        # crops
        s_crops = _crop_and_resize(image, sb, self.crop_size)
        o_crops = _crop_and_resize(image, ob, self.crop_size)
        # union box
        u = torch.stack([
            torch.minimum(sb[:,0], ob[:,0]),
            torch.minimum(sb[:,1], ob[:,1]),
            torch.maximum(sb[:,2], ob[:,2]),
            torch.maximum(sb[:,3], ob[:,3])
        ], dim=1)
        u_crops = _crop_and_resize(image, u, self.crop_size)

        # encode via CLIP image tower
        with torch.no_grad():
            s_feat = self.image_encoder(s_crops.type(self.dtype))  # [M,C]
            o_feat = self.image_encoder(o_crops.type(self.dtype))
            u_feat = self.image_encoder(u_crops.type(self.dtype))
        # normalize
        s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)
        o_feat = o_feat / o_feat.norm(dim=-1, keepdim=True)
        u_feat = u_feat / u_feat.norm(dim=-1, keepdim=True)
        geom = self.spatial(sb, ob, H, W)  # [M,128]
        return s_feat, o_feat, u_feat, geom

    def forward_pairs(self, image: torch.Tensor, boxes: torch.Tensor, pairs: torch.Tensor,
                      pred_text_feats: torch.Tensor) -> torch.Tensor:
        """Compute predicate logits for all pairs in a single image.
        pred_text_feats: [P,C] normalized text features for all predicates.
        Returns logits: [M,P]
        """
        s_feat, o_feat, u_feat, geom = self._pair_visual(image, boxes, pairs)
        vis = torch.cat([s_feat, o_feat, u_feat, geom], dim=-1)
        vis = self.fuse(vis)
        vis = vis / vis.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (vis @ pred_text_feats.t())
        return logits

    def forward(self, images: torch.Tensor, boxes_list: List[torch.Tensor], pairs_list: List[torch.Tensor]):
        """Compute per-image top-K selection, then full logits.
        Returns list of tensors [M_i, P] (or pruned [M_i, K] if cfg.USE_TOPK_LOGITS)
        """
        device = images.device
        B = images.size(0)
        # Build text features once for all predicates with current prompts
        prompts = self.prompt_learner()               # [P, L, D]
        text_feats = self.text_encoder(prompts, self.tokenized_prompts)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)  # [P,C]

        # Global image features for top-K selection
        with torch.no_grad():
            gimg = self.image_encoder(images.type(self.dtype))  # [B,C]
            gimg = gimg / gimg.norm(dim=-1, keepdim=True)
        topk_idx = self.selector(gimg, text_feats)  # [B,K]

        all_logits = []
        for i in range(B):
            img = images[i:i+1]
            boxes = boxes_list[i]
            H, W = img.shape[-2:]
            boxes = _boxes_to_int_xyxy(boxes, H, W).to(img.device)
            pairs = pairs_list[i].to(img.device)

            # Restrict to top-K predicate prototypes for speed
            P_sel = text_feats[topk_idx[i]]  # [K,C]
            logits_i = self.forward_pairs(img, boxes, pairs, P_sel)  # [M_i,K]

            # Map back to full P by scattering (optional)
            if getattr(self, "_scatter_fullP", False):
                P = text_feats.size(0)
                full = torch.empty((logits_i.size(0), P), device=img.device).fill_(-1e4)
                full[:, topk_idx[i]] = logits_i
                logits_i = full
            all_logits.append(logits_i)
        return all_logits, topk_idx


# -------------------------------------------------
# Trainer
# -------------------------------------------------
@TRAINER_REGISTRY.register()
class DPSGG(TrainerX):
    """Dual Prompt SGG trainer.
    Text-side: relation RelPromptLearner.
    Image-side: can be vanilla CLIP or VPT-Deep if you swap image tower in.
    """
    def check_cfg(self, cfg):
        assert cfg.TRAINER.DPSGG.PREC in ["fp16", "fp32", "amp"]
        assert cfg.INPUT.SIZE[0] == cfg.INPUT.SIZE[1]

    def build_model(self):
        cfg = self.cfg
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.DPSGG.PREC in ["fp32", "amp"]:
            clip_model.float()

        # predicate class names supplied by dataset
        if hasattr(self.dm.dataset, "classnames"):
            pred_names = list(self.dm.dataset.classnames)
        elif hasattr(self.dm.dataset, "pred_id2name"):
            id2n = self.dm.dataset.pred_id2name
            pred_names = [id2n[i] for i in sorted(id2n)]
        else:
            raise AttributeError("Dataset must expose predicate names as `classnames` or `pred_id2name`. See VisualGenomeSGG.")
        self.pred_names = pred_names

        self.model = DPSGGModel(cfg, pred_names, clip_model, crop_size=cfg.TRAINER.DPSGG.CROP_SIZE)

        # Freeze everything except prompts + fusion + spatial (typical DPT style)
        for name, p in self.model.named_parameters():
            trainable = ("prompt_learner" in name) or ("fuse" in name) or ("spatial" in name)
            p.requires_grad_(trainable)

        self.model.to(self.device)
        opt = build_optimizer([p for p in self.model.parameters() if p.requires_grad], cfg.OPTIM)
        sch = build_lr_scheduler(opt, cfg.OPTIM)
        self.register_model("dpsgg", self.model, opt, sch)

        self.multilabel = bool(cfg.TRAINER.DPSGG.MULTILABEL)
        self.prec = cfg.TRAINER.DPSGG.PREC
        self.scaler = torch.cuda.amp.GradScaler() if self.prec == "amp" else None

    # --------------- data parsing ---------------
    def parse_batch_train(self, batch):
        # Expect batch is dict-like or tuple. Adjust here as needed.
        if isinstance(batch, dict):
            images = batch["image"]
            boxes  = batch["boxes"]
            pairs  = batch["pair_idx"]
            if self.multilabel:
                rel = batch.get("rel_multi_hot")  # list[T[M,P]]
            else:
                rel = batch["rel_labels"]  # list[T[M]]
        else:
            # tuple layout: (image, boxes, obj_labels, pair_idx, rel)
            images, boxes, _obj, pairs, rel = batch
        images = images.to(self.device)
        return images, boxes, pairs, rel

    parse_batch_test = parse_batch_train

    # --------------- train step ---------------
    def forward_backward(self, batch):
        images, boxes, pairs, rel = self.parse_batch_train(batch)
        prec = self.prec
        if prec == "amp":
            with torch.cuda.amp.autocast():
                logits_list, topk_idx = self.model(images, boxes, pairs)
                loss = self._compute_loss(logits_list, rel)
            self.model_optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.model_optimizer)
            self.scaler.update()
        else:
            logits_list, topk_idx = self.model(images, boxes, pairs)
            loss = self._compute_loss(logits_list, rel)
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()
        self.visitables = {"loss": loss.item()}

    # --------------- loss ---------------
    def _compute_loss(self, logits_list: List[torch.Tensor], rel_gt_list):
        if self.multilabel:
            # BCE with optional pos_weight for tail
            total = 0.0
            for logits, target in zip(logits_list, rel_gt_list):
                total = total + F.binary_cross_entropy_with_logits(logits, target.to(logits.device).float())
            return total / max(1, len(logits_list))
        else:
            total = 0.0
            for logits, target in zip(logits_list, rel_gt_list):
                total = total + F.cross_entropy(logits, target.to(logits.device))
            return total / max(1, len(logits_list))

    # --------------- inference ---------------
    @torch.no_grad()
    def model_inference(self, batch):
        images, boxes, pairs, _ = self.parse_batch_test(batch)
        logits_list, topk_idx = self.model(images, boxes, pairs)
        return logits_list  # list of [M_i, K] (or P if scatter is enabled)


# -----------------------
# Config extension helper
# -----------------------
def extend_cfg(cfg):
    cfg.TRAINER.DPSGG = type("DPSGGCfg", (), {})()
    cfg.TRAINER.DPSGG.PREC = "fp16"
    cfg.TRAINER.DPSGG.N_CTX_PRED = 12
    cfg.TRAINER.DPSGG.CTX_INIT_PRED = ""
    cfg.TRAINER.DPSGG.CLASS_TOKEN_POSITION_PRED = "middle"  # {front,middle,end}
    cfg.TRAINER.DPSGG.TOPK = 50
    cfg.TRAINER.DPSGG.MULTILABEL = False
    cfg.TRAINER.DPSGG.CROP_SIZE = 224
    return cfg
