# -*- coding: utf-8 -*-
# Visual Genome (VG) SGG loader for Dassl.pytorch
# 기대 경로:
# ROOT/
#   visual_genome/
#     images/1.jpg, 2.jpg, ...
#     vg/train.json, val.json, test.json, rel.json

import os, json, pickle, numpy as np
from collections import defaultdict

from dassl.data.datasets import DatasetBase, Datum, DATASET_REGISTRY
from dassl.utils import mkdir_if_missing


@DATASET_REGISTRY.register()
class VisualGenomeSGG(DatasetBase):
    dataset_dir = "C:/Users/MILAB_server3/Desktop/jin/Datasets/visual_genome" 

    def __init__(self, cfg):
        self.root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.img_dir = os.path.join(self.dataset_dir, "images")
        self.vg_dir  = os.path.join(self.dataset_dir, "vg")
        assert os.path.isdir(self.img_dir), f"Not found: {self.img_dir}"
        assert os.path.isdir(self.vg_dir),  f"Not found: {self.vg_dir}"

        # 라벨 맵(술어) 로딩: 여러 포맷을 허용
        rel_map_path = os.path.join(self.vg_dir, "rel.json")
        self.pred_name2id, self.pred_id2name = _load_pred_maps(rel_map_path)

        # 스플릿 로드
        train = self._load_split(os.path.join(self.vg_dir, "train.json"))
        val   = self._load_split(os.path.join(self.vg_dir, "val.json"))
        test  = self._load_split(os.path.join(self.vg_dir, "test.json"))

        super().__init__(train_x=train, val=val, test=test)

    # ---------- internals ----------
    def _load_split(self, json_path):
        cache_dir  = os.path.join(self.dataset_dir, "cache_sgg")
        mkdir_if_missing(cache_dir)
        cache_file = os.path.join(cache_dir, os.path.basename(json_path) + ".pkl")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        items = []
        for rec in records:
            # 1) 이미지 경로 결정
            image_id, impath = _resolve_image_path(rec, self.img_dir)

            # 2) 객체 박스/라벨 파싱 (여러 포맷 허용)
            boxes, obj_labels, obj_names = _parse_objects(rec)

            # 3) 관계 파싱 (여러 포맷 허용)
            relations = _parse_relations(rec, self.pred_name2id)

            # 방어: 인덱스 범위/타입 정리
            boxes = np.asarray(boxes, dtype=np.float32)
            obj_labels = np.asarray(obj_labels, dtype=np.int64)
            relations = np.asarray([
                trip for trip in relations
                if 0 <= trip[0] < len(boxes) and 0 <= trip[1] < len(boxes) and trip[2] >= 0
            ], dtype=np.int64)

            meta = dict(
                image_id=int(image_id) if str(image_id).isdigit() else image_id,
                boxes=boxes,                # [N,4] (xyxy, 픽셀 or 정규화 상관없음; 모델에서 통일)
                obj_labels=obj_labels,      # [N]
                obj_names=obj_names,        # List[str]
                relations=relations         # [M,3] (subj_idx, obj_idx, predicate_id)
            )
            items.append(Datum(impath=impath, label=-1, classname=None, metadata=meta))

        with open(cache_file, "wb") as f:
            pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)
        return items


# ---------- helpers (입력 포맷 다양성 대응) ----------

def _load_pred_maps(path):
    """
    rel.json 포맷 다양성 대응:
    - {"predicate_to_idx": {"on":1,...}, "idx_to_predicate": {"1":"on",...}}
    - {"idx_to_name": ["__background__", "on", "in", ...]}
    - {"predicates": ["on","in",...]}
    """
    if not os.path.exists(path):
        # 맵이 없다면 추후 동적 생성 가능(여기선 일단 빈 맵)
        return {}, {}

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    name2id, id2name = {}, {}

    if "predicate_to_idx" in obj:
        name2id = {k.strip().lower(): int(v) for k, v in obj["predicate_to_idx"].items()}
        # 역맵
        for n, i in name2id.items():
            id2name[int(i)] = n

    elif "idx_to_predicate" in obj:
        id2name = {int(k): str(v).strip().lower() for k, v in obj["idx_to_predicate"].items()}
        name2id = {v: k for k, v in id2name.items()}

    elif "idx_to_name" in obj and isinstance(obj["idx_to_name"], list):
        id2name = {i: str(n).strip().lower() for i, n in enumerate(obj["idx_to_name"])}
        name2id = {v: k for k, v in id2name.items()}

    elif "predicates" in obj and isinstance(obj["predicates"], list):
        id2name = {i: str(n).strip().lower() for i, n in enumerate(obj["predicates"])}
        name2id = {v: k for k, v in id2name.items()}

    return name2id, id2name


def _resolve_image_path(rec, img_dir):
    """
    우선순위:
    - 'img_path' 절대/상대 경로
    - 'file_name' 상대 경로
    - 'image_id' 기반 "{image_id}.jpg"
    """
    if "img_path" in rec:
        p = rec["img_path"]
        return _infer_image_id(p), p if os.path.isabs(p) else os.path.join(img_dir, p)
    if "file_name" in rec:
        p = os.path.join(img_dir, rec["file_name"])
        return _infer_image_id(p), p
    # default: 1.jpg, 2.jpg ...
    image_id = rec.get("image_id", rec.get("img_id", rec.get("id")))
    assert image_id is not None, "image_id/img_id/id 중 하나가 필요합니다."
    return image_id, os.path.join(img_dir, f"{image_id}.jpg")


def _infer_image_id(path):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem if not stem.isdigit() else int(stem)


def _parse_objects(rec):
    """
    가능한 포맷:
    A) 'boxes': [[x1,y1,x2,y2],...], 'labels':[...], 'names':[...]
    B) 'objects': [{'x':..,'y':..,'w':..,'h':..,'name':'person'}, ...]
    C) 'objects': [{'bbox':[x1,y1,x2,y2],'category':'person'}, ...]
    """
    boxes, labels, names = [], [], []

    # Case A
    if "boxes" in rec:
        boxes = rec["boxes"]
        labels = rec.get("labels", rec.get("obj_labels", [-1] * len(boxes)))
        names  = rec.get("names", rec.get("obj_names", [""] * len(boxes)))
        boxes = [_to_xyxy(b) for b in boxes]
        names = [str(n).strip().lower() for n in names]
        return boxes, labels, names

    # Case B/C
    if "objects" in rec:
        for obj in rec["objects"]:
            if "bbox" in obj:
                b = _to_xyxy(obj["bbox"])
            else:
                # x,y,w,h -> xyxy
                x, y = float(obj["x"]), float(obj["y"])
                w, h = float(obj.get("w", obj.get("width", 0))), float(obj.get("h", obj.get("height", 0)))
                b = [x, y, x + w, y + h]
            name = obj.get("name", obj.get("category", ""))
            boxes.append(b)
            names.append(str(name).strip().lower())
            labels.append(int(obj.get("label", obj.get("category_id", -1))))
        return boxes, labels, names

    # Fallback (없으면 빈 항목)
    n = 0
    return [], [], []


def _parse_relations(rec, pred_name2id):
    """
    가능한 포맷:
    A) 'relations': [[s,o,pid], ...]
    B) 'rels' 또는 'relationships': [{'subject':i,'object':j,'predicate':'on' or pid}, ...]
    """
    out = []

    # Case A
    if "relations" in rec and isinstance(rec["relations"], list) and rec["relations"] and isinstance(rec["relations"][0], (list, tuple)):
        for s, o, p in rec["relations"]:
            out.append([int(s), int(o), int(p)])
        return out

    # Case B
    for key in ("rels", "relationships", "relation"):
        if key in rec and isinstance(rec[key], list):
            for r in rec[key]:
                s = int(r.get("subject", r.get("subject_id", -1)))
                o = int(r.get("object",  r.get("object_id", -1)))
                p = r.get("predicate", r.get("predicate_id", r.get("rel_id", -1)))
                if isinstance(p, str):
                    p = pred_name2id.get(p.strip().lower(), -1)
                out.append([s, o, int(p)])
            return out

    return out


def _to_xyxy(b):
    """
    허용 입력:
    - [x1,y1,x2,y2]
    - [x,y,w,h]
    - {'x1':..,'y1':..,'x2':..,'y2':..} 등
    """
    if isinstance(b, dict):
        if all(k in b for k in ("x1", "y1", "x2", "y2")):
            return [float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])]
        if all(k in b for k in ("x", "y", "w", "h")):
            return [float(b["x"]), float(b["y"]), float(b["x"]) + float(b["w"]), float(b["y"]) + float(b["h"])]
    if len(b) == 4:
        x1, y1, x2, y2 = [float(t) for t in b]
        # w,h 포맷 추정: x2>x1 && y2>y1 이면 xyxy로 간주
        if x2 <= x1 or y2 <= y1:
            x2, y2 = x1 + x2, y1 + y2
        return [x1, y1, x2, y2]
    raise ValueError(f"Unexpected bbox format: {b}")
