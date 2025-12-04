import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog



VG_CLASSES = [
    "airplane", "animal", "arm", "bag", "banana", "basket", "beach", "bear", "bed", "bench",
    "bike", "bird", "board", "boat", "book", "boot", "bottle", "bowl", "box", "boy",
    "branch", "building", "bus", "cabinet", "cap", "car", "cat", "chair", "child", "clock",
    "coat", "counter", "cow", "cup", "curtain", "desk", "dog", "door", "drawer", "ear",
    "elephant", "engine", "eye", "face", "fence", "finger", "flag", "flower", "food", "fork",
    "fruit", "giraffe", "girl", "glass", "glove", "guy", "hair", "hand", "handle", "hat",
    "head", "helmet", "hill", "horse", "house", "jacket", "jean", "kid", "kite", "lady",
    "lamp", "laptop", "leaf", "leg", "letter", "light", "logo", "man", "men", "motorcycle",
    "mountain", "mouth", "neck", "nose", "number", "orange", "pant", "paper", "paw", "people",
    "person", "phone", "pillow", "pizza", "plane", "plant", "plate", "player", "pole", "post",
    "pot", "racket", "railing", "rock", "roof", "room", "screen", "seat", "sheep", "shelf",
    "shirt", "shoe", "short", "sidewalk", "sign", "sink", "skateboard", "ski", "skier", "sneaker",
    "snow", "sock", "stand", "street", "surfboard", "table", "tail", "tie", "tile", "tire",
    "toilet", "towel", "tower", "track", "train", "tree", "truck", "trunk", "umbrella", "vase",
    "vegetable", "vehicle", "wave", "wheel", "window", "windshield", "wing", "wire", "woman", "zebra",
]


VG_REL_CLASSES = {"1": "above", "2": "across", "3": "against", "4": "along", "5": "and", "6": "at", "7": "attached to", "8": "behind", "9": "belonging to", "10": "between", 
"11": "carrying", "12": "covered in", "13": "covering", "14": "eating", "15": "flying in", "16": "for", "17": "from", "18": "growing on", "19": "hanging from", "20": "has", 
"21": "holding", "22": "in", "23": "in front of", "24": "laying on", "25": "looking at", "26": "lying on", "27": "made of", "28": "mounted on", "29": "near", "30": "of", 
"31": "on", "32": "on back of", "33": "over", "34": "painted on", "35": "parked on", "36": "part of", "37": "playing", "38": "riding", "39": "says", "40": "sitting on", 
"41": "standing on", "42": "to", "43": "under", "44": "using", "45": "walking in", "46": "walking on", "47": "watching", "48": "wearing", "49": "wears", "50": "with"}




def register_visual_genome(root="datasets/visual_genome"):
    # 경로 설정 (사용자 환경에 맞게 수정 필요)
    img_root = os.path.join(root, "images")
    ann_root = os.path.join(root, "annotations")

    # Train Set 등록
    register_coco_instances(
        "vg_train",
        {},
        os.path.join(ann_root, "train.json"),
        img_root
    )
    # 메타데이터 설정 (클래스 이름 주입)
    MetadataCatalog.get("vg_train").thing_classes = VG_CLASSES

    # Val Set 등록
    register_coco_instances(
        "vg_val",
        {},
        os.path.join(ann_root, "val.json"),
        img_root
    )
    MetadataCatalog.get("vg_val").thing_classes = VG_CLASSES
    
    print(f"[Info] Visual Genome (VG150) registered. {len(VG_CLASSES)} classes.")


# ============================================================
# 2. Open Images (OID) 등록 설정
# ============================================================

def register_open_images(root="datasets"):
    img_root = os.path.join(root, "open_images/images")
    ann_root = os.path.join(root, "open_images/annotations")
    
    # Open Images는 클래스가 많으므로 JSON 파일에 있는 'categories' 정보를 그대로 사용하는 것이 좋습니다.
    # register_coco_instances는 기본적으로 JSON의 categories 이름을 메타데이터로 로드합니다.
    
    # Train Set
    register_coco_instances(
        "oid_train",
        {},
        os.path.join(ann_root, "oidv7_train.json"),
        img_root
    )
    
    # Val Set
    register_coco_instances(
        "oid_val",
        {},
        os.path.join(ann_root, "oidv7_val.json"),
        img_root
    )
    print("[Info] Open Images registered.")

# # 이 파일이 직접 실행될 때 등록 테스트
# if __name__ == "__main__":
#     register_visual_genome()
#     # register_open_images()
    
#     # 잘 등록되었는지 확인
#     meta = MetadataCatalog.get("vg_train")
#     print("VG Classes example:", meta.thing_classes[:5])