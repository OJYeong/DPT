from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.data import MetadataCatalog
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
import torch
import copy

# 앞서 작성한 DPT 컴포넌트 임포트 (경로는 파일 위치에 따라 수정 필요)
from detectron2.modeling.backbone.clip_backbone import build_clip_language_encoder

from detectron2.modeling.dpt_sgg import PromptLearner, TextEncoder
from .clip_rcnn import CLIPFastRCNN # 기존 클래스 상속

@META_ARCH_REGISTRY.register()
class DPTCLIPFastRCNN(CLIPFastRCNN):
    """
    RegionCLIP에 DPT(Dual Prompt Tuning)를 적용한 클래스
    기존의 고정된(Frozen) 텍스트 임베딩 대신, 학습 가능한 PromptLearner를 통해
    매 forward마다 동적으로 텍스트 임베딩을 생성하여 사용합니다.
    """

    def __init__(self, cfg):
        # 1. 부모 클래스(CLIPFastRCNN)의 __init__ 호출하여 기본 설정 로드
        super().__init__(cfg=cfg)

        # 2. 데이터셋 메타데이터에서 클래스 이름 가져오기
        # DPT는 "a photo of a [CLASS]" 형태의 프롬프트를 만들기 위해 클래스 이름 리스트가 필요합니다.
        train_dataset_name = cfg.DATASETS.TRAIN[0]
        classnames = MetadataCatalog.get(train_dataset_name).thing_classes
        
        # 3. CLIP 모델 접근
        # RegionCLIP의 language_encoder 내부에 있는 clip_model을 가져옵니다.
        if self.lang_encoder is None:
            print("[DPTCLIP] Language encoder is None. Building it now...")
            self.lang_encoder = build_clip_language_encoder(cfg)
            
        # 2. CLIP 모델 객체 추출
        # RegionCLIP 구현체에 따라 clip_model이 위치한 경로가 다를 수 있어 안전하게 접근
        if hasattr(self.lang_encoder, 'clip_model'):
             clip_model = self.lang_encoder.clip_model
        elif hasattr(self.lang_encoder, 'model'): # 일부 구현체 대응
             clip_model = self.lang_encoder.model
        else:
             clip_model = self.lang_encoder
             
        # 4. DPT 컴포넌트(PromptLearner, TextEncoder) 초기화
        # 여기서 classnames와 clip_model을 전달하여 프롬프트 학습기를 만듭니다.
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.text_encoder = TextEncoder(clip_model)
        
        # 5. Gradient 설정 (중요)
        # 기존 Image Encoder와 Text Encoder는 고정(Freeze)하고, PromptLearner만 학습시킵니다.
        # RegionCLIP의 백본 등은 Config설정에 따라 이미 freeze 되어 있을 수 있으나, 여기서 확실히 처리합니다.
        for name, param in self.prompt_learner.named_parameters():
            param.requires_grad = True
        
        # TextEncoder는 CLIP의 Transformer를 그대로 쓰므로 gradient를 끕니다.
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False

    def get_text_embeddings(self):
        """
        학습 가능한 프롬프트를 텍스트 인코더에 통과시켜
        현재 시점의 텍스트 임베딩(Classifier Weights)을 생성합니다.
        Returns:
            text_features (Tensor): [Num_Classes, Feature_Dim]
        """
        # 1. 프롬프트 생성 (Learned Context + Class Name)
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # 2. 텍스트 인코더 통과
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        # 3. 정규화 (Cosine Similarity 계산을 위해 필수)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, batched_inputs):
        """
        매 학습 스텝(Iteration)마다 실행되는 함수입니다.
        RegionCLIP의 로직을 그대로 따르되, 텍스트 임베딩을 동적으로 생성하여 주입합니다.
        """
        # [수정] Inference(테스트) 모드일 경우 별도 처리
        if not self.training:
            return self.inference(batched_inputs)

        # GT 인스턴스 준비
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        # [DPT 핵심 수정 1] 텍스트 임베딩 동적 생성
        # 이번 배치를 위한 텍스트 임베딩을 생성합니다. (PromptLearner 업데이트 반영)
        text_features = self.get_text_embeddings()

        # [DPT 핵심 수정 2] ROI Heads에 텍스트 임베딩 주입
        # ROIHeads가 객체 분류를 할 때 이 text_features를 사용하도록 설정해야 합니다.
        # 주의: 사용하시는 ROIHeads(예: CLIPRes5ROIHeads)가 'text_embeddings' 속성을 
        # 참조하도록 구현되어 있어야 합니다. 여기서는 속성 주입 방식을 사용합니다.
        if hasattr(self.roi_heads, 'box_predictor'):
             # 일반적인 구조라면 box_predictor 내부에 classifier weight가 있을 수 있음
             # 혹은 ROIHead 자체가 text_embeddings 속성을 가지고 있을 수 있음
             self.roi_heads.text_embeddings = text_features
        else:
             # 속성이 없다면 강제로 할당하여 ROIHeads 내부에서 self.text_embeddings로 쓰게 함
             self.roi_heads.text_embeddings = text_features


        # --- 아래부터는 RegionCLIP의 기존 forward 로직 복사 ---
        
        # 1. Localization Branch: Offline Modules (RPN 등)을 사용하여 Region Proposal 추출
        with torch.no_grad():  
            if self.clip_crop_region_type == "GT":  # Ground-Truth 박스 사용
                proposals = []
                for r_i, b_input in enumerate(batched_inputs): 
                    this_gt = copy.deepcopy(b_input["instances"])
                    gt_boxes = this_gt._fields['gt_boxes'].to(self.device)
                    # GT를 Proposal 형식으로 변환
                    this_gt._fields = {
                        'proposal_boxes': gt_boxes, 
                        'objectness_logits': torch.ones(gt_boxes.tensor.size(0)).to(self.device)
                    }
                    proposals.append(this_gt)                
            elif self.clip_crop_region_type == "RPN": # 학습된 RPN 사용
                if self.offline_backbone.training or self.offline_proposal_generator.training:
                    self.offline_backbone.eval() 
                    self.offline_proposal_generator.eval()  
                images = self.offline_preprocess_image(batched_inputs)
                features = self.offline_backbone(images.tensor)
                if self.offline_proposal_generator is not None:
                    proposals, _ = self.offline_proposal_generator(images, features, None)     

        # 2. Recognition Branch: 현재 학습 중인 Backbone으로 이미지 특징 추출
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # 3. ROI Heads: Proposal 영역의 특징을 잘라내고(Crop), 분류(Classification) 및 Loss 계산
        # 여기서 주입한 text_features가 분류기 가중치(Classifier Weights)로 사용됩니다.
        if self.use_clip_c4: # C4 + ResNet weights from CLIP
            if self.use_clip_attpool:
                _, detector_losses = self.roi_heads(
                    images, features, proposals, gt_instances, 
                    res5=self.backbone.layer4, attnpool=self.backbone.attnpool
                )
            else: # Mean pool
                _, detector_losses = self.roi_heads(
                    images, features, proposals, gt_instances, res5=self.backbone.layer4
                )
        else:  # Regular detector setting
            if self.use_clip_attpool:
                _, detector_losses = self.roi_heads(
                    images, features, proposals, gt_instances, attnpool=self.backbone.bottom_up.attnpool
                )
            else:
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        # 4. 시각화 (Visualization)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        
        # [선택사항] Image-Text Level Matching Loss (Pretraining 시 사용)
        # RegionCLIP의 image_text_matching 메소드는 기본적으로 Caption을 사용합니다.
        # DPT 학습 시에는 이 Loss를 끌 수도 있고, 켤 수도 있습니다.
        # 켠다면 get_text_embeddings() 결과를 활용하도록 image_text_matching 오버라이딩이 필요합니다.
        
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        테스트/평가 시 실행되는 함수입니다.
        forward와 마찬가지로 동적 텍스트 임베딩을 생성하여 ROI Heads에 주입해야 합니다.
        """
        assert not self.training

        # [DPT 핵심 수정] Inference 시에도 학습된 Prompt로 텍스트 임베딩 생성 및 주입
        text_features = self.get_text_embeddings()
        self.roi_heads.text_embeddings = text_features
        
        # --- 아래는 RegionCLIP 기존 inference 로직 복사 ---

        # 1. Proposal 추출
        if self.clip_crop_region_type == "GT":
            proposals = []
            for r_i, b_input in enumerate(batched_inputs): 
                this_gt = copy.deepcopy(b_input["instances"])
                gt_boxes = this_gt._fields['gt_boxes'].to(self.device)
                this_gt._fields = {'proposal_boxes': gt_boxes}
                proposals.append(this_gt)                
        elif self.clip_crop_region_type == "RPN":
            images = self.offline_preprocess_image(batched_inputs)
            features = self.offline_backbone(images.tensor)
            if detected_instances is None:
                if self.offline_proposal_generator is not None:
                    proposals, _ = self.offline_proposal_generator(images, features, None)     
    
        # 2. 이미지 특징 추출
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # 3. ROI Heads 실행 (결과 예측)
        if self.use_clip_c4:
            if self.use_clip_attpool:
                results, _ = self.roi_heads(
                    images, features, proposals, None, 
                    res5=self.backbone.layer4, attnpool=self.backbone.attnpool
                )
            else:
                results, _ = self.roi_heads(
                    images, features, proposals, None, res5=self.backbone.layer4
                )
        else:
            if self.use_clip_attpool:
                results, _  = self.roi_heads(
                    images, features, proposals, None, attnpool=self.backbone.bottom_up.attnpool
                )
            else:
                results, _  = self.roi_heads(images, features, proposals, None)
        
        # 4. Post processing (이미지 크기 복원 등)
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return CLIPFastRCNN._postprocess(results, batched_inputs)
        else:
            return results