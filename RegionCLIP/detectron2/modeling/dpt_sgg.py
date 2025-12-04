import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.utils.registry import Registry

from detectron2.data.clip_datasets.clip_prompt_engineering import SimpleTokenizer, tokenize

_tokenizer = SimpleTokenizer()

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        
        if hasattr(clip_model, "visual"):
            self.dtype = clip_model.visual.conv1.weight.dtype
        else:
            self.dtype = clip_model.token_embedding.weight.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.MODEL.DPT.N_CTX if hasattr(cfg.MODEL, 'DPT') else 16
        ctx_init = cfg.MODEL.DPT.CTX_INIT if hasattr(cfg.MODEL, 'DPT') else ""
        csc = cfg.MODEL.DPT.CSC if hasattr(cfg.MODEL, 'DPT') else False
        class_token_position = cfg.MODEL.DPT.CLASS_TOKEN_POSITION if hasattr(cfg.MODEL, 'DPT') else "end"

        if hasattr(clip_model, "visual"):
            dtype = clip_model.visual.conv1.weight.dtype
        else:
            dtype = clip_model.token_embedding.weight.dtype
            
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            # [수정 2] 단순 공백 분리 대신 실제 토크나이저로 길이 계산 (BPE 고려)
            n_ctx = len(_tokenizer.encode(ctx_init))
            
            # [수정 3] _tokenizer.tokenize -> tokenize 함수 직접 호출
            # tokenize 함수는 [1, 77] 텐서를 반환하므로 cat으로 묶어도 안전함
            prompt = tokenize(ctx_init).to(clip_model.token_embedding.weight.device)
            
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            # embedding: [1, 77, dim], index 0 is SOS, so take 1:1+n_ctx
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if csc:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # [수정 4] _tokenizer.tokenize -> tokenize 함수 직접 호출
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.to(clip_model.token_embedding.weight.device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [prefix, ctx, suffix], dim=1
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [prefix_i, class_i, ctx_i, suffix_i], dim=1
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts