import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange
from typing import Tuple, Dict, Any  # Added for clarity in comments
from models.GPT2_arch import AccustumGPT2Model
from layers.RASS_Blocks import Retriever
from peft import LoraConfig, TaskType
from peft import get_peft_model
import torch.nn.functional as F


class PatchTokenEmbedding(nn.Module):
    def __init__(self, c_in: int, patch_size: int, d_model: int, method: str = 'conv'):
        super(PatchTokenEmbedding, self).__init__()
        assert method in ['linear', 'conv']
        self.method = method
        self.patch_size = patch_size
        patch_dim = c_in * patch_size

        if method == 'linear':
            self.tokenProj = nn.Linear(patch_dim, d_model)
        else:
            # Note: Using kernel_size=3 mixes information from adjacent patches.
            # If you want a pure per-patch projection (like 'linear'),
            # you might consider kernel_size=1.
            self.tokenProj = nn.Conv1d(in_channels=patch_dim, out_channels=d_model,
                                       kernel_size=3, padding=1, padding_mode='zeros', bias=False)
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, L, C] (Batch, Seq_Len, Channels)
        b, _, c = x.shape

        # Reshape to create patches
        # [B, L, C] -> [B, N, P, C] where N = L/P
        # Then flatten patch dim: [B, N, P*C]
        patches = x.view(b, -1, self.patch_size, c).flatten(2)

        if self.method == 'linear':
            # Project each patch: [B, N, P*C] -> [B, N, D_model]
            tokens = self.tokenProj(patches)
        else:
            # Conv1d expects [B, C_in, L_in]
            # [B, N, P*C] -> [B, P*C, N]
            patches_permuted = patches.permute(0, 2, 1)
            # [B, P*C, N] -> [B, D_model, N]
            tokens_conv = self.tokenProj(patches_permuted)
            # [B, D_model, N] -> [B, N, D_model]
            tokens = tokens_conv.transpose(1, 2)

        return tokens


class RetrievalIntegrateBlock(nn.Module):
    def __init__(self, l: int, d: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"Embedding dimension d ({d}) must be divisible by n_heads ({n_heads})")

        self.global_token = nn.Parameter(torch.randn(1, 3, d))
        self.x_encoder = nn.Linear(l, d)
        self.y_encoder = nn.Linear(l, d)

        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, candidate: torch.Tensor, text_token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # candidate: [B, N, L, C+1]
        # text_token: [1, 1, D]
        b, n, _, _ = candidate.shape

        # === 核心修改：全局 Token 增强 ===
        q_base = self.global_token + text_token
        q = q_base.expand(b * n, -1, -1)

        candidate_flat = rearrange(candidate, 'b n l c -> (b n) c l')
        cand_x = candidate_flat[:, -1:, :]
        cand_y = candidate_flat[:, :-1, :]
        feat_x = self.x_encoder(cand_x)
        feat_y = self.y_encoder(cand_y)
        kv = torch.cat([feat_x, feat_y], dim=1)

        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)

        attn_out, attn_weights = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm, need_weights=True)
        x_res = q_norm + self.attn_drop(attn_out)

        x_out = rearrange(x_res, '(b n) m d -> b n m d', b=b)
        return x_out, attn_weights


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.device = device
        self.top_n = configs.top_n
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 1. 加载 GPT2 与 PEFT
        self.gpt2 = AccustumGPT2Model.from_pretrained('./models/gpt2', output_attentions=True,
                                                      output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )
        self.gpt2 = get_peft_model(self.gpt2, peft_config)

        # 2. 获取 Embeddings 层
        # self.word_embeddings = self.gpt2.get_input_embeddings()

        # 3. Tokenizer
        # self.tokenizer = GPT2Tokenizer.from_pretrained('./models/gpt2')
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        #
        # # 4. 文本压缩层
        self.text_compressor = nn.Sequential(
            nn.Linear(configs.max_length, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, 3)
        )

        # 其他组件保持不变...
        self.retriever = Retriever(configs.seq_len, configs.enc_in, configs.stride_ret, configs.top_n)
        self.RIB = RetrievalIntegrateBlock(configs.seq_len, configs.d_model, configs.n_heads, configs.dropout)
        self.soft_global_tokens = nn.Parameter(torch.randn(1, 1, configs.d_model))
        self.token_emb = PatchTokenEmbedding(configs.enc_in, configs.patch_size, configs.d_model)
        self.act = nn.GELU()
        self.pred_head = nn.Linear(configs.d_model, 1)

    def forward(self, x, raw_text_emb):

        B, L, C = x.shape
        # 检索增强路径
        top_n_similarity, top_n_candidate, top_n_similarity_raw, top_n_candidate_raw = self.retriever(x)


        # [1, max_text_len, D] -> [1, D, max_text_len] -> [1, D, 1] -> [1, 1, D]
        compressed_text_emb = self.text_compressor(raw_text_emb.transpose(1, 2)).transpose(1, 2)
        prompt_tokens_raw, _ = self.RIB(top_n_candidate_raw, compressed_text_emb)
        # print("prompt_tokens shape:", prompt_tokens.shape)

        # 后续逻辑保持不变...
        weights = F.softmax(top_n_similarity_raw, dim=-1)
        prompt_tokens_weighted_raw = torch.einsum('bn,bnmd->bmd', weights, prompt_tokens_raw)
        soft_prompt_tokens = self.soft_global_tokens.expand(B, -1, -1)
        patches = self.token_emb(x)

        if self.training:
            # 训练逻辑 (与之前一致)
            prompt_tokens, _ = self.RIB(top_n_candidate, compressed_text_emb)
            prompt_tokens_extend = rearrange(prompt_tokens, 'b n m d -> (b n) m d')
            all_prompt_tokens = torch.cat([prompt_tokens_weighted_raw, prompt_tokens_extend], dim=0)
            all_soft_prompt_tokens = repeat(soft_prompt_tokens, 'b n d -> (b t) n d', t=1 + self.top_n)
            all_patches = repeat(patches, 'b n d -> (b t) n d', t=1 + self.top_n)
            all_tokens = torch.cat((all_prompt_tokens, all_soft_prompt_tokens, all_patches), dim=1)
            all_outputs, att = self.gpt2(inputs_embeds=all_tokens, output_attentions=True)
            all_outputs = self.pred_head(self.act(all_outputs[:, -1:, :])).reshape(B, -1)
        else:
            # 测试逻辑
            tokens = torch.cat((prompt_tokens_weighted_raw, soft_prompt_tokens, patches), dim=1)
            outputs, att = self.gpt2(inputs_embeds=tokens, output_attentions=True)
            # att_tensor = torch.stack(att)
            #
            # att_numpy = att_tensor.detach().cpu().numpy()
            #
            # # 3. 保存为 npy 文件
            # np.save("attention_weights.npy", att_numpy)
            #
            # print(f"保存成功，数组形状为: {att_numpy.shape}")

            all_outputs = self.pred_head(self.act(outputs[:, -1:, :])).reshape(B, -1)

        return {
            'output': all_outputs,
            'top_n_similarity': top_n_similarity,
            'logit_scale': self.logit_scale.exp().view(1).repeat(B)
        }