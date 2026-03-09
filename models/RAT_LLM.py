import numpy as np
import torch
import torch.nn as nn
from models.GPT2_arch import AccustumGPT2Model
from layers.RAT_LLM_Blocks import Retriever, MTRM
from peft import LoraConfig, TaskType
from peft import get_peft_model

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

       
        self.retriever = Retriever(configs.seq_len, configs.enc_in, configs.rel_stride, configs.top_n, configs.ratio, configs.metric)
        self.mtrm = MTRM(configs.top_n, configs.seq_len, configs.enc_in, configs.patch_size, configs.d_model, configs.max_length, configs.hid_m, configs.m, configs.n_heads, configs.dropout)
        self.act = nn.GELU()
        self.pred_head = nn.Linear(configs.d_model, 1)

    def forward(self, x, text_emb):
        """
        前向传播，使用新的MTRM整合所有预处理步骤

        Args:
            x: 输入序列 [B, L, C]
            raw_text_emb: 原始文本嵌入 [1, max_text_len, D]

        Returns:
            dict: 包含预测输出和相似度分数的字典
        """
        B, _, _ = x.shape

        # 1. 检索增强路径
        retriever_outputs = self.retriever(x)

        # 2. 使用新的MTRM处理所有后续步骤，获取准备输入GPT2的tokens
        tokens = self.mtrm(x, text_emb, retriever_outputs)

        # 3. 传递给GPT2
        outputs, att = self.gpt2(inputs_embeds=tokens, output_attentions=True)

        # 4. 预测头
        all_outputs = self.pred_head(self.act(outputs[:, -1:, :])).reshape(B, -1)
        # 5. 获取相似度分数用于返回
        # retriever_outputs格式: (top_n_similarity, top_n_candidate, top_n_similarity_raw, top_n_candidate_raw)
        # 在测试模式下，top_n_similarity可能为None，使用top_n_similarity_raw
        top_n_similarity = retriever_outputs[0] if retriever_outputs[0] is not None else retriever_outputs[2]

        return {
            'output': all_outputs,
            'top_n_similarity': top_n_similarity,
            'logit_scale': self.logit_scale.exp().view(1).repeat(B)
        }