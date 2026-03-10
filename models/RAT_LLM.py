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

        # 1. Load GPT2 and apply PEFT (LoRA)
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
        Forward pass using MTRM to integrate all preprocessing steps.

        Args:
            x: Input sequence [B, L, C]
            text_emb: Text embeddings [1, max_text_len, D]

        Returns:
            dict: Dictionary containing prediction outputs and similarity scores
        """
        B, _, _ = x.shape
        # 1. Retrieval-augmented path
        retriever_outputs = self.retriever(x)
        print(retriever_outputs[0].device, retriever_outputs[0].sum().item())

        # 2. Use MTRM to process all subsequent steps and get tokens ready for GPT2
        tokens = self.mtrm(x, text_emb, retriever_outputs)

        # 3. Pass to GPT2
        outputs, att = self.gpt2(inputs_embeds=tokens, output_attentions=True)

        # 4. Prediction head
        all_outputs = self.pred_head(self.act(outputs[:, -1:, :])).reshape(B, -1)

        # 5. Get similarity scores for return
        # retriever_outputs format: (top_n_similarity, top_n_candidate, top_n_similarity_raw, top_n_candidate_raw)
        # In test mode, top_n_similarity may be None, use top_n_similarity_raw instead
        top_n_similarity = retriever_outputs[0] if retriever_outputs[0] is not None else retriever_outputs[2]

        return {
            'output': all_outputs,
            'top_n_similarity': top_n_similarity,
            'logit_scale': self.logit_scale.exp().view(1).repeat(B)
        }