import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import repeat

# from .similar_utils import *
# from copy import deepcopy
#
# from .losses import mape_loss, mase_loss, smape_loss

loss_dict = {
    "l1": nn.L1Loss(),
    "smooth_l1": nn.SmoothL1Loss(),
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "kl": nn.KLDivLoss(reduction='batchmean'),
}

# def custom_loss(outputs2, batch_y):
#     return (outputs2-batch_y)**2

class cmLoss(nn.Module):
    def __init__(self, task_loss, similarity_loss, task_w=1.0, similarity_w=0.1, teacher_temp=0.5):
        super(cmLoss, self).__init__()
        self.task_w = task_w
        self.similarity_w = similarity_w
        self.task_loss = loss_dict[task_loss]
        self.similarity_loss = loss_dict[similarity_loss]
        self.teacher_temp = teacher_temp

    def forward(self, outputs, batch_y):
        output, top_n_similarity, logit_scale = (
            outputs["output"],
            outputs["top_n_similarity"],
            outputs["logit_scale"]
        )
        # print(std_scores)
        task_loss = self.task_loss(output[:, 0].unsqueeze(1), batch_y)

        # 2. Similarity Loss (KL Divergence)
        # --- Student (Retriever) ---
        # 使用模型学到的 logit_scale。
        # Cosine Similarity [-1, 1] * Scale [e.g., 10 ~ 100] -> Logits
        logit_scale = logit_scale.mean()
        q_logits = top_n_similarity * logit_scale
        q_log_prob = F.log_softmax(q_logits, dim=1)

        # --- Teacher (Forecaster) ---
        cand_preds = output[:, 1:]  # [B, Top_N]
        batch_y_expanded = batch_y.expand(-1, cand_preds.shape[1])


        mse_per_cand = (cand_preds - batch_y_expanded) ** 2  # [B, Top_N]
        p_logits = -mse_per_cand / self.teacher_temp
        p_prob = F.softmax(p_logits.detach(), dim=1)  # Detach

        # 监控一下 Entropy，确保 Teacher 不是 One-hot
        with torch.no_grad():
            entropy = -(p_prob * torch.log(p_prob + 1e-6)).sum(dim=1).mean()
            if entropy < 0.1: print("Warning: Teacher distribution collapsing!")

        similarity_loss = self.similarity_loss(q_log_prob, p_prob)

        total_loss = self.task_w * task_loss + self.similarity_w * similarity_loss
        return total_loss, task_loss, similarity_loss

class cmLoss_rl(nn.Module):
    def __init__(self):
        super(cmLoss_rl, self).__init__()
        self.ACTOR_WEIGHT = 1.0
        self.CRITIC_WEIGHT = 0.5
        self.PRED_WEIGHT = 1.0
        self.loss_fn_pred = nn.MSELoss(reduction='none')
        self.loss_fn_critic = nn.MSELoss()

    def forward(self, outputs, batch_y):
        output, baseline, sampled_log_probs = (
            outputs["output"],
            outputs["baseline"],
            outputs["sampled_log_probs"],
        )
        L_pred_raw = self.loss_fn_pred(output, batch_y)
        L_pred_per_item = L_pred_raw.squeeze(-1)

        L_pred = L_pred_per_item.mean()

        R_t = -L_pred_per_item.detach()
        A_t = R_t - baseline.squeeze(1).detach()
        L_actor = -(sampled_log_probs.squeeze(1) * A_t).mean()
        L_critic = self.loss_fn_critic(baseline.squeeze(1), R_t)

        # --- 5. 合并总损失 (与之前相同) ---
        total_loss = (
                (self.PRED_WEIGHT * L_pred) +
                (self.CRITIC_WEIGHT * L_critic) +
                (self.ACTOR_WEIGHT * L_actor)
        )

        return total_loss, L_pred, L_actor, L_critic


if __name__ == '__main__':
    a = torch.randn(64, 8)
    b = torch.randn(64, 1)
    loss = F.softmax(-(b-a)**2, dim=1)
    print(loss,loss.shape)
    print(loss[0])