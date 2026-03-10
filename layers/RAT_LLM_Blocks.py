import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class StandardConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (dilation * (kernel_size - 1)) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x):
        # x: [B, C, L]
        out = self.conv(x)
        if out.shape[-1] != x.shape[-1]:
            out = out[:, :, :x.shape[-1]]
        return out



class MultiScaleFeatureExtractor(nn.Module):
    """
    Feature extractor designed for industrial multivariate time series.
    Integrates: multi-scale dilated convolutions + channel mixing.
    """

    def __init__(self, c_in, d_model=64):
        super().__init__()
        # Branch A: 1x1 conv for point-wise coupling
        self.branch1 = nn.Conv1d(c_in, d_model // 4, kernel_size=1)

        # Branch B: small receptive field (k=3, d=1)
        self.branch2 = StandardConv1d(c_in, d_model // 4, kernel_size=3, dilation=1)

        # Branch C: medium receptive field (k=3, d=2)
        self.branch3 = StandardConv1d(c_in, d_model // 4, kernel_size=3, dilation=2)

        # Branch D: large receptive field (k=3, d=4), captures long-term dependencies
        self.branch4 = StandardConv1d(c_in, d_model // 4, kernel_size=3, dilation=4)

        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.final_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x):
        # Input x: [B, L, C] -> transpose to [B, C, L] for Conv1d
        x = x.permute(0, 2, 1)

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        feat = torch.cat([b1, b2, b3, b4], dim=1)  # [B, d_model, L]

        feat = self.act(feat)

        # [B, d_model, L] -> [B, d_model, 16] -> [B, 16*d_model]
        feat_vec = self.pool(feat).flatten(1)

        # [B, d_model]
        out = self.final_proj(feat_vec)
        out = self.norm(out)
        return out

class Retriever(nn.Module):
    """
    Retriever module that uses MultiScaleFeatureExtractor to process raw sequences.
    Maintains efficient separation between update_index (offline) and forward (online).
    """

    def __init__(self,
                 seq_len: int,
                 c_in: int,
                 rel_stride: int,
                 top_k: int,
                 ratio: float = 0.5,
                 metric: str = 'cosine'
                 ):
        super(Retriever, self).__init__()

        if metric not in ['cosine', 'euclidean']:
            raise ValueError("metric must be 'cosine' or 'euclidean'")

        self.c_in = c_in
        self.c_db = c_in + 1  # Database contains (X, Y), so channels = c_in + 1
        self.seq_len = seq_len
        self.rel_stride = rel_stride
        self.k = top_k
        self.ratio = ratio
        self.metric = metric

        # --- Core encoder replaced with powerful Extractor ---
        self.encoder = MultiScaleFeatureExtractor(c_in)

        # --- Cache buffers ---
        self.register_buffer('ax_cache', None, persistent=False)
        self.register_buffer('ax_sq_cache', None, persistent=False)
        self.register_buffer('all_original_windows_cache', None, persistent=False)
        self.register_buffer('total_windows_in_db', torch.tensor(0, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def update_index(self, datasets: list[torch.Tensor], batch_size_enc=10000):
        """
        Build index for retrieval.

        Args:
            datasets: list of tensors with shape [Time, c_db]
            batch_size_enc: batch size for encoding to prevent OOM
        """
        self.eval()  # Switch to eval mode, disable BatchNorm statistics update
        # print("Retriever: Building index with MultiScale Extractor...")

        all_feature_windows_list = []
        all_original_windows_list = []

        # 1. Prepare sliding window data
        for df in datasets:
            if df.shape[1] != self.c_db:
                raise ValueError(f"Expected {self.c_db} channels, got {df.shape[1]}")

            # Extract feature part (X) for encoding
            # unfold: [Time, C] -> [N_wins, C, SeqLen] -> permute to [N_wins, SeqLen, C]
            # Note: Extractor expects input with shape [B, L, C]
            feat_wins = df[:, :self.c_in].unfold(0, self.seq_len, self.rel_stride).permute(0, 2, 1)
            all_feature_windows_list.append(feat_wins)

            # Extract complete part (X+Y) for retrieval return
            # [N_wins, C_db, SeqLen]
            orig_wins = df.unfold(0, self.seq_len, self.rel_stride)
            all_original_windows_list.append(orig_wins)

        # Concatenate all windows (note: may consume memory for large datasets, but required for module buffer)
        all_feature_inputs = torch.cat(all_feature_windows_list, dim=0)  # [Total, L, C_in]
        all_original_windows = torch.cat(all_original_windows_list, dim=0)  # [Total, C_db, L]

        # Update raw data cache
        self.all_original_windows_cache = all_original_windows
        self.total_windows_in_db.fill_(all_original_windows.shape[0])

        del all_feature_windows_list, all_original_windows_list  # Free temporary memory

        # 2. Encode features in batches (avoid OOM)
        ax_list = []
        total_samples = all_feature_inputs.shape[0]

        for i in range(0, total_samples, batch_size_enc):
            batch = all_feature_inputs[i: i + batch_size_enc]  # [B, L, C]
            # Pass through Extractor
            embeddings = self.encoder(batch)  # [B, d_model]
            ax_list.append(embeddings)

        ax = torch.cat(ax_list, dim=0)  # [Total, d_model]
        print('ax', ax.sum().item())
        # 3. Normalization and cache processing
        if self.metric == 'cosine':
            # Cosine similarity requires normalization
            ax = F.normalize(ax, p=2, dim=1)
            ax_sq = None
        else:
            # Euclidean distance requires squared sum term
            ax_sq = torch.sum(ax ** 2, dim=1)

        self.ax_cache = ax
        self.ax_sq_cache = ax_sq

        print(f"Retriever: Index built. Total windows: {self.total_windows_in_db.item()}")

    def forward(self, x: torch.Tensor):
        """
        Online retrieval.

        Args:
            x: Query tensor with shape [B, L, C_in]
        """
        print(f"Weight sum: {self.encoder.branch1.weight.sum().item()}")
        print(f"Internal d_model checking: {self.encoder.final_proj.out_features}")
        B, L, C = x.shape
        device = x.device

        if self.ax_cache is None:
            raise RuntimeError("Index not built! Call update_index() first.")

        # 1. Encode query
        # Gradients are enabled during training/inference
        bx = self.encoder(x)  # [B, d_model]

        # 2. Normalization
        if self.metric == 'cosine':
            bx = F.normalize(bx, p=2, dim=1)
            bx_sq = None
        else:
            bx_sq = torch.sum(bx ** 2, dim=1)

        # 3. Compute similarity scores
        # ax_cache: [Total, d_model]
        # dot_product: [B, Total]
        dot_product = torch.matmul(bx, self.ax_cache.T)

        if self.metric == 'cosine':
            score_matrix = dot_product
        else:
            # Euclidean: -(x-y)^2 = 2xy - x^2 - y^2
            distance = bx_sq.unsqueeze(1) - 2 * dot_product + self.ax_sq_cache.unsqueeze(0)
            score_matrix = -distance

        # 4. Top-K retrieval
        k_selected = min(self.k, self.total_windows_in_db.item())
        topk_scores, topk_indices = torch.topk(score_matrix, k=k_selected, dim=1)

        # 5. Retrieve corresponding raw data
        # indices: [B, K] -> retrieved: [B, K, C_db, L]
        retrieved_windows = self.all_original_windows_cache[topk_indices]

        # Adjust format to match Transformer input: [B, K, L, C_db]
        selected_windows = retrieved_windows.permute(0, 1, 3, 2)

        scores_raw = topk_scores.clone()
        windows_raw = selected_windows.clone()

        # 7. Negative sampling during training
        if self.training and self.ratio > 0:
            replace_mask = torch.rand(B, k_selected, device=device) < self.ratio
            num_replace = replace_mask.sum().item()

            if num_replace > 0:
                rand_indices = torch.randint(0, self.total_windows_in_db.item(), (num_replace,), device=device)
                # Get row indices corresponding to mask (for extracting random scores from score_matrix)
                batch_rows = torch.arange(B, device=device).view(B, 1).expand(B, k_selected)[replace_mask]

                rand_scores = score_matrix[batch_rows, rand_indices]
                topk_scores[replace_mask] = rand_scores

                rand_wins = self.all_original_windows_cache[rand_indices].permute(0, 2, 1)
                selected_windows[replace_mask] = rand_wins

            # Return 4 items in training mode
            return topk_scores, selected_windows, scores_raw, windows_raw

        # Return None for first two items in non-training mode or when ratio is 0
        return None, None, scores_raw, windows_raw


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


class RetrievalRepresentation(nn.Module):
    def __init__(self, seq_len, d_model, max_length, hid_m, m, n_heads, dropout):
        super(RetrievalRepresentation, self).__init__()
        self.global_token = nn.Parameter(torch.randn(1, m, d_model))
        self.x_encoder = nn.Linear(seq_len, d_model)
        self.y_encoder = nn.Linear(seq_len, d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)

        self.text_compressor = nn.Sequential(
            nn.Linear(max_length, hid_m),
            nn.GELU(),
            nn.Linear(hid_m, m)
        )
    def forward(self, candidate: torch.Tensor, text_emb: torch.Tensor):
        """
        Process retrieved candidates with text augmentation.

        Args:
            candidate: Retrieved windows [B, N, L, C+1]
            text_emb: Text embeddings [1, max_length, D]

        Returns:
            Augmented retrieval representations [B, N, m, d_model]
        """
        b, n, _, _ = candidate.shape

        # === Core enhancement: Global Token with Text Augmentation ===
        q_base = self.global_token
        text_emb_compressed = self.text_compressor(text_emb.transpose(1, 2)).transpose(1, 2)
        q_aug = q_base + text_emb_compressed
        q = q_aug.expand(b * n, -1, -1)

        candidate_flat = rearrange(candidate, 'b n l c -> (b n) c l')
        cand_x = candidate_flat[:, -1:, :]
        cand_y = candidate_flat[:, :-1, :]
        feat_x = self.x_encoder(cand_x)
        feat_y = self.y_encoder(cand_y)
        kv = torch.cat([feat_x, feat_y], dim=1)

        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)

        attn_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm, need_weights=True)
        x_res = q_norm + self.attn_drop(attn_out)

        x_out = rearrange(x_res, '(b n) m d -> b n m d', b=b)
        return x_out



class MTRM(nn.Module):
    def __init__(self, top_n, seq_len, enc_in, patch_size, d_model, max_length, hid_m, m, n_heads, dropout):
        super(MTRM, self).__init__()

        self.sep_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.patch_emb = PatchTokenEmbedding(enc_in, patch_size, d_model)
        self.r_r = RetrievalRepresentation(seq_len, d_model, max_length, hid_m, m, n_heads, dropout)
        
        self.top_n = top_n


    def forward(self, x: torch.Tensor, text_emb: torch.Tensor, retriever_outputs: tuple) -> torch.Tensor:
        """
        Forward pass integrating all functionality.

        Args:
            x: Raw input sequence [B, L, C]
            text_emb: Text embeddings [1, max_text_len, D]
            retriever_outputs: Tuple from retriever
                (top_n_similarity, top_n_candidate, top_n_similarity_raw, top_n_candidate_raw)

        Returns:
            tokens: Tokens ready for GPT2 input [B, N, D] or [B*(1+top_n), N, D] (in training mode)
        """
        # Unpack retriever outputs
        _, top_n_candidate, top_n_similarity_raw, top_n_candidate_raw = retriever_outputs

        B, _, _ = x.shape

        # 1. Text compression (handled in RetrievalRepresentation)

        # 2. Process raw candidate sequences through RetrievalRepresentation
        rel_tokens_raw = self.r_r(top_n_candidate_raw, text_emb)

        # 3. Weighted averaging
        weights = F.softmax(top_n_similarity_raw, dim=-1)
        rel_tokens_weighted = torch.einsum('bn,bnmd->bmd', weights, rel_tokens_raw)

        # 4. Patch embedding
        patches = self.patch_emb(x)

        # 5. Separator token
        sep_token = self.sep_token.expand(B, -1, -1)

        # 6. Return different tokens based on train/test mode
        if self.training:
            # Training mode: additionally process top_n_candidate (may include negative sampling)
            # Note: In test mode, top_n_candidate may be None
            if top_n_candidate is not None:
                rel_tokens = self.r_r(top_n_candidate, text_emb)
                rel_tokens_extend = rearrange(rel_tokens, 'b n m d -> (b n) m d')
                all_rel_tokens = torch.cat([rel_tokens_weighted, rel_tokens_extend], dim=0)
                all_sep_token = repeat(sep_token, 'b n d -> (b t) n d', t=1 + self.top_n)
                all_patches = repeat(patches, 'b n d -> (b t) n d', t=1 + self.top_n)
                all_tokens = torch.cat((all_rel_tokens, all_sep_token, all_patches), dim=1)
                return all_tokens
            else:
                # Fallback to test mode if top_n_candidate is None (should not happen in training)
                tokens = torch.cat((rel_tokens_weighted, sep_token, patches), dim=1)
                return tokens
        else:
            # Test mode
            tokens = torch.cat((rel_tokens_weighted, sep_token, patches), dim=1)
            return tokens
