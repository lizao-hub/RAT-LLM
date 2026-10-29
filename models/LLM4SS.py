"""GPT-2-based soft sensors: strict channel-independent vs. encode-then-fuse.

Input
-----
x: float tensor [batch, lookback_length, num_channels]

Output
------
y_hat: float tensor [batch, target_dim]

The model uses a fixed natural-language process prompt as a prefix, then appends
patch embeddings for one numerical channel. GPT-2 processes each channel in a
separate batch item, so no channel can attend to another channel inside GPT-2.

Model A predicts the same quality target independently from each channel and
averages those predictions with a fixed arithmetic mean (no learned cross-channel
feature fusion). Model B first obtains the same per-channel GPT-2 representations,
then uses a trainable Transformer over the channel dimension before prediction.

This is a GPT4MTS-inspired adaptation for supervised soft sensing, not an exact
reproduction of the original GPT4MTS forecasting objective.

Notes
-----
当前 conda `pytorch` 环境没有安装 transformers，也无法下载 GPT-2 预训练权重，
因此默认使用 `models/GPT2_mock.py` 中的 mock GPT-2（随机初始化权重、结构一致）。
mock 的输入输出形状保持相同：`inputs_embeds [B, L, D] -> last_hidden_state [B, L, D]`。
装好 transformers 且能访问权重后，把 `use_mock_gpt2=False` 即可切回真实 GPT-2。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # 作为包导入（python run.py，模型经由 exp 层加载）
    from .GPT2_mock import build_gpt2_backbone, build_tokenizer
except ImportError:  # 直接运行本文件（python models/LLM4SS.py，sys.path 为 models/）
    from GPT2_mock import build_gpt2_backbone, build_tokenizer


class GPT2PerChannelEncoder(nn.Module):
    """Shared GPT-2 encoder applied independently to every numerical channel."""

    def __init__(
        self,
        num_channels: int,
        channel_names: Optional[Sequence[str]] = None,
        process_prompt: str = (
            "Industrial soft sensing task. Estimate the target quality variable "
            "from historical process measurements. Use the stated process context "
            "as background knowledge; do not invent unprovided causal relationships."
        ),
        target_description: str = "the target quality variable",
        model_name: str = "openai-community/gpt2",
        patch_len: int = 16,
        stride: int = 8,
        max_prompt_tokens: int = 96,
        freeze_transformer_blocks: bool = True,
        dropout: float = 0.1,
        use_mock_gpt2: bool = True,
        gpt2_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError("num_channels must be >= 1")
        if patch_len < 1 or stride < 1:
            raise ValueError("patch_len and stride must be >= 1")

        if channel_names is None:
            channel_names = [f"channel_{i}" for i in range(num_channels)]
        if len(channel_names) != num_channels:
            raise ValueError(
                f"channel_names has {len(channel_names)} entries, "
                f"but num_channels={num_channels}"
            )

        self.num_channels = num_channels
        self.patch_len = patch_len
        self.stride = stride
        # mock 模式：随机初始化的本地 GPT-2，无需预训练权重与 transformers。
        self.gpt2 = build_gpt2_backbone(
            model_name, use_mock=use_mock_gpt2, **(gpt2_kwargs or {})
        )
        self.d_model = int(self.gpt2.config.hidden_size)
        self.max_positions = int(self.gpt2.config.max_position_embeddings)

        # GPT-2 本身没有 PAD token；这里用 EOS 作为 padding token（mock tokenizer 自带 PAD）。
        self.tokenizer = build_tokenizer(
            model_name,
            use_mock=use_mock_gpt2,
            vocab_size=int(self.gpt2.config.vocab_size),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        prompts = []
        for name in channel_names:
            prompts.append(
                f"{process_prompt}\n"
                f"Target quality variable: {target_description}.\n"
                f"Current input variable: {name}.\n"
                "Estimate the target quality from the historical values of this "
                "input variable and the static process context. This encoder sees "
                "only this channel's numerical observations."
            )

        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
            return_tensors="pt",
        )
        # Buffers move with model.to(device), but are re-created from the prompt
        # when a model is instantiated, so they need not be stored in state_dict.
        self.register_buffer("prompt_input_ids", tokenized["input_ids"], persistent=False)
        self.register_buffer(
            "prompt_attention_mask", tokenized["attention_mask"], persistent=False
        )

        self.patch_projection = nn.Linear(patch_len, self.d_model)
        self.patch_dropout = nn.Dropout(dropout)

        if freeze_transformer_blocks:
            # GPT4MTS-style parameter-efficient setting: freeze GPT-2 parameters,
            # then train its absolute position embeddings and LayerNorms. The
            # numerical patch projection remains trainable in either setting.
            for p in self.gpt2.parameters():
                p.requires_grad = False
            for p in self.gpt2.wpe.parameters():
                p.requires_grad = True
            for module in self.gpt2.modules():
                if isinstance(module, nn.LayerNorm):
                    for p in module.parameters():
                        p.requires_grad = True
        else:
            for p in self.gpt2.parameters():
                p.requires_grad = True

    def _to_patches(self, x_channel: torch.Tensor) -> torch.Tensor:
        """Convert [B*M, L] into [B*M, num_patches, patch_len]."""
        length = x_channel.shape[-1]
        if length <= self.patch_len:
            left_pad = self.patch_len - length
        else:
            # Left-pad just enough so that the final patch ends at the latest
            # observed timestamp; this avoids dropping the most recent values.
            remainder = (length - self.patch_len) % self.stride
            left_pad = (self.stride - remainder) % self.stride
        if left_pad:
            x_channel = F.pad(x_channel, (left_pad, 0))
        return x_channel.unfold(dimension=1, size=self.patch_len, step=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one GPT-2 hidden vector per channel: [B, M, d_model]."""
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [batch, lookback_length, num_channels], "
                f"got shape={tuple(x.shape)}"
            )
        batch_size, lookback, num_channels = x.shape
        if num_channels != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {num_channels}"
            )
        if lookback < 1:
            raise ValueError("lookback_length must be >= 1")

        # [B, L, M] -> [B, M, L] -> [B*M, L]. Each channel becomes an
        # independent batch item; attention never mixes different channels.
        x_channel = x.transpose(1, 2).contiguous().reshape(batch_size * num_channels, lookback)
        patches = self._to_patches(x_channel)                     # [B*M, N, patch_len]
        patch_embeds = self.patch_dropout(self.patch_projection(patches))  # [B*M, N, D]
        num_patches = patch_embeds.shape[1]

        prompt_len = self.prompt_input_ids.shape[1]
        if prompt_len + num_patches > self.max_positions:
            raise ValueError(
                f"Prompt tokens ({prompt_len}) + time patches ({num_patches}) = "
                f"{prompt_len + num_patches}, exceeding GPT-2 position limit "
                f"{self.max_positions}. Reduce max_prompt_tokens, patch count, "
                "or lookback length / stride."
            )

        # Repeat each channel's semantic prompt for every sample in the batch.
        # Ordering matches x_channel: [sample0/channel0..M, sample1/channel0..M, ...].
        prompt_ids = (
            self.prompt_input_ids.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .reshape(batch_size * num_channels, prompt_len)
        )
        prompt_mask = (
            self.prompt_attention_mask.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .reshape(batch_size * num_channels, prompt_len)
        )
        text_embeds = self.gpt2.wte(prompt_ids)
        inputs_embeds = torch.cat([text_embeds, patch_embeds], dim=1)

        patch_mask = torch.ones(
            batch_size * num_channels,
            num_patches,
            dtype=prompt_mask.dtype,
            device=prompt_mask.device,
        )
        attention_mask = torch.cat([prompt_mask, patch_mask], dim=1)

        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        # The final temporal-patch token can attend to all earlier prompt and
        # temporal-patch tokens under GPT-2's causal attention mask.
        channel_hidden = outputs.last_hidden_state[:, -1, :]  # [B*M, D]
        return channel_hidden.reshape(batch_size, num_channels, self.d_model)


class GPT2StrictChannelIndependentSoftSensor(nn.Module):
    """方案 A: 每通道独立估计同一质量指标，再用固定均值聚合。

    No trainable operation mixes features from different channels. The fixed
    arithmetic mean combines only the per-channel scalar/vector predictions.
    """

    def __init__(
        self,
        num_channels: int,
        target_dim: int = 1,
        channel_names: Optional[Sequence[str]] = None,
        process_prompt: str = (
            "Industrial soft sensing task. Estimate the target quality variable "
            "from historical process measurements. Use the stated process context "
            "as background knowledge; do not invent unprovided causal relationships."
        ),
        target_description: str = "the target quality variable",
        model_name: str = "openai-community/gpt2",
        patch_len: int = 16,
        stride: int = 8,
        max_prompt_tokens: int = 96,
        freeze_transformer_blocks: bool = True,
        dropout: float = 0.1,
        use_mock_gpt2: bool = True,
        gpt2_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.encoder = GPT2PerChannelEncoder(
            num_channels=num_channels,
            channel_names=channel_names,
            process_prompt=process_prompt,
            target_description=target_description,
            model_name=model_name,
            patch_len=patch_len,
            stride=stride,
            max_prompt_tokens=max_prompt_tokens,
            freeze_transformer_blocks=freeze_transformer_blocks,
            dropout=dropout,
            use_mock_gpt2=use_mock_gpt2,
            gpt2_kwargs=gpt2_kwargs,
        )
        d_model = self.encoder.d_model
        # Shared prediction head: same mapping is applied to every channel.
        self.channel_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, target_dim),
        )

    def forward(
        self, x: torch.Tensor, return_per_channel: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h = self.encoder(x)                          # [B, M, D]
        y_each = self.channel_head(h)                # [B, M, target_dim]
        y_hat = y_each.mean(dim=1)                   # fixed, non-learned mean
        if return_per_channel:
            return y_hat, y_each
        return y_hat


class GPT2EncodeThenFuseSoftSensor(nn.Module):
    """方案 B: 各通道独立经 GPT-2 编码，再跨通道融合隐藏状态预测质量。"""

    def __init__(
        self,
        num_channels: int,
        target_dim: int = 1,
        channel_names: Optional[Sequence[str]] = None,
        process_prompt: str = (
            "Industrial soft sensing task. Estimate the target quality variable "
            "from historical process measurements. Use the stated process context "
            "as background knowledge; do not invent unprovided causal relationships."
        ),
        target_description: str = "the target quality variable",
        model_name: str = "openai-community/gpt2",
        patch_len: int = 16,
        stride: int = 8,
        max_prompt_tokens: int = 96,
        freeze_transformer_blocks: bool = True,
        dropout: float = 0.1,
        use_mock_gpt2: bool = True,
        gpt2_kwargs: Optional[dict] = None,
        fusion_layers: int = 1,
        fusion_heads: int = 8,
    ) -> None:
        super().__init__()
        if fusion_layers < 1:
            raise ValueError("fusion_layers must be >= 1")
        self.encoder = GPT2PerChannelEncoder(
            num_channels=num_channels,
            channel_names=channel_names,
            process_prompt=process_prompt,
            target_description=target_description,
            model_name=model_name,
            patch_len=patch_len,
            stride=stride,
            max_prompt_tokens=max_prompt_tokens,
            freeze_transformer_blocks=freeze_transformer_blocks,
            dropout=dropout,
            use_mock_gpt2=use_mock_gpt2,
            gpt2_kwargs=gpt2_kwargs,
        )
        d_model = self.encoder.d_model
        if d_model % fusion_heads != 0:
            raise ValueError(
                f"GPT-2 hidden size {d_model} must be divisible by fusion_heads={fusion_heads}"
            )

        # The channel embeddings [B, M, D] are now tokens in a new Transformer.
        # This is the explicit cross-channel interaction that Model A does not have.
        fusion_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=fusion_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.channel_fusion = nn.TransformerEncoder(
            fusion_block,
            num_layers=fusion_layers,
            enable_nested_tensor=False,
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, target_dim),
        )

    def forward(self, x: torch.Tensor, return_channel_hidden: bool = False):
        h_channel = self.encoder(x)                  # [B, M, D], independent encoding
        h_fused = self.channel_fusion(h_channel)     # [B, M, D], channel mixing starts here
        pooled = h_fused.mean(dim=1)                 # [B, D]
        y_hat = self.regressor(pooled)               # [B, target_dim]
        if return_channel_hidden:
            return y_hat, h_channel, h_fused
        return y_hat


def _pick_fusion_heads(desired: int, hidden: int) -> int:
    """在不超过 ``desired`` 的范围内，找能整除 ``hidden`` 的注意力头数。"""
    for heads in range(max(int(desired), 1), 0, -1):
        if hidden % heads == 0:
            return heads
    return 1


class Model(nn.Module):
    """把 ``run.py`` 的 args 适配成方案 A / 方案 B，供 exp 层统一调用。

    这样 ``exp/exp_soft_sensor.py`` 只依赖 ``Model(args)`` +
    ``forward(x) -> [B, target_dim]`` 这一约定，不需要知道 GPT-2 的细节。

    从 args 读取的参数
    ------------------
    - ``enc_in``:            输入通道数（必填）
    - ``target_dim``:        输出维度，默认 1
    - ``llm4ss_variant``:    ``'A'``（严格通道独立+均值）或 ``'B'``（先编码再融合）
    - ``patch_size`` / ``patch_stride``: 时序 patch 长度 / 步长
    - ``max_length``:        提示词最大 token 数
    - ``target``:            目标质量变量名，会写进提示词
    - ``channel_names``:     可选，各通道的工艺名称，会写进提示词
    - ``use_mock_gpt2``:     1=本地 mock GPT-2（默认，离线可跑），0=真实 GPT-2
    - ``gpt2_hidden`` / ``gpt2_layers`` / ``gpt2_heads``: mock GPT-2 尺寸（0=用默认）
    - ``n_heads`` / ``fusion_layers``: 方案 B 跨通道融合 Transformer 的头数 / 层数
    - ``dropout``
    """

    def __init__(self, configs) -> None:
        super().__init__()

        variant = str(getattr(configs, "llm4ss_variant", "A")).strip().upper()
        if variant not in ("A", "B"):
            raise ValueError(f"llm4ss_variant 只能是 'A' 或 'B'，收到 {variant!r}")

        num_channels = int(getattr(configs, "enc_in", 0) or 0)
        if num_channels < 1:
            raise ValueError("LLM4SS 需要 args.enc_in >= 1（输入通道数）")

        use_mock_gpt2 = bool(int(getattr(configs, "use_mock_gpt2", 1)))
        gpt2_kwargs = {}
        for arg_name, config_key in (
            ("gpt2_hidden", "hidden_size"),
            ("gpt2_layers", "num_layers"),
            ("gpt2_heads", "num_heads"),
        ):
            value = int(getattr(configs, arg_name, 0) or 0)
            if value > 0:
                gpt2_kwargs[config_key] = value

        common = dict(
            num_channels=num_channels,
            target_dim=int(getattr(configs, "target_dim", 1)),
            channel_names=getattr(configs, "channel_names", None),
            target_description=str(
                getattr(configs, "target", "the target quality variable")
            ),
            patch_len=int(getattr(configs, "patch_size", 16)),
            stride=int(getattr(configs, "patch_stride", 8)),
            max_prompt_tokens=int(getattr(configs, "max_length", 96)),
            freeze_transformer_blocks=bool(
                int(getattr(configs, "freeze_transformer_blocks", 1))
            ),
            dropout=float(getattr(configs, "dropout", 0.1)),
            use_mock_gpt2=use_mock_gpt2,
            gpt2_kwargs=gpt2_kwargs or None,
        )

        if variant == "A":
            self.model = GPT2StrictChannelIndependentSoftSensor(**common)
        else:
            # mock 尺寸已知；真实 GPT-2 按 768 估计，构造后会由编码器再校验
            hidden = int(gpt2_kwargs.get("hidden_size", 768))
            fusion_heads = _pick_fusion_heads(
                int(getattr(configs, "n_heads", 8) or 8), hidden
            )
            self.model = GPT2EncodeThenFuseSoftSensor(
                fusion_layers=int(getattr(configs, "fusion_layers", 1)),
                fusion_heads=fusion_heads,
                **common,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: [B, L, C]`` -> ``y_hat: [B, target_dim]``。"""
        return self.model(x)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device],
    grad_clip: Optional[float] = 1.0,
) -> float:
    """Minimal MSE training loop. Each batch should be (x, y)."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = F.mse_loss(prediction, y)
        loss.backward()
        if grad_clip is not None:
            trainable_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if trainable_params:
                nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_mse(
    model: nn.Module,
    loader,
    device: Union[str, torch.device],
) -> float:
    model.eval()
    total_squared_error = 0.0
    total_elements = 0
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        prediction = model(x)
        total_squared_error += F.mse_loss(prediction, y, reduction="sum").item()
        total_elements += y.numel()
    return total_squared_error / max(total_elements, 1)


if __name__ == "__main__":
    # Minimal shape smoke test. Runs fully offline on the mock GPT-2
    # (models/GPT2_mock.py); no pretrained weights are downloaded.
    # Replace these sample settings and the random data with your real dataset.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = [f"process_variable_{i + 1}" for i in range(10)]
    process_prompt = (
        "Process: chemical production. Target: product quality index. "
        "Known context: temperature, pressure and flow are process measurements. "
        "Only use the supplied history and this static process description."
    )
    x = torch.randn(4, 96, 10)  # [batch, lookback, channels]

    # Small mock backbone keeps this smoke test fast; drop gpt2_kwargs to use the
    # GPT-2-sized mock (hidden 768, 2 layers) or set use_mock_gpt2=False for real GPT-2.
    gpt2_kwargs = {"hidden_size": 128, "num_layers": 2, "num_heads": 4}

    model_a = GPT2StrictChannelIndependentSoftSensor(
        num_channels=10,
        target_dim=1,
        channel_names=names,
        process_prompt=process_prompt,
        target_description="product quality index",
        patch_len=16,
        stride=8,
        gpt2_kwargs=gpt2_kwargs,
    ).to(device)
    model_b = GPT2EncodeThenFuseSoftSensor(
        num_channels=10,
        target_dim=1,
        channel_names=names,
        process_prompt=process_prompt,
        target_description="product quality index",
        patch_len=16,
        stride=8,
        fusion_heads=8,
        gpt2_kwargs=gpt2_kwargs,
    ).to(device)

    with torch.no_grad():
        pred_a = model_a(x.to(device))
        pred_b = model_b(x.to(device))
    print("Model A prediction shape:", tuple(pred_a.shape))
    print("Model B prediction shape:", tuple(pred_b.shape))

    # Quick gradient check: the prediction heads must receive gradients.
    y = torch.randn(4, 1, device=device)
    loss = F.mse_loss(model_a(x.to(device)), y)
    loss.backward()
    print("Model A trainable params:",
          sum(p.numel() for p in model_a.parameters() if p.requires_grad))
    print("Model A head grad ok:", model_a.channel_head[2].weight.grad is not None)
