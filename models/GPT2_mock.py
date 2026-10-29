"""纯 PyTorch 实现的 mock GPT-2，用于在没有预训练权重时跑通软测量流程。

为什么需要它
------------
本项目的 conda `pytorch` 环境没有安装 `transformers`，也无法联网下载
`openai-community/gpt2` 权重。为了能先调试 `models/LLM4SS.py` 的网络结构、
attention mask 和冻结策略，这里提供一个结构与接口都对齐真实 GPT-2 的 mock。

与真实 ``transformers.GPT2Model`` 的对应关系
--------------------------------------------
- 结构：``wte`` / ``wpe`` / ``h[]`` / ``ln_f``，每个 block 内为
  ``ln_1`` -> ``attn(c_attn, c_proj)`` -> ``ln_2`` -> ``mlp(c_fc, c_proj)``，
  与 GPT-2 命名一致，便于沿用同一套参数冻结代码。
- 接口：``forward(inputs_embeds=..., attention_mask=..., return_dict=True)``，
  返回对象带 ``last_hidden_state``。
- **形状**：输入 ``inputs_embeds: [B, L, hidden]``，输出
  ``last_hidden_state: [B, L, hidden]``，输入输出形状完全相同（forward 内会断言）。
- 差异：权重随机初始化（这就是 "mock" 的含义），默认层数更少、跑得更快；
  不支持 generation / KV cache / 多卡切分，只会忽略 ``use_cache``、``head_mask`` 等参数。

换回真实 GPT-2：安装 ``transformers`` 后把 ``use_mock_gpt2`` 设为 ``False`` 即可，
调用方（``models/LLM4SS.py``）不需要改其他代码。
"""

from __future__ import annotations

import hashlib
import math
import re
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MockGPT2Config",
    "GPT2Model_mock",
    "MockGPT2Tokenizer",
    "MockGPT2Output",
    "build_gpt2_backbone",
    "build_tokenizer",
]


# ---------------------------------------------------------------------------
# 配置：同时提供 HF GPT-2 的短名（n_embd/n_layer...）与长名（hidden_size/...）
# ---------------------------------------------------------------------------
class MockGPT2Config:
    """最小可用的 GPT-2 配置，字段名与 HF ``GPT2Config`` 对齐。"""

    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 2,
        num_heads: int = 12,
        max_position_embeddings: int = 1024,
        vocab_size: int = 50257,
        n_inner: Optional[int] = None,
        dropout: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_epsilon: float = 1e-5,
        attention_dropout: Optional[float] = None,
        **extra,
    ) -> None:
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} 必须能被 num_heads={num_heads} 整除"
            )
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_layers)
        self.num_attention_heads = int(num_heads)
        self.max_position_embeddings = int(max_position_embeddings)
        self.vocab_size = int(vocab_size)
        self.n_inner = int(n_inner) if n_inner is not None else 4 * self.hidden_size
        self.dropout = float(dropout)
        self.initializer_range = float(initializer_range)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        # mock 默认不做 attention dropout 的额外区分，统一用 dropout
        self.attention_dropout = (
            float(attention_dropout) if attention_dropout is not None else self.dropout
        )
        # HF 里的短名别名
        self.n_embd = self.hidden_size
        self.n_layer = self.num_hidden_layers
        self.n_head = self.num_attention_heads
        self.n_positions = self.max_position_embeddings
        # forward 的默认开关
        self.use_cache = False
        self.output_attentions = False
        self.output_hidden_states = False
        self.return_dict = True
        self.torch_dtype = None
        self.activation_function = "gelu_new"
        for key, value in extra.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return (
            f"MockGPT2Config(hidden_size={self.hidden_size}, "
            f"num_layers={self.num_hidden_layers}, num_heads={self.num_attention_heads}, "
            f"max_position_embeddings={self.max_position_embeddings}, "
            f"vocab_size={self.vocab_size})"
        )


# ---------------------------------------------------------------------------
# 输出容器：兼容 ``outputs.last_hidden_state`` 与 ``outputs[0]`` 两种取法
# ---------------------------------------------------------------------------
class MockGPT2Output:
    """对齐 ``BaseModelOutputWithPastAndCrossAttentions`` 的最小返回对象。"""

    def __init__(
        self,
        last_hidden_state: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        hidden_states: Optional[Tuple[torch.Tensor, ...]] = None,
        attentions: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> None:
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions

    def to_tuple(self) -> Tuple:
        return tuple(
            value
            for value in (
                self.last_hidden_state,
                self.past_key_values,
                self.hidden_states,
                self.attentions,
            )
            if value is not None
        )

    def __getitem__(self, index: int):
        return self.to_tuple()[index]

    def __len__(self) -> int:
        return len(self.to_tuple())


# ---------------------------------------------------------------------------
# 组件：因果自注意力 / MLP / Block
# ---------------------------------------------------------------------------
class _MockCausalSelfAttention(nn.Module):
    """标准 GPT-2 多头因果自注意力（支持 key padding mask）。"""

    def __init__(self, config: MockGPT2Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, hidden = x.shape
        query, key, value = self.c_attn(x).split(hidden, dim=2)
        # [B, L, D] -> [B, heads, L, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积 + 因果掩码：位置 i 只能看到 <= i 的位置
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=attn_weights.device
        ).tril()
        attn_weights = attn_weights.masked_fill(
            ~causal_mask, torch.finfo(attn_weights.dtype).min
        )
        # padding 位置（attention_mask==0）不参与 attention
        if attention_mask is not None:
            keep = attention_mask.view(batch_size, 1, 1, seq_len).to(torch.bool).expand(
                -1, -1, seq_len, -1
            )
            attn_weights = attn_weights.masked_fill(
                ~keep, torch.finfo(attn_weights.dtype).min
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)          # [B, heads, L, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden)
        return self.resid_dropout(self.c_proj(attn_output))


class _MockMLP(nn.Module):
    """GPT-2 的 4x 扩张两层 MLP（GELU）。"""

    def __init__(self, config: MockGPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class _MockGPT2Block(nn.Module):
    """pre-LN Transformer block，与 GPT-2 一致。"""

    def __init__(self, config: MockGPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = _MockCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = _MockMLP(config)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------
class GPT2Model_mock(nn.Module):
    """形状与接口对齐 ``transformers.GPT2Model`` 的本地 mock。

    输入 ``inputs_embeds: [B, L, hidden]``，输出 ``last_hidden_state: [B, L, hidden]``，
    两者形状严格相同。
    """

    def __init__(self, config: Optional[MockGPT2Config] = None) -> None:
        super().__init__()
        self.config = config if config is not None else MockGPT2Config()

        self.wte = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.wpe = nn.Embedding(
            self.config.max_position_embeddings, self.config.hidden_size
        )
        self.drop = nn.Dropout(self.config.dropout)
        self.h = nn.ModuleList(
            [_MockGPT2Block(self.config) for _ in range(self.config.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_epsilon)

        self.apply(self._init_weights)
        # GPT-2 的残差投影按层数缩放，训练更稳
        std = self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and name.endswith("c_proj"):
                module.weight.data.normal_(mean=0.0, std=std)
        if getattr(self.wte, "padding_idx", None) is not None:
            self.wte.weight.data[self.wte.padding_idx].zero_()

    # ---- 构造 -------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_name_or_path: Optional[str] = None, config: Optional[MockGPT2Config] = None, **config_kwargs) -> "GPT2Model_mock":
        """仿 `GPT2Model.from_pretrained` 的签名，但**不读取任何权重**。

        参数 ``model_name_or_path`` 只为保持调用方代码不变而接受（会被忽略）。
        其余关键字参数用于构造 ``MockGPT2Config``，例如
        ``GPT2Model_mock.from_pretrained(hidden_size=256, num_layers=2)``。
        """
        if model_name_or_path is not None:
            warnings.warn(
                "GPT2Model_mock 使用随机初始化权重，忽略预训练权重 "
                f"'{model_name_or_path}'；装好 transformers 后请改用真实 GPT-2。",
                stacklevel=2,
            )
        if config is None:
            config = MockGPT2Config(**config_kwargs)
        return cls(config)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # ---- 工具 -------------------------------------------------------------
    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def max_position_embeddings(self) -> int:
        return self.config.max_position_embeddings

    # ---- 与 HF 模型一致的embedding取用接口 --------------------------------
    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.wte = new_embeddings

    # ---- 前向 -------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MockGPT2Output, Tuple[torch.Tensor, ...]]:
        """``inputs_embeds: [B, L, hidden]`` -> ``last_hidden_state: [B, L, hidden]``。"""
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("input_ids 与 inputs_embeds 不能同时提供")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("必须提供 input_ids 或 inputs_embeds 之一")
            inputs_embeds = self.wte(input_ids)
        if inputs_embeds.is_floating_point() is False:
            raise TypeError("inputs_embeds 必须是浮点张量")

        batch_size, seq_len, hidden = inputs_embeds.shape
        if hidden != self.config.hidden_size:
            raise ValueError(
                f"inputs_embeds 的最后一维应为 hidden_size={self.config.hidden_size}，"
                f"实际为 {hidden}"
            )
        if seq_len < 1:
            raise ValueError("序列长度必须 >= 1")
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"序列长度 {seq_len} 超过最大位置数 {self.config.max_position_embeddings}；"
                "请缩短输入或增大 max_position_embeddings"
            )

        # 位置编码：mock 只支持从头开始的连续位置（不做 KV cache 增量）
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        if position_embeds.shape[1] != seq_len:
            raise ValueError("position_ids 长度必须与序列长度一致")

        hidden_states = self.drop(inputs_embeds + position_embeds)
        if token_type_ids is not None:
            hidden_states = hidden_states + self.wte(token_type_ids)

        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            if attention_mask.shape[1] != seq_len:
                raise ValueError(
                    f"attention_mask 长度 {attention_mask.shape[1]} 与序列长度 {seq_len} 不一致"
                )

        collect_states = bool(output_hidden_states)
        collect_attentions = bool(output_attentions)
        all_hidden_states: List[torch.Tensor] = []
        all_attentions: List[torch.Tensor] = []

        for block in self.h:
            if collect_states:
                all_hidden_states.append(hidden_states)
            hidden_states = block(hidden_states, attention_mask)

        hidden_states = self.ln_f(hidden_states)

        # 核心约定：输出与输入形状严格相同
        if hidden_states.shape != inputs_embeds.shape:
            raise RuntimeError(
                f"mock GPT-2 输出形状 {tuple(hidden_states.shape)} 与输入形状 "
                f"{tuple(inputs_embeds.shape)} 不一致"
            )
        if collect_states:
            all_hidden_states.append(hidden_states)

        return_dict = self.config.return_dict if return_dict is None else return_dict
        if not return_dict:
            return tuple(
                value
                for value in (
                    hidden_states,
                    None,  # past_key_values：mock 不支持
                    tuple(all_hidden_states) if collect_states else None,
                    tuple(all_attentions) if collect_attentions else None,
                )
                if value is not None
            )

        return MockGPT2Output(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=tuple(all_hidden_states) if collect_states else None,
            attentions=tuple(all_attentions) if collect_attentions else None,
        )


# ---------------------------------------------------------------------------
# 分词器 mock：接口对齐 HF tokenizer，但按哈希确定性地生成 id
# ---------------------------------------------------------------------------
class MockGPT2Tokenizer:
    """模仿 ``transformers.AutoTokenizer`` 调用约定的离线分词器。

    只做 ``空格切分 -> 稳定哈希取模 -> 截断/补齐``，不加载任何词表文件；
    同一个字符串永远映射到同一个 id，保证多次运行结果可复现。
    """

    def __init__(self, vocab_size: int = 50257, max_length: Optional[int] = None) -> None:
        self.vocab_size = int(vocab_size)
        self.max_length = max_length
        self.pad_token = "<|pad|>"
        self.eos_token = "<|endoftext|>"
        self.bos_token = self.eos_token
        self.unk_token = "<|unk|>"
        self.pad_token_id = self._token_to_id(self.pad_token)
        self.eos_token_id = self._token_to_id(self.eos_token)
        self.bos_token_id = self.eos_token_id

    @classmethod
    def from_pretrained(cls, model_name_or_path: Optional[str] = None, vocab_size: int = 50257, **kwargs) -> "MockGPT2Tokenizer":
        """仿 ``AutoTokenizer.from_pretrained`` 的签名，忽略模型名与本地词表。"""
        if model_name_or_path is not None:
            warnings.warn(
                "MockGPT2Tokenizer 不加载真实词表，忽略 "
                f"'{model_name_or_path}'；装好 transformers 后请改用真实 tokenizer。",
                stacklevel=2,
            )
        return cls(vocab_size=vocab_size, **kwargs)

    def _token_to_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.vocab_size

    def tokenize(self, text: str) -> List[int]:
        tokens = re.findall(r"\S+", text)
        return [self._token_to_id(token) for token in tokens]

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        ids = self.tokenize(text)
        limit = max_length if max_length is not None else self.max_length
        if limit is not None:
            ids = ids[:limit]
        return ids

    def __call__(
        self,
        texts: Union[str, Sequence[str]],
        padding: Union[bool, str] = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> dict:
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        limit = max_length if max_length is not None else self.max_length

        encoded = [self.tokenize(text) for text in texts]
        if truncation and limit is not None:
            encoded = [ids[:limit] for ids in encoded]
        if not encoded:
            raise ValueError("至少需要一条文本")

        target_len = max(len(ids) for ids in encoded)
        if padding == "max_length" and limit is not None:
            target_len = limit
        if padding is False:
            if len({len(ids) for ids in encoded}) > 1 and return_tensors is not None:
                raise ValueError("padding=False 时批次内长度必须一致")
            target_len = max(len(ids) for ids in encoded)

        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        for ids in encoded:
            if len(ids) > target_len:
                raise ValueError(
                    f"样本长度 {len(ids)} 超过目标长度 {target_len}，请开启 truncation"
                )
            pad_count = target_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_count)
            attention_mask.append([1] * len(ids) + [0] * pad_count)

        result = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_tensors == "pt":
            result = {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}
        return result


# ---------------------------------------------------------------------------
# 统一入口：mock / 真实 GPT-2 一行切换
# ---------------------------------------------------------------------------
def build_gpt2_backbone(
    model_name: str = "openai-community/gpt2",
    use_mock: bool = True,
    **mock_config_kwargs,
) -> nn.Module:
    """构造 GPT-2 backbone（默认返回 mock，离线可用）。

    Parameters
    ----------
    model_name:
        真实 GPT-2 的名称或本地路径；mock 模式下会被忽略。
    use_mock:
        ``True`` 使用 ``GPT2Model_mock``（随机权重）；``False`` 需要已安装
        ``transformers`` 并能够访问/已缓存对应权重。
    **mock_config_kwargs:
        mock 模式下的 ``MockGPT2Config`` 覆盖项，如 ``hidden_size=256``。
    """
    if use_mock:
        return GPT2Model_mock.from_pretrained(model_name, **mock_config_kwargs)

    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise ImportError(
            "use_mock=False 需要安装 transformers，或先使用 use_mock=True 的 mock 版本"
        ) from exc
    return GPT2Model.from_pretrained(model_name)


def build_tokenizer(
    model_name: str = "openai-community/gpt2",
    use_mock: bool = True,
    vocab_size: Optional[int] = None,
):
    """构造与 GPT-2 配套的 tokenizer（默认返回离线 mock）。"""
    if use_mock:
        return MockGPT2Tokenizer.from_pretrained(
            model_name, vocab_size=vocab_size if vocab_size is not None else 50257
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise ImportError(
            "use_mock=False 需要安装 transformers，或先使用 use_mock=True 的 mock 版本"
        ) from exc
    return AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    # 自检：形状、mask、梯度、以及和 LLM4SS 的衔接
    torch.manual_seed(0)
    config = MockGPT2Config(hidden_size=64, num_layers=2, num_heads=4, vocab_size=1024)
    backbone = GPT2Model_mock(config)

    x = torch.randn(2, 7, config.hidden_size, requires_grad=True)
    mask = torch.ones(2, 7)
    mask[0, -2:] = 0  # 模拟 padding
    out = backbone(inputs_embeds=x, attention_mask=mask, use_cache=False, return_dict=True)

    print("config:", config)
    print("input :", tuple(x.shape))
    print("output:", tuple(out.last_hidden_state.shape))
    assert out.last_hidden_state.shape == x.shape, "输入输出形状必须一致"

    loss = out.last_hidden_state.sum()
    loss.backward()
    # inputs_embeds 路径不会经过 wte，因此 wte 没有梯度是正常的
    unused = {name for name, p in backbone.named_parameters() if p.grad is None}
    print("grad ok:", x.grad is not None, "| 无梯度参数（应为空或仅 wte）:", unused or "无")

    tokenizer = MockGPT2Tokenizer(vocab_size=config.vocab_size)
    batch = tokenizer(["hello world", "a much longer prompt here"], padding=True,
                      truncation=True, max_length=8, return_tensors="pt")
    print("tokenized input_ids:", tuple(batch["input_ids"].shape),
          "| attention_mask:", batch["attention_mask"].tolist())
    with torch.no_grad():
        tokens_out = backbone(input_ids=batch["input_ids"],
                              attention_mask=batch["attention_mask"])
    print("token path output:", tuple(tokens_out.last_hidden_state.shape))
    print("mock GPT-2 self-check passed.")
