"""M2 完整模型。M4/M5 直接使用本文件中你已完成的实现。"""

from dataclasses import asdict, dataclass

from torch import nn

from .attention import MultiHeadSelfAttention
from .layers import Embedding, Linear, RMSNorm, SwiGLU, silu


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    d_ff: int = 704
    rope_theta: float = 10000.0
    use_rope: bool = True
    use_norm: bool = True
    ffn_type: str = "swiglu"
    tie_embeddings: bool = False

    def __post_init__(self):
        for name in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if self.d_model % self.num_heads:
            raise ValueError("d_model 必须被 num_heads 整除")
        if self.use_rope and (self.d_model // self.num_heads) % 2:
            raise ValueError("使用 RoPE 时每个 head 的维度必须为偶数")
        if self.ffn_type not in ("swiglu", "silu"):
            raise ValueError("ffn_type 只能为 swiglu 或 silu")


class SiLUFeedForward(nn.Module):
    """M5 可选对照：两层 FFN；配置 d_ff=4*d_model 可近似匹配 SwiGLU 参数量。"""

    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x):
        return self.w2(silu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, device=None, dtype=None):
        super().__init__()
        c = config
        self.attn = MultiHeadSelfAttention(
            c.d_model, c.num_heads, c.context_length, c.rope_theta, c.use_rope, device, dtype
        )
        self.ln1 = RMSNorm(c.d_model, device=device, dtype=dtype) if c.use_norm else nn.Identity()
        self.ln2 = RMSNorm(c.d_model, device=device, dtype=dtype) if c.use_norm else nn.Identity()
        ffn_cls = SwiGLU if c.ffn_type == "swiglu" else SiLUFeedForward
        self.ffn = ffn_cls(c.d_model, c.d_ff, device, dtype)

    def forward(self, x, token_positions=None):
        """T10：两段 pre-norm 残差。输入和输出均为 [...,T,D]。

        第一段：y = x + attn(ln1(x), token_positions)
        第二段：在 y 上执行 ln2、ffn，再加回 y。
        """
        raise NotImplementedError("TODO T10: TransformerBlock.forward")


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig, device=None, dtype=None):
        super().__init__()
        self.config = config
        c = config
        self.token_embeddings = Embedding(c.vocab_size, c.d_model, device, dtype)
        self.layers = nn.ModuleList([TransformerBlock(c, device, dtype) for _ in range(c.num_layers)])
        self.ln_final = RMSNorm(c.d_model, device=device, dtype=dtype) if c.use_norm else nn.Identity()
        self.lm_head = Linear(c.d_model, c.vocab_size, device, dtype)
        if c.tie_embeddings:
            # 共享参数的初始尺度会影响优化；M5 对照中必须记录此初始化差异。
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, token_ids):
        if token_ids.shape[-1] > self.config.context_length or token_ids.shape[-1] == 0:
            raise ValueError("输入长度必须在 [1, context_length] 内")
        return self.predict(token_ids)

    def predict(self, token_ids):
        """T11：embedding → 顺序遍历 layers → ln_final → lm_head。

        token_ids [B,T] → logits [B,T,V]。这里不调用 softmax。
        每层内部会创建默认的位置索引。使用 for block in self.layers 即可。
        """
        raise NotImplementedError("TODO T11: TransformerLM.predict")

    def config_dict(self):
        return asdict(self.config)
