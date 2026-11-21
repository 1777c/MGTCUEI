import math
import torch
from torch import nn
from .utils import ContractError


def rms(x):
    # 缩放范数可避免很大的有限 FP32 值溢出。vector_norm 在零点定义有限次梯度，贡献路径保留梯度。
    x = x.float()
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1.0)
    return scale * (torch.linalg.vector_norm(x / scale, dim=-1, keepdim=True) / math.sqrt(x.shape[-1]))


class ResidualGate(nn.Module):
    def __init__(self, mode="fixed", value=0.01, epsilon=1e-6):
        super().__init__()
        self.mode = mode
        self.epsilon = epsilon
        if mode not in ("unit", "fixed", "learned") or not 0 < value < 1 or epsilon <= 0:
            raise ContractError("gate configuration")
        self.register_buffer("fixed", torch.tensor(1.0 if mode == "unit" else value, dtype=torch.float32))
        if mode == "learned":
            # 中心化 logit 与同一 sigmoid 门控代数等价。以 FP64 计算固定偏移，可令初始有效 FP32 lambda 与 Fixed 按位相同，可训练增量仍为 FP32。
            self.register_buffer(
                "initial_logit", torch.tensor(math.log(value / (1 - value)), dtype=torch.float64)
            )
            self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.last = {}

    def scale(self):
        return (
            (self.initial_logit + self.alpha.double()).sigmoid().float()
            if self.mode == "learned"
            else self.fixed
        )

    def forward(self, h, c, mask):
        if (
            h.shape != c.shape
            or mask.shape != h.shape[:2]
            or not torch.isfinite(h).all()
            or not torch.isfinite(c).all()
        ):
            raise ContractError("residual input schema/finite")
        with torch.autocast(device_type=h.device.type, enabled=False):
            rh = rms(h).detach()
            rc = rms(c)
            delta = self.scale() * rh * (c.float() / rc.clamp_min(self.epsilon))
            delta = torch.where(mask[:, :, None].bool(), delta, torch.zeros_like(delta))
            y = (h.float() + delta).to(h.dtype)
            actual = y.float() - h.float()
            valid = mask.bool()
            with torch.no_grad():
                self.last = {
                    "lambda": float(self.scale()),
                    "hidden_rms": float(rh.squeeze(-1)[valid].mean()),
                    "delta_rms": float(rms(delta).squeeze(-1)[valid].mean()),
                    "delta_hidden_ratio": float(
                        (rms(delta) / rh.clamp_min(self.epsilon)).squeeze(-1)[valid].mean()
                    ),
                    "clamp_rate": float((rc.squeeze(-1)[valid] < self.epsilon).float().mean()),
                    "pre_cast_ratio": float(
                        (rms(delta) / rh.clamp_min(self.epsilon)).squeeze(-1)[valid].mean()
                    ),
                    "effective_delta_ratio": float(
                        (rms(actual) / rh.clamp_min(self.epsilon)).squeeze(-1)[valid].mean()
                    ),
                    "cosine": float(
                        torch.nn.functional.cosine_similarity(h.float(), actual, dim=-1)[valid].mean()
                    ),
                }
            return y
