# ste_fake_quant_demo.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Fake Quant with STE (autograd.Function) --------
class FakeQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits, symmetric, per_channel, ch_axis, eps):
        # 计算 scale/zero_point（简单版：逐次前向用min/max估计）
        if per_channel:
            # 按通道统计 min/max
            reduce_dims = [d for d in range(x.dim()) if d != ch_axis]
            x_min = x.amin(dim=reduce_dims, keepdim=True)
            x_max = x.amax(dim=reduce_dims, keepdim=True)
        else:
            x_min = x.min()
            x_max = x.max()

        if symmetric:
            # 对称量化：零点为0
            qmax = 2 ** (num_bits - 1) - 1
            m = torch.maximum(x_max.abs(), x_min.abs())
            scale = torch.maximum(m / qmax, torch.tensor(eps, device=x.device, dtype=x.dtype))
            zero_point = 0.0
            x_clamp_min = -qmax * scale
            x_clamp_max =  qmax * scale
        else:
            # 非对称：映射到 [0, 2^b-1]
            qmin, qmax = 0, 2 ** num_bits - 1
            scale = torch.maximum((x_max - x_min) / max(qmax - qmin, 1), torch.tensor(eps, device=x.device, dtype=x.dtype))
            zero_point = torch.clamp(torch.round(qmin - x_min / scale), qmin, qmax)
            x_clamp_min = (0.0 - zero_point) * scale
            x_clamp_max = (qmax - zero_point) * scale

        # 保存范围用于反向时做简单截断（可选）
        ctx.save_for_backward(x, x_clamp_min, x_clamp_max)
        ctx.num_bits = num_bits
        ctx.symmetric = symmetric
        ctx.per_channel = per_channel
        ctx.ch_axis = ch_axis
        ctx.eps = eps

        # 量化 -> 反量化
        if symmetric:
            q = torch.round(x / scale)
            q = torch.clamp(q, -(2**(num_bits-1)), 2**(num_bits-1)-1)
            x_hat = q * scale
        else:
            q = torch.round(x / scale + zero_point)
            q = torch.clamp(q, 0, 2**num_bits - 1)
            x_hat = (q - zero_point) * scale

        return x_hat

    @staticmethod
    def backward(ctx, grad_output):
        # STE：忽略round的梯度，近似 d/dx (x) = 1
        x, x_clamp_min, x_clamp_max = ctx.saved_tensors
        # 在可表示范围内直通，范围外可选择截断为0（常见简化）
        pass_through = (x >= x_clamp_min) & (x <= x_clamp_max)
        grad_input = grad_output * pass_through.to(grad_output.dtype)
        # 非张量参数的梯度为 None
        return grad_input, None, None, None, None, None

# -------- 便捷模块封装 --------
class FakeQuantize(nn.Module):
    def __init__(self, num_bits=8, symmetric=True, per_channel=False, ch_axis=1, eps=1e-8):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.eps = eps

    def forward(self, x):
        return FakeQuantizeFunction.apply(
            x, self.num_bits, self.symmetric, self.per_channel, self.ch_axis, self.eps
        )

# 将权重也做伪量化（常见做法：前向时对权重做伪量化）
class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 w_num_bits=8, a_num_bits=8, symmetric=True):
        super().__init__(in_features, out_features, bias=bias)
        self.w_fake = FakeQuantize(num_bits=w_num_bits, symmetric=symmetric, per_channel=True, ch_axis=0)
        self.a_fake = FakeQuantize(num_bits=a_num_bits, symmetric=symmetric, per_channel=False)

    def forward(self, x):
        # 激活伪量化（前一层输出）
        x_q = self.a_fake(x)
        # 权重伪量化（对 weight 做 per-channel，对应 out_features 维度）
        w_q = self.w_fake(self.weight)
        b = self.bias
        return F.linear(x_q, w_q, b)

# -------- 一个极简可跑的训练 demo --------
def synthetic_regression(n=1024, in_dim=16, out_dim=1, noise=0.1, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, in_dim, generator=g)
    w_true = torch.randn(in_dim, out_dim, generator=g)
    y = X @ w_true + noise * torch.randn(n, out_dim, generator=g)
    return X.to(device), y.to(device)

class TinyQATNet(nn.Module):
    def __init__(self, in_dim=16, hidden=32, out_dim=1):
        super().__init__()
        self.fc1 = QuantLinear(in_dim, hidden, w_num_bits=8, a_num_bits=8, symmetric=True)
        self.act = nn.ReLU()
        self.fc2 = QuantLinear(hidden, out_dim, w_num_bits=8, a_num_bits=8, symmetric=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    X, y = synthetic_regression(n=2048, in_dim=16, out_dim=1, noise=0.1, seed=123, device=device)
    model = TinyQATNet(in_dim=16, hidden=32, out_dim=1).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for step in range(1, 301):
        # 小批量
        idx = torch.randint(0, X.size(0), (128,), device=device)
        xb, yb = X[idx], y[idx]

        pred = model(xb)
        loss = loss_fn(pred, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            with torch.no_grad():
                mse_all = loss_fn(model(X), y).item()
            print(f"step {step:4d} | batch_loss={loss.item():.4f} | full_MSE={mse_all:.4f}")

    # 简单对比：关闭伪量化看一下（推理阶段常用做法）
    model.eval()
    print("\nDisable fake-quant (eval) and re-evaluate:")
    def disable_fake(m):
        if isinstance(m, FakeQuantize):
            # 用恒等映射替换（最简单方式：注册一个不做事的forward）
            m.forward = lambda x: x
    model.apply(disable_fake)
    with torch.no_grad():
        mse_no_fake = loss_fn(model(X), y).item()
    print(f"Full-data MSE without fake-quant: {mse_no_fake:.4f}")

if __name__ == "__main__":
    main()
