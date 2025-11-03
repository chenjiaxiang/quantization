import torch
from torch import nn
from torch.autograd import Function

# ---------------------------
# 工具：打印计算图（从某个 grad_fn 出发，追溯 next_functions）
# ---------------------------
def print_graph(fn, indent=0, seen=None):
    if fn is None:
        print(" " * indent + "None")
        return
    if seen is None:
        seen = set()
    if fn in seen:
        print(" " * indent + f"[{type(fn).__name__}] (visited)")
        return
    seen.add(fn)

    name = type(fn).__name__
    print(" " * indent + f"[{name}]")
    # next_functions: list of (Function or None, index)
    for i, (next_fn, _) in enumerate(fn.next_functions):
        prefix = " " * (indent + 2) + f"--> edge[{i}] "
        if next_fn is None:
            print(prefix + "None")
        else:
            print(prefix + f"{type(next_fn).__name__}")
            print_graph(next_fn, indent + 6, seen)

# ---------------------------
# 自定义一个 Function，用来在 forward/backward 打印细节
# 同时演示 save_for_backward（前向如何为反向保存中间量）
# ---------------------------
class DemoLinear(Function):
    @staticmethod
    def forward(ctx, x, W, b):
        # 保存给 backward 用的中间量
        ctx.save_for_backward(x, W)
        y = x @ W.t() + b  # 简单线性层
        print("\n[Forward] DemoLinear.forward")
        print(f"  x.requires_grad={x.requires_grad}, W.requires_grad={W.requires_grad}, b.requires_grad={b.requires_grad}")
        print(f"  y.shape={y.shape}")
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, W = ctx.saved_tensors
        # 线性层的标准梯度
        grad_x = grad_out @ W
        grad_W = grad_out.t() @ x
        grad_b = grad_out.sum(0)

        print("\n[Backward] DemoLinear.backward (executed)")
        print(f"  grad_out.shape={grad_out.shape}")
        print(f"  produces: grad_x({grad_x.shape}), grad_W({grad_W.shape}), grad_b({grad_b.shape})")
        return grad_x, grad_W, grad_b

# 便捷包装（像 nn.Linear 一样使用）
def demo_linear(x, W, b):
    return DemoLinear.apply(x, W, b)

# ---------------------------
# 小模型：Linear -> ReLU -> Mean
# 我们在关键张量上注册 backward hook，打印反向执行顺序
# ---------------------------
def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    B, In, Out = 4, 3, 2
    # 叶子张量（需要梯度）：x, W, b
    x = torch.randn(B, In, device=device, requires_grad=True)
    W = torch.randn(Out, In, device=device, requires_grad=True)
    b = torch.randn(Out, device=device, requires_grad=True)

    # 前向
    y_lin = demo_linear(x, W, b)            # 有 grad_fn: DemoLinearBackward
    y_relu = torch.relu(y_lin)              # 有 grad_fn: ReluBackward*
    z = y_relu.mean()                       # 有 grad_fn: MeanBackward*

    # -------- 前向：打印各张量 grad_fn 和 next_functions 结构 --------
    def show_tensor_info(name, t):
        print(f"{name}: shape={tuple(t.shape)} | requires_grad={t.requires_grad}")
        if t.grad_fn is not None:
            print(f"  grad_fn: {type(t.grad_fn).__name__}")
        else:
            print("  grad_fn: None (leaf tensor)")
        # 对叶子张量（x/W/b）会挂 AccumulateGrad；对中间张量挂具体 *Backward 节点

    print("\n=== Forward graph & grad_fn ===")
    show_tensor_info("x (leaf)", x)
    show_tensor_info("W (leaf)", W)
    show_tensor_info("b (leaf)", b)
    show_tensor_info("y_lin", y_lin)
    show_tensor_info("y_relu", y_relu)
    show_tensor_info("z (loss)", z)

    print("\n=== Traverse backward graph from z.grad_fn ===")
    print_graph(z.grad_fn)

    # -------- 反向：注册 hook 以观察执行顺序 --------
    # 对中间张量和叶子张量都注册 hook，打印梯度何时被计算到
    def make_hook(name):
        def _hook(grad):
            # 这里的打印顺序基本就是反向传播中，这个张量的梯度被“就绪并回传”的时刻
            print(f"[Hook] grad for {name}: shape={tuple(grad.shape)}, norm={grad.norm().item():.6f}")
        return _hook

    # 对中间结果注册 hook
    y_lin.register_hook(make_hook("y_lin"))
    y_relu.register_hook(make_hook("y_relu"))
    # 对叶子也注册：x、W、b 的梯度在 AccumulateGrad 节点处被写入
    x.register_hook(make_hook("x (leaf)"))
    W.register_hook(make_hook("W (leaf)"))
    b.register_hook(make_hook("b (leaf)"))

    print("\n=== Backward start ===")
    z.backward()  # 触发反向：MeanBackward -> ReluBackward -> DemoLinearBackward -> AccumulateGrad

    print("\n=== Gradients accumulated on leaves ===")
    print(f"x.grad.shape={tuple(x.grad.shape)}, norm={x.grad.norm().item():.6f}")
    print(f"W.grad.shape={tuple(W.grad.shape)}, norm={W.grad.norm().item():.6f}")
    print(f"b.grad.shape={tuple(b.grad.shape)}, norm={b.grad.norm().item():.6f}")

if __name__ == "__main__":
    main()