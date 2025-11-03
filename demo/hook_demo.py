import torch
import torch.nn as nn

# 定义一个简单线性层
layer = nn.Linear(3, 2, bias=False)

# ---------- 前向前 hook：劫持输入 ----------
def pre_hook(module, inputs):
    print(f"[Forward pre-hook] 原始输入: {inputs[0]}")
    new_input = torch.zeros_like(inputs[0])
    print(f"[Forward pre-hook] 修改后输入: {new_input}")
    # 必须返回 tuple
    return (new_input,)

layer.register_forward_pre_hook(pre_hook)

# ---------- 反向 hook：打印梯度 ----------
def backward_hook(module, grad_input, grad_output):
    print(f"[Backward hook] grad_input: {grad_input}")
    print(f"[Backward hook] grad_output: {grad_output}")

layer.register_full_backward_hook(backward_hook)

# ---------- 前向 + 反向 ----------
x = torch.randn(1, 3, requires_grad=True)
y = layer(x)
loss = y.sum()
loss.backward()
