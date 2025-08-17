# -*- coding: utf-8 -*-
import os
import sys
import torch
from torch import Tensor

# —————————— 全局配置 ——————————
MODEL_DIR = "/home/jin/MoE/vLLM/model/OLMoE-1B-7B-0125"
TOP_K = 4

# 路由历史：list of tuples (layer_idx, [expert_ids per token])
ROUTE_HISTORY = []

# 控制何时开始记录
RECORD = False

# —————————— 1) 导入 vLLM 并实例化模型 ——————————
from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL_DIR,
    tensor_parallel_size=1,   # 单卡模式
)
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    repetition_penalty=1.05,
    max_tokens=256
)

# —————————— 2) 给每个 OlmoeMoE 实例注册 forward-hook ——————————
from vllm.model_executor.models.olmoe import OlmoeMoE
import re

def olm_hook(module: OlmoeMoE, inp, out):
    """
    1) 始终保存 flat hidden_states 以便后续 topk
    2) 当 RECORD=True 时，把本 layer 的 top-k 记录到 ROUTE_HISTORY
    """
    flat: Tensor = inp[0]  # [batch_tokens, hidden_size]
    module._hook_flat_hidden = flat

    if RECORD:
        logits, _ = module.gate(flat)                  # [batch_tokens, num_experts]
        idx = torch.topk(logits, k=TOP_K, dim=-1).indices  # [batch_tokens, TOP_K]

        # 从 module.__repr__ 里解析 layer idx
        name = module.__repr__()
        m = re.search(r"layers\.(\d+)\.mlp", name)
        layer_idx = int(m.group(1)) if m else -1

        ROUTE_HISTORY.append((layer_idx, idx.detach().cpu().tolist()))

# 遍历所有层，注册 hook
# llm.model 可能是个 wrapper，具体路径取决于版本；通常是 llm.model.model
olm_model = llm.model.model
for layer in olm_model.layers:
    layer.mlp.register_forward_hook(olm_hook)


# —————————— 3) 拼装 Prompt 文本 ——————————
convs = [
    [
        {'role':'system','content':'You are a helpful assistant.'},
        {'role':'user','content':'请介绍一下浙江工商大学？'}
    ],
    [
        {'role':'system','content':'You are a helpful assistant.'},
        {'role':'user','content':'现有的编程语言有哪些？'}
    ]
]
texts = []
for conv in convs:
    s = ""
    for m in conv:
        s += f"[{m['role'].upper()}]: {m['content']}\n"
    s += "[ASSISTANT]:"
    texts.append(s)


# —————————— 4) 运行推理并收集路由 ——————————
# 在 generate 前开启记录
RECORD = True
outputs = llm.generate(texts, sampling_params)
RECORD = False  # 生成完毕后关闭记录


# —————————— 5) 打印生成结果和路由 ——————————
for i, out in enumerate(outputs):
    # 根据版本兼容取文本
    if hasattr(out, "text"):
        gen = out.text
    elif hasattr(out, "outputs") and out.outputs:
        gen = out.outputs[0].text
    else:
        gen = "<no text>"
    print("="*60)
    print(f"🧪 Prompt #{i+1}\n{texts[i]}\n")
    print(f"🤖 Model Output:\n{gen.strip()}\n")

print("\n" + "="*20 + " Top-4 Routing History " + "="*20)
# 按 layer 分组打印
by_layer = defaultdict(list)
for layer_idx, token_list in ROUTE_HISTORY:
    by_layer[layer_idx].append(token_list)

for layer_idx in sorted(by_layer):
    print(f"Layer {layer_idx}:")
    for token_idx, experts in enumerate(by_layer[layer_idx]):
        print(f" Token {token_idx}: {experts}")
    print()