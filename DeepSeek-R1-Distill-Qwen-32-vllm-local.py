import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re

# 可选：指定使用 GPU（如果你有 GPU 并且安装了 GPU 版 vLLM 和 PyTorch，可以省略此行）
# os.environ['VLLM_TARGET_DEVICE'] = 'cpu'

# 模型路径（本地）
model_dir = '/home/jin/MoE/vLLM/model/DeepSeek-R1-Distill-Qwen-32B'

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

# 构建多个 Chat Prompt（支持多轮）
prompts = [
    [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '请介绍一下你的工作是基于什么架构的？'}
    ],
    # [
    #     {'role': 'system', 'content': 'You are a helpful assistant.'},
    #     {'role': 'user', 'content': '你觉得浙江工商大学好吗'}
    # ]
]

# 应用 prompt 模板
texts = [
    tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
    for p in prompts
]

# 初始化 vLLM 模型（GPU 会自动使用）
llm = LLM(
    model=model_dir,
    tensor_parallel_size=8,  # 有多卡可以设为 2、4、8...
    # device="cuda",
)

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    repetition_penalty=1.05,
    max_tokens=512
)

# 执行批量推理
outputs = llm.generate(texts, sampling_params)

# 输出清晰格式化结果
for idx, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text.strip()

    # 解析 system prompt
    system_content = prompt.replace(tokenizer.bos_token, '').split('<｜User｜>')[0].strip()

    # 提取第一个 user 输入（适配单轮/多轮）
    user_match = re.search(r'<｜User｜>(.*?)((?=<｜)|$)', prompt, re.DOTALL)
    user_content = user_match.group(1).strip() if user_match else "（未识别）"

    print("=" * 60)
    print(f"🧪 Prompt #{idx+1}")
    print(f"🧠 System Prompt:\n{system_content}\n")
    print(f"🙋‍♂️ User Input:\n{user_content}\n")
    print(f"🤖 LLM Output:\n{generated_text}")
    print("=" * 60 + "\n")

