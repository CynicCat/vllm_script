# import os
# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams

# # 设置环境变量：指定使用 CPU（如果你用的是 GPU，可删除这行）
# # os.environ['VLLM_TARGET_DEVICE'] = 'cpu'

# # 模型路径（本地目录）
# model_dir = './Qwen2-0.5B'

# # 初始化 tokenizer（从本地加载）
# tokenizer = AutoTokenizer.from_pretrained(
#     model_dir,
#     local_files_only=True,
# )

# # 构建 Chat 模式的 prompt
# messages = [
#     {'role': 'system', 'content': '你是一个聊天机器人.'},
#     {'role': 'user', 'content': '刘芊昨天去看了孙燕姿的演唱会，她是不是激动开心到现在？'}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
# )

# # 初始化 vLLM 模型（CPU 推理）
# llm = LLM(
#     model=model_dir,
#     tensor_parallel_size=1,  # CPU 不使用张量并行
#     device='cuda',
# )

# # 设置采样参数
# sampling_params = SamplingParams(
#     temperature=0.7,
#     top_p=0.8,
#     repetition_penalty=1.05,
#     max_tokens=512
# )

# # 推理
# outputs = llm.generate([text], sampling_params)

# # 打印结果
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f'Prompt 提示词: {prompt!r}\n大模型推理输出: {generated_text!r}')

import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 可选：指定使用 GPU（如果你有 GPU 并且安装了 GPU 版 vLLM 和 PyTorch，可以省略此行）
# os.environ['VLLM_TARGET_DEVICE'] = 'cpu'

# 模型路径（本地）
model_dir = '../model/Qwen2-0.5B'

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

# 构建多个 Chat Prompt（支持多轮）
prompts = [
    [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '请介绍一下2025年孙燕姿上海演唱会'}
    ],
    [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '请分析一下：刘芊昨天晚上去看了孙燕姿在上海的就在日落以后演唱会，她是不是昨晚开始🧠一直兴奋到现在？'}
    ]
]

# 应用 prompt 模板
texts = [
    tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
    for p in prompts
]

# 初始化 vLLM 模型（GPU 会自动使用）
llm = LLM(
    model=model_dir,
    tensor_parallel_size=2,  # 有多卡可以设为 2、4、8...
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

    # 尝试从 prompt 中解析出 system 和 user 内容
    system_content = ""
    user_content = ""
    try:
        segments = prompt.split("<|im_start|>")
        for seg in segments:
            if seg.startswith("system"):
                system_content = seg.replace("system", "").replace("<|im_end|>", "").strip()
            elif seg.startswith("user"):
                user_content = seg.replace("user", "").replace("<|im_end|>", "").strip()
    except Exception:
        user_content = prompt

    # 输出
    print("=" * 60)
    print(f"🧪 Prompt #{idx+1}")
    print(f"🧠 System Prompt:\n{system_content}\n")
    print(f"🙋‍♂️ User Input:\n{user_content}\n")
    print(f"🤖 LLM Output:\n{generated_text}")
    print("=" * 60 + "\n")
