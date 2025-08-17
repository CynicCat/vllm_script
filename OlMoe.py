from vllm import LLM, SamplingParams

# 输入几个问题
prompts = [
    "你好，你是谁？",
    "法国的首都在哪里？",
]

# 设置初始化采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# 加载模型，确保路径正确
llm = LLM(model="'../model/OLMoE-1B-7B-0125'", trust_remote_code=True, max_model_len=4096)

# 展示输出结果
outputs = llm.generate(prompts, sampling_params)

# 打印输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")