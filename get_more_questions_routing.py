# import os
# import json
# import pandas as pd
# from vllm import LLM, SamplingParams
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# # ——— 配置 ——————————————
# # 本地存放问题的 CSV（前面抽样合并后的结果）
# PROMPT_CSV = "/home/jin/TraceGen_test1/prompt_questions/only_hard_1000.csv"

# # 你本地部署的 MoE 模型路径
# MODEL_PATH = "/home/jin/moe_router/models/deepseek-moe-16b-base"

# # 每次请求的 batch 大小
# BATCH_SIZE = 8

# # 生成配置：温度、top-p、最大生成长度等
# SAMPLING_PARAMS = SamplingParams(
#     temperature=0.8,
#     top_p=0.95,
#     max_tokens=150
# )

# # 输出结果文件
# OUTPUT_JSON = "/home/jin/TraceGen_test1/prompt_questions/answers/deepseek-moe-16b_responses_hard.json"

# # ——— 主流程 ——————————————
# def main():
#     # 1. 读取所有 prompt
#     df = pd.read_csv(PROMPT_CSV)
#     prompts = df["question"].tolist()

#     # 2. 初始化本地 vLLM
#     llm = LLM(
#         model=MODEL_PATH,
#         tensor_parallel_size=1,
#         trust_remote_code=True,
#         max_model_len=4096,
#     )

#     # 3. 批量调用模型生成回答
#     results = []
#     n_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE

#     for batch_idx in range(n_batches):
#         start = batch_idx * BATCH_SIZE
#         end = min(start + BATCH_SIZE, len(prompts))
#         batch_prompts = prompts[start:end]

#         # 3.1 调用 vLLM
#         outputs = llm.generate(batch_prompts, SAMPLING_PARAMS)

#         # 3.2 收集回答文本
#         for prompt, out in zip(batch_prompts, outputs):
#             reply = out.outputs[0].text
#             results.append({
#                 "prompt": prompt,
#                 "response": reply,
#             })

#         print(f"[{batch_idx+1}/{n_batches}] generated {len(batch_prompts)} prompts")

#     # 4. 保存到 JSON
#     os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
#     with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

#     print(f"✅ All done! Saved {len(results)} responses to {OUTPUT_JSON}")

# if __name__ == "__main__":
#     main()

import os
import time
import json
import pandas as pd
from vllm import LLM, SamplingParams

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

# ——— 配置 ——————————————
# 使用随机抽样后的 1k 问题 CSV 文件
# PROMPT_CSV = "/mnt/debugger/longbin/moe_routing_trace_generator/only_hard_10000.csv" # 困难 需推理数据集
PROMPT_CSV = "/mnt/debugger/longbin/moe_routing_trace_generator/only_easy_10000.csv"  # 普通

# 本地部署的 MoE 模型路径
MODEL_PATH = "/mnt/debugger/longbin/model/Qwen3-235B-A22B-Thinking-2507-FP4"

# 批大小
BATCH_SIZE = 8

# 生成参数设定
SAMPLING_PARAMS = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1000 # 需推理数据集才需要 1400，普通数据集可以设置为 150
)

# 输出结果文件
OUTPUT_JSON = "/mnt/debugger/longbin/moe_routing_trace_generator/answers/Qwen3-235B-A22B-Thinking-2507-FP4_easy.json"

# ——— 主流程 ——————————————
def main():
    # 1. 读取随机抽样的 1k 问题
    df = pd.read_csv(PROMPT_CSV)
    prompts = df['question'].tolist()

    # 2. 初始化 vLLM
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=8,
        trust_remote_code=True,
        max_model_len=4096,
        enable_expert_parallel=True,      # 专家并行，也切 8 份
    )
    #llm.start_profilt()

    # 3. 流式生成回答并保存（避免内存累积）
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    total_processed = 0
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        f.write('[\n')
        first_item = True
        
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i + BATCH_SIZE]
            outputs = llm.generate(batch, SAMPLING_PARAMS)
            
            for prompt, out in zip(batch, outputs):
                result = {
                    'prompt': prompt,
                    'response': out.outputs[0].text
                }
                
                # 写入JSON项（除第一项外都要加逗号）
                if not first_item:
                    f.write(',\n')
                json.dump(result, f, ensure_ascii=False, indent=2)
                first_item = False
                total_processed += 1
            
            # 强制刷新到磁盘，确保数据安全
            f.flush()
            print(f"Processed batch {i//BATCH_SIZE + 1}/{(len(prompts)-1)//BATCH_SIZE + 1}, total: {total_processed}")
        
        f.write('\n]')
    #llm.stop_profile()
    time.sleep(10)
    print(f"✅ 完成！共保存 {total_processed} 条回答至 {OUTPUT_JSON}")
    
if __name__ == '__main__':
    main()

