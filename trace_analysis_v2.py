import json
from collections import defaultdict

def analyze_aten_operations(trace_file_path):
    with open(trace_file_path, "r") as f:
        data = json.load(f)

    # 使用字典统计每个 aten 操作的总耗时和事件次数
    aten_ops = defaultdict(lambda: {"total_duration": 0.0, "count": 0})

    for event in data.get("traceEvents", []):
        name = event.get("name", "unknown")
        duration = event.get("dur", 0.0)

        # 只处理以 "aten::" 开头的操作
        if name.startswith("aten::"):
            aten_ops[name]["total_duration"] += duration
            aten_ops[name]["count"] += 1

    # 输出结果：按总耗时降序排列
    sorted_ops = sorted(
        aten_ops.items(),
        key=lambda x: x[1]["total_duration"],
        reverse=True
    )

    print("✅ 分析结果（仅包含 aten:: 开头的操作）:")
    for name, stats in sorted_ops:
        avg_dur = stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0.0
        print(f"[{name}] 总耗时: {stats['total_duration']:.2f} ms | 出现次数: {stats['count']} | 平均耗时: {avg_dur:.2f} ms")

    return sorted_ops

# 示例用法：替换为你的 trace 文件路径
if __name__ == "__main__":
    analyze_aten_operations("GraphGenLab_38093.1755841256503050592.pt.trace.json")

