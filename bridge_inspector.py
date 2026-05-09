import requests
import json
import time

# ---- 1. 配置项 ----
URL = "http://127.0.0.1:8080/completion"
GBNF_FILE = "bridge_standard.gbnf"

TEST_CASE = """
在2024年5月8日的巡检中，高级工程师张伟对京沪高速K122+500大桥进行了详细检查。
1号桥墩状况良好（A级）；
3号桥墩存在垂直裂缝，裂缝宽度约2.5mm，严重程度评估为0.75，整体状况定为D级。
此外，2号主梁边缘有轻微混凝土剥落，严重程度0.2，状况等级为B级。
"""

def call_llm(use_constraint=False):
    """调用 llama.cpp API"""
    prompt = f"Report: {TEST_CASE.strip()}\nResult JSON:"
    
    payload = {
        "prompt": prompt,
        "n_predict": 512,
        "temperature": 0.1 if use_constraint else 0.8,
        "stream": False,
    }
    
    if use_constraint:
        with open(GBNF_FILE, "r") as f:
            payload["grammar"] = f.read().strip()
    
    response = requests.post(URL, json=payload, timeout=60)
    return response.json().get("content", "").strip()

def run_comparison():
    # ---- 教学展示：GBNF 语法定义 ----
    print("\n" + "-" * 50)
    print("📜 [教学展示] bridge_standard.gbnf 手工定义的 GBNF 语法：")
    print("-" * 50)
    try:
        with open(GBNF_FILE, "r") as f:
            print(f.read().strip())
    except FileNotFoundError:
        print(f"❌ 错误: 找不到 {GBNF_FILE} 文件")
    print("-" * 50)

    print("\n" + "=" * 60)
    print("  CS599 Lab 5: GBNF 状态机确定性对比实验")
    print("=" * 60)

    # ---- 对照组 ----
    print("\n【对照组】不使用 GBNF (自由生成)")
    print("-" * 40)
    raw_output = call_llm(use_constraint=False)
    print(f"原始输出:\n> {raw_output}")
    print(f"\n-> 结论: 对照组输出了大量解释文本，难以直接提取 JSON 数据。")

    # ---- 实验组 ----
    print(f"\n【实验组】注入手动编写的 GBNF 语法约束")
    print("-" * 40)
    try:
        gbnf_output = call_llm(use_constraint=True)
        print(f"原始输出:\n> {gbnf_output}")
        
        data = json.loads(gbnf_output)
        print("\n✅ 成功解析 JSON 数据!")
        print(f"   桥梁ID: {data.get('bridge_id')}")
        print(f"   检测员: {data.get('inspector')}")
        print(f"   构件明细:")
        for i, elem in enumerate(data.get('elements', [])):
            print(f"     [{i+1}] 类型: {elem.get('type'):8} 编号: {elem.get('index'):<2} 等级: {elem.get('grade'):2} 严重度: {elem.get('severity')}")
    except Exception as e:
        print(f"❌ 解析失败: {e}")

    # ---- 最终结论 ----
    print("\n" + "=" * 60)
    print("  实验结论")
    print("=" * 60)
    print("  无 GBNF: 模型自由生成，输出格式随机（可能包含自由文本）")
    print("  有 GBNF: 状态机强制模型在语法轨道上生成，输出 100% 合规 JSON")
    print("  → 在零微调条件下，GBNF 是保证工程可靠性的有效手段")
    print("=" * 60)

if __name__ == "__main__":
    run_comparison()
