import requests
import json
import time
import statistics
import matplotlib.pyplot as plt

API_URL = "http://localhost:8080/completion"

# 实验设计：同一组 prompt，分别在低 temperature 和高 temperature 下运行
# 原理：低 temp 时模型始终取高概率 token（确定性高）
#       高 temp 时模型被迫采样低概率 token（暴露真实分布不确定性）
PROMPTS = [
    {"name": "事实", "prompt": "Q: 太阳系中最大的行星是哪一颗？\nA: "},
    {"name": "诱导", "prompt": "Q: 请解释爱因斯坦在 1995 年发表的关于互联网协议的论文核心观点。\nA: "},
    {"name": "虚构", "prompt": "Q: 请用专业术语详细描述反重力引擎中量子真空极化场的数学推导过程。\nA: "},
    {"name": "模糊标准", "prompt": "Q: 根据 2014 年 ISO/IEC 29182-7 标准第 4.3.2 节的附录 C，第三个注意事项是什么？\nA: "},
    {"name": "噪声", "prompt": "Q: xk7#mZ@9!qrT$vL2pN&bF5jW*sD8cY^hK0aE\nA: "},
]

TEMPERATURES = [0.1, 0.7, 1.5]
TOP_N = 3
MAX_TOKENS = 64  # 缩短生成量，聚焦前几个 token 的 logprob 分布

# 熔断阈值
AVG_THRESHOLD = -1.5

def get_logprobs(prompt, temperature):
    payload = {
        "prompt": prompt,
        "n_predict": MAX_TOKENS,
        "temperature": temperature,
        "n_probs": TOP_N,
        "stream": False,
    }

    start = time.time()
    response = requests.post(API_URL, json=payload, timeout=120)
    elapsed = time.time() - start
    data = response.json()

    content = data.get("content", "").strip()
    logprobs = []
    for token_info in data.get("completion_probabilities", []):
        logprobs.append(token_info["logprob"])

    return content, logprobs, elapsed


def run_experiment():
    print("=" * 70)
    print("  CS599 Lab 5: Log-probs 幻觉拦截实验（Circuit Breaker）")
    print("=" * 70)
    print(f"策略: 同一 prompt，三种 temperature 下对比 logprob 分布")
    print(f"熔断规则: 平均 logprobs < {AVG_THRESHOLD}")
    print(f"核心假设: 高 temperature 下，模型被迫采样低概率 token，")
    print(f"           对于知识盲区的问题，logprob 会显著降低\n")

    all_results = []

    for p in PROMPTS:
        print(f"\n{'='*60}")
        print(f"📋 {p['name']}: {p['prompt'][:70]}...")
        print("-" * 60)

        prompt_results = {"name": p["name"], "runs": []}

        for temp in TEMPERATURES:
            content, logprobs, elapsed = get_logprobs(p["prompt"], temp)

            if logprobs:
                avg_lp = statistics.mean(logprobs)
                min_lp = min(logprobs)
                low_count = sum(1 for lp in logprobs if lp < -3.0)

                trigger = avg_lp < AVG_THRESHOLD
                status = "🔴 熔断" if trigger else "✅ 通过"

                print(f"  temp={temp:.1f}  avg={avg_lp:.4f}  min={min_lp:.4f}  "
                      f"low(<-3.0)={low_count}  {status}")
                print(f"         生成: {content[:80]}...")

                prompt_results["runs"].append({
                    "temp": temp,
                    "avg": avg_lp,
                    "min": min_lp,
                    "low_count": low_count,
                    "content": content,
                    "trigger": trigger,
                })
            else:
                print(f"  temp={temp:.1f}  ❌ 无 logprobs 数据")

        all_results.append(prompt_results)

    # 汇总报告
    print(f"\n\n{'='*70}")
    print("  📊 实验总结报告")
    print("=" * 70)
    print(f"\n{'用例':<8} | {'temp=0.1':>9} | {'temp=0.7':>9} | {'temp=1.5':>9} | {'最差':>9} | {'熔断?'}")
    print("-" * 70)

    for pr in all_results:
        avgs = [r["avg"] for r in pr["runs"]]
        vals = [f"{a:.4f}" if a is not None else "N/A" for a in avgs]
        worst = min(avgs) if avgs else 0
        any_trigger = any(r["trigger"] for r in pr["runs"])
        icon = "🔴" if any_trigger else "  "
        print(f"{pr['name']:<8} | {vals[0]:>9} | {vals[1]:>9} | {vals[2]:>9} | {worst:>9.4f} | {icon}")

    print("-" * 70)

    total_fires = sum(1 for pr in all_results for r in pr["runs"] if r["trigger"])
    print(f"\n总熔断次数: {total_fires}/{len(PROMPTS)*len(TEMPERATURES)}")

    # 绘制对比图
    plot_heatmap(all_results)


def plot_heatmap(all_results):
    names = [pr["name"] for pr in all_results]
    temps = TEMPERATURES

    data = []
    for pr in all_results:
        row = [r["avg"] for r in pr["runs"]]
        data.append(row)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=-3, vmax=0)

    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"t={t}" for t in temps])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Prompt Type")
    ax.set_title("Model Confidence (avg logprob) across Temperature")
    plt.colorbar(im, label="Avg Log-prob")

    # 数值标注
    for i in range(len(names)):
        for j in range(len(temps)):
            val = data[i][j]
            color = "white" if val < -1.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    plt.tight_layout()
    plt.savefig("cs599_logprobs_confidence.png", dpi=150)
    print("\n📊 置信度热力图已保存: cs599_logprobs_confidence.png")


if __name__ == "__main__":
    run_experiment()
