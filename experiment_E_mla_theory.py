import matplotlib.pyplot as plt
import numpy as np
import time
import requests
import statistics
import subprocess
import os

# ============================================================
# Part A: 理论计算 — MHA vs GQA vs MLA 的 KV Cache 对比
# ============================================================

def calculate_kv_cache_size(seq_len, num_layers, num_kv_heads, head_dim, arch_type="MHA"):
    """计算 FP16 下 KV Cache 大小 (MB)"""
    bytes_per_param = 2
    if arch_type == "MHA":
        # K + V 各一份: 2 × seq × layers × heads × dim × 2
        size = 2 * seq_len * num_layers * num_kv_heads * head_dim * bytes_per_param
    elif arch_type == "GQA":
        size = 2 * seq_len * num_layers * num_kv_heads * head_dim * bytes_per_param
    elif arch_type == "MLA":
        # DeepSeek MLA: 每层每 token 仅存一个 512 维潜向量（不含 K+V 分头存储）
        latent_dim = 512
        size = seq_len * num_layers * latent_dim * bytes_per_param
    return size / (1024 * 1024)

# 模型参数
configs = {
    "MHA":   {"layers": 32, "kv_heads": 32, "head_dim": 128, "label": "MHA (32L × 32H × 128)"},
    "GQA":   {"layers": 28, "kv_heads": 4,  "head_dim": 128, "label": "GQA (Qwen2.5-7B, 28L × 4KV × 128)"},
    "MLA":   {"layers": 60, "kv_heads": 0,  "head_dim": 512, "label": "MLA (DeepSeek-V2, 60L × 512 latent)"},
}

seq_lengths = [1024, 4096, 16384, 32768, 65536, 128000]

# 计算各架构的 KV Cache
results = {}
for name, cfg in configs.items():
    results[name] = [calculate_kv_cache_size(s, cfg["layers"], cfg["kv_heads"], cfg["head_dim"], name)
                     for s in seq_lengths]

# --- 图 1: 对数坐标总览 ---
plt.figure(figsize=(10, 6))
for name, cfg in configs.items():
    plt.plot(seq_lengths, results[name], marker={'MHA':'o','GQA':'s','MLA':'x'}[name],
             label=cfg["label"], linewidth=2, color={'MHA':'#3498db','GQA':'#2ecc71','MLA':'#e74c3c'}[name])
plt.yscale('log')
plt.xlabel('Sequence Length')
plt.ylabel('KV Cache Memory Usage (MB)')
plt.title('KV Cache Comparison: MHA vs GQA vs MLA')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('cs599_mla_comparison.png', dpi=150)

print("✅ Part A: 架构对比图已生成: cs599_mla_comparison.png")
print(f"\n{'Context':<12} | {'MHA':<10} | {'GQA':<10} | {'MLA':<10} | {'MLA/GQA':<8} | {'MLA/MHA':<8}")
print("-" * 65)
for i, s in enumerate(seq_lengths):
    mha = results["MHA"][i]
    gqa = results["GQA"][i]
    mla = results["MLA"][i]
    print(f"{s:<12} | {mha:<10.0f} | {gqa:<10.0f} | {mla:<10.0f} | {mla/gqa:<8.2f}x | {mla/mha:<8.2f}x")

# --- 图 2: 每层每 Token 的 KV 开销（核心指标）---
per_token_per_layer = {}
for name, cfg in configs.items():
    if name == "MLA":
        per_token_per_layer[name] = 512 * 2 / 1024  # KB
    else:
        per_token_per_layer[name] = 2 * cfg["kv_heads"] * cfg["head_dim"] * 2 / 1024  # KB

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(configs.keys(), [per_token_per_layer[k] for k in configs.keys()],
              color=['#3498db', '#2ecc71', '#e74c3c'])
ax.set_ylabel('KV Cost per Token per Layer (KB)')
ax.set_title('Per-Token-Per-Layer KV Cache Overhead')
ax.grid(axis='y', ls='--', alpha=0.5)
for bar, val in zip(bars, per_token_per_layer.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.0f} KB',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('cs599_mla_per_token_kv.png', dpi=150)
print("\n📊 每层每 Token KV 开销图: cs599_mla_per_token_kv.png")

# ============================================================
# Part B: 固定显存下的最大并发请求数（核心教学指标）
# ============================================================

def compute_max_concurrent(avail_mem_mb, seq_len, layers, kv_heads, head_dim, arch_type):
    """计算在可用显存下，最多能同时服务多少个请求（仅考虑 KV Cache）。"""
    kv_per_request_mb = calculate_kv_cache_size(seq_len, layers, kv_heads, head_dim, arch_type)
    if kv_per_request_mb <= 0:
        return 0
    return int(avail_mem_mb / kv_per_request_mb)

# 假设可用显存（M3 约 18GB 总内存，模型占 ~5GB，剩余 ~13GB）
AVAILABLE_MEM_GB = 13.0
AVAILABLE_MEM_MB = AVAILABLE_MEM_GB * 1024

print("\n" + "=" * 60)
print("  Part B: 固定 GPU 显存下的并发承载能力对比")
print("=" * 60)
print(f"\n可用显存: {AVAILABLE_MEM_GB:.0f} GB = {AVAILABLE_MEM_MB:.0f} MB")

test_seq_lengths = [2048, 8192, 32768]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(test_seq_lengths))
width = 0.25
arch_order = ["MHA", "GQA", "MLA"]
colors = {'MHA': '#3498db', 'GQA': '#2ecc71', 'MLA': '#e74c3c'}

all_capacities = {}
for name, cfg in configs.items():
    capacities = [compute_max_concurrent(AVAILABLE_MEM_MB, s, cfg["layers"],
                                         cfg["kv_heads"], cfg["head_dim"], name)
                  for s in test_seq_lengths]
    all_capacities[name] = capacities
    ax.bar(x + arch_order.index(name) * width, capacities, width,
           label=cfg["label"], color=colors[name])
    # 标注数值
    for j, cap in enumerate(capacities):
        if cap > 0:
            ax.text(x[j] + arch_order.index(name) * width, cap + max(capacities) * 0.015,
                    str(cap), ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Sequence Length (tokens)')
ax.set_ylabel('Max Concurrent Requests')
ax.set_title(f'Max Concurrent Requests at {AVAILABLE_MEM_GB:.0f} GB Available Memory')
ax.set_xticks(x + width)
ax.set_xticklabels([f"{s//1024}K" for s in test_seq_lengths])
ax.legend(fontsize=9)
ax.grid(axis='y', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('cs599_concurrent_capacity.png', dpi=150)
print("📊 并发能力对比图: cs599_concurrent_capacity.png")

# 打印表格
print(f"\n{'Context':<12} | {'MHA':<8} | {'GQA':<8} | {'MLA':<8} | {'MLA vs GQA':<12} | {'MLA vs MHA':<12}")
print("-" * 70)
for i, s in enumerate(test_seq_lengths):
    mha_c = all_capacities["MHA"][i]
    gqa_c = all_capacities["GQA"][i]
    mla_c = all_capacities["MLA"][i]
    vs_gqa = f"{mla_c/gqa_c:.1f}x" if gqa_c > 0 else "N/A"
    vs_mha = f"{mla_c/mha_c:.1f}x" if mha_c > 0 else "N/A"
    print(f"{s:<12} | {mha_c:<8} | {gqa_c:<8} | {mla_c:<8} | {vs_gqa:<12} | {vs_mha:<12}")

# 核心结论
print("\n💡 核心结论:")
for i, s in enumerate(test_seq_lengths):
    mla_c = all_capacities["MLA"][i]
    gqa_c = all_capacities["GQA"][i]
    if gqa_c > 0:
        print(f"   {s//1024}K context: MLA 支持 {mla_c} 个并发，GQA 支持 {gqa_c} 个 → {mla_c/gqa_c:.1f}x")

print(f"\n 重要发现: 在 7B 级别模型中，MLA 与 GQA 的总 KV Cache 相近")
print(f"   因为 MLA 虽然每层开销小 (512 字节 vs 4096 字节)，但 DeepSeek-V2 有 60 层 (vs 28 层)")
print(f"   → MLA 的核心优势不在 KV Cache 大小，而在计算效率：")
print(f"     MLA 将 Q/K/V 投影到低维潜空间，Attention 计算的矩阵乘法规模大幅缩小")
print(f"     这意味着: 相同硬件下，MLA 模型的 Prefill/Decoding 速度更快")
print(f"\n→ 对比: 当模型规模扩大到 671B (DeepSeek-V3) 时，MLA 的优势才真正爆发：")
print(f"   如果 DeepSeek-V3 使用 GQA，KV Cache 将是天文数字；")
print(f"   MLA 让 671B 模型也能在有限显存下高效推理。")

# ============================================================
# Part C: llama.cpp 实测 — 多 Slot 并发压力下的 TPS 衰减
# ============================================================

API_URL = "http://localhost:8080"

MODEL_PATH = "/Users/qixin/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

def kill_server():
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(3)

def start_server_with_slots(n_slots=16, total_ctx=65536):
    """重启 llama-server，使用多 slot 以观察并发压力下的 TPS 衰减。

    注意：Apple M3 GPU 在 --parallel > 16 且总 context pool 过大时会直接
    触发 kIOGPUCommandBufferCallbackErrorOutOfMemory，因此采用保守配置：
    n_slots=16, total_ctx=65536 → 每 slot 4096 tokens。
    """
    kill_server()
    ctx_per_slot = total_ctx // n_slots
    cmd = [
        "llama-server", "-m", MODEL_PATH,
        "-c", str(total_ctx),
        "--parallel", str(n_slots),
        "--port", "8080", "--n-gpu-layers", "99", "--metrics",
    ]
    print(f"  正在重启 llama-server: {n_slots} slots, n_ctx={total_ctx} (每 slot {ctx_per_slot})")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(45):
        time.sleep(2)
        try:
            r = requests.get(f"{API_URL}/health", timeout=3)
            if r.status_code == 200:
                return proc, ctx_per_slot
        except:
            pass
    print("  ⚠️ 服务器启动超时")
    return None, 0

def measure_tps_at_context(context_len, gen_tokens=64, samples=1, warmup=1):
    """测量指定 context 长度的 TPS。"""
    prompt = "repeat " * context_len
    payload = {
        "prompt": prompt,
        "n_predict": gen_tokens,
        "temperature": 0.0,
        "stream": False
    }
    tps_list = []
    for i in range(warmup + samples):
        try:
            r = requests.post(f"{API_URL}/completion", json=payload, timeout=600)
            if r.status_code != 200:
                if i < warmup:
                    return None, r.json().get("error", {}).get("message", "err")
                continue
            t = r.json().get("timings", {}).get("predicted_per_second", 0)
            if t > 0 and i >= warmup:
                tps_list.append(t)
        except Exception as e:
            if i < warmup:
                return None, str(e)
    return statistics.median(tps_list) if tps_list else None, None

def run_part_c():
    """通过多并发长上下文请求制造 GPU 计算压力，观察 TPS 衰减。

    原理：
    - Qwen2.5-7B (GQA) 单请求在 2048 context 下占用 ~112 MB KV Cache
    - 16 个 slot 全满 = 16 × 112 MB ≈ 1.8 GB，远低于 M3 可用显存（~13 GB）
    - 因此 TPS 衰减主要来自 GPU 计算 contention（多个 decode batch 共享 GPU），
      而非显存溢出。在数据中心 GPU 上，若进一步增大 context 至显存瓶颈，
      会出现更剧烈的 TPS 衰减。

    Apple M3 硬件限制：
    - 当 --parallel > 16 且总 context pool > 65536 时，GPU Metal 后端
      直接报 kIOGPUCommandBufferCallbackErrorOutOfMemory。
    - 本实验采用保守配置 n_slots=16, total_ctx=65536，
      每 slot 4096 tokens，每请求使用 2048 tokens（留有生成空间）。
    """
    print("\n" + "=" * 60)
    print("  Part C: 多并发请求的 TPS 衰减实测（Apple M3 环境）")
    print("=" * 60)
    print("  目标: 通过增加并发请求数观察 per-request TPS 衰减")
    print("  方法: 重启 llama-server 为 16 slots (total_ctx=65536, 每 slot 4096)")
    print("  注意: Apple M3 在更大配置下会 GPU OOM，因此采用保守参数")
    print("  预期: 随着并发请求数增加，per-request TPS 应显著下降")

    proc, ctx_per_slot = start_server_with_slots(n_slots=16, total_ctx=65536)
    if not proc:
        print("\n❌ 服务器启动失败")
        return

    props = requests.get(f"{API_URL}/props", timeout=5).json()
    n_ctx = props["default_generation_settings"]["n_ctx"]
    model = props.get("model_alias", "unknown")
    print(f"  ✅ 服务器就绪 | 模型: {model} | 每 Slot n_ctx: {n_ctx}")

    # 每请求使用 2048 tokens（在 4096 per-slot 限制内）
    ctx_per_request = 2048
    kv_per_request_mb = 2 * ctx_per_request * 28 * 4 * 128 * 2 / (1024*1024)
    gen_tokens = 16  # 短生成，专注于测量 decoding TPS

    print(f"  每请求 context: {ctx_per_request} tokens")
    print(f"  估算: 每请求 KV Cache ≈ {kv_per_request_mb:.0f} MB")
    print(f"  可用显存 ≈ 13 GB = {13*1024:.0f} MB")
    safe_requests = int(13 * 1024 / kv_per_request_mb)
    print(f"  显存安全并发数 ≈ {safe_requests}（远超 slot 数，KV Cache 不是瓶颈）")
    print(f"  测试并发级别: 1, 2, 4, 8, 16 个同时请求")
    print(f"  衰减来源: GPU 计算 contention（多个 decode batch 共享 GPU SM）")

    # 测试不同并发级别下的 TPS
    concurrency_levels = [1, 2, 4, 8, 16]

    print(f"\n{'并发请求数':<12} | {'总 KV Cache':<12} | {'平均 TPS':<10} | {'相对基准':<10} | {'状态'}")
    print("-" * 60)

    baseline_tps = None
    results_conc = []
    results_tps = []
    results_ratio = []
    results_kv = []

    for n_conc in concurrency_levels:
        total_kv_mb = n_conc * kv_per_request_mb
        total_kv_gb = total_kv_mb / 1024

        # 发送 n_conc 个并发请求，测量各自的 TPS
        prompt = "repeat " * ctx_per_request
        payload = {
            "prompt": prompt,
            "n_predict": gen_tokens,
            "temperature": 0.0,
            "stream": False
        }

        tps_list = []
        errors = 0

        # 使用多线程并发
        import concurrent.futures
        def send_one():
            try:
                r = requests.post(f"{API_URL}/completion", json=payload, timeout=300)
                if r.status_code == 200:
                    t = r.json().get("timings", {}).get("predicted_per_second", 0)
                    return t if t > 0 else None
            except:
                pass
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_conc) as executor:
            futures = [executor.submit(send_one) for _ in range(n_conc)]
            for f in concurrent.futures.as_completed(futures):
                result = f.result()
                if result:
                    tps_list.append(result)
                else:
                    errors += 1

        if tps_list:
            avg_tps = statistics.median(tps_list)
            if baseline_tps is None:
                baseline_tps = avg_tps
            ratio = (avg_tps / baseline_tps) * 100
            status = f"OK ({errors} errors)" if errors else "OK"
            print(f"{n_conc:<12} | {total_kv_gb:<10.1f} GB | {avg_tps:<10.2f} | {ratio:<10.1f}% | {status}")
            results_conc.append(n_conc)
            results_tps.append(avg_tps)
            results_ratio.append(ratio)
            results_kv.append(total_kv_gb)
        else:
            print(f"{n_conc:<12} | {total_kv_gb:<10.1f} GB | {'-':<10} | {'-':<10} | 全部失败")

    if not results_tps:
        print("\n⚠️ 无有效数据")
        proc.terminate()
        return

    # 绘制衰减曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(results_conc, results_tps, marker='o', color='#e74c3c', linewidth=2, markersize=8)
    ax1.set_xlabel('Concurrent Requests')
    ax1.set_ylabel('Decoding TPS (per request)')
    ax1.set_title(f'Per-Request TPS vs Concurrency ({model})')
    ax1.grid(True, ls='--', alpha=0.5)

    ax2.plot(results_conc, results_ratio, marker='s', color='#2ecc71', linewidth=2, markersize=8)
    ax2.axhline(y=100, color='gray', ls='--', alpha=0.5)
    ax2.axhline(y=70, color='red', ls='--', alpha=0.5, label='30% degradation threshold')
    ax2.set_xlabel('Concurrent Requests')
    ax2.set_ylabel('Relative to Baseline (%)')
    ax2.set_title('TPS Retention Rate')
    ax2.legend()
    ax2.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('cs599_kv_cache_pressure.png', dpi=150)
    print("\n📊 多并发压力衰减图: cs599_kv_cache_pressure.png")

    last_ratio = results_ratio[-1]
    print(f"\n💡 实测结论:")
    if last_ratio > 90:
        print(f"   TPS 衰减 <10%（1→{results_conc[-1]} 并发），说明 GPU 计算资源充足")
    elif last_ratio > 70:
        print(f"   TPS 衰减 10%-30%（从 {results_tps[0]:.1f} 降至 {results_tps[-1]:.1f} TPS）")
        print(f"   GPU 调度开销开始影响性能")
    else:
        print(f"   TPS 衰减 >30%（从基准 {results_tps[0]:.1f} 降至 {results_tps[-1]:.1f} TPS），")
        print(f"   多个 decode batch 共享 GPU 导致显著的计算 contention！")
        print(f"   注意: 总 KV Cache 仅 {results_kv[-1]:.1f} GB << 13 GB 可用显存")
        print(f"   → 衰减来源是 GPU SM 算力分摊，而非显存溢出")
        print(f"   → 在数据中心 GPU（如 A100）上，若 context 增大到显存瓶颈，")
        print(f"     会出现更剧烈的 TPS 衰减（内存换页 + 计算 contention 叠加）")

    # 系统吞吐量分析
    if len(results_tps) >= 2:
        sys_tp_1 = results_tps[0] * 1  # 1 并发
        sys_tp_n = results_tps[-1] * results_conc[-1]  # N 并发
        print(f"\n  系统吞吐量对比:")
        print(f"    1 并发: {sys_tp_1:.1f} tokens/s（单请求）")
        print(f"    {results_conc[-1]} 并发: {sys_tp_n:.1f} tokens/s（总产出）")
        if sys_tp_n > sys_tp_1:
            print(f"    → 系统吞吐量提升 {sys_tp_n/sys_tp_1:.1f}x，并行加速有效")
        else:
            print(f"    → 系统吞吐量反而下降，GPU 计算 contention 导致性能退化")

    proc.terminate()

if __name__ == "__main__":
    # Part A and B run at module level (theory + capacity analysis)
    # Part C runs as the empirical measurement against the active llama-server
    run_part_c()
