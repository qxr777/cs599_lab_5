import requests
import time

API_URL = "http://localhost:8080/completion"

# 构造一个长公共前缀（约 2048 tokens，每词约1 token）
COMMON_PREFIX = ("The analysis of the bridge structure requires careful consideration of "
                 "load distribution, material fatigue, environmental stress factors, "
                 "structural integrity over time, and the impact of dynamic forces including "
                 "wind, traffic, and seismic activity. The inspection protocol mandates "
                 "a comprehensive evaluation of all primary and secondary structural elements "
                 "including piers, girders, deck slabs, abutments, bearings, expansion joints, "
                 "and cable systems. Each component must be assessed against established "
                 "safety standards and historical performance data. The engineering report "
                 "shall document all findings with quantitative measurements, photographic "
                 "evidence, and recommended maintenance schedules. " * 5)

UNIQUE_SUFFIX_A = "Question A: What is the primary material concern for pier number three?"
UNIQUE_SUFFIX_B = "Question B: What is the recommended inspection frequency for the deck slab?"

def measure_ttft(prompt, label):
    payload = {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0,
        "stream": False,
    }
    start = time.time()
    resp = requests.post(API_URL, json=payload, timeout=120)
    ttft = time.time() - start
    timings = resp.json().get("timings", {})
    prompt_ms = timings.get("prompt_ms", 0)
    cached = resp.json().get("tokens_cached", 0)
    return ttft, prompt_ms, cached

def run_experiment():
    print("=" * 60)
    print("  Prefix Caching 效果对比实验")
    print("=" * 60)
    print(f"公共前缀长度: 约 {len(COMMON_PREFIX.split())} 个 tokens")
    print()

    # --- 第1步：冷启动请求（无缓存） ---
    prompt1 = COMMON_PREFIX + UNIQUE_SUFFIX_A
    ttft1, pms1, cached1 = measure_ttft(prompt1, "冷启动")
    print(f"【第1步】冷启动请求（不同前缀）")
    print(f"  TTFT: {ttft1*1000:.0f}ms (prefill: {pms1:.0f}ms, cached: {cached1})")
    print()

    # --- 第2步：相同前缀请求（命中缓存） ---
    prompt2 = COMMON_PREFIX + UNIQUE_SUFFIX_B
    ttft2, pms2, cached2 = measure_ttft(prompt2, "缓存命中")
    print(f"【第2步】相同前缀请求（命中 KV Cache）")
    print(f"  TTFT: {ttft2*1000:.0f}ms (prefill: {pms2:.0f}ms, cached: {cached2})")
    print()

    # --- 第3步：再次不同前缀（验证不是偶然） ---
    prompt3 = "Short random context: " + "X" * 2000 + " Question C?"
    ttft3, pms3, cached3 = measure_ttft(prompt3, "不相关")
    print(f"【第3步】不同前缀请求（不缓存，验证冷启动一致性）")
    print(f"  TTFT: {ttft3*1000:.0f}ms (prefill: {pms3:.0f}ms, cached: {cached3})")
    print()

    # --- 汇总 ---
    speedup = ttft3 / ttft2 if ttft2 > 0 else 0
    print("=" * 60)
    print("  结论")
    print("=" * 60)
    print(f"  无缓存（冷启动）: TTFT {ttft3*1000:.0f}ms（完整 prefill {pms3:.0f}ms）")
    print(f"  有缓存（命中）  : TTFT {ttft2*1000:.0f}ms（仅处理增量 {pms2:.0f}ms，缓存 {cached2} tokens）")
    print(f"  鸿沟: 命中缓存后 TTFT 降低 {speedup:.1f}x")
    print()
    print("  教学意义: 在真实业务中（如多轮对话、批量审计），")
    print("  只要请求共享公共前缀（系统提示词/长文档），")
    print("  Prefix Caching 就能将 TTFT 从秒级降到毫秒级。")


if __name__ == "__main__":
    run_experiment()
