import time
import requests
import statistics

# 实验配置
API_URL = "http://localhost:8080/completion"
# 不同的输入 Token 长度（近似值，通过重复单词构造）
PROMPT_LENGTHS = [128, 512, 1024, 2048, 4096]
SAMPLES_PER_LEN = 3

def test_prefill_latency(token_count):
    # 构造指定长度的输入（每个单词约占1个token）
    prompt = "Context " * token_count
    payload = {
        "prompt": f"Based on this: {prompt} \nSummarize in one word:",
        "n_predict": 1,  # 强制只生成1个token，以隔离 Prefill 阶段
        "temperature": 0.0,
        "stream": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload, timeout=120)
        end_time = time.time()
        
        data = response.json()
        # 优先读取 llama.cpp 返回的内部精确时间
        timings = data.get("timings", {})
        prefill_ms = timings.get("prompt_ms", (end_time - start_time) * 1000)
        return prefill_ms
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None

def run_experiment():
    print("=" * 60)
    print("  CS599 Lab 5: Experiment A - Prefill 算力瓶颈测试")
    print("=" * 60)
    print(f"{'输入长度 (Tokens)':<20} | {'平均预填充耗时 (ms)':<20} | {'速度 (Tokens/s)'}")
    print("-" * 60)

    for length in PROMPT_LENGTHS:
        latencies = []
        for _ in range(SAMPLES_PER_LEN):
            lat = test_prefill_latency(length)
            if lat: latencies.append(lat)
        
        if latencies:
            avg_lat = statistics.mean(latencies)
            tps = (length / avg_lat) * 1000
            print(f"{length:<20} | {avg_lat:<20.2f} | {tps:.2f}")

    print("-" * 60)
    print("💡 结论提示：Prefill 阶段是并行的。随着输入长度翻倍，耗时应呈线性增长。")
    print("如果增长是非线性的，说明 GPU 算力已达到瓶颈。")

if __name__ == "__main__":
    run_experiment()
