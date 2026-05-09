import time
import requests
import statistics

# 实验配置
API_URL = "http://localhost:8080/completion"
GEN_LENGTHS = [64, 128, 256, 512]
SAMPLES_PER_LEN = 2

# 假设模型大小（Qwen2.5-7B Q4_K_M 约 4.7 GB）
MODEL_SIZE_GB = 4.7 

def test_decoding_speed(gen_len):
    payload = {
        "prompt": "Once upon a time in a distant galaxy,",
        "n_predict": gen_len,
        "temperature": 0.7,
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        data = response.json()
        
        timings = data.get("timings", {})
        # 内部解码时间（不含预填充）
        predict_ms = timings.get("predicted_ms")
        predict_count = timings.get("predicted_n")
        
        if predict_ms and predict_count:
            tps = (predict_count / predict_ms) * 1000
            return tps, predict_ms
        return None, None
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None, None

def run_experiment():
    print("=" * 60)
    print("  CS599 Lab 5: Experiment B - Decoding 带宽瓶颈测试")
    print("=" * 60)
    print(f"{'生成长度 (Tokens)':<20} | {'生成速率 (TPS)':<20} | {'预估带宽消耗 (GB/s)'}")
    print("-" * 60)

    for length in GEN_LENGTHS:
        speeds = []
        for _ in range(SAMPLES_PER_LEN):
            tps, _ = test_decoding_speed(length)
            if tps: speeds.append(tps)
        
        if speeds:
            avg_tps = statistics.mean(speeds)
            # 理论计算：每生成一个 Token，需要搬运一次完整的模型参数到显存/内存
            # 带宽消耗 (GB/s) = 平均 TPS * 模型大小 (GB)
            estimated_bw = avg_tps * MODEL_SIZE_GB
            print(f"{length:<20} | {avg_tps:<20.2f} | {estimated_bw:.2f}")

    print("-" * 60)
    print(f"💡 结论分析：")
    print(f"1. 注意观察不同生成长度下的 TPS，它们应该是基本恒定的。")
    print(f"2. 你的设备理论内存带宽通常在 100-400 GB/s 之间。")
    print(f"3. 如果预估带宽接近物理带宽上限，说明你的 GPU 已处于‘喂不饱’状态。")

if __name__ == "__main__":
    run_experiment()
