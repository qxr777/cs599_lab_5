import asyncio
import aiohttp
import time
import statistics

# 实验配置
API_URL = "http://localhost:8080/completion"
CONCURRENCY_LEVELS = [1, 4, 8, 16, 32] # 模拟不同数量的并发用户
REQUESTS_PER_USER = 1
GEN_LEN = 128

async def send_request(session, user_id):
    payload = {
        "prompt": "Tell me a long story about a bridge construction.",
        "n_predict": GEN_LEN,
        "stream": False
    }
    
    start = time.time()
    try:
        async with session.post(API_URL, json=payload) as response:
            data = await response.json()
            end = time.time()
            
            # 获取内部解码数据
            timings = data.get("timings", {})
            tps = timings.get("predicted_per_second", 0)
            return {
                "user_id": user_id,
                "latency": end - start,
                "tps": tps,
                "success": True
            }
    except Exception as e:
        return {"user_id": user_id, "success": False, "error": str(e)}

async def run_stress_level(concurrency):
    print(f"\n🚀 正在测试并发级别: {concurrency} 用户")
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(concurrency)]
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
    total_time = end_time - start_time
    success_results = [r for r in results if r["success"]]
    
    if not success_results:
        print("❌ 所有请求均失败")
        return

    avg_latency = statistics.mean([r["latency"] for r in success_results])
    # 系统总吞吐量 = (总生成 Token 数) / (总实验时间)
    total_tokens = sum([GEN_LEN for r in success_results])
    system_throughput = total_tokens / total_time
    
    print(f"✅ 完成！成功率: {len(success_results)}/{concurrency}")
    print(f"⏱️ 平均首位用户感知延迟: {avg_latency:.2f}s")
    print(f"📊 系统总吞吐量: {system_throughput:.2f} tokens/s")

async def main():
    print("=" * 60)
    print("  CS599 Lab 5: Experiment D - PagedAttention & 高并发性能测试")
    print("=" * 60)
    print("说明：观察随着并发增加，系统总吞吐量是否能保持高位，还是会由于内存压力崩溃。")
    
    for level in CONCURRENCY_LEVELS:
        await run_stress_level(level)
        await asyncio.sleep(2) # 给服务器一点喘息时间

if __name__ == "__main__":
    asyncio.run(main())
