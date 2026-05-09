import asyncio
import aiohttp
import time
import json
import statistics
import matplotlib.pyplot as plt
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# 1. 极简 OTLP 配置 (绕过版本不兼容的 SDK 对象)
SAMPLES_PER_CONCURRENCY = 3  # 每个并发级别采样多次取平均
GEN_LENGTH = 32              # 缩短生成长度，专注于测试 TTFT
resource = Resource(attributes={"service.name": "CS599-Stress-Eval"})
provider = TracerProvider(resource=resource)
# 确保 Phoenix Server 已启动并在 6006 端口监听
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

async def send_request(session, user_id):
    with tracer.start_as_current_span(f"User-{user_id}-Req") as span:
        url = "http://localhost:8080/completion"
        # 语法约束
        try:
            with open("bridge_standard.gbnf", 'r', encoding='utf-8') as f:
                grammar = f.read()
        except:
            grammar = ""

        prompt_text = "发现桥轴线 4.2 位置出现贯穿裂缝，请输出 JSON 审计结果。"
        payload = {
            "prompt": prompt_text,
            "n_predict": GEN_LENGTH,
            "temperature": 0,
            "stream": True,
            "grammar": grammar
        }

        # --- 记录 Prompt 到 Span ---
        span.set_attribute("prompt", prompt_text)
        span.set_attribute("n_predict", GEN_LENGTH)
        span.set_attribute("temperature", payload["temperature"])

        start_time = time.time()
        ttft = None
        full_content = ""

        try:
            async with session.post(url, json=payload, timeout=30) as response:
                async for line in response.content:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        try:
                            chunk = json.loads(line_str.replace('data: ', ''))
                            if ttft is None:
                                ttft = (time.time() - start_time) * 1000
                            full_content += chunk.get("content", "")
                        except: continue

            # --- 记录返回结果到 Span ---
            span.set_attribute("ttft_ms", round(ttft, 2))
            span.set_attribute("response", full_content)
            span.set_attribute("status", "success" if ttft else "timeout")
            return ttft
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("status", "error")
            span.set_attribute("error", str(e))
            return None

async def run_batch(concurrency):
    async with aiohttp.ClientSession() as session:
        all_ttfts = []
        for _ in range(SAMPLES_PER_CONCURRENCY):
            tasks = [send_request(session, i) for i in range(concurrency)]
            ttfts = await asyncio.gather(*tasks)
            all_ttfts.extend([t for t in ttfts if t is not None])
        return statistics.mean(all_ttfts) if all_ttfts else None

def finalize_report(concurrencies, latencies):
    plt.figure(figsize=(10, 6))
    plt.plot(concurrencies, latencies, marker='o', linestyle='-', color='#2980b9', linewidth=2)
    plt.fill_between(concurrencies, latencies, color='#3498db', alpha=0.1)
    plt.title('CS599 Bridge Audit: Concurrency vs Latency')
    plt.xlabel('Concurrent Users')
    plt.ylabel('Mean TTFT (ms)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('cs599_stress_eval_final.png')
    print("\n✅ 图表已保存: cs599_stress_eval_final.png")

async def run_experiment():
    print("🚦 启动专业版 [并发压力追踪器]...")
    
    # --- 预热阶段 ---
    print("🔥 正在进行 Warm-up 预热...")
    async with aiohttp.ClientSession() as session:
        await send_request(session, 0)
    print("✅ 预热完成。")
    
    concurrencies = [1, 2, 4, 8, 16]
    latencies = []
    
    for c in concurrencies:
        print(f"执行 {c} 并发测试中...")
        avg_ttft = await run_batch(c)
        if avg_ttft: 
            latencies.append(avg_ttft)
            print(f"   -> 平均 TTFT: {avg_ttft:.2f}ms")
        else:
            print(f"   -> ⚠️ {c} 并发测试失败")
        
    # 确保绘图时数据对齐：只绘制成功获取到延迟的并发级别
    valid_concurrencies = concurrencies[:len(latencies)]
    
    if latencies:
        finalize_report(valid_concurrencies, latencies)
        
    print("\n🏁 测试完成！基础 Trace 数据已注入 Phoenix。")
    print("💡 提示：在 Phoenix UI 的 Traces 列表中关注请求耗时 (Latency) 的变化即可。")

if __name__ == "__main__":
    asyncio.run(run_experiment())
