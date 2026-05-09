# CS599 Lab 5：桥梁安全审计 LLM 推理性能与可靠性实验

> [!NOTE]
> **首席架构师寄语**：不要机械地执行命令，要理解比特（Bit）如何在硅片（Silicon）中流动。本实验旨在帮助你通过亲手测量，真正看清大语言模型在硬件上运行的"物理边界"与"逻辑行为"。

---

## 目录

- [1. 实验概述](#1-实验概述)
- [2. 环境准备](#2-环境准备)
- [3. 实验一：GBNF 确定性约束注入](#3-实验一gbnf-确定性约束注入)
- [4. 实验二：Prefill 计算密度验证](#4-实验二prefill-计算密度验证)
- [5. 实验三：Decoding 内存带宽瓶颈验证](#5-实验三decoding-内存带宽瓶颈验证)
- [6. 实验四：高并发压力与排队延迟分析](#6-实验四高并发压力与排队延迟分析)
- [7. 实验五：并行架构吞吐量验证（PagedAttention）](#7-实验五并行架构吞吐量验证pagedattention)
- [8. 实验六：注意力架构对比分析（MLA 理论 + 量化）](#8-实验六注意力架构对比分析mla-理论-量化)
- [9. 实验总结与综合思考](#9-实验总结与综合思考)
- [附录 A：术语表](#附录-a术语表)
- [附录 B：故障排查](#附录-b故障排查)

---

## 1. 实验概述

### 1.1 实验目标

本实验以"桥梁安全审计"为业务场景，围绕本地部署的 llama.cpp 推理引擎，完成以下六个相互关联的子实验：

| 实验 | 核心问题 | 性能维度 |
|------|----------|----------|
| 一 | 如何让 LLM 输出 100% 合规的结构化数据？ | **可靠性** |
| 二 | 输入越长，处理时间如何变化？为什么？ | **计算密度（Compute-bound）** |
| 三 | 生成速度是否随长度变化？瓶颈在哪里？ | **内存带宽（Memory-bound）** |
| 四 | 并发用户增多时，延迟如何变化？何时"爆炸"？ | **并发调度与排队论** |
| 五 | 并行处理相比串行，吞吐量提升多少？ | **系统级并行效率** |
| 六 | MHA/GQA/MLA 的 KV Cache 与计算效率差异？MLA 的核心动机是什么？ | **理论+量化** |

### 1.2 前置知识

- 理解 Transformer 的基本架构（Attention、KV Cache）
- 了解 GPU 内存带宽与算力（FLOPs）的基本概念
- 具备 Python 基础，了解 `asyncio` 异步编程模型
- 已安装并可用 `llama-server`（llama.cpp 推理服务器）

### 1.3 系统架构概览

```
┌─────────────────┐     HTTP POST      ┌──────────────────────────┐
│  实验脚本 (Py)   │ ──────────────────▶ │  llama-server (:8080)    │
│  requests/aiohttp │ ◀────────────────── │  Qwen2.5-7B Q4_K_M       │
└─────────────────┘    JSON Response     │  --parallel 8            │
                                         │  --ctx-size 8192         │
┌─────────────────┐                      │  --n-gpu-layers 99       │
│ Arize Phoenix   │ ◀──── OTLP Trace ─── │                          │
│  (:6006)        │                      └──────────────────────────┐
└─────────────────┘
```

---

## 2. 环境准备

### 2.1 Python 虚拟环境

```bash
python3 -m venv venv_cs599
source venv_cs599/bin/activate
pip install openai aiohttp matplotlib pandas requests arize-phoenix
```

### 2.2 部署 llama.cpp 推理服务器

#### 2.2.1 获取模型

下载 `Qwen2.5-7B-Instruct-Q4_K_M.gguf` 至 `~/models/` 目录。Q4_K_M 是 4-bit 量化格式，模型文件大小约 4.7 GB。

#### 2.2.2 启动推理服务

```bash
llama-server -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --parallel 8 \
    --ctx-size 8192 \
    --port 8080 \
    --n-gpu-layers 99
```

#### 2.2.3 关键参数物理意义

| 参数 | 物理含义 | 对实验的影响 |
|------|----------|-------------|
| `--parallel 8` | **物理槽位数（Slots）**。决定了同时处理多少个"正在进行中"的请求。每个 Slot 独立维护自己的 KV Cache 和生成状态。 | 实验四、五中，当并发请求数 > 8 时，超出的请求必须在服务端内部排队 |
| `--ctx-size 8192` | **KV Cache 总池大小**。所有 Slot 共享的 Token 存储空间，单位为 Token。 | 限制单个请求的最大上下文长度。实验二的 prompt 长度不应超过此值 |
| `--n-gpu-layers 99` | **GPU 卸载层数**。99 表示全部层载入 GPU。 | 确保实验数据稳定。若部分层在 CPU，Prefill/Decoding 速度将显著下降 |

#### 2.2.4 验证服务就绪

```bash
curl http://localhost:8080/health
# 应返回 {"status":"ok"}
```

### 2.3（可选）启动 Arize Phoenix

实验四会将 OpenTelemetry Trace 发送到 Phoenix，用于可视化追踪请求延迟。

```bash
phoenix serve
# 默认监听 http://localhost:6006
```

---

## 3. 实验一：GBNF 确定性约束注入

### 3.1 学习目标

- 理解 LLM 自由生成时为何会输出非结构化文本
- 掌握 GBNF（GGML BNF）语法约束的工作原理
- 验证 GBNF 能否将概率性输出转化为确定性 JSON

### 3.2 理论解析

**问题本质**：LLM 是一个自回归概率模型。给定前缀，它输出下一个 token 的概率分布。如果不加约束，模型会按照训练时的语言习惯自由采样——这意味着它可能在 JSON 前后添加"好的，这是结果："之类的闲聊文本。

**GBNF 的解决思路**：在每次 token 采样之前，系统用 GBNF 定义的语法规则构造一个**有限状态机**。状态机会将所有"不符合当前语法路径"的 token 概率设为负无穷（`-inf`），使其不可能被采样到。

可以类比为：
- **普通采样**：在空旷草地上开车，方向随意，结果不可预测
- **GBNF 约束**：在铁轨上开火车，即使引擎再强，也只能沿轨道行驶

### 3.3 代码分析

打开 `bridge_inspector.py`，关注以下关键点：

```python
# 对照组：自由生成（temperature=0.8，无约束）
raw_output = call_llm(use_constraint=False)

# 实验组：GBNF 约束（temperature=0.1，加载 grammar）
gbnf_output = call_llm(use_constraint=True)
```

GBNF 语法定义在 `bridge_standard.gbnf` 中：

```
root ::= report
report ::= "{" "\"bridge_id\":" string ",\"inspector\":" string ",\"elements\":" elements "}"
elements ::= "[" (element ("," element)*)? "]"
element ::= "{" "\"type\":" etype ",\"index\":" number ",\"grade\":" grade ",\"severity\":" float "}"
etype ::= "\"pier\"" | "\"girder\"" | "\"deck\"" | "\"abutment\""
grade ::= "\"A\"" | "\"B\"" | "\"C\"" | "\"D\"" | "\"E\""
```

这条语法规定了：输出必须是一个包含 `bridge_id`、`inspector` 和 `elements` 数组的 JSON 对象，且每个 element 的类型、等级和严重度都在预设枚举范围内。

### 3.4 执行步骤

```bash
python bridge_inspector.py
```

### 3.5 预期结果

- **对照组**：模型输出一大段自由文本，可能包含解释、描述等非 JSON 内容
- **实验组**：输出严格匹配 GBNF 语法，可被 `json.loads()` 直接解析

示例输出片段：
```
{"bridge_id":"京沪高速K122+500","inspector":"张伟","elements":[{"type":"pier","index":1,"grade":"A","severity":0.1},{"type":"pier","index":3,"grade":"D","severity":0.75}]}
```

### 3.6 思考题

1. 如果将 `temperature` 在 GBNF 约束下从 0.1 调高到 1.0，输出还会变化吗？为什么？
2. GBNF 约束是否影响模型的推理质量（内容准确性），还是只影响格式？

---

## 4. 实验二：Prefill 计算密度验证

### 4.1 学习目标

- 理解 Prefill 阶段的计算特性
- 验证输入 Token 数量与处理时间的线性关系
- 建立"Compute-bound"的直观认知

### 4.2 理论解析

**Prefill（预填充）阶段**：当 LLM 收到完整 prompt 后，需要将所有输入 token 的 embedding 通过整个 Transformer 网络进行一次前向传播，生成初始 KV Cache。

**为何是 Compute-bound**：
- 输入 prompt 是一次性并行处理的，所有 token 的 attention 计算可以打包成一个大的矩阵乘法
- 计算量随输入长度 $N$ 呈 $O(N^2)$ 增长（self-attention），但由于矩阵并行，实际耗时近似线性增长
- 数据已经全部在 GPU 显存中，不需要反复从内存搬运

公式：`prompt_tokens / prompt_ms × 1000 = Tokens/s`，该值应接近硬件的矩阵算力上限。

### 4.3 代码分析

`experiment_A_compute.py` 的核心逻辑：

```python
payload = {
    "prompt": f"Based on this: {prompt} \nSummarize in one word:",
    "n_predict": 1,  # 只生成 1 个 token，隔离 Prefill 阶段
    "temperature": 0.0,
    "stream": False
}
timings = data.get("timings", {})
prefill_ms = timings.get("prompt_ms")  # llama.cpp 返回的纯净计算时间
```

- `n_predict=1`：强制只生成 1 个 token，确保总时间几乎全部来自 Prefill
- `prompt_ms`：从 llama.cpp 的服务端计时器获取，排除了网络传输和 Python 脚本的开销
- 测试长度：128、512、1024、2048、4096 tokens，每个长度采样 3 次取均值

### 4.4 执行步骤

```bash
python experiment_A_compute.py
```

### 4.5 预期结果

输出表格如下：

```
输入长度 (Tokens)    | 平均预填充耗时 (ms)   | 速度 (Tokens/s)
-------------------------------------------------------------
128                  | 5.xx                  | 2xxxx.xx
512                  | 2x.xx                 | 2xxxx.xx
1024                 | 4x.xx                 | 2xxxx.xx
2048                 | 9x.xx                 | 2xxxx.xx
4096                 | 19x.xx                | 2xxxx.xx
```

**关键观察**：
- 随着输入长度翻倍，耗时近似翻倍（线性关系）
- Tokens/s 应基本保持恒定，说明 GPU 算力稳定
- 如果短 prompt 的 Tokens/s 异常偏高，是因为固定开销（kernel launch）在小矩阵上占比更大

### 4.6 思考题

1. 如果将 `--ctx-size` 从 8192 降低到 2048，实验结果会怎样？
2. 在消费级 GPU（如 RTX 4060）和数据中心 GPU（如 A100）上，Prefill 速度可能相差 10 倍。这是因为什么硬件指标的差异？

---

## 5. 实验三：Decoding 内存带宽瓶颈验证

### 5.1 学习目标

- 理解 Decoding 阶段与 Prefill 阶段的本质区别
- 验证生成速率 TPS 的恒定性
- 建立"Memory-bound"的直观认知

### 5.2 理论解析

**Decoding（自回归生成）阶段**：每生成 1 个 token，都需要执行一次完整的前向传播——读取全部模型权重，计算 attention，输出下一个 token 的概率。

**为何是 Memory-bound**：
- 每次只生成 1 个 token，计算量很小（向量-矩阵运算），但必须从显存/内存中**搬运整个模型**（Qwen2.5-7B Q4_K_M 约 4.7 GB）
- 瓶颈不在计算能力，而在**内存带宽**（Memory Bandwidth）
- 公式：`带宽消耗 (GB/s) = TPS × 模型大小 (GB)`

**Prefill vs Decoding 对比**：

| | Prefill | Decoding |
|---|---------|----------|
| 计算模式 | 矩阵-矩阵乘法（高并行） | 向量-矩阵乘法（低并行） |
| 瓶颈 | GPU 算力（FLOPs） | 内存带宽（GB/s） |
| 速度单位 | Tokens/s（吞吐） | TPS/token（逐个） |
| 输入增加 | 耗时线性增长 | 几乎不影响 |

### 5.3 代码分析

`experiment_B_bandwidth.py` 的核心逻辑：

```python
payload = {
    "prompt": "Once upon a time in a distant galaxy,",  # 短 prompt
    "n_predict": gen_len,     # 强制生成 N 个 token
    "temperature": 0.7,
    "stream": False
}
timings = data.get("timings", {})
tps = timings.get("predicted_per_second")  # 直接从 llama.cpp 获取
estimated_bw = avg_tps * MODEL_SIZE_GB     # 推算带宽消耗
```

- 使用极短的 prompt，使 Prefill 时间可以忽略
- `n_predict` 从 64 到 512 递增，观察 TPS 是否恒定
- 估算带宽消耗 = TPS × 4.7 GB，与设备理论带宽上限对比

### 5.4 执行步骤

```bash
python experiment_B_bandwidth.py
```

### 5.5 预期结果

```
生成长度 (Tokens)    | 生成速率 (TPS)       | 预估带宽消耗 (GB/s)
-------------------------------------------------------------
64                   | 25.xx               | 11x.xx
128                  | 25.xx               | 11x.xx
256                  | 25.xx               | 11x.xx
512                  | 25.xx               | 11x.xx
```

**关键观察**：
- 不同生成长度下的 TPS 应**基本恒定**，说明 Decoding 速度与长度无关
- 预估带宽消耗如果接近设备内存带宽上限（如 MacBook M 系列约 100-200 GB/s，RTX 4090 约 1008 GB/s），说明 GPU 已"喂不饱"

### 5.6 思考题

1. 如果将模型从 Q4_K_M（4-bit 量化）换成 FP16，TPS 会如何变化？为什么？
2. 为什么"输入快、输出慢"是 LLM 推理的普遍现象？这跟 Attention 的计算方式有什么关系？

---

## 6. 实验四：高并发压力与排队延迟分析

### 6.1 学习目标

- 理解 llama-server 的 Slot 调度机制
- 观察并发用户数增加时 TTFT 的变化规律
- 识别"排队爆炸"的临界点
- 掌握使用 OpenTelemetry 追踪 LLM 推理请求

### 6.2 理论解析

**llama-server 的并发模型**：
- 启动时通过 `--parallel N` 分配 N 个 Slot
- 当并发请求数 ≤ N 时，每个请求分配到独立的 Slot，并行处理
- 当并发请求数 > N 时，超出的请求在服务器内部 FIFO 排队，等待空 Slot

**排队延迟的 Waterfall 效应**：

```
并发数 1-8:  TTFT 缓慢上升（仅受 GPU 算力分摊影响）
并发数 16+:  TTFT 剧增（Slot 耗尽，请求排队，产生瀑布式叠加延迟）
```

**TTFT（Time To First Token）**：从请求发出到收到第一个生成 token 的时间。它由三部分组成：

```
TTFT = 网络延迟 + 排队等待时间 + Prefill 计算时间
```

在并发压力下，**排队等待时间**成为主导项。

### 6.3 代码分析

`experiment_C_stress_eval.py` 的核心组件：

#### 6.3.0 OTLP 初始化配置

```python
resource = Resource(attributes={"service.name": "CS599-Stress-Eval"})
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(OTLPSpanExporter(
    endpoint="http://localhost:6006/v1/traces"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)
```

这段代码将 Tracer 的 Span 批量发送到 Phoenix 的 OTLP 接收端（`localhost:6006`）。

#### 6.3.1 异步并发请求 + Span 属性注入

```python
async def send_request(session, user_id):
    with tracer.start_as_current_span(f"User-{user_id}-Req") as span:
        # 请求前：记录 prompt 和配置参数
        span.set_attribute("prompt", prompt_text)
        span.set_attribute("n_predict", GEN_LENGTH)

        async with session.post(url, json=payload, timeout=30) as response:
            async for line in response.content:
                if ttft is None:  # 首次收到数据时记录时间
                    ttft = (time.time() - start_time) * 1000
                full_content += chunk.get("content", "")

        # 请求后：记录响应数据和性能指标
        span.set_attribute("ttft_ms", round(ttft, 2))
        span.set_attribute("response", full_content)
        span.set_attribute("status", "success" if ttft else "timeout")
        return ttft
    except Exception as e:
        span.record_exception(e)
        span.set_attribute("status", "error")
        return None
```

**关键点**：
- `span.set_attribute("prompt", ...)` — 在 Phoenix 中可直接看到每个请求的输入
- `span.set_attribute("response", ...)` — 记录完整返回内容，用于验证 GBNF 约束是否生效
- `span.set_attribute("ttft_ms", ...)` — 将 TTFT 作为 Span 属性，便于在 Phoenix 中按耗时排序
- `span.record_exception(e)` — 异常 Span 会被 Phoenix 标记为错误，方便快速定位问题请求

#### 6.3.2 批量执行

```python
async def run_batch(concurrency):
    tasks = [send_request(session, i) for i in range(concurrency)]
    ttfts = await asyncio.gather(*tasks)  # 同时触发 N 个请求
    return statistics.mean(all_ttfts)
```

### 6.4 执行步骤

1. （可选）启动 Phoenix：`phoenix serve`
2. 运行实验：

```bash
python experiment_C_stress_eval.py
```

### 6.5 预期结果

图表 `cs599_stress_eval_final.png` 应呈现以下趋势：

```
TTFT (ms)
  ↑
  |                                    ***
  |                               ****
  |                          ****
  |                    ****
  |               ****
  |          ****
  |     ****
  |  ***
  +--------------------------------→ 并发数
    1    2    4    8   16
```

- **1-8 并发**：TTFT 缓慢线性增长（GPU 算力分摊）
- **16 并发**：TTFT 出现明显拐点（超出 8 个 Slot，请求开始排队）

### 6.6 思考题

1. 如果将 `--parallel` 从 8 调整到 16，"排队爆炸"的拐点会出现在哪个并发数？
2. TTFT 和总延迟（Total Latency）有什么区别？在高并发场景下哪个指标更能反映用户体验？

---

## 7. 实验五：并行架构吞吐量验证（PagedAttention）

### 7.1 学习目标

- 理解 PagedAttention 的基本原理
- 对比串行处理与并行处理的吞吐量差异
- 评估系统在不同并发级别下的吞吐效率

### 7.2 理论解析

**PagedAttention 原理**：

传统的 LLM 推理中，KV Cache 在显存中是连续分配的。这导致了两个问题：
1. **内存碎片**：不同请求的序列长度不同，连续分配会产生大量无法利用的碎片空间
2. **并发受限**：每个 Slot 需要预分配最大长度的 KV Cache，即使实际使用很少

PagedAttention（源自 vLLM 论文）的解决方案：
- 将 KV Cache 分割为固定大小的 **Page（页）**
- 按需分配，类似操作系统的虚拟内存分页
- 多个请求可以共享同一个 Block Table，动态映射到物理 Page
- 结果：显存利用率从 ~20% 提升到 ~90%+，支持更多并发 Slot

**本实验中的体现**：
llama.cpp 的 `--parallel` 机制本身就体现了并行 Slot 的思想。当多个请求并行处理时：
- 每个 Slot 独立维护自己的 KV Cache
- GPU 通过 batched 推理同时服务多个 Slot
- 总吞吐量（系统每秒生成的总 Token 数）远高于串行

### 7.3 代码分析

`experiment_D_paged_attention.py` 的核心逻辑：

```python
# 测量总实验时间
start_time = time.time()
results = await asyncio.gather(*tasks)  # 并行处理
total_time = end_time - start_time

# 系统吞吐量 = 总 Token 数 / 总实验时间
total_tokens = sum([GEN_LEN for r in success_results])
system_throughput = total_tokens / total_time
```

- 测试并发级别：1、4、8、16 个用户
- 每个用户请求生成 128 个 token
- 记录系统总吞吐量（tokens/s）

### 7.4 执行步骤

```bash
python experiment_D_paged_attention.py
```

### 7.5 预期结果

```
并发 1:  系统吞吐量 ~25 tokens/s  (单一 Slot 全速运行)
并发 4:  系统吞吐量 ~80 tokens/s  (4 个 Slot 并行，接近 4x 线性加速)
并发 8:  系统吞吐量 ~120 tokens/s (8 个 Slot 满载，加速比开始衰减)
并发 16: 系统吞吐量 ~120 tokens/s (8 个 Slot 满载，8 个排队，吞吐量不再增长)
```

**关键观察**：
- 吞吐量随并发数增长而提升，但在 Slot 耗尽后趋于平缓
- 并行处理的总吞吐量远高于串行（逐个处理相同请求的耗时累加）

### 7.6 思考题

1. 为什么 8 Slot 并行时的吞吐量不是 1 Slot 的 8 倍？（提示：考虑 GPU SM 利用率和 batch overhead）
2. PagedAttention 主要解决的是 Prefill 阶段还是 Decoding 阶段的内存问题？为什么？

---

## 8. 实验六：注意力架构对比分析（MLA 理论 + 量化）

### 8.1 学习目标

- 理解 MHA、GQA 和 MLA 的架构差异及其对 KV Cache 的影响
- 量化对比三种架构在不同上下文长度下的显存占用
- 通过 llama.cpp 实测单 Slot TPS vs Context 长度的关系
- 理解 DeepSeek-V3 采用 MLA 降低推理成本的根本原因

### 8.2 理论解析

**KV Cache 是 LLM 推理的"内存税"**：
在 Decoding 阶段，为了避免重复计算已有 token 的 K 和 V，系统会将它们缓存下来。KV Cache 的大小为：

```
KV Cache = 2 × seq_len × num_layers × num_heads × head_dim × bytes_per_param
```

**三种架构的演进**：

| 架构 | 核心思路 | 每层每 Token KV 开销 | 代表模型 |
|------|---------|---------------------|----------|
| **MHA** | 标准多头注意力 | `2 × heads × head_dim × 2 字节` | GPT-3 |
| **GQA** | 多 Q head 共享 KV head | `2 × kv_heads × head_dim × 2 字节` | Qwen2.5-7B |
| **MLA** | 投影到低维潜空间 | `latent_dim × 2 字节` | DeepSeek-V2/V3 |

- **MHA**：KV Cache 最大，计算开销也最大
- **GQA**：KV Cache 大幅缩小（缩小倍数 = Q heads / KV heads，本实验中为 8 倍）
- **MLA**：每层每 Token 开销从 GQA 的 8 KB 降至 1 KB（512 维潜向量）

**MLA 的双重优势**：

1. **内存效率**：每层每 Token 的 KV 开销大幅降低（512 字节 vs GQA 的 4096 字节）
2. **计算效率**：Attention 计算在低维潜空间进行，矩阵乘法规模缩小约 8 倍

⚠️ **重要提醒**：在 7B 级别模型中，MLA 的总 KV Cache 不一定比 GQA 小——因为 DeepSeek-V2 有 60 层而 Qwen2.5 仅 28 层，层数差异几乎抵消了单层开销的优势（MLA/GQA ≈ 1.07x）。**MLA 真正的爆发点在大规模模型（如 DeepSeek-V3 的 671B 参数），在那里 KV Cache 和计算开销都会呈数量级放大。**

### 8.3 Part A：理论计算代码

```python
# MHA：标准多头注意力
size = 2 * seq_len * num_layers * num_heads * head_dim * bytes_per_param

# GQA：分组查询注意力（Qwen2.5-7B: 28 layers, 4 KV heads, head_dim=128）
size = 2 * seq_len * num_layers * num_kv_heads * head_dim * bytes_per_param

# MLA：多头潜注意力（DeepSeek-V2: 60 layers, latent_dim=512）
size = seq_len * num_layers * latent_dim * bytes_per_param
```

实验参数设置：
- MHA: `num_layers=32, num_heads=32, head_dim=128`
- GQA: `num_layers=28, num_kv_heads=4, head_dim=128`（Qwen2.5-7B）
- MLA: `num_layers=60, latent_dim=512`（DeepSeek-V2）
- 序列长度覆盖 1024 到 128000（128k）

### 8.4 Part B：并发承载能力量化

将理论计算转化为生产环境真正关心的指标：**相同显存下能同时服务多少个请求？**

```python
def compute_max_concurrent(avail_mem_mb, seq_len, layers, kv_heads, head_dim, arch_type):
    kv_per_request_mb = calculate_kv_cache_size(seq_len, layers, kv_heads, head_dim, arch_type)
    return int(avail_mem_mb / kv_per_request_mb)
```

同时生成"每层每 Token KV 开销"对比图，直观展示 MLA 的底层优势。

### 8.5 Part C：llama.cpp 实测 — 多并发请求的 TPS 衰减

通过增加并发请求数，观察 per-request TPS 的衰减曲线。

**原理**：
- 每请求使用 2048 context tokens，KV Cache ≈ 112 MB
- 16 个请求全满 = 16 × 112 MB ≈ 1.8 GB，远低于 M3 可用显存（~13 GB）
- 因此 TPS 衰减主要来自 **GPU 计算 contention**（多个 decode batch 共享 GPU SM），而非显存溢出
- ⚠️ Apple M3 在 `--parallel > 16` 且总 context pool > 65536 时会触发 GPU Metal OOM，故采用保守配置

**测试流程**：
1. Part C 会自动 kill 并重启 llama-server（`--parallel 16 -c 65536`）
2. 发送 1、2、4、8、16 个并发请求（每个请求 2048 context + 16 generated tokens）
3. 观察 per-request TPS 随并发数的衰减曲线

**思考**：在数据中心 GPU（如 A100）上，若 context 增大到 KV Cache 占满显存，会出现更剧烈的衰减（内存换页 + 计算 contention 叠加）。

### 8.6 执行步骤

```bash
python experiment_E_mla_theory.py
```

### 8.7 预期结果

**Part A 输出**（`cs599_mla_comparison.png`）：

| Context | MHA (MB) | GQA (MB) | MLA (MB) | MLA/GQA |
|---------|----------|----------|----------|---------|
| 1024 | 512 | 56 | 60 | 1.07x |
| 4096 | 2048 | 224 | 240 | 1.07x |
| 128000 | 64000 | 7000 | 7500 | 1.07x |

注意：MLA 与 GQA 的总 KV Cache 相近（MLA/GQA ≈ 1.07x），这是因为 DeepSeek-V2 的 60 层几乎抵消了单层开销的优势。

**Part B 输出**（13 GB 可用显存下的最大并发请求数）：

| Context | MHA | GQA | MLA | MLA vs GQA |
|---------|-----|-----|-----|------------|
| 2K | 13 | 118 | 110 | 0.9x |
| 8K | 3 | 29 | 27 | 0.9x |
| 32K | 0 | 7 | 6 | 0.9x |

在 7B 级别模型中，MLA 与 GQA 的并发能力相近。MLA 的优势将在大规模模型中完全体现。

**Part C 输出**：

```
并发请求数    | 总 KV Cache  | 平均 TPS     | 相对基准    | 状态
------------------------------------------------------------
1            | 0.1 GB      | 17.29       | 100.0%     | OK
2            | 0.2 GB      | 7.31        | 42.3%      | OK
4            | 0.4 GB      | 5.67        | 32.8%      | OK
8            | 0.9 GB      | 1.73        | 10.0%      | OK
16           | 1.8 GB      | 0.45        | 2.6%       | OK
```

当并发数从 1 增加到 16 时，per-request TPS 从 17.29 降至 0.45（衰减 97.4%），远超 30% 阈值。
注意总 KV Cache 仅 1.8 GB << 13 GB 可用显存，说明衰减来源是 **GPU SM 算力分摊**，而非显存溢出。

系统吞吐量对比：
- 1 并发：17.3 tokens/s（单请求）
- 16 并发：7.2 tokens/s（总产出）
- **系统总吞吐量反而下降**，说明在消费级 GPU 上，并发 decode 的 batch overhead 超过了并行加速的收益。

### 8.8 思考题

1. 为什么在 7B 级别模型中，MLA 的总 KV Cache 并不比 GQA 小？层数差异是如何抵消单层开销优势的？
2. 如果你的推理服务器内存有限（如只有 16 GB），在面对 32k 以上长上下文时，应该优先选择哪种架构的模型？
3. DeepSeek-V3 有 671B 参数。估算一下如果用 GQA 架构，128k 上下文下的 KV Cache 将有多大？MLA 解决了什么问题？

---

## 9. 实验总结与综合思考

### 9.1 核心结论回顾

通过六个实验，我们建立了从"代码"到"硅片"的完整认知链：

```
实验一（GBNF）      → 输出可靠性：给 LLM 铺铁轨，将概率生成变为确定性输出
实验二（Prefill）   → 输入端物理特性：Compute-bound，GPU 算力决定处理速度
实验三（Decoding）  → 输出端物理特性：Memory-bound，内存带宽决定生成速度
实验四（并发）      → 系统调度边界：Slot 耗尽引发排队延迟瀑布式增长
实验五（并行）      → 吞吐量扩展：并行 Slot 可提升整体产出，但存在加速上限
实验六（MLA）       → 理论+量化：MLA/GQA/MHA 架构对比 + 固定显存下并发承载能力
```

### 9.2 综合性能模型

将实验二和实验三的观察整合为一个简化的 LLM 推理时间模型：

```
总延迟 ≈ Prefill时间 + (生成Token数 / TPS) + 排队延迟

其中：
  Prefill时间 ∝ 输入Token数（线性，GPU 算力受限）
  TPS ≈ 常数（内存带宽受限，与生成长度无关）
  排队延迟 ≈ 0（并发数 ≤ Slot 数）
  排队延迟 >> 0（并发数 > Slot 数，FIFO 排队）
```

### 9.3 开放性思考题

1. **成本估算**：如果你的桥梁审计系统需要服务 1000 个并发用户，每份报告平均生成 200 个 token，假设每台服务器 8 个 Slot、TPS = 25，你需要多少台服务器？平均 TTFT 是多少？

2. **架构选型**：对比 MHA、GQA、MLA 三种架构，如果你要为一个长上下文（64k+）推理服务选型，会考虑哪些因素？KV Cache 大小是否是唯一考量？

3. **生产级优化**：本实验仅涉及了 llama.cpp 的基础配置。在生产环境中，还可以通过 Continuous Batching、Speculative Decoding、Prefix Caching 等技术进一步优化。查阅相关资料，简述这三种技术分别解决了什么问题。

---

## 附录 A：术语表

| 术语 | 全称 | 含义 |
|------|------|------|
| LLM | Large Language Model | 大语言模型 |
| Prefill | Pre-filling | LLM 处理输入 prompt 的并行计算阶段 |
| Decoding | Autoregressive Decoding | LLM 逐个生成 token 的自回归阶段 |
| KV Cache | Key-Value Cache | 缓存已有 token 的 K/V 状态，避免重复计算 |
| TTFT | Time To First Token | 从请求发出到收到第一个生成 token 的时间 |
| TPS | Tokens Per Second | 每秒生成的 token 数量 |
| GBNF | GGML BNF | llama.cpp 的语法约束格式，基于巴科斯-诺尔范式 |
| MHA | Multi-Head Attention | 标准多头注意力 |
| GQA | Grouped-Query Attention | 分组查询注意力 |
| MLA | Multi-Head Latent Attention | 多头潜注意力（DeepSeek 提出） |
| PagedAttention | — | vLLM 提出的 KV Cache 分页管理机制 |
| Slot | — | llama-server 的并发请求处理槽位 |
| FP16 | Half-Precision Floating Point | 16 位浮点数，每个参数占 2 字节 |
| Q4_K_M | 4-bit Quantization (K-quants, Medium) | 4-bit 量化格式之一 |

---

## 附录 B：故障排查

### B.1 llama-server 无法启动

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `llama-server: command not found` | 未安装 llama.cpp | 从 https://github.com/ggml-org/llama.cpp 编译安装，或 `brew install llama.cpp` |
| `error: failed to load model` | 模型文件路径错误 | 确认 `~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf` 存在 |
| `CUDA out of memory` | GPU 显存不足 | 降低 `--ctx-size`，或使用更小的量化模型 |
| GPU 加速未生效 | 未正确安装 CUDA 支持 | 确认 `llama-server` 输出中包含 `BLAS = CUDA` 等 GPU 后端信息 |

### B.2 实验脚本运行失败

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `Connection refused` | llama-server 未启动 | 先确认 `curl http://localhost:8080/health` 返回 `{"status":"ok"}` |
| 请求超时 | prompt 过长导致 Prefill 时间超过默认 timeout | 增大 `timeout` 参数，或减少 prompt 长度 |
| JSON 解析失败 | GBNF 约束未生效或 model 版本不同 | 确认 `bridge_standard.gbnf` 路径正确，检查 llama-server 日志 |
| Phoenix 无数据 | OTLP endpoint 配置错误 | 确认 `phoenix serve` 已在 6006 端口运行 |

### B.3 实验数据异常

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| Prefill 速度非线性增长 | GPU 显存不足导致 kernel 分片执行 | 降低 `--ctx-size` 或减少最大 prompt 长度 |
| Decoding TPS 波动大 | 后台进程抢占内存带宽 | 关闭其他 GPU/CPU 密集型应用 |
| 并发实验 TTFT 无拐点 | `--parallel` 设置过大 | 减小 `--parallel` 值以观察排队效应 |

---

**版本**: v6.0-Restructured
**最后更新**: 2026-05-09
