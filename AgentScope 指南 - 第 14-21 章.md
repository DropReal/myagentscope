# AgentScope 使用详细指南 - 第 14-21 章

## 14. RAG 知识增强

### 14.1 RAG 模块组成

AgentScope RAG 模块由三个核心组件构成：

| 组件 | 职责 | 内置实现 |
|------|------|---------|
| **Reader** | 读取数据并分块 | `TextReader`, `PDFReader`, `ImageReader`, `WordReader`, `ExcelReader`, `PowerPointReader` |
| **Knowledge** | 知识库查询和存储算法 | `SimpleKnowledge`, `KnowledgeBase` |
| **Store** | 与向量数据库交互 | `QdrantStore`, `MilvusLiteStore` |

### 14.2 Reader 使用

**TextReader 示例**:

```python
from agentscope.rag import TextReader
import asyncio

async def example_text_reader():
    reader = TextReader(
        chunk_size=512,        # 每块大小
        split_by="paragraph",  # 按段落分割
    )
    
    documents = await reader(
        text=(
            "我的名字是李明，今年 28 岁。\n"
            "我居住在中国杭州，是一名算法工程师。\n"
            "我父亲的名字是李强，是一名医生。\n"
        ),
    )
    
    print(f"文本被分块为 {len(documents)} 个 Document 对象")
    for idx, doc in enumerate(documents):
        print(f"Document {idx}:")
        print(f"\tScore: {doc.score}")
        print(f"\tMetadata: {doc.metadata}")

asyncio.run(example_text_reader())
```

**Document 对象结构**:

```python
class Document:
    metadata: DocMetadata  # 包含 content, doc_id, chunk_id, total_chunks
    embedding: list[float] | None  # 向量表示
    score: float | None  # 检索相关性分数
```

**自定义 Reader**:

```python
from agentscope.rag import ReaderBase, Document

class CustomReader(ReaderBase):
    """自定义 Reader"""
    
    async def __call__(self, source: str) -> list[Document]:
        """读取并分块"""
        # 实现自定义读取逻辑
        documents = []
        # ...
        return documents
```

### 14.3 向量存储

**QdrantStore**:

```python
from agentscope.rag import QdrantStore
from agentscope.embedding import DashScopeTextEmbedding

# 内存模式
qdrant_store = QdrantStore(
    location=":memory:",
    collection_name="my_knowledge",
    dimensions=1024,  # 必须与 embedding 模型输出维度一致
)

# 本地模式
qdrant_local = QdrantStore(
    location="./qdrant_data",
    collection_name="my_knowledge",
    dimensions=1024,
)

# 远程模式
qdrant_remote = QdrantStore(
    url="http://localhost:6333",
    api_key="your-qdrant-key",
    collection_name="my_knowledge",
    dimensions=1024,
)
```

**MilvusLiteStore**:

```python
from agentscope.rag import MilvusLiteStore

milvus_store = MilvusLiteStore(
    uri="./milvus_data.db",  # 本地数据库文件
    collection_name="my_knowledge",
    dimensions=1024,
)
```

### 14.4 知识库管理

**SimpleKnowledge**:

```python
from agentscope.rag import SimpleKnowledge
from agentscope.embedding import DashScopeTextEmbedding

async def build_knowledge_base():
    # 读取文档
    reader = TextReader()
    documents = await reader(text="...")
    
    # 创建知识库
    knowledge = SimpleKnowledge(
        embedding_model=DashScopeTextEmbedding(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            model_name="text-embedding-v4",
            dimensions=1024,
        ),
        embedding_store=QdrantStore(
            location=":memory:",
            collection_name="test_collection",
            dimensions=1024,
        ),
    )
    
    # 添加文档
    await knowledge.add_documents(documents)
    
    # 检索
    docs = await knowledge.retrieve(
        query="李明的父亲是谁？",
        limit=3,
        score_threshold=0.5,
    )
    
    for doc in docs:
        print(f"Score: {doc.score}, Content: {doc.metadata.content}")

asyncio.run(build_knowledge_base())
```

### 14.5 与 ReActAgent 集成

**方式 1: 智能体自主控制（Agentic Manner）**

```python
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit

async def agentic_rag_example():
    # 创建智能体
    agent = ReActAgent(
        name="Friday",
        sys_prompt="你是有帮助的助手",
        model=dashscope_model,
        formatter=DashScopeChatFormatter(),
        toolkit=Toolkit(),
    )
    
    # 第一次交流（不提供检索工具）
    await agent(
        Msg("user", "李明是我最好的朋友。", "user"),
    )
    
    # 注册检索工具
    agent.toolkit.register_tool_function(
        knowledge.retrieve_knowledge,
        func_description="用于检索与给定查询相关的文档。当你需要查找有关李明的信息时使用。",
    )
    
    # 第二次交流（智能体自主决定检索）
    await agent(
        Msg("user", "你知道他父亲是谁吗？", "user"),
    )
    # 智能体会改写查询为"李明的父亲"并检索
```

**方式 2: 通用方式（Generic Manner）**

```python
async def generic_rag_example():
    # 直接传递 knowledge 参数
    agent = ReActAgent(
        name="Friday",
        sys_prompt="你是有帮助的助手",
        model=dashscope_model,
        formatter=DashScopeChatFormatter(),
        knowledge=knowledge,  # 每次回复自动检索
    )
    
    await agent(
        Msg("user", "你知道李明的父亲是谁吗？", "user"),
    )
    
    # 查看检索信息如何插入到记忆
    content = (await agent.memory.get_memory())[1].content
    print(content)
    # <retrieved_knowledge>...</retrieved_knowledge>
```

**两种方式对比**:

| 集成方式 | 优点 | 缺点 |
|---------|------|------|
| **智能体自主控制** | 灵活性高，智能体自主决定何时检索，避免不必要查询 | 对 LLM 能力要求高 |
| **通用方式** | 实现简单，对 LLM 要求低 | 每次都查询，可能引入不必要检索 |

### 14.6 多模态 RAG

```python
from agentscope.rag import ImageReader, SimpleKnowledge
from agentscope.embedding import DashScopeMultiModalEmbedding

async def multimodal_rag_example():
    # 读取图片
    reader = ImageReader()
    docs = await reader(image_url="./example.png")
    
    # 创建多模态知识库
    knowledge = SimpleKnowledge(
        embedding_model=DashScopeMultiModalEmbedding(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            model_name="multimodal-embedding-v1",
            dimensions=1024,
        ),
        embedding_store=QdrantStore(
            location=":memory:",
            collection_name="multimodal_collection",
            dimensions=1024,
        ),
    )
    
    await knowledge.add_documents(docs)
    
    # 使用多模态模型
    agent = ReActAgent(
        name="Friday",
        sys_prompt="你是有帮助的助手",
        model=DashScopeChatModel(
            model_name="qwen3-vl-plus",  # 视觉语言模型
            api_key=os.environ["DASHSCOPE_API_KEY"],
        ),
        formatter=DashScopeChatFormatter(),
        knowledge=knowledge,
    )
    
    await agent(
        Msg("user", "你知道我的名字吗？", "user"),
    )
```

---

## 15. 嵌入模型

### 15.1 OpenAI 嵌入

```python
from agentscope.embedding import OpenAIEmbedding

async def openai_embedding_example():
    embedding_model = OpenAIEmbedding(
        model_name="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    
    texts = [
        "什么是 AgentScope？",
        "AgentScope 是一个多智能体框架",
    ]
    
    response = await embedding_model(texts)
    
    print(f"嵌入维度：{len(response.embeddings[0])}")
    print(f"Token 使用：{response.usage}")

asyncio.run(openai_embedding_example())
```

### 15.2 DashScope 嵌入

```python
from agentscope.embedding import DashScopeTextEmbedding

async def dashscope_embedding_example():
    embedding_model = DashScopeTextEmbedding(
        model_name="text-embedding-v2",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    )
    
    texts = ["法国的首都是哪里？", "巴黎是法国的首都"]
    response = await embedding_model(texts)
    
    print(f"嵌入：{response.embeddings}")

asyncio.run(dashscope_embedding_example())
```

### 15.3 嵌入缓存

```python
from agentscope.embedding import FileEmbeddingCache, OpenAIEmbedding

# 创建缓存
cache = FileEmbeddingCache(cache_path="./embedding_cache.json")

# 使用缓存
embedding_model = OpenAIEmbedding(
    model_name="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
    cache=cache,
)

# 相同文本不会重复调用 API
texts = ["相同文本", "相同文本"]
response1 = await embedding_model(texts)  # 调用 API
response2 = await embedding_model(texts)  # 从缓存读取

# 保存缓存
cache.save()
```

### 15.4 多模态嵌入

```python
from agentscope.embedding import DashScopeMultiModalEmbedding

multimodal_embedding = DashScopeMultiModalEmbedding(
    model_name="multimodal-embedding-v1",
    api_key=os.environ["DASHSCOPE_API_KEY"],
)

# 文本嵌入
text_response = await multimodal_embedding(texts=["文本"])

# 图像嵌入
image_response = await multimodal_embedding(images=["./image.jpg"])

# 视频嵌入
video_response = await multimodal_embedding(videos=["./video.mp4"])
```

---

## 16. 评估系统

### 16.1 评估框架组件

```
评估框架
├── 基准测试 (Benchmark)
│   └── 任务 (Task)
│       └── 指标 (Metric)
├── 评估器 (Evaluator)
│   ├── RayEvaluator（并行/分布式）
│   └── GeneralEvaluator（顺序）
└── 解决方案 (Solution)
```

### 16.2 自定义评估指标

```python
from agentscope.evaluate import MetricBase, MetricType, MetricResult, SolutionOutput, Task

class AccuracyMetric(MetricBase):
    """准确率评估指标"""
    
    name = "accuracy"
    metric_type = MetricType.HIGHER_IS_BETTER
    
    def __init__(self, ground_truth: str):
        super().__init__(
            name="accuracy",
            metric_type=MetricType.HIGHER_IS_BETTER,
            description="检查答案是否正确",
        )
        self.ground_truth = ground_truth
    
    async def __call__(self, solution: SolutionOutput) -> MetricResult:
        expected = self.ground_truth
        actual = solution.output
        
        is_correct = str(expected).lower() == str(actual).lower()
        return MetricResult(
            name=self.name,
            result=1.0 if is_correct else 0.0,
            message="正确" if is_correct else "错误",
        )
```

### 16.3 定义任务

```python
from agentscope.evaluate import Task

task = Task(
    id="math_problem_1",
    input="2 + 2 等于多少？",
    ground_truth="4",
    tags={
        "difficulty": "easy",
        "category": "math",
    },
    metrics=[
        AccuracyMetric(ground_truth="4"),
    ],
    metadata={},
)
```

### 16.4 定义基准测试

```python
from typing import Generator
from agentscope.evaluate import BenchmarkBase

class ToyBenchmark(BenchmarkBase):
    def __init__(self):
        super().__init__(
            name="Toy bench",
            description="示例基准测试",
        )
        self.dataset = self._load_data()
    
    @staticmethod
    def _load_data() -> list[Task]:
        dataset = []
        for item in TOY_BENCHMARK:
            dataset.append(
                Task(
                    id=item["id"],
                    input=item["question"],
                    ground_truth=item["ground_truth"],
                    tags=item.get("tags", {}),
                    metrics=[CheckEqual(item["ground_truth"])],
                ),
            )
        return dataset
    
    def __iter__(self) -> Generator[Task, None, None]:
        for task in self.dataset:
            yield task
    
    def __len__(self) -> int:
        return len(self.dataset)
```

### 16.5 评估器使用

**GeneralEvaluator**:

```python
from agentscope.evaluate import GeneralEvaluator, FileEvaluatorStorage

async def solution_generation(task: Task, pre_hook) -> SolutionOutput:
    # 定义智能体解决方案生成逻辑
    agent = ReActAgent(...)
    res = await agent(Msg("user", task.input, "user"))
    return SolutionOutput(
        success=True,
        output=res.get_text_content(),
        trajectory=[],
    )

async def main():
    evaluator = GeneralEvaluator(
        name="Toy benchmark evaluation",
        benchmark=ToyBenchmark(),
        n_repeat=1,
        storage=FileEvaluatorStorage(save_dir="./results"),
        n_workers=4,  # 并行工作进程数
    )
    
    await evaluator.run(solution_generation)

asyncio.run(main())
```

**RayEvaluator（分布式）**:

```python
from agentscope.evaluate import RayEvaluator

ray_evaluator = RayEvaluator(
    name="Distributed evaluation",
    benchmark=LargeBenchmark(),
    n_repeat=3,
    storage=FileEvaluatorStorage(save_dir="./results"),
    n_workers=16,  # 16 个工作进程
)

await ray_evaluator.run(solution_generation)
```

### 16.6 ACEBench

```python
from agentscope.evaluate import ACEBenchmark

ace_benchmark = ACEBenchmark(
    data_path="./ace_data/",
)

results = await ace_benchmark.run(agent)

print(f"总体得分：{results.overall_score}")
print(f"各维度得分：{results.dimension_scores}")
```

---

## 17. 链路追踪与监控

### 17.1 连接到 AgentScope Studio

```python
import agentscope
from agentscope.tracing import setup_tracing

# 方式 1: 通过 init 连接
agentscope.init(
    project="my_project",
    name="experiment_001",
    studio_url="http://localhost:3000",
)

# 方式 2: 单独设置链路
setup_tracing(
    endpoint="http://localhost:4318/v1/traces",  # OpenTelemetry 端点
)
```

### 17.2 追踪装饰器

```python
from agentscope.tracing import trace

# 追踪异步函数
@trace
async def process_data(data: dict) -> dict:
    return processed_data

# 追踪同步函数
@trace
def sync_operation(x: int, y: int) -> int:
    return x + y

# 自定义追踪名称
@trace(name="custom_operation")
async def my_function():
    pass
```

### 17.3 手动追踪

```python
from agentscope.tracing import start_span

async def manual_tracing():
    with start_span("custom_operation") as span:
        span.set_attribute("input_size", 100)
        span.add_event("processing_started")
        
        result = await process_data()
        
        span.set_attribute("output_size", len(result))
        span.add_event("processing_completed")
        
        return result
```

---

## 18. AgentScope Studio

### 18.1 安装和启动

```bash
# 安装
pip install agentscope-studio

# 启动
agentscope-studio --port 3000
```

### 18.2 功能特性

- 实时查看智能体对话
- 链路追踪可视化
- 评估结果展示
- 会话历史记录

---

## 19. MCP 集成

### 19.1 HTTP 无状态客户端

```python
from agentscope.mcp import HttpStatelessClient
from agentscope.tool import Toolkit

async def mcp_example():
    # 创建 MCP 客户端
    http_client = HttpStatelessClient(
        name="gaode_maps",
        transport="streamable_http",
        url=f"https://mcp.amap.com/mcp?key={os.environ['GAODE_API_KEY']}",
        timeout=30,
    )
    
    # 列出可用工具
    tools = await http_client.list_tools()
    print(f"可用工具：{[t.name for t in tools]}")
    
    # 获取可调用函数
    maps_geo = await http_client.get_callable_function("maps_geo")
    
    # 直接调用
    result = await maps_geo(address="天安门广场", city="北京")
    print(result)
    
    # 注册到工具包
    toolkit = Toolkit()
    await toolkit.register_mcp_client(
        http_client,
        group_name="basic",
        enable_funcs=["maps_geo", "maps_direction"],
    )

asyncio.run(mcp_example())
```

### 19.2 精细 MCP 控制

```python
# 获取工具作为本地函数
func = await client.get_callable_function(func_name="maps_geo")

# 方式 1: 直接调用
await func(address="...")

# 方式 2: 注册到工具包
toolkit.register_tool_function(func)

# 方式 3: 包装为复杂工具
async def enhanced_geo(address: str):
    result = await func(address=address)
    return process(result)
```

---

## 20. 智能体钩子函数

### 20.1 支持的钩子

| 钩子 | 触发时机 | 类支持 |
|------|---------|-------|
| `pre_reply` / `post_reply` | 回复前后 | `AgentBase`, `ReActAgentBase` |
| `pre_observe` / `post_observe` | 观察消息前后 | `AgentBase`, `ReActAgentBase` |
| `pre_print` / `post_print` | 打印消息前后 | `AgentBase`, `ReActAgentBase` |
| `pre_reasoning` / `post_reasoning` | 推理前后 | `ReActAgentBase` |
| `pre_acting` / `post_acting` | 行动前后 | `ReActAgentBase` |

### 20.2 注册钩子

```python
from agentscope.agent import ReActAgent

agent = ReActAgent(...)

# 注册实例钩子
def save_logging(msg):
    # 保存日志
    pass

agent.register_instance_hook(
    hook_name="pre_print",
    hook_id="save_logging",
    hook_func=save_logging,
)

# 注册类钩子
@ReActAgent.register_class_hook("post_reply")
def log_reply(agent, msg):
    print(f"回复：{msg.get_text_content()}")
```

---

## 21. 中间件

（中间件详细内容）
