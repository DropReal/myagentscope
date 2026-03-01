# AgentScope 使用详细指南 - 第 22-28 章及附录

## 22. A2A 智能体

A2A（Agent-to-Agent）协议支持智能体间远程通信。

### 22.1 A2aAgent 使用

```python
from agentscope.agent import A2aAgent
from agentscope.message import Msg

async def a2a_example():
    # 创建 A2A 智能体
    a2a_agent = A2aAgent(
        name="RemoteAgent",
        agent_url="http://remote-server:8080/agent",
        api_key="your-api-key",
    )
    
    # 发送消息到远程智能体
    response = await a2a_agent(
        Msg("user", "执行远程任务", "user"),
    )
    
    print(response.get_text_content())
```

### 22.2 A2A 协议配置

```python
a2a_agent = A2aAgent(
    name="RemoteAgent",
    agent_url="http://localhost:8080/agent",
    timeout=30,
    max_retries=3,
    headers={"Authorization": "Bearer token"},
)
```

---

## 23. 实时智能体

实时智能体支持 WebSocket 连接和低延迟交互。

### 23.1 RealtimeAgent

```python
from agentscope.agent import RealtimeAgent

async def realtime_example():
    agent = RealtimeAgent(
        name="RealtimeAssistant",
        model=realtime_model,
        ws_url="ws://localhost:8080/realtime",
    )
    
    # 实时流式交互
    await agent.connect()
    await agent.send_audio(audio_data)
    async for response in agent.receive():
        print(response)
```

---

## 24. TTS 语音合成

### 24.1 DashScope TTS

```python
from agentscope.tts import DashScopeTextToSpeech

async def tts_example():
    tts = DashScopeTextToSpeech(
        model_name="sambert-zhichu-v1",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    )
    
    # 文本转语音
    audio_data = await tts(
        text="你好，欢迎使用 AgentScope。",
    )
    
    # 保存音频文件
    with open("output.wav", "wb") as f:
        f.write(audio_data)
```

### 24.2 OpenAI TTS

```python
from agentscope.tts import OpenAITextToSpeech

openai_tts = OpenAITextToSpeech(
    model_name="tts-1",
    api_key=os.environ["OPENAI_API_KEY"],
    voice="alloy",
)

audio = await openai_tts(text="Hello world")
```

---

## 25. Tuner 模型调优

### 25.1 调优配置

```python
from agentscope.tuner import tune, DatasetConfig, TunerModelConfig, AlgorithmConfig

# 数据集配置
dataset = DatasetConfig(
    path="my_dataset",
    split="train",
)

# 模型配置
model = TunerModelConfig(
    model_path="Qwen/Qwen3-0.6B",
    max_model_len=16384,
)

# 算法配置
algorithm = AlgorithmConfig(
    algorithm_type="multi_step_grpo",
    group_size=8,
    batch_size=32,
    learning_rate=1e-6,
)

# 工作流函数
async def run_react_agent(
    task: dict,
    model: ChatModelBase,
    auxiliary_models: dict[str, ChatModelBase],
) -> WorkflowOutput:
    agent = ReActAgent(
        name="react_agent",
        sys_prompt="你是数学问题求解助手",
        model=model,
        formatter=OpenAIChatFormatter(),
    )
    
    response = await agent.reply(
        msg=Msg("user", task["question"], role="user"),
    )
    
    return WorkflowOutput(response=response)

# 评判函数
async def judge_function(
    task: dict,
    response: Msg,
    auxiliary_models: dict[str, ChatModelBase],
) -> JudgeOutput:
    ground_truth = task["answer"]
    reward = 1.0 if ground_truth in response.get_text_content() else 0.0
    return JudgeOutput(reward=reward)

# 运行调优
tune(
    workflow_func=run_react_agent,
    judge_func=judge_function,
    model=model,
    train_dataset=dataset,
    algorithm=algorithm,
)
```

---

## 26. 高级应用案例

### 26.1 多智能体辩论系统

```python
from agentscope.pipeline import MsgHub, sequential_pipeline, fanout_pipeline
from agentscope.agent import ReActAgent

async def debate_system():
    """多智能体辩论系统"""
    
    # 创建辩论双方
    pro_agent = ReActAgent(
        name="正方",
        sys_prompt="你是辩论的正方，必须支持给定论点。",
        model=model,
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    con_agent = ReActAgent(
        name="反方",
        sys_prompt="你是辩论的反方，必须反对给定论点。",
        model=model,
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 裁判
    judge_agent = ReActAgent(
        name="裁判",
        sys_prompt="你是辩论裁判，给出公正评判。",
        model=model,
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    topic = "人工智能应该被严格监管"
    opening_msg = Msg("主持人", f"辩论主题：{topic}", "assistant")
    
    # 开篇立论
    pro_opening = await pro_agent(opening_msg)
    con_opening = await con_agent(pro_opening)
    
    # 自由辩论（3 轮）
    last_msg = con_opening
    for round_num in range(3):
        pro_response = await pro_agent(last_msg)
        con_response = await con_agent(pro_response)
        last_msg = con_response
    
    # 总结陈词
    pro_closing = await pro_agent(Msg("主持人", "请做最后总结", "assistant"))
    con_closing = await con_agent(pro_closing)
    
    # 裁判评判
    debate_history = [
        opening_msg, pro_opening, con_opening,
        pro_closing, con_closing,
    ]
    
    judgment = await judge_agent(
        Msg(
            "主持人",
            f"基于以下辩论历史给出评判:\n" + 
            "\n".join([f"{m.name}: {m.get_text_content()}" for m in debate_history]),
            "assistant",
        )
    )
    
    print(f"裁判评判:\n{judgment.get_text_content()}")
```

### 26.2 代码审查工作流

```python
from agentscope.pipeline import sequential_pipeline

async def code_review_workflow():
    """多阶段代码审查工作流"""
    
    # 各阶段审查智能体
    syntax_checker = ReActAgent(
        name="语法检查器",
        sys_prompt="检查代码语法错误、类型问题和基本规范。",
        model=model,
    )
    
    security_checker = ReActAgent(
        name="安全检查器",
        sys_prompt="检查安全漏洞，如 SQL 注入、XSS 等。",
        model=model,
    )
    
    performance_checker = ReActAgent(
        name="性能检查器",
        sys_prompt="识别性能问题，如低效循环、内存泄漏。",
        model=model,
    )
    
    fix_suggester = ReActAgent(
        name="修复建议器",
        sys_prompt="提供具体的代码修复建议。",
        model=model,
    )
    
    code_to_review = """
def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    return result
"""
    
    initial_msg = Msg("user", f"请审查这段代码:\n{code_to_review}", "user")
    
    # 顺序执行审查流程
    syntax_report = await syntax_checker(initial_msg)
    security_report = await security_checker(syntax_report)
    performance_report = await performance_checker(security_report)
    fix_suggestions = await fix_suggester(performance_report)
    
    print(f"修复建议:\n{fix_suggestions.get_text_content()}")
```

### 26.3 研究助手（RAG + 工具）

```python
from agentscope.rag import TextReader, QdrantStore, KnowledgeBase
from agentscope.embedding import OpenAIEmbedding
from agentscope.tool import Toolkit, execute_python_code

async def research_assistant():
    """智能研究助手"""
    
    # 构建知识库
    embedding = OpenAIEmbedding(
        model_name="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    
    paper_reader = TextReader()
    papers = paper_reader.read("./papers/ml_papers.txt")
    
    vector_store = QdrantStore(
        collection_name="research_papers",
        path="./qdrant_data",
        embedding=embedding,
    )
    
    knowledge_base = KnowledgeBase(
        store=vector_store,
        readers=[paper_reader],
    )
    await knowledge_base.add_documents(papers)
    
    # 创建工具包
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)
    
    # 研究助手智能体
    research_agent = ReActAgent(
        name="研究助手",
        sys_prompt="""你是智能研究助手：
        1. 基于知识库中的论文回答问题
        2. 可以执行 Python 代码进行数据分析
        3. 提供准确的引用来源""",
        model=OpenAIChatModel(
            model_name="gpt-4",
            api_key=os.environ["OPENAI_API_KEY"],
        ),
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
        toolkit=toolkit,
        knowledge=knowledge_base,
        enable_rewrite_query=True,
    )
    
    # 进行研究查询
    queries = [
        "Transformer 架构的核心创新是什么？",
        "请总结注意力机制的发展历程",
        "写一篇关于 BERT 的 Python 分析代码",
    ]
    
    for query in queries:
        response = await research_agent(Msg("user", query, "user"))
        print(f"问题：{query}")
        print(f"回答:\n{response.get_text_content()}\n")
```

### 26.4 任务规划与执行系统

```python
import json
import re
from agentscope.pipeline import sequential_pipeline

async def task_planning_system():
    """任务规划与执行系统"""
    
    # 规划器
    planner = ReActAgent(
        name="规划器",
        sys_prompt="""你是任务规划专家。
        将复杂任务分解为可执行的子任务。
        输出 JSON 格式：
        {"subtasks": [{"id": 1, "description": "...", "type": "code|shell|research"}]}""",
        model=model,
    )
    
    # 执行器
    executor = ReActAgent(
        name="执行器",
        sys_prompt="执行具体的代码和命令。",
        model=model,
        toolkit=Toolkit(),
    )
    executor.toolkit.register_tool_function(execute_python_code)
    
    # 验证器
    validator = ReActAgent(
        name="验证器",
        sys_prompt="验证执行结果是否满足任务要求。",
        model=model,
    )
    
    complex_task = """
    创建一个数据分析项目：
    1. 生成包含 1000 行模拟销售数据的 CSV 文件
    2. 分析数据并计算关键指标
    3. 创建可视化图表
    4. 生成分析报告
    """
    
    # 任务规划
    plan_response = await planner(
        Msg("user", f"请规划以下任务:\n{complex_task}", "user"),
    )
    
    # 解析规划
    plan_text = plan_response.get_text_content()
    json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
    if json_match:
        plan = json.loads(json_match.group())
    else:
        raise ValueError("未找到有效 JSON")
    
    # 执行子任务
    last_result = None
    for subtask in plan["subtasks"]:
        print(f"执行子任务 {subtask['id']}: {subtask['description']}")
        
        exec_msg = Msg(
            "planner",
            f"执行任务:\n{subtask['description']}",
            "assistant",
        )
        
        if last_result:
            exec_msg.content += f"\n上一步结果:\n{last_result.get_text_content()}"
        
        exec_result = await executor(exec_msg)
        print(f"执行结果:\n{exec_result.get_text_content()}")
        last_result = exec_result
    
    # 验证结果
    validation_result = await validator(
        Msg(
            "executor",
            f"原始任务:\n{complex_task}\n\n执行结果:\n{last_result.get_text_content()}",
            "assistant",
        )
    )
    print(f"验证结果:\n{validation_result.get_text_content()}")
```

---

## 27. 最佳实践

### 27.1 性能优化

**使用缓存**:
```python
from agentscope.embedding import FileEmbeddingCache

cache = FileEmbeddingCache(cache_path="./cache.json")
embedding_model = OpenAIEmbedding(..., cache=cache)
```

**批量处理**:
```python
# 批量调用嵌入
texts = ["text1", "text2", "text3"]
response = await embedding_model(texts)  # 一次调用
```

**并行执行**:
```python
from agentscope.pipeline import fanout_pipeline

# 并发执行多个智能体
responses = await fanout_pipeline(
    agents=[agent1, agent2, agent3],
    msg=question,
    enable_gather=True,  # 使用 asyncio.gather
)
```

**记忆压缩**:
```python
agent = ReActAgent(
    compression_config=ReActAgent.CompressionConfig(
        enable=True,
        trigger_threshold=8000,  # 8000 tokens 触发
        keep_recent=3,
    ),
)
```

### 27.2 成本控制

**Token 监控**:
```python
from agentscope.token import OpenAITokenCounter

counter = OpenAITokenCounter(model_name="gpt-4")
total_tokens = counter.count_messages(messages)
```

**模型选择**:
- 简单任务：使用小模型（Qwen-Turbo）
- 复杂推理：使用大模型（GPT-4, Qwen-Max）
- 压缩/摘要：使用小模型

**缓存策略**:
- 嵌入缓存
- 工具结果缓存
- 常见问题答案缓存

### 27.3 安全考虑

**工具权限控制**:
```python
toolkit.create_tool_group(
    group_name="dangerous_ops",
    description="危险操作工具",
    active=False,  # 默认不激活
    notes="谨慎使用",
)
```

**输入验证**:
```python
def safe_tool(user_input: str) -> ToolResponse:
    # 验证输入
    if not re.match(r'^[a-zA-Z0-9_]+$', user_input):
        raise ValueError("无效输入")
    # ...
```

**资源限制**:
```python
toolkit.register_tool_function(
    execute_shell_command,
    preset_kwargs={"timeout": 30},  # 30 秒超时
)
```

### 27.4 调试技巧

**启用详细日志**:
```python
agentscope.init(
    logging_level="DEBUG",
    logging_path="./logs",
)
```

**链路追踪**:
```python
agentscope.init(studio_url="http://localhost:3000")
# 在 Studio 查看执行链路
```

**异常处理**:
```python
try:
    response = await agent(msg)
except ModelResponseError as e:
    logger.error(f"模型错误：{e.details}")
except ToolExecutionError as e:
    logger.error(f"工具错误：{e.error_message}")
```

---

## 28. 常见问题

### 28.1 模型调用失败

**问题**: 模型调用返回空响应或超时

**解决方法**:
1. 检查 API 密钥是否正确
2. 验证网络连接
3. 增加超时时间：`model_kwargs={"timeout": 60}`
4. 减少上下文长度
5. 检查模型配额

### 28.2 记忆泄漏

**问题**: 长时间运行后内存占用过高

**解决方法**:
1. 启用记忆压缩
2. 定期清理旧对话：`await memory.clear()`
3. 使用 Redis 等外部存储
4. 限制记忆大小

### 28.3 工具执行卡住

**问题**: 工具调用长时间无响应

**解决方法**:
1. 添加工具执行超时
2. 检查工具依赖
3. 添加异常处理
4. 使用异步工具调用

### 28.4 RAG 检索不准确

**问题**: 检索到的文档不相关

**解决方法**:
1. 调整 `chunk_size`
2. 更换嵌入模型
3. 调整 `score_threshold`
4. 使用查询重写（`enable_rewrite_query=True`）

---

## 附录 A: 资源链接

- **官方文档**: https://doc.agentscope.io
- **中文文档**: https://doc.agentscope.io/zh_CN
- **GitHub 仓库**: https://github.com/agentscope-ai/agentscope
- **AgentScope Studio**: https://github.com/agentscope-ai/studio
- **示例代码**: https://github.com/agentscope-ai/agentscope/tree/main/examples
- **社区讨论**: https://github.com/agentscope-ai/agentscope/discussions
- **Discord**: https://discord.gg/eYMpfnkG8h

---

## 附录 B: API 快速参考

### 核心模块

```python
# 消息
from agentscope.message import Msg, TextBlock, ImageBlock, ToolUseBlock

# 模型
from agentscope.model import OpenAIChatModel, DashScopeChatModel, AnthropicChatModel

# 智能体
from agentscope.agent import ReActAgent, UserAgent, AgentBase

# 记忆
from agentscope.memory import InMemoryMemory, RedisMemory, AsyncSQLAlchemyMemory

# 工具
from agentscope.tool import Toolkit, ToolResponse, execute_python_code

# 格式化器
from agentscope.formatter import OpenAIChatFormatter, DashScopeChatFormatter

# 管道
from agentscope.pipeline import MsgHub, sequential_pipeline, fanout_pipeline

# RAG
from agentscope.rag import TextReader, QdrantStore, KnowledgeBase

# 嵌入
from agentscope.embedding import OpenAIEmbedding, DashScopeTextEmbedding

# 评估
from agentscope.evaluate import GeneralEvaluator, MetricBase, Task

# 追踪
from agentscope.tracing import trace, setup_tracing
```

---

## 附录 C: 版本历史

| 版本 | 日期 | 主要更新 |
|------|------|---------|
| v1.0 | 2025 | 正式版发布 |
| v2.0 | 2026-03-01 | 完全基于官方教程重写 |

---

*本指南基于 AgentScope v1.0+ 官方文档编写*
*最后更新：2026 年 3 月 1 日*
