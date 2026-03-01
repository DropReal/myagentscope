# AgentScope 使用详细指南 - 第 6-13 章

## 6. 工具系统

工具系统允许智能体调用外部能力，如代码执行、文件操作、Web 搜索等。

### 6.1 工具系统特性

AgentScope 全面支持工具 API 使用，具有以下特性：

| 特性 | 描述 |
|------|------|
| 自动解析 | 从文档字符串**自动**解析工具函数参数 |
| 同步/异步 | 支持**同步和异步**工具函数 |
| 流式响应 | 支持**流式**工具响应（同步或异步生成器） |
| 动态扩展 | 支持对工具 JSON Schema 的**动态扩展** |
| 实时中断 | 支持用户实时**中断**工具执行 |
| 自主管理 | 支持智能体的**自主工具管理** |

### 6.2 内置工具函数

```python
from agentscope.tool import (
    # 代码执行
    execute_python_code,
    
    # Shell 命令
    execute_shell_command,
    
    # 文件操作
    view_text_file,
    write_text_file,
    insert_text_file,
    
    # DashScope 多模态
    dashscope_text_to_image,
    dashscope_text_to_audio,
    dashscope_image_to_text,
    
    # OpenAI 多模态
    openai_text_to_image,
    openai_text_to_audio,
    openai_edit_image,
    openai_create_image_variation,
    openai_image_to_text,
    openai_audio_to_text,
)
```

### 6.3 自定义工具函数

**基础工具函数**:

```python
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock

def search_web(query: str, num_results: int = 5) -> ToolResponse:
    """搜索网络信息
    
    Args:
        query (str):
            搜索查询字符串。
        num_results (int):
            返回结果数量，默认 5。
    
    Returns:
        ToolResponse: 搜索结果
    """
    # 实现搜索逻辑
    results = [f"结果{i}: 关于'{query}'的信息" for i in range(num_results)]
    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(results))],
    )
```

**异步工具函数**:

```python
async def async_search(query: str) -> ToolResponse:
    """异步搜索工具"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/search?q={query}") as resp:
            data = await resp.json()
            return ToolResponse(
                content=[TextBlock(type="text", text=str(data))],
            )
```

**流式工具函数**:

```python
from typing import AsyncGenerator

async def streaming_tool(query: str) -> AsyncGenerator[ToolResponse, None]:
    """流式工具函数示例"""
    # 第一部分响应
    yield ToolResponse(
        content=[TextBlock(type="text", text="正在搜索...")],
        stream=True,
    )
    
    # 模拟搜索过程
    await asyncio.sleep(1)
    
    # 完整响应（累积式）
    yield ToolResponse(
        content=[
            TextBlock(type="text", text="正在搜索..."),
            TextBlock(type="text", text="找到结果：AgentScope"),
        ],
        stream=True,
        is_last=True,
    )
```

**重要**: 流式工具函数的响应应该是**累积的**，即当前块包含之前所有块的内容。

### 6.4 工具模块（Toolkit）

**基本用法**:

```python
from agentscope.tool import Toolkit
from agentscope.message import ToolUseBlock
import asyncio

async def example_toolkit():
    # 创建工具包
    toolkit = Toolkit()
    
    # 注册工具函数
    toolkit.register_tool_function(search_web)
    
    # 获取工具 JSON Schema
    schemas = toolkit.get_json_schemas()
    print(schemas)
    # 输出：[{"type": "function", "function": {"name": "search_web", ...}}]
    
    # 执行工具函数
    result = await toolkit.call_tool_function(
        ToolUseBlock(
            type="tool_use",
            id="123",
            name="search_web",
            input={"query": "AgentScope", "num_results": 3},
        ),
    )
    
    # 获取工具响应
    async for tool_response in result:
        print(tool_response)

asyncio.run(example_toolkit())
```

**预设参数**:

```python
# 注册时预设参数（如 API 密钥）
toolkit.register_tool_function(
    search_web,
    preset_kwargs={"api_key": "your-secret-key"},
)

# 预设参数会从 JSON schema 中移除，调用时自动传入
```

**清空工具包**:

```python
toolkit.clear()
```

### 6.5 动态扩展 JSON Schema

```python
from agentscope.tool import Toolkit
from pydantic import BaseModel, Field
from typing import Any

# 工具函数需要接受 **kwargs
def tool_with_extension(**kwargs: Any) -> ToolResponse:
    """一个工具函数"""
    return ToolResponse(
        content=[TextBlock(type="text", text=f"参数：{kwargs}")],
    )

# 定义扩展模型
class ThinkingModel(BaseModel):
    """用于 CoT 推理的扩展字段"""
    thinking: str = Field(
        description="总结当前状态并决定下一步做什么。",
    )

# 注册并扩展
toolkit = Toolkit()
toolkit.register_tool_function(tool_with_extension)
toolkit.set_extended_model("tool_with_extension", ThinkingModel)

# 获取扩展后的 schema
schemas = toolkit.get_json_schemas()
print(schemas)
# 现在 schema 包含 thinking 字段
```

**应用场景**:
- 动态 structured-output
- CoT（思维链）推理
- 添加元数据字段

### 6.6 中断工具执行

基于 asyncio 取消机制实现：

**非流式工具中断**:

```python
async def interruptible_tool() -> ToolResponse:
    """可被中断的工具函数"""
    await asyncio.sleep(1)  # 模拟长时间任务
    
    # 模拟中断
    raise asyncio.CancelledError()
    
    # 以下代码不会执行
    return ToolResponse(content=[...])

# 中断后返回：
# ToolResponse(
#     content=[{"type": "text", "text": "<system-info>The tool call has been interrupted by the user.</system-info>"}],
#     is_interrupted=True,
# )
```

**流式工具中断**:

```python
async def streaming_interruptible_tool() -> AsyncGenerator[ToolResponse, None]:
    """可被中断的流式工具"""
    # 第一部分响应
    yield ToolResponse(
        content=[TextBlock(type="text", text="1234")],
        stream=True,
    )
    
    # 模拟中断
    raise asyncio.CancelledError()
    
    # 不会执行
    yield ToolResponse(...)

# 中断后：
# 块 0: {"text": "1234"}, is_interrupted=False
# 块 1: {"text": "1234<system-info>interrupted...</system-info>"}, is_interrupted=True
```

### 6.7 自动工具管理

通过**工具组**（Group）和**元工具函数**（Meta Tool）实现：

**创建工具组**:

```python
from agentscope.tool import Toolkit

toolkit = Toolkit()

# 创建浏览器使用工具组
toolkit.create_tool_group(
    group_name="browser_use",
    description="用于网页浏览的工具函数。",
    active=False,  # 默认不激活
    notes="""1. 使用 ``navigate`` 打开网页。
2. 当需要用户身份验证时，请向用户询问凭据
3. ...""",
)

# 注册工具到组
def navigate(url: str) -> ToolResponse:
    """导航到网页"""
    pass

def click_element(element_id: str) -> ToolResponse:
    """点击网页元素"""
    pass

toolkit.register_tool_function(navigate, group_name="browser_use")
toolkit.register_tool_function(click_element, group_name="browser_use")
```

**管理工具组**:

```python
# 激活/停用工具组
toolkit.update_tool_groups(group_names=["browser_use"], active=True)

# 获取已激活工具组的注意事项
notes = toolkit.get_activated_notes()
print(notes)
```

**元工具函数**:

```python
# 注册元工具函数
toolkit.register_tool_function(toolkit.reset_equipped_tools)

# 智能体可以调用此函数来动态管理工具组
# reset_equipped_tools 的 schema:
# {
#     "name": "reset_equipped_tools",
#     "parameters": {
#         "properties": {
#             "browser_use": {
#                 "type": "boolean",
#                 "description": "用于网页浏览的工具函数。",
#                 "default": false
#             }
#         }
#     }
# }
```

**使用示例**:

```python
# 智能体调用
response = await toolkit.call_tool_function(
    ToolUseBlock(
        type="tool_use",
        id="456",
        name="reset_equipped_tools",
        input={"browser_use": True},
    ),
)

async for tool_response in response:
    print(tool_response.content[0]["text"])
    # 输出："Now tool groups 'browser_use' are activated. 
    #        You MUST follow these notes: ..."
```

---

## 7. 记忆管理

记忆模块负责存储和管理消息对象。

### 7.1 记忆基类

所有记忆类继承自 `MemoryBase`，提供以下方法：

| 方法 | 描述 |
|------|------|
| `add` | 向记忆中添加 `Msg` 对象 |
| `delete` | 从记忆中删除 `Msg` 对象 |
| `delete_by_mark` | 通过标记从记忆中删除 `Msg` 对象 |
| `size` | 记忆的大小 |
| `clear` | 清空记忆内容 |
| `get_memory` | 以 `Msg` 对象列表形式获取记忆内容 |
| `update_messages_mark` | 更新记忆中消息的标记 |
| `state_dict` | 获取记忆的状态字典 |
| `load_state_dict` | 加载记忆的状态字典 |

### 7.2 标记（Mark）机制

**标记**是与记忆中每条消息关联的字符串标签，用于分类、过滤和检索。

```python
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg

memory = InMemoryMemory()

# 添加带标记的消息
await memory.add(
    Msg("system", "内部提示", "system"),
    marks="hint",  # 添加标记
)

# 获取带特定标记的消息
hint_msgs = await memory.get_memory(mark="hint")

# 获取排除特定标记的消息
filtered = await memory.get_memory(exclude_mark="hint")

# 通过标记删除
deleted_count = await memory.delete_by_mark("hint")
```

**记忆标记类型**:

| 标记 | 用途 |
|------|------|
| `hint` | 一次性提示消息 |
| `COMPRESSED` | 已压缩的记忆（用于记忆压缩） |
| 自定义 | 用户自定义标记 |

### 7.3 InMemoryMemory - 内存记忆

```python
from agentscope.memory import InMemoryMemory
import json

async def in_memory_example():
    memory = InMemoryMemory()
    
    # 添加消息
    await memory.add(
        Msg("Alice", "生成一份关于 AgentScope 的报告", "user"),
    )
    
    # 添加带标记的消息
    await memory.add(
        Msg("system", "<system-hint>先创建计划</system-hint>", "system"),
        marks="hint",
    )
    
    # 获取记忆
    msgs = await memory.get_memory()
    for msg in msgs:
        print(f"- {msg}")
    
    # 导出状态
    state = memory.state_dict()
    print(json.dumps(state, indent=2, ensure_ascii=False))
    
    # 加载状态
    memory.load_state_dict(state)
    
    # 清空记忆
    await memory.clear()
```

### 7.4 AsyncSQLAlchemyMemory - 关系数据库记忆

支持 SQLite、PostgreSQL、MySQL 等：

**SQLite 示例**:

```python
from sqlalchemy.ext.asyncio import create_async_engine
from agentscope.memory import AsyncSQLAlchemyMemory

async def sqlalchemy_example():
    # 创建异步引擎
    engine = create_async_engine("sqlite+aiosqlite:///./test_memory.db")
    
    # 创建记忆实例
    memory = AsyncSQLAlchemyMemory(
        engine_or_session=engine,
        user_id="user_1",      # 可选
        session_id="session_1", # 可选
    )
    
    # 添加消息
    await memory.add(Msg("user", "你好", "user"))
    
    # 完成后关闭
    await memory.close()
```

**使用上下文管理器**:

```python
async with AsyncSQLAlchemyMemory(
    engine_or_session=engine,
    user_id="user_1",
    session_id="session_1",
) as memory:
    await memory.add(Msg("user", "你好", "user"))
    # 退出上下文时自动关闭
```

**生产环境配置（FastAPI + 连接池）**:

```python
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

app = FastAPI()

# 创建带连接池的引擎
engine = create_async_engine(
    "sqlite+aiosqlite:///./memory.db",
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
)

async_session_maker = async_sessionmaker(
    engine,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

async def get_db():
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

@app.post("/chat")
async def chat_endpoint(
    user_id: str,
    session_id: str,
    input: str,
    db_session = Depends(get_db),
):
    from agentscope.agent import ReActAgent
    from agentscope.memory import AsyncSQLAlchemyMemory
    
    agent = ReActAgent(
        memory=AsyncSQLAlchemyMemory(
            engine_or_session=db_session,
            user_id=user_id,
            session_id=session_id,
        ),
    )
    # ...
```

### 7.5 RedisMemory - NoSQL 记忆

```python
from redis.asyncio import ConnectionPool
from agentscope.memory import RedisMemory

async def redis_example():
    # 创建连接池
    redis_pool = ConnectionPool(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True,
        max_connections=10,
    )
    
    # 创建 Redis 记忆
    memory = RedisMemory(
        connection_pool=redis_pool,
        user_id="user_1",
        session_id="session_1",
    )
    
    # 使用
    await memory.add(Msg("user", "你好", "user"))
    
    # 获取客户端
    client = memory.get_client()
    await client.aclose()
```

### 7.6 自定义记忆

```python
from agentscope.memory import MemoryBase
from agentscope.message import Msg
from typing import List, Optional

class CustomMemory(MemoryBase):
    """自定义记忆实现"""
    
    async def add(
        self,
        msgs: Msg | List[Msg],
        marks: str | List[str] | None = None,
    ) -> None:
        """添加消息"""
        pass
    
    async def delete(self, msg_ids: List[str]) -> int:
        """删除消息"""
        pass
    
    async def delete_by_mark(self, mark: str) -> int:
        """通过标记删除"""
        pass
    
    def size(self) -> int:
        """记忆大小"""
        pass
    
    async def clear(self) -> None:
        """清空记忆"""
        pass
    
    async def get_memory(
        self,
        mark: str | None = None,
        exclude_mark: str | None = None,
    ) -> List[Msg]:
        """获取记忆"""
        pass
    
    async def update_messages_mark(
        self,
        msg_ids: List[str],
        marks: List[str],
    ) -> None:
        """更新消息标记"""
        pass
    
    def state_dict(self) -> dict:
        """获取状态字典"""
        pass
    
    async def load_state_dict(self, state_dict: dict) -> None:
        """加载状态字典"""
        pass
```

---

## 8. 长期记忆

长期记忆支持跨会话保存用户偏好和历史。

### 8.1 Mem0LongTermMemory

```python
from agentscope.memory import Mem0LongTermMemory
from agentscope.message import Msg

async def mem0_example():
    # 创建长期记忆
    long_term_memory = Mem0LongTermMemory(
        user_id="user_123",
        agent_name="Assistant",
        config={"version": "v1.1"},
    )
    
    async with long_term_memory:
        # 记录对话
        await long_term_memory.record(
            msgs=[
                Msg("user", "我喜欢喝龙井茶", "user"),
                Msg("assistant", "记住了", "assistant"),
            ],
        )
        
        # 检索相关记忆
        memories = await long_term_memory.retrieve(
            msg=Msg("user", "我喜欢喝什么茶？", "user"),
        )
        print(memories)
```

**高级配置**:

```python
from mem0 import MemoryConfig

advanced_config = MemoryConfig(
    version="v1.1",
    vector_store={
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        },
    },
)

mem0 = Mem0LongTermMemory(
    user_id="user_123",
    agent_name="Assistant",
    mem0_config=advanced_config,
)
```

### 8.2 ReMePersonalLongTermMemory

```python
from agentscope.memory import ReMePersonalLongTermMemory
from agentscope.embedding import DashScopeTextEmbedding
from agentscope.model import DashScopeChatModel
import os

async def reme_example():
    # 创建个人长期记忆
    personal_memory = ReMePersonalLongTermMemory(
        agent_name="Friday",
        user_name="user_123",
        model=DashScopeChatModel(
            model_name="qwen3-max",
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
        ),
        embedding_model=DashScopeTextEmbedding(
            model_name="text-embedding-v4",
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            dimensions=1024,
        ),
        vector_store_dir="./vector_store",
    )
    
    async with personal_memory:
        # 方式 1: record_to_memory（工具函数）
        await personal_memory.record_to_memory(
            thinking="用户分享旅行偏好",
            content=[
                "我在杭州旅行时喜欢住民宿",
                "我喜欢早上去西湖",
            ],
        )
        
        # 方式 2: retrieve_from_memory（工具函数）
        result = await personal_memory.retrieve_from_memory(
            keywords=["杭州旅行"],
        )
        
        # 方式 3: record（直接方法）
        await personal_memory.record(
            msgs=[
                Msg("user", "我是软件工程师", "user"),
            ],
        )
        
        # 方式 4: retrieve（直接方法）
        memories = await personal_memory.retrieve(
            msg=Msg("user", "你知道我的职业吗？", "user"),
        )
```

### 8.3 ReMeTaskLongTermMemory

```python
from agentscope.memory import ReMeTaskLongTermMemory

async def task_memory_example():
    # 任务长期记忆
    task_memory = ReMeTaskLongTermMemory(
        agent_name="TaskAssistant",
        user_name="workspace_123",
        model=dashscope_model,
        embedding_model=dashscope_embedding,
    )
    
    async with task_memory:
        # 记录成功的任务执行（带评分）
        await task_memory.record(
            msgs=[
                Msg("user", "API 返回 404", "user"),
                Msg("assistant", "检查路由定义", "assistant"),
                Msg("user", "找到拼写错误了", "user"),
            ],
            score=0.95,  # 高分表示成功轨迹
        )
        
        # 检索相关经验
        experiences = await task_memory.retrieve(
            msg=Msg("user", "如何调试 API 错误？", "user"),
        )
```

---

## 9. 会话管理

### 9.1 状态字典

```python
from agentscope.memory import InMemoryMemory

memory = InMemoryMemory()

# 导出状态
state = memory.state_dict()
# {
#     "_compressed_summary": "",
#     "content": [
#         [msg_dict, marks],
#         ...
#     ]
# }

# 加载状态
memory.load_state_dict(state)
```

### 9.2 用户和会话管理

```python
from agentscope.memory import AsyncSQLAlchemyMemory

# 为每个用户/会话创建独立记忆
memory = AsyncSQLAlchemyMemory(
    engine_or_session=engine,
    user_id="user_123",
    session_id="session_456",
)
```

---

## 10. 格式化处理

格式化器将 AgentScope 消息转换为不同 LLM API 格式。

### 10.1 支持的格式化器

| 格式化器 | 适用模型 |
|---------|---------|
| `OpenAIChatFormatter` | OpenAI, Azure, DeepSeek, vLLM |
| `AnthropicChatFormatter` | Anthropic Claude |
| `DashScopeChatFormatter` | 阿里云 DashScope |
| `GeminiChatFormatter` | Google Gemini |
| `OllamaChatFormatter` | Ollama |
| `DeepSeekChatFormatter` | DeepSeek |
| `DashScopeMultiAgentFormatter` | 多智能体场景 |

### 10.2 格式化器使用

```python
from agentscope.formatter import OpenAIChatFormatter, AnthropicChatFormatter
from agentscope.message import Msg, TextBlock, ImageBlock, URLSource

async def example_format():
    # 创建格式化器
    openai_formatter = OpenAIChatFormatter()
    anthropic_formatter = AnthropicChatFormatter()
    
    # 创建消息
    messages = [
        Msg("system", "你是有帮助的助手", "system"),
        Msg(
            "user",
            [
                TextBlock(type="text", text="描述这张图片"),
                ImageBlock(
                    type="image",
                    source=URLSource(
                        type="url",
                        url="https://example.com/image.jpg",
                    ),
                ),
            ],
            "user",
        ),
    ]
    
    # 转换为 OpenAI 格式
    openai_formatted = await openai_formatter.format(messages)
    # [{"role": "system", "content": "..."}, {"role": "user", "content": [...]}]
    
    # 转换为 Anthropic 格式
    anthropic_formatted = await anthropic_formatter.format(messages)

asyncio.run(example_format())
```

---

## 11. Token 计数

### 11.1 OpenAI Token 计数器

```python
from agentscope.token import OpenAITokenCounter

counter = OpenAITokenCounter(model_name="gpt-4")

# 计算文本 Token 数
text = "这是一段测试文本"
num_tokens = counter.count_tokens(text)

# 计算消息列表的 Token 数
messages = [
    {"role": "system", "content": "你是助手"},
    {"role": "user", "content": "你好"},
]
total_tokens = counter.count_messages(messages)
```

### 11.2 DashScope Token 计数器

```python
from agentscope.token import DashScopeTokenCounter

counter = DashScopeTokenCounter(model_name="qwen-max")
num_tokens = counter.count_tokens("测试文本")
```

### 11.3 自定义 Token 计数器

```python
from agentscope.token import TokenCounterBase

class CustomTokenCounter(TokenCounterBase):
    def count_tokens(self, text: str) -> int:
        return len(text) // 4  # 简单估算
    
    def count_messages(self, messages: list) -> int:
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.get("content", ""))
        return total
```

---

## 12. 工作流编排

### 12.1 MsgHub - 消息广播中心

```python
from agentscope.pipeline import MsgHub, sequential_pipeline
from agentscope.agent import ReActAgent
from agentscope.message import Msg

async def example_msghub():
    alice = create_agent("Alice", "研究员")
    bob = create_agent("Bob", "工程师")
    charlie = create_agent("Charlie", "产品经理")
    
    # 创建消息广播中心
    async with MsgHub(
        participants=[alice, bob, charlie],
        announcement=Msg("Host", "讨论新的 AI 功能", "assistant"),
        enable_auto_broadcast=True,
    ) as hub:
        # 顺序对话
        await sequential_pipeline([alice, bob, charlie])
        
        # 动态添加/删除参与者
        david = create_agent("David", "设计师")
        hub.add(david)
        hub.delete(charlie)
        
        # 手动广播
        await hub.broadcast(Msg("Host", "总结", "assistant"))
```

### 12.2 顺序管道

```python
from agentscope.pipeline import sequential_pipeline

# 等价于：
# msg = await alice(msg)
# msg = await bob(msg)
# msg = await charlie(msg)

msg = await sequential_pipeline(
    agents=[alice, bob, charlie],
    msg=None,
)
```

### 12.3 扇出管道

```python
from agentscope.pipeline import fanout_pipeline

# 并发执行（默认）
msgs = await fanout_pipeline(
    agents=[alice, bob, charlie],
    msg=question,
    enable_gather=True,  # 使用 asyncio.gather 并发
)

# 顺序执行
msgs = await fanout_pipeline(
    agents=[alice, bob, charlie],
    msg=question,
    enable_gather=False,  # 逐个顺序执行
)
```

### 12.4 流式获取打印消息

```python
from agentscope.pipeline import stream_printing_messages

async for msg, last in stream_printing_messages(
    agents=[agent],
    coroutine_task=agent(Msg("user", "你好", "user")),
):
    print(msg, last)
    if last:
        print()  # 最后一块
```

---

## 13. 计划模块

（计划模块详细内容，包括任务分解、多步规划等）
