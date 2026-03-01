# AgentScope 多智能体开发框架深度指南

> 本文档提供 AgentScope 框架的完整使用说明，涵盖从基础概念到高级应用的所有模块，包含详细的代码示例和最佳实践。

## 目录

1. [框架概述与核心概念](#1-框架概述与核心概念)
2. [安装与环境配置](#2-安装与环境配置)
3. [消息系统](#3-消息系统)
4. [模型集成](#4-模型集成)
5. [智能体系统](#5-智能体系统)
6.  [工具系统](#6-工具系统)
7. [记忆管理](#7-记忆管理)
8. [会话管理](#8-会话管理)
9. [格式化处理](#9-格式化处理)
10. [工作流编排](#10-工作流编排)
11. [RAG 知识增强](#11-rag 知识增强)
12. [嵌入模型](#12-嵌入模型)
13. [Token 计数](#13-token 计数)
14. [评估系统](#14-评估系统)
15. [链路追踪与监控](#15-链路追踪与监控)
16. [异常处理](#16-异常处理)
17. [MCP 集成](#17-mcp 集成)
18. [高级应用案例](#18-高级应用案例)

---

## 1. 框架概述与核心概念

### 1.1 什么是 AgentScope

AgentScope 是一个功能强大的多智能体系统构建框架，采用面向智能体的编程范式，提供以下核心特性：

- **模型无关性**：统一的接口支持 OpenAI、Anthropic、Gemini、DashScope、Ollama 等主流 LLM 提供商
- **模块化设计**：消息、记忆、工具、模型等组件高度解耦，可自由组合
- **分布式支持**：原生支持多智能体协作和消息路由
- **透明控制**：提供完整的链路追踪和调试能力
- **实时交互**：支持流式输出和人机协作

### 1.2 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                      AgentScope 架构                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Agent     │  │   Agent     │  │   Agent     │          │
│  │  (智能体)    │  │  (智能体)    │  │  (智能体)    │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                  │
│              ┌───────────▼───────────┐                      │
│              │     MsgHub/消息路由    │                      │
│              └───────────┬───────────┘                      │
│                          │                                  │
│    ┌──────────┬──────────┼──────────┬──────────┐           │
│    │          │          │          │          │           │
│ ┌──▼──┐  ┌───▼───┐  ┌───▼───┐  ┌───▼───┐  ┌──▼──┐        │
│ │Model│  │Memory │  │ Tool  │  │Formatter│ │RAG  │        │
│ └─────┘  └───────┘  └───────┘  └─────────┘ └─────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 设计哲学

**面向智能体编程 (Agent-Oriented Programming)**

传统编程关注函数和对象，而 AgentScope 以智能体为基本单元：
- 智能体是自主的决策和执行单元
- 智能体之间通过消息进行通信
- 智能体可以感知环境、推理、行动和记忆

**关注点分离**

每个模块职责单一：
- `message`: 定义消息格式和内容块
- `model`: 封装 LLM API 调用
- `agent`: 实现智能体行为和决策逻辑
- `memory`: 管理短期和长期记忆
- `tool`: 提供外部能力集成
- `formatter`: 处理消息格式转换
- `pipeline`: 编排多智能体工作流

---

## 2. 安装与环境配置

### 2.1 基础安装

```bash
# 安装稳定版
pip install agentscope

# 安装开发版（获取最新功能）
pip install git+https://github.com/agentscope-ai/agentscope.git

# 安装完整依赖（包含所有可选功能）
pip install "agentscope[full]"
```

### 2.2 可选依赖

```bash
# RAG 功能依赖
pip install agentscope[rag]

# Redis 记忆后端
pip install agentscope[redis]

# SQLAlchemy 记忆后端
pip install agentscope[sqlalchemy]

# 分布式评估（Ray）
pip install agentscope[ray]

# MCP 集成
pip install agentscope[mcp]
```

### 2.3 环境配置

创建 `.env` 文件配置 API 密钥：

```bash
# OpenAI
OPENAI_API_KEY=sk-your-openai-key

# Anthropic
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# DashScope (阿里云通义千问)
DASHSCOPE_API_KEY=your-dashscope-key

# Google Gemini
GOOGLE_API_KEY=your-google-key

# Azure OpenAI
AZURE_OPENAI_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

### 2.4 初始化 AgentScope

```python
import agentscope
from agentscope import logger

# 基础初始化
agentscope.init(
    project="my_agent_project",      # 项目名称
    name="experiment_001",            # 实验名称
    run_id="run_20240101_001",        # 运行 ID（用于区分不同运行）
    logging_path="./logs",            # 日志保存路径
    logging_level="INFO",             # 日志级别：DEBUG, INFO, WARNING, ERROR
)

# 连接 AgentScope Studio（可视化监控）
agentscope.init(
    project="my_agent_project",
    studio_url="http://localhost:3000",  # Studio 地址
    tracing_url="http://localhost:4318/v1/traces",  # OpenTelemetry 端点
)

# 验证初始化
logger.info("AgentScope 初始化完成")
```

---

## 3. 消息系统

消息是 AgentScope 中智能体间通信的基本单位。

### 3.1 基础消息类型

#### Msg - 消息基类

```python
from agentscope.message import Msg

# 简单文本消息
text_msg = Msg(
    name="user",                      # 发送者名称
    content="你好，请帮我分析这段代码",  # 消息内容
    role="user",                      # 角色：user, assistant, system
)

# 查看消息
print(text_msg.name)                  # "user"
print(text_msg.content)               # "你好，请帮我分析这段代码"
print(text_msg.role)                  # "user"
print(text_msg.timestamp)             # 消息时间戳
print(text_msg.id)                    # 消息唯一 ID
```

#### 文本块 (TextBlock)

```python
from agentscope.message import TextBlock

# 创建文本块
text_block = TextBlock(
    type="text",
    text="这是一段纯文本内容",
)

# 在消息中使用文本块
msg = Msg(
    name="assistant",
    content=[text_block],
    role="assistant",
)
```

#### 图像块 (ImageBlock)

```python
from agentscope.message import ImageBlock, URLSource, Base64Source

# 方式 1: URL 来源的图像
image_from_url = ImageBlock(
    type="image",
    source=URLSource(
        type="url",
        url="https://example.com/image.png",
    ),
)

# 方式 2: Base64 编码的图像
import base64
with open("image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

image_from_base64 = ImageBlock(
    type="image",
    source=Base64Source(
        type="base64",
        data=image_data,
        media_type="image/png",
    ),
)

# 多模态消息
multimodal_msg = Msg(
    name="user",
    content=[
        TextBlock(type="text", text="请描述这张图片的内容"),
        image_from_url,
    ],
    role="user",
)
```

#### 音频块 (AudioBlock)

```python
from agentscope.message import AudioBlock

audio_msg = Msg(
    name="user",
    content=[
        AudioBlock(
            type="audio",
            source=Base64Source(
                type="base64",
                data="UklGRiQAAABXQVZFZm10IBAAAAABAA...",  # Base64 音频数据
                media_type="audio/wav",
            ),
        ),
    ],
    role="user",
)
```

#### 工具使用块 (ToolUseBlock)

```python
from agentscope.message import ToolUseBlock

tool_use = ToolUseBlock(
    type="tool_use",
    id="call_abc123",              # 工具调用 ID
    name="execute_python_code",     # 工具名称
    input={"code": "print('Hello')"},  # 工具参数
)

tool_msg = Msg(
    name="assistant",
    content=[tool_use],
    role="assistant",
)
```

#### 工具结果块 (ToolResultBlock)

```python
from agentscope.message import ToolResultBlock

tool_result = ToolResultBlock(
    type="tool_result",
    id="call_abc123",               # 对应工具调用的 ID
    name="execute_python_code",     # 工具名称
    output=[TextBlock(type="text", text="Hello")],  # 工具输出
)

result_msg = Msg(
    name="system",
    content=[tool_result],
    role="system",
)
```

#### 思考块 (ThinkingBlock)

```python
from agentscope.message import ThinkingBlock

thinking = ThinkingBlock(
    type="thinking",
    thinking="用户询问代码问题，我需要先理解代码结构，然后分析潜在问题...",
)

thinking_msg = Msg(
    name="assistant",
    content=[thinking],
    role="assistant",
)
```

### 3.2 消息操作方法

```python
from agentscope.message import Msg, TextBlock

# 创建消息
msg = Msg(
    name="assistant",
    content=[
        TextBlock(type="text", text="你好"),
        TextBlock(type="text", text="有什么可以帮助你的？"),
    ],
    role="assistant",
)

# 提取文本内容
text_content = msg.get_text_content()
# 输出："你好有什么可以帮助你的？"

# 获取特定类型的内容块
text_blocks = msg.get_content_blocks("text")
# 返回所有 TextBlock 列表

# 检查是否包含特定类型的内容块
has_text = msg.has_content_blocks("text")  # True
has_image = msg.has_content_blocks("image")  # False

# 转换为字典
msg_dict = msg.to_dict()
# {"name": "assistant", "content": [...], "role": "assistant", ...}

# 从字典恢复
restored_msg = Msg.from_dict(msg_dict)

# 序列化/反序列化
import json
json_str = msg.to_json()
msg_from_json = Msg.from_json(json_str)
```

### 3.3 消息对话历史

```python
from agentscope.message import Msg

# 构建对话历史
conversation = [
    Msg("system", "你是一个有帮助的 AI 助手", "system"),
    Msg("user", "你好", "user"),
    Msg("assistant", "你好！有什么我可以帮助你的吗？", "assistant"),
    Msg("user", "请解释一下 Python 的装饰器", "user"),
]

# 提取最近 N 条消息
recent_messages = conversation[-3:]

# 按角色过滤
user_messages = [m for m in conversation if m.role == "user"]
assistant_messages = [m for m in conversation if m.role == "assistant"]

# 提取所有用户输入
user_inputs = [m.get_text_content() for m in user_messages]
```

---

## 4. 模型集成

AgentScope 提供统一的模型接口，支持多种 LLM 提供商。

### 4.1 支持的模型 API

| API | 类名 | 流式 | 工具调用 | 视觉 | 推理 |
|-----|------|------|---------|------|------|
| OpenAI | `OpenAIChatModel` | ✅ | ✅ | ✅ | ✅ |
| Azure OpenAI | `OpenAIChatModel` | ✅ | ✅ | ✅ | ✅ |
| Anthropic | `AnthropicChatModel` | ✅ | ✅ | ✅ | ✅ |
| DashScope | `DashScopeChatModel` | ✅ | ✅ | ✅ | ✅ |
| Gemini | `GeminiChatModel` | ✅ | ✅ | ✅ | ✅ |
| Ollama | `OllamaChatModel` | ✅ | ✅ | ✅ | ❌ |
| DeepSeek | `OpenAIChatModel` | ✅ | ✅ | ✅ | ✅ |

### 4.2 OpenAI 模型配置

```python
from agentscope.model import OpenAIChatModel
import os

# 标准 OpenAI
openai_model = OpenAIChatModel(
    model_name="gpt-4o",                    # 模型名称
    api_key=os.environ["OPENAI_API_KEY"],   # API 密钥
    stream=True,                            # 启用流式输出
    generate_kwargs={                       # 生成参数
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 0.9,
    },
)

# Azure OpenAI
azure_model = OpenAIChatModel(
    model_name="gpt-4",
    api_key=os.environ["AZURE_OPENAI_KEY"],
    client_type="azure",                    # 指定 Azure 客户端
    client_kwargs={
        "api_version": "2024-02-01",
        "azure_endpoint": "https://your-resource.openai.azure.com/",
    },
    stream=True,
)

# OpenAI 兼容的本地模型（vLLM、DeepSeek 等）
vllm_model = OpenAIChatModel(
    model_name="llama-3-70b",
    api_key="not-needed",                   # 本地模型通常不需要密钥
    client_kwargs={
        "base_url": "http://localhost:8000/v1",  # vLLM 端点
    },
    stream=True,
)

# DeepSeek
deepseek_model = OpenAIChatModel(
    model_name="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    client_kwargs={
        "base_url": "https://api.deepseek.com/v1",
    },
    stream=True,
)
```

### 4.3 Anthropic Claude 模型

```python
from agentscope.model import AnthropicChatModel
import os

claude_model = AnthropicChatModel(
    model_name="claude-sonnet-4-20250514",  # 模型名称
    api_key=os.environ["ANTHROPIC_API_KEY"],
    stream=True,
    generate_kwargs={
        "max_tokens": 4096,
        "temperature": 0.5,
    },
)
```

### 4.4 阿里云 DashScope（通义千问）

```python
from agentscope.model import DashScopeChatModel
import os

qwen_model = DashScopeChatModel(
    model_name="qwen-max",                  # 通义千问模型
    api_key=os.environ["DASHSCOPE_API_KEY"],
    stream=True,
    generate_kwargs={
        "temperature": 0.7,
        "max_tokens": 2000,
    },
)

# 使用示例
async def example_qwen():
    messages = [{"role": "user", "content": "你好"}]
    response = await qwen_model(messages)
    print(response.content)
```

### 4.5 Google Gemini 模型

```python
from agentscope.model import GeminiChatModel
import os

gemini_model = GeminiChatModel(
    model_name="gemini-2.0-flash-exp",
    api_key=os.environ["GOOGLE_API_KEY"],
    stream=True,
    generate_kwargs={
        "temperature": 0.7,
        "max_output_tokens": 2048,
    },
)
```

### 4.6 Ollama 本地模型

```python
from agentscope.model import OllamaChatModel

ollama_model = OllamaChatModel(
    model_name="llama3.2",                  # Ollama 模型标签
    stream=True,
    client_kwargs={
        "host": "http://localhost:11434",   # Ollama 服务地址
    },
)
```

### 4.7 模型调用与响应处理

```python
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg, ChatResponse
import asyncio
import os

async def example_model_usage():
    model = OpenAIChatModel(
        model_name="gpt-4",
        api_key=os.environ["OPENAI_API_KEY"],
        stream=True,
    )
    
    # 方式 1: 非流式调用
    response = await model(
        messages=[
            {"role": "system", "content": "你是有帮助的助手"},
            {"role": "user", "content": "Hello"},
        ],
    )
    
    print(f"响应 ID: {response.id}")
    print(f"创建时间：{response.created_at}")
    print(f"内容：{response.content}")
    print(f"Token 使用：{response.usage}")
    
    # 方式 2: 流式调用
    async def example_streaming():
        stream_response = await model(
            messages=[{"role": "user", "content": "讲个故事"}],
        )
        
        # 流式响应是异步生成器
        async for chunk in stream_response:
            print(chunk.content, end="", flush=True)
    
    # 方式 3: 带工具调用的模型调用
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    response_with_tools = await model(
        messages=[{"role": "user", "content": "北京天气如何"}],
        tools=tools,
        tool_choice="auto",
    )
    
    # 检查是否包含工具调用
    for block in response_with_tools.content:
        if block.type == "tool_use":
            print(f"需要调用工具：{block.name}")
            print(f"参数：{block.input}")

asyncio.run(example_model_usage())
```

### 4.8 响应结构详解

```python
from agentscope.model import ChatResponse
from agentscope.message import TextBlock, ToolUseBlock, ThinkingBlock

# ChatResponse 结构
response = ChatResponse(
    content=[
        ThinkingBlock(
            type="thinking",
            thinking="我需要先搜索相关信息...",
        ),
        TextBlock(
            type="text",
            text="让我来帮你查询这个信息。",
        ),
        ToolUseBlock(
            type="tool_use",
            id="call_123",
            name="google_search",
            input={"query": "AgentScope 框架"},
        ),
    ],
    id="response_20240101_001",
    created_at="2024-01-01 12:00:00.000",
    type="chat",
    usage={
        "input_tokens": 100,
        "output_tokens": 50,
    },
    metadata={"model": "gpt-4"},
)

# 访问响应内容
print(response.id)              # 响应 ID
print(response.created_at)      # 创建时间
print(response.type)            # 响应类型
print(response.usage)           # Token 使用情况
print(response.metadata)        # 元数据

# 处理多块内容
for block in response.content:
    if block.type == "text":
        print(f"文本：{block.text}")
    elif block.type == "thinking":
        print(f"思考：{block.thinking}")
    elif block.type == "tool_use":
        print(f"工具调用：{block.name}")
```

---

## 5. 智能体系统

智能体是 AgentScope 的核心，负责感知、推理、决策和行动。

### 5.1 ReActAgent - 推理 - 行动智能体

ReActAgent 是 AgentScope 的核心智能体实现，支持推理 - 行动循环、工具调用、记忆和 RAG。

```python
from agentscope.agent import ReActAgent, UserAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit, execute_python_code, execute_shell_command
from agentscope.message import Msg
import asyncio
import os

async def example_react_agent():
    # 1. 创建工具包
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)
    toolkit.register_tool_function(execute_shell_command)
    
    # 2. 创建模型实例
    model = OpenAIChatModel(
        model_name="gpt-4",
        api_key=os.environ["OPENAI_API_KEY"],
        stream=True,
    )
    
    # 3. 创建记忆
    memory = InMemoryMemory()
    
    # 4. 创建格式化器
    formatter = OpenAIChatFormatter()
    
    # 5. 创建 ReActAgent
    agent = ReActAgent(
        name="Assistant",                           # 智能体名称
        sys_prompt="你是一个有帮助的 AI 助手，可以执行代码和回答问题。",  # 系统提示
        model=model,                                # LLM 模型
        memory=memory,                              # 短期记忆
        formatter=formatter,                        # 消息格式化器
        toolkit=toolkit,                            # 工具包
        max_iters=10,                               # 最大推理 - 行动迭代次数
        parallel_tool_calls=True,                   # 允许并行工具调用
    )
    
    # 6. 创建用户智能体（用于终端交互）
    user = UserAgent(name="user")
    
    # 7. 对话循环
    msg = None
    while True:
        # 用户输入
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break
        
        # 智能体响应
        msg = await agent(msg)
        print(f"{agent.name}: {msg.get_text_content()}")

asyncio.run(example_react_agent())
```

### 5.2 ReActAgent 高级配置

```python
from agentscope.agent import ReActAgent
from agentscope.token import OpenAITokenCounter

# 带记忆压缩的配置
agent_with_compression = ReActAgent(
    name="SmartAssistant",
    sys_prompt="你是一个智能助手",
    model=model,
    formatter=OpenAIChatFormatter(),
    memory=InMemoryMemory(),
    toolkit=toolkit,
    # 记忆压缩配置
    compression_config=ReActAgent.CompressionConfig(
        enable=True,                                    # 启用压缩
        agent_token_counter=OpenAITokenCounter(
            model_name="gpt-4"
        ),                                              # Token 计数器
        trigger_threshold=8000,                         # 触发压缩的 Token 阈值
        keep_recent=3,                                  # 保留最近 3 条消息不压缩
    ),
)

# 带 RAG 的配置
from agentscope.rag import KnowledgeBase

rag_agent = ReActAgent(
    name="KnowledgeAssistant",
    sys_prompt="基于知识库回答问题",
    model=model,
    formatter=OpenAIChatFormatter(),
    memory=InMemoryMemory(),
    knowledge=knowledge_base,                           # 知识库
    enable_rewrite_query=True,                          # 启用查询重写
)

# 带长期记忆的配置
from agentscope.memory import Mem0LongTermMemory

long_term_memory = Mem0LongTermMemory(
    user_id="user_123",
    config={"version": "v1.1"},
)

agent_with_ltm = ReActAgent(
    name="PersonalAssistant",
    sys_prompt="你记住用户跨会话的偏好",
    model=model,
    formatter=OpenAIChatFormatter(),
    memory=InMemoryMemory(),
    long_term_memory=long_term_memory,                  # 长期记忆
    long_term_memory_mode="both",                       # 智能体控制和静态模式
)
```

### 5.3 UserAgent - 用户交互

```python
from agentscope.agent import UserAgent
from agentscope.message import Msg

# 标准用户代理（命令行输入）
user_agent = UserAgent(
    name="user",
    input_hint="请输入你的问题（输入'exit'退出）: ",
)

# 对话示例
async def conversation():
    user = UserAgent(name="user")
    msg = None
    
    while True:
        msg = await user(msg)  # 等待用户输入
        if msg.get_text_content() == "exit":
            break
        print(f"用户说：{msg.content}")
```

### 5.4 自定义智能体

```python
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
import asyncio

class CustomAgent(AgentBase):
    """自定义智能体示例"""
    
    def __init__(
        self,
        name: str,
        model: OpenAIChatModel,
        sys_prompt: str = None,
    ):
        super().__init__(name=name)
        self.model = model
        self.sys_prompt = sys_prompt
        self.history = []
    
    async def __call__(self, msg: Msg = None) -> Msg:
        """处理输入消息并生成响应"""
        
        # 添加消息到历史
        if msg:
            self.history.append(msg)
        
        # 构建提示
        messages = []
        if self.sys_prompt:
            messages.append({"role": "system", "content": self.sys_prompt})
        
        # 添加历史消息
        for h_msg in self.history[-10:]:  # 只保留最近 10 条
            messages.append({
                "role": h_msg.role,
                "content": h_msg.get_text_content(),
            })
        
        # 调用模型
        response = await self.model(messages)
        
        # 创建响应消息
        response_msg = Msg(
            name=self.name,
            content=response.content,
            role="assistant",
        )
        
        # 保存响应到历史
        self.history.append(response_msg)
        
        return response_msg

# 使用示例
async def use_custom_agent():
    model = OpenAIChatModel(
        model_name="gpt-4",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    
    custom_agent = CustomAgent(
        name="MyAgent",
        model=model,
        sys_prompt="你是一个专业的代码审查助手",
    )
    
    # 调用智能体
    response = await custom_agent(
        Msg(name="user", content="请审查这段代码：...", role="user")
    )
    print(response.content)

asyncio.run(use_custom_agent())
```

### 5.5 智能体工厂模式

```python
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit
import os

async def create_agent(name: str, role: str, sys_prompt: str = None) -> ReActAgent:
    """智能体工厂函数"""
    
    if sys_prompt is None:
        sys_prompt = f"你是{name}，一个{role}。请专业地完成任务。"
    
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)
    
    return ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=OpenAIChatModel(
            model_name="gpt-4",
            api_key=os.environ["OPENAI_API_KEY"],
            stream=True,
        ),
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
        toolkit=toolkit,
        max_iters=10,
    )

# 创建多个专业智能体
researcher = await create_agent("研究员", "数据分析师")
engineer = await create_agent("工程师", "软件开发专家")
writer = await create_agent("作家", "技术文档撰写人")
```

---

## 6. 工具系统

工具系统允许智能体调用外部能力，如代码执行、文件操作、Web 搜索等。

### 6.1 内置工具

```python
from agentscope.tool import (
    execute_python_code,      # 执行 Python 代码
    execute_shell_command,    # 执行 Shell 命令
    view_text_file,          # 查看文本文件
    write_text_file,         # 写入文本文件
    read_text_file,          # 读取文本文件
    ToolResponse,            # 工具响应
)

# 执行 Python 代码
async def example_python():
    result = await execute_python_code(
        code="print('Hello, World!')\n2 + 2",
    )
    print(result.content)

# 执行 Shell 命令
async def example_shell():
    result = await execute_shell_command(
        command="ls -la",
        timeout=30,  # 超时时间（秒）
    )
    print(result.content)

# 文件操作
async def example_file():
    # 写入文件
    await write_text_file(
        file_path="./output.txt",
        content="Hello, File!",
    )
    
    # 读取文件
    content = await view_text_file(
        file_path="./output.txt",
    )
    print(content)
```

### 6.2 自定义工具函数

```python
from agentscope.tool import Toolkit, ToolResponse
from agentscope.message import TextBlock
from pydantic import BaseModel, Field

# 方式 1: 简单函数
def search_web(query: str, num_results: int = 5) -> ToolResponse:
    """搜索网络信息
    
    Args:
        query: 搜索查询字符串
        num_results: 返回结果数量，默认 5
        
    Returns:
        搜索结果
    """
    # 实现搜索逻辑
    results = [f"结果{i}: 关于'{query}'的信息" for i in range(num_results)]
    return ToolResponse(
        content=[TextBlock(type="text", text="\n".join(results))],
    )

# 方式 2: 带 Pydantic 参数验证的函数
class WeatherParams(BaseModel):
    location: str = Field(description="城市名称")
    date: str = Field(description="日期，格式 YYYY-MM-DD", default="today")

async def get_weather(location: str, date: str = "today") -> ToolResponse:
    """获取天气信息"""
    # 实现天气查询逻辑
    weather_info = f"{location} {date}的天气：晴，25°C"
    return ToolResponse(
        content=[TextBlock(type="text", text=weather_info)],
    )

# 注册工具
toolkit = Toolkit()
toolkit.register_tool_function(search_web)
toolkit.register_tool_function(get_weather)
```

### 6.3 工具包管理

```python
from agentscope.tool import Toolkit
from agentscope.tool import (
    execute_python_code,
    execute_shell_command,
    view_text_file,
    write_text_file,
)

# 创建工具包
toolkit = Toolkit()

# 注册内置工具
toolkit.register_tool_function(execute_python_code)
toolkit.register_tool_function(execute_shell_command)
toolkit.register_tool_function(view_text_file)
toolkit.register_tool_function(write_text_file)

# 注册自定义工具
toolkit.register_tool_function(search_web)

# 创建工具组（用于组织和管理）
toolkit.create_tool_group(
    group_name="file_operations",
    description="文件操作工具",
    active=False,  # 默认不激活
    notes="谨慎使用文件修改工具",
)

# 将工具分配到组
toolkit.register_tool_function(
    view_text_file,
    group_name="file_operations",
)
toolkit.register_tool_function(
    write_text_file,
    group_name="file_operations",
)

# 激活/停用工具组
toolkit.update_tool_groups(["file_operations"], active=True)

# 获取激活工具的 JSON Schema（用于模型调用）
tool_schemas = toolkit.get_json_schemas()
print(tool_schemas)
# 输出格式：
# [
#   {"type": "function", "function": {"name": "...", "parameters": {...}}},
#   ...
# ]

# 获取特定工具
specific_tool = toolkit.get_tool("execute_python_code")

# 删除工具
toolkit.remove_tool("search_web")
```

### 6.4 工具扩展示例

```python
from agentscope.tool import Toolkit, ToolResponse
from agentscope.message import TextBlock
import requests
import aiohttp
import asyncio

# HTTP 请求工具
async def http_request(
    method: str,
    url: str,
    headers: dict = None,
    body: str = None,
) -> ToolResponse:
    """发送 HTTP 请求"""
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method=method,
            url=url,
            headers=headers,
            data=body,
        ) as response:
            content = await response.text()
            return ToolResponse(
                content=[TextBlock(
                    type="text",
                    text=f"状态码：{response.status}\n内容：{content[:1000]}",
                )],
            )

# 数据库查询工具（SQLite 示例）
import sqlite3

def query_database(query: str, db_path: str = "data.db") -> ToolResponse:
    """执行 SQL 查询"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        return ToolResponse(
            content=[TextBlock(
                type="text",
                text=f"查询结果：{len(results)} 行\n{str(results[:10])}",
            )],
        )
    except Exception as e:
        return ToolResponse(
            content=[TextBlock(type="text", text=f"错误：{str(e)}")],
        )
    finally:
        conn.close()

# 注册工具
toolkit = Toolkit()
toolkit.register_tool_function(http_request)
toolkit.register_tool_function(query_database)
```

---

## 7. 记忆管理

记忆系统负责存储和检索智能体的对话历史和知识。

### 7.1 短期记忆

#### InMemoryMemory - 内存存储

```python
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg

# 创建内存实例
memory = InMemoryMemory()

# 添加消息
await memory.add(Msg("user", "你好", "user"))
await memory.add(Msg("assistant", "你好！有什么可以帮助你的？", "assistant"))

# 添加带标记的消息（用于分类和过滤）
await memory.add(
    Msg("system", "内部提示：用户喜欢简洁的回答", "system"),
    marks="hint",  # 标记
)

# 获取所有记忆
messages = await memory.get_memory()
print(messages)

# 获取排除特定标记的记忆
filtered = await memory.get_memory(exclude_mark="hint")

# 按标记删除记忆
await memory.delete_by_mark(mark="hint")

# 清空记忆
await memory.clear()

# 获取记忆状态
state = memory.state_dict()
print(state)
# {"chat_history_storage": [...], "memory_storage": [...], "max_token": 28000}

# 加载记忆状态
memory.load_state_dict(state)
```

#### RedisMemory - Redis 持久化存储

```python
from agentscope.memory import RedisMemory

# 创建 Redis 记忆
redis_memory = RedisMemory(
    url="redis://localhost:6379/0",      # Redis 连接 URL
    namespace="agent_memory",            # 命名空间（用于隔离）
    ttl=3600,                            # 过期时间（秒），None 表示永不过期
)

# 使用方式与 InMemoryMemory 相同
await redis_memory.add(Msg("user", "Hello", "user"))
messages = await redis_memory.get_memory()

# 适用于生产环境，支持多进程/分布式共享记忆
```

#### AsyncSQLAlchemyMemory - 数据库存储

```python
from agentscope.memory import AsyncSQLAlchemyMemory

# SQLite（本地存储）
sql_memory = AsyncSQLAlchemyMemory(
    url="sqlite+aiosqlite:///memory.db",
)

# PostgreSQL（生产环境）
pg_memory = AsyncSQLAlchemyMemory(
    url="postgresql+asyncpg://user:password@localhost:5432/agent_db",
)

# MySQL
mysql_memory = AsyncSQLAlchemyMemory(
    url="mysql+aiomysql://user:password@localhost:3306/agent_db",
)

# 使用方式相同
await sql_memory.add(Msg("user", "测试", "user"))
```

### 7.2 长期记忆

#### Mem0LongTermMemory

```python
from agentscope.memory import Mem0LongTermMemory
from agentscope.message import Msg

# 创建长期记忆
long_term_memory = Mem0LongTermMemory(
    user_id="user_123",                    # 用户 ID
    agent_name="Assistant",                # 智能体名称
    config={"version": "v1.1"},            # Mem0 配置
)

# 使用上下文管理器（推荐）
async with long_term_memory:
    # 记录对话
    await long_term_memory.record(
        msgs=[
            Msg(role="user", content="我喜欢喝龙井茶", name="user"),
            Msg(role="assistant", content="记住了", name="assistant"),
        ],
    )
    
    # 检索相关记忆
    memories = await long_term_memory.retrieve(
        msg=Msg(role="user", content="我喜欢喝什么茶？", name="user"),
    )
    print(memories)

# 高级 Mem0 配置
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

custom_mem0 = Mem0LongTermMemory(
    user_id="user_123",
    agent_name="Assistant",
    mem0_config=advanced_config,
)
```

#### ReMePersonalLongTermMemory

```python
from agentscope.memory import ReMePersonalLongTermMemory
from agentscope.embedding import DashScopeTextEmbedding
from agentscope.model import DashScopeChatModel
from agentscope.message import Msg
import os

# 创建个人长期记忆
personal_memory = ReMePersonalLongTermMemory(
    agent_name="Friday",
    user_name="user_123",
    model=DashScopeChatModel(
        model_name="qwen3-max",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=False,
    ),
    embedding_model=DashScopeTextEmbedding(
        model_name="text-embedding-v4",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        dimensions=1024,
    ),
    vector_store_dir="./vector_store",     # 向量存储目录
)

async with personal_memory:
    # 方式 1: record_to_memory（工具函数）
    result = await personal_memory.record_to_memory(
        thinking="用户分享旅行偏好",
        content=[
            "我在杭州旅行时喜欢住民宿",
            "我喜欢早上去西湖",
            "我喜欢喝龙井茶",
        ],
    )
    
    # 方式 2: retrieve_from_memory（工具函数）
    result = await personal_memory.retrieve_from_memory(
        keywords=["杭州旅行", "茶偏好"],
    )
    
    # 方式 3: record（直接方法）
    await personal_memory.record(
        msgs=[
            Msg(role="user", content="我是软件工程师", name="user"),
            Msg(role="assistant", content "记住了", name="assistant"),
        ],
    )
    
    # 方式 4: retrieve（直接方法）
    memories = await personal_memory.retrieve(
        msg=Msg(role="user", content="你知道我的职业吗？", name="user"),
    )
    print(memories)
```

#### ReMeTaskLongTermMemory

```python
from agentscope.memory import ReMeTaskLongTermMemory
from agentscope.message import Msg

# 任务长期记忆（用于记录任务执行经验）
task_memory = ReMeTaskLongTermMemory(
    agent_name="TaskAssistant",
    user_name="workspace_123",  # 作为工作空间 ID
    model=dashscope_model,
    embedding_model=dashscope_embedding,
)

async with task_memory:
    # 记录成功的任务执行（带评分）
    await task_memory.record(
        msgs=[
            Msg(role="user", content="API 返回 404", name="user"),
            Msg(role="assistant", content="检查路由定义...", name="assistant"),
            Msg(role="user", content="找到拼写错误了！", name="user"),
        ],
        score=0.95,  # 高分表示成功的执行轨迹
    )
    
    # 检索相关经验
    experiences = await task_memory.retrieve(
        msg=Msg(role="user", content="如何调试 API 错误？", name="user"),
    )
    print(experiences)
```

### 7.3 记忆压缩

```python
from agentscope.agent import ReActAgent
from agentscope.token import OpenAITokenCounter

# 配置记忆压缩
agent = ReActAgent(
    name="CompressingAgent",
    sys_prompt="你是一个智能助手",
    model=model,
    formatter=OpenAIChatFormatter(),
    memory=InMemoryMemory(),
    compression_config=ReActAgent.CompressionConfig(
        enable=True,
        agent_token_counter=OpenAITokenCounter(model_name="gpt-4"),
        trigger_threshold=8000,     # 超过 8000 tokens 触发压缩
        keep_recent=3,              # 保留最近 3 条消息不压缩
    ),
)

# 压缩策略：
# 1. 当对话历史超过阈值时自动触发
# 2. 保留最近 N 条消息保持上下文连贯性
# 3. 使用 LLM 总结早期对话
```

---

## 8. 会话管理

### 8.1 基础会话

```python
from agentscope.session import Session
from agentscope.agent import ReActAgent
from agentscope.message import Msg

# 创建会话
session = Session(
    session_id="session_001",
    max_turns=10,  # 最大对话轮数
)

# 添加智能体到会话
session.add_agent(agent)

# 进行对话
response = await session.run(
    Msg(name="user", content="你好", role="user")
)

# 获取会话历史
history = session.get_history()

# 保存会话
session.save("./sessions/session_001.json")

# 加载会话
loaded_session = Session.load("./sessions/session_001.json")
```

### 8.2 多轮对话

```python
from agentscope.agent import ReActAgent, UserAgent

async def multi_turn_conversation():
    agent = ReActAgent(...)
    user = UserAgent(name="user")
    
    msg = None
    turn_count = 0
    max_turns = 10
    
    while turn_count < max_turns:
        # 用户输入
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break
        
        # 智能体响应
        msg = await agent(msg)
        print(f"{agent.name}: {msg.get_text_content()}")
        
        turn_count += 1
```

---

## 9. 格式化处理

格式化器负责将 AgentScope 的消息格式转换为不同 LLM API 所需的格式。

### 9.1 支持的格式化器

```python
from agentscope.formatter import (
    OpenAIChatFormatter,      # OpenAI/Azure/DeepSeek/vLLM
    AnthropicChatFormatter,   # Anthropic Claude
    DashScopeChatFormatter,   # 阿里云 DashScope
    GeminiChatFormatter,      # Google Gemini
    OllamaChatFormatter,      # Ollama
    DeepSeekChatFormatter,    # DeepSeek
)
```

### 9.2 格式化器使用

```python
from agentscope.formatter import OpenAIChatFormatter, AnthropicChatFormatter
from agentscope.message import Msg, TextBlock, ImageBlock, URLSource
import asyncio

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
    print("OpenAI 格式:")
    print(openai_formatted)
    # [
    #   {"role": "system", "content": "你是有帮助的助手"},
    #   {"role": "user", "content": [
    #     {"type": "text", "text": "描述这张图片"},
    #     {"type": "image_url", "image_url": {"url": "..."}}
    #   ]}
    # ]
    
    # 转换为 Anthropic 格式
    anthropic_formatted = await anthropic_formatter.format(messages)
    print("Anthropic 格式:")
    print(anthropic_formatted)
    # [
    #   {"role": "system", "content": "你是有帮助的助手"},
    #   {"role": "user", "content": [
    #     {"type": "text", "text": "描述这张图片"},
    #     {"type": "image", "source": {"type": "url", "url": "..."}}
    #   ]}
    # ]
```

### 9.3 自定义格式化器

```python
from agentscope.formatter import FormatterBase
from agentscope.message import Msg

class CustomFormatter(FormatterBase):
    """自定义格式化器"""
    
    async def format(self, messages: list[Msg]) -> list[dict]:
        """将消息转换为目标格式"""
        formatted = []
        
        for msg in messages:
            formatted_msg = {
                "role": msg.role,
                "name": msg.name,
                "content": msg.get_text_content(),
                "timestamp": msg.timestamp,
            }
            formatted.append(formatted_msg)
        
        return formatted

# 使用自定义格式化器
custom_formatter = CustomFormatter()
formatted = await custom_formatter.format(messages)
```

---

## 10. 工作流编排

工作流编排用于管理多智能体协作和消息路由。

### 10.1 MsgHub - 消息广播中心

```python
from agentscope.pipeline import MsgHub, sequential_pipeline, fanout_pipeline
from agentscope.agent import ReActAgent
from agentscope.message import Msg
import asyncio
import os

async def example_msghub():
    # 创建多个智能体
    alice = ReActAgent(
        name="Alice",
        sys_prompt="你是研究员 Alice",
        model=OpenAIChatModel(model_name="gpt-4", api_key=os.environ["OPENAI_API_KEY"]),
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    bob = ReActAgent(
        name="Bob",
        sys_prompt="你是工程师 Bob",
        model=OpenAIChatModel(model_name="gpt-4", api_key=os.environ["OPENAI_API_KEY"]),
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    charlie = ReActAgent(
        name="Charlie",
        sys_prompt="你是产品经理 Charlie",
        model=OpenAIChatModel(model_name="gpt-4", api_key=os.environ["OPENAI_API_KEY"]),
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 创建消息广播中心
    async with MsgHub(
        participants=[alice, bob, charlie],
        announcement=Msg("Host", "讨论新的 AI 功能", "assistant"),
        enable_auto_broadcast=True,  # 启用自动广播
    ) as hub:
        # 顺序对话 - 智能体按顺序发言
        await sequential_pipeline([alice, bob, charlie])
        
        # 动态添加/删除参与者
        david = await create_agent("David", "设计师")
        hub.add(david)
        hub.delete(charlie)
        
        # 继续对话
        await sequential_pipeline([alice, bob, david])
        
        # 手动广播消息
        await hub.broadcast(Msg("Host", "让我们总结一下", "assistant"))

asyncio.run(example_msghub())
```

### 10.2 顺序工作流

```python
from agentscope.pipeline import sequential_pipeline

# 顺序执行 - 每个智能体依次处理消息
async def sequential_example():
    agents = [agent1, agent2, agent3]
    initial_msg = Msg("user", "处理这个任务", "user")
    
    # 消息依次经过每个智能体
    result = await sequential_pipeline(
        agents=agents,
        msg=initial_msg,
    )
    
    print(f"最终结果：{result.get_text_content()}")
```

### 10.3 并行工作流

```python
from agentscope.pipeline import fanout_pipeline

# 并行执行 - 多个智能体同时处理同一消息
async def fanout_example():
    agents = [agent1, agent2, agent3]
    question = Msg("user", "你对 AI 安全有什么看法？", "user")
    
    # 所有智能体同时收到消息并生成响应
    responses = await fanout_pipeline(
        agents=agents,
        msg=question,
    )
    
    # responses 是来自每个智能体的 Msg 对象列表
    for i, response in enumerate(responses):
        print(f"智能体{i}的回答：{response.get_text_content()}")
```

### 10.4 流式输出

```python
from agentscope.pipeline import stream_printing_messages

# 实时流式打印智能体响应
async def streaming_example():
    async for chunk in stream_printing_messages(agent(initial_msg)):
        print(chunk.get_text_content(), end="", flush=True)
    # 输出会实时显示，类似打字机效果
```

### 10.5 复杂工作流编排

```python
from agentscope.pipeline import MsgHub, sequential_pipeline, fanout_pipeline

async def complex_workflow():
    """复杂工作流示例：头脑风暴 -> 评估 -> 决策"""
    
    # 阶段 1: 头脑风暴（并行）
    brainstorm_agents = [creative1, creative2, creative3]
    ideas = await fanout_pipeline(
        agents=brainstorm_agents,
        msg=Msg("user", "提出新产品创意", "user"),
    )
    
    # 阶段 2: 评估（顺序）
    evaluator1, evaluator2 = technical_eval, market_eval
    eval_result = await sequential_pipeline(
        agents=[evaluator1, evaluator2],
        msg=Msg("system", f"评估这些创意：{ideas}", "system"),
    )
    
    # 阶段 3: 决策（会议讨论）
    decision_makers = [ceo, cto, cpo]
    async with MsgHub(
        participants=decision_makers,
        announcement=Msg("system", f"基于评估做决策：{eval_result}", "system"),
        enable_auto_broadcast=True,
    ) as hub:
        final_decision = await sequential_pipeline(decision_makers)
    
    return final_decision
```

---

## 11. RAG 知识增强

检索增强生成（RAG）允许智能体基于外部知识库回答问题。

### 11.1 文档读取器

```python
from agentscope.rag import (
    TextReader,      # 文本文件读取
    PDFReader,       # PDF 文件读取
    WordReader,      # Word 文档读取
    ExcelReader,     # Excel 表格读取
)

# 创建读取器
text_reader = TextReader()
pdf_reader = PDFReader()
word_reader = WordReader()
excel_reader = ExcelReader()

# 读取文档
text_docs = text_reader.read("./documents/guide.txt")
pdf_docs = pdf_reader.read("./documents/manual.pdf")
word_docs = word_reader.read("./documents/report.docx")

# 文档结构
# 每个文档包含：
# - content: 文本内容
# - metadata: 元数据（文件名、页码等）
```

### 11.2 向量存储

#### QdrantStore

```python
from agentscope.rag import QdrantStore
from agentscope.embedding import OpenAIEmbedding

# 创建嵌入模型
embedding = OpenAIEmbedding(
    model_name="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
)

# 创建 Qdrant 向量存储
qdrant_store = QdrantStore(
    collection_name="my_knowledge",
    path="./qdrant_data",  # 本地存储路径
    embedding=embedding,
)

# 或者连接远程 Qdrant
qdrant_remote = QdrantStore(
    collection_name="remote_kb",
    url="http://localhost:6333",
    api_key="your-qdrant-key",
    embedding=embedding,
)
```

#### MilvusLiteStore

```python
from agentscope.rag import MilvusLiteStore

# Milvus Lite（嵌入式，无需单独部署）
milvus_store = MilvusLiteStore(
    collection_name="my_knowledge",
    uri="./milvus_data.db",  # 本地数据库文件
    embedding=embedding,
)
```

### 11.3 知识库管理

```python
from agentscope.rag import KnowledgeBase, SimpleKnowledge

# 方式 1: 完整知识库
knowledge_base = KnowledgeBase(
    store=qdrant_store,
    readers=[text_reader, pdf_reader],
)

# 添加文档
await knowledge_base.add_documents(text_docs)
await knowledge_base.add_documents(pdf_docs)

# 从路径直接添加
await knowledge_base.add_from_path("./documents/")  # 添加整个目录

# 检索相关文档
relevant_docs = await knowledge_base.search(
    query="如何使用 AgentScope",
    top_k=5,  # 返回最相关的 5 个文档
)

# 方式 2: 简单知识库（快速设置）
simple_kb = SimpleKnowledge(
    embedding=embedding,
    documents=[
        "AgentScope 是一个多智能体框架",
        "它支持各种 LLM 提供商",
        "RAG 功能实现知识增强回答",
    ],
)
```

### 11.4 带 RAG 的智能体

```python
from agentscope.agent import ReActAgent
from agentscope.rag import KnowledgeBase

# 创建带 RAG 的智能体
rag_agent = ReActAgent(
    name="KnowledgeAssistant",
    sys_prompt="基于提供的知识库回答问题。如果知识库中没有相关信息，请明确说明。",
    model=model,
    formatter=OpenAIChatFormatter(),
    memory=InMemoryMemory(),
    knowledge=knowledge_base,  # 附加知识库
    enable_rewrite_query=True,  # 启用查询重写（提高检索效果）
)

# 智能体会自动：
# 1. 接收用户问题
# 2. 从知识库检索相关文档
# 3. 将检索结果作为上下文提供给 LLM
# 4. 生成基于知识的回答
```

### 11.5 RAG 完整示例

```python
from agentscope.rag import (
    TextReader, PDFReader, QdrantStore, KnowledgeBase
)
from agentscope.embedding import OpenAIEmbedding
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
import asyncio
import os

async def rag_full_example():
    # 1. 创建嵌入模型
    embedding = OpenAIEmbedding(
        model_name="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    
    # 2. 创建文档读取器
    text_reader = TextReader()
    pdf_reader = PDFReader()
    
    # 3. 创建向量存储
    qdrant_store = QdrantStore(
        collection_name="product_kb",
        path="./qdrant_data",
        embedding=embedding,
    )
    
    # 4. 创建知识库
    knowledge_base = KnowledgeBase(
        store=qdrant_store,
        readers=[text_reader, pdf_reader],
    )
    
    # 5. 添加文档
    text_docs = text_reader.read("./docs/product_manual.txt")
    await knowledge_base.add_documents(text_docs)
    
    # 6. 创建 RAG 智能体
    model = OpenAIChatModel(
        model_name="gpt-4",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    
    rag_agent = ReActAgent(
        name="ProductAssistant",
        sys_prompt="你是产品支持助手，基于知识库回答用户问题。",
        model=model,
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
        knowledge=knowledge_base,
        enable_rewrite_query=True,
    )
    
    # 7. 提问
    response = await rag_agent(
        Msg("user", "这个产品如何安装？", "user")
    )
    print(response.get_text_content())

asyncio.run(rag_full_example())
```

---

## 12. 嵌入模型

嵌入模型用于将文本转换为向量表示，支持 RAG、语义搜索等功能。

### 12.1 OpenAI 嵌入

```python
from agentscope.embedding import OpenAIEmbedding
import asyncio
import os

async def openai_embedding_example():
    embedding_model = OpenAIEmbedding(
        model_name="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    
    texts = [
        "什么是 AgentScope？",
        "AgentScope 是一个多智能体框架",
    ]
    
    # 生成嵌入
    response = await embedding_model(texts)
    
    print(f"嵌入 ID: {response.id}")
    print(f"创建时间：{response.created_at}")
    print(f"Token 使用：{response.usage}")
    print(f"嵌入向量维度：{len(response.embeddings[0])}")
    print(f"嵌入向量：{response.embeddings}")

asyncio.run(openai_embedding_example())
```

### 12.2 DashScope 嵌入

```python
from agentscope.embedding import DashScopeTextEmbedding
import asyncio
import os

async def dashscope_embedding_example():
    embedding_model = DashScopeTextEmbedding(
        model_name="text-embedding-v2",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    )
    
    texts = [
        "法国的首都是哪里？",
        "巴黎是法国的首都",
    ]
    
    response = await embedding_model(texts)
    
    print(f"嵌入向量：{response.embeddings}")

asyncio.run(dashscope_embedding_example())
```

### 12.3 嵌入缓存

```python
from agentscope.embedding import FileEmbeddingCache, OpenAIEmbedding

# 创建文件缓存
cache = FileEmbeddingCache(
    cache_path="./embedding_cache.json",
)

# 带缓存的嵌入模型
embedding_model = OpenAIEmbedding(
    model_name="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
    cache=cache,  # 启用缓存
)

# 相同的文本不会重复调用 API
texts = ["相同文本", "相同文本"]
response1 = await embedding_model(texts)  # 调用 API
response2 = await embedding_model(texts)  # 从缓存读取

# 清空缓存
cache.clear()

# 保存缓存
cache.save()
```

### 12.4 多模态嵌入

```python
from agentscope.embedding import DashScopeMultiModalEmbedding

# 多模态嵌入（文本、图像、视频）
multimodal_embedding = DashScopeMultiModalEmbedding(
    model_name="multimodal-embedding-v1",
    api_key=os.environ["DASHSCOPE_API_KEY"],
)

# 文本嵌入
text_response = await multimodal_embedding(
    texts=["这是一段文本"],
)

# 图像嵌入
image_response = await multimodal_embedding(
    images=["./image.jpg"],
)

# 视频嵌入
video_response = await multimodal_embedding(
    videos=["./video.mp4"],
)
```

---

## 13. Token 计数

Token 计数器用于跟踪 LLM 调用的 Token 使用情况，支持成本估算和记忆压缩。

### 13.1 OpenAI Token 计数器

```python
from agentscope.token import OpenAITokenCounter

# 创建计数器
counter = OpenAITokenCounter(model_name="gpt-4")

# 计算文本 Token 数
text = "这是一段测试文本"
num_tokens = counter.count_tokens(text)
print(f"Token 数：{num_tokens}")

# 计算消息列表的 Token 数
messages = [
    {"role": "system", "content": "你是助手"},
    {"role": "user", "content": "你好"},
]
total_tokens = counter.count_messages(messages)
print(f"总 Token 数：{total_tokens}")
```

### 13.2 DashScope Token 计数器

```python
from agentscope.token import DashScopeTokenCounter

counter = DashScopeTokenCounter(model_name="qwen-max")
num_tokens = counter.count_tokens("测试文本")
```

### 13.3 自定义 Token 计数器

```python
from agentscope.token import TokenCounterBase

class CustomTokenCounter(TokenCounterBase):
    """自定义 Token 计数器"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        # 实现你的 Token 计算逻辑
    
    def count_tokens(self, text: str) -> int:
        """计算文本的 Token 数"""
        # 简单示例：按字符数估算
        return len(text) // 4
    
    def count_messages(self, messages: list) -> int:
        """计算消息列表的 Token 数"""
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.get("content", ""))
        return total

# 使用自定义计数器
custom_counter = CustomTokenCounter("my-model")
```

### 13.4 在记忆压缩中使用

```python
from agentscope.agent import ReActAgent
from agentscope.token import OpenAITokenCounter

# 配置带 Token 计数的记忆压缩
agent = ReActAgent(
    name="TokenAwareAgent",
    sys_prompt="你是智能助手",
    model=model,
    memory=InMemoryMemory(),
    compression_config=ReActAgent.CompressionConfig(
        enable=True,
        agent_token_counter=OpenAITokenCounter(model_name="gpt-4"),
        trigger_threshold=8000,      # 8000 tokens 触发压缩
        keep_recent=3,               # 保留最近 3 条消息
    ),
)

# 当对话历史超过 8000 tokens 时，自动压缩早期对话
```

---

## 14. 评估系统

评估系统用于基准测试智能体性能。

### 14.1 评估基础组件

```python
from agentscope.evaluate import (
    GeneralEvaluator,      # 通用评估器
    RayEvaluator,          # 分布式评估器（基于 Ray）
    BenchmarkBase,         # 基准测试基类
    Task,                  # 任务定义
    SolutionOutput,        # 解决方案输出
    MetricBase,            # 评估指标基类
    MetricResult,          # 评估结果
    MetricType,            # 指标类型
    ACEBenchmark,          # ACE 基准测试
)
```

### 14.2 自定义评估指标

```python
from agentscope.evaluate import MetricBase, MetricType, MetricResult
from agentscope.evaluate import Task, SolutionOutput

class AccuracyMetric(MetricBase):
    """准确率评估指标"""
    
    name = "accuracy"
    metric_type = MetricType.HIGHER_IS_BETTER  # 越高越好
    
    async def compute(
        self,
        task: Task,
        solution: SolutionOutput,
    ) -> MetricResult:
        """计算准确率"""
        expected = task.metadata.get("expected_answer")
        actual = solution.output
        
        # 比较答案
        is_correct = str(expected).lower() == str(actual).lower()
        
        return MetricResult(
            name=self.name,
            value=1.0 if is_correct else 0.0,
            metadata={
                "expected": expected,
                "actual": actual,
            },
        )

class BLEUScoreMetric(MetricBase):
    """BLEU 分数（文本生成质量）"""
    
    name = "bleu_score"
    metric_type = MetricType.HIGHER_IS_BETTER
    
    async def compute(
        self,
        task: Task,
        solution: SolutionOutput,
    ) -> MetricResult:
        from nltk.translate.bleu_score import sentence_bleu
        
        reference = task.metadata.get("reference_text")
        candidate = solution.output
        
        # 计算 BLEU 分数
        score = sentence_bleu([reference.split()], candidate.split())
        
        return MetricResult(
            name=self.name,
            value=score,
            metadata={"reference": reference, "candidate": candidate},
        )
```

### 14.3 通用评估器

```python
from agentscope.evaluate import GeneralEvaluator, Task, SolutionOutput
from agentscope.agent import ReActAgent
import asyncio

async def evaluation_example():
    # 创建评估器
    evaluator = GeneralEvaluator(
        metrics=[AccuracyMetric()],  # 使用自定义指标
        num_workers=4,                # 并行评估的工作进程数
    )
    
    # 定义任务
    tasks = [
        Task(
            id="task_1",
            prompt="2 + 2 等于多少？",
            metadata={"expected_answer": "4"},
        ),
        Task(
            id="task_2",
            prompt="法国的首都是哪里？",
            metadata={"expected_answer": "巴黎"},
        ),
        Task(
            id="task_3",
            prompt="Python 中如何定义函数？",
            metadata={"expected_answer": "使用 def 关键字"},
        ),
    ]
    
    # 创建待评估的智能体
    agent = ReActAgent(...)
    
    # 定义解决方案函数
    async def solve(task: Task) -> SolutionOutput:
        response = await agent(task.prompt)
        return SolutionOutput(
            task_id=task.id,
            output=response.get_text_content(),
        )
    
    # 运行评估
    results = await evaluator.evaluate(
        tasks=tasks,
        solve_fn=solve,
    )
    
    # 打印结果
    for result in results:
        print(f"任务 {result.task_id}:")
        print(f"  指标：{result.metrics}")
        print(f"  输出：{result.output}")
    
    # 统计总体表现
    avg_accuracy = sum(
        r.metrics["accuracy"].value for r in results
    ) / len(results)
    print(f"平均准确率：{avg_accuracy * 100:.2f}%")

asyncio.run(evaluation_example())
```

### 14.4 分布式评估（Ray）

```python
from agentscope.evaluate import RayEvaluator

# Ray 评估器用于大规模基准测试
ray_evaluator = RayEvaluator(
    metrics=[AccuracyMetric(), BLEUScoreMetric()],
    num_workers=16,  # 16 个并行工作进程
)

# 使用方式与 GeneralEvaluator 相同
# 适用于数百上千任务的评估
results = await ray_evaluator.evaluate(
    tasks=large_task_set,
    solve_fn=solve,
)
```

### 14.5 ACE 基准测试

```python
from agentscope.evaluate import ACEBenchmark

# ACE 基准测试（Agent 能力评估）
ace_benchmark = ACEBenchmark(
    data_path="./ace_data/",  # 基准测试数据路径
)

# 运行基准测试
results = await ace_benchmark.run(agent)

# 分析结果
print(f"总体得分：{results.overall_score}")
print(f"各维度得分：{results.dimension_scores}")
```

---

## 15. 链路追踪与监控

### 15.1 OpenTelemetry 集成

```python
import agentscope
from agentscope.tracing import setup_tracing, trace
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
import asyncio
import os

async def tracing_example():
    # 方式 1: 连接到 AgentScope Studio
    agentscope.init(
        project="traced_agents",
        name="experiment_001",
        studio_url="http://localhost:3000",  # 链路发送到 Studio
    )
    
    # 方式 2: 连接到外部 OpenTelemetry 收集器
    setup_tracing(
        endpoint="http://localhost:4318/v1/traces",  # Jaeger, Zipkin 等
    )
    
    # 创建智能体 - 所有操作自动被追踪
    agent = ReActAgent(
        name="TracedAssistant",
        sys_prompt="你是有帮助的助手",
        model=OpenAIChatModel(
            model_name="gpt-4",
            api_key=os.environ["OPENAI_API_KEY"],
            stream=True,
        ),
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 自定义追踪函数
    @trace
    async def custom_operation(data: str) -> str:
        """这个操作会被追踪"""
        # 你的业务逻辑
        return f"处理结果：{data}"
    
    # 执行被追踪的操作
    result = await custom_operation("测试数据")
    response = await agent("你好，请帮助我")
    
    # 查看链路：
    # - AgentScope Studio 仪表板
    # - Jaeger UI（如果使用 Jaeger）
    # - Arize Phoenix
    # - Langfuse

asyncio.run(tracing_example())
```

### 15.2 追踪装饰器

```python
from agentscope.tracing import trace

# 追踪异步函数
@trace
async def process_data(data: dict) -> dict:
    """数据处理函数（会被追踪）"""
    # 业务逻辑
    return processed_data

# 追踪同步函数
@trace
def sync_operation(x: int, y: int) -> int:
    """同步操作（会被追踪）"""
    return x + y

# 带自定义追踪名称
@trace(name="custom_operation_name")
async def my_function():
    pass

# 追踪类方法
class DataProcessor:
    @trace
    async def process(self, data):
        return data
```

### 15.3 手动追踪

```python
from agentscope.tracing import start_span, get_current_span

# 手动创建追踪跨度
async def manual_tracing():
    with start_span("custom_operation") as span:
        # 添加属性
        span.set_attribute("input_size", 100)
        span.set_attribute("user_id", "user_123")
        
        # 添加事件
        span.add_event("processing_started")
        
        # 业务逻辑
        result = await process_data()
        
        # 记录结果
        span.set_attribute("output_size", len(result))
        span.add_event("processing_completed")
        
        return result
```

---

## 16. 异常处理

### 16.1 异常类型

```python
from agentscope.exception import (
    ModelResponseError,       # 模型响应错误
    ToolExecutionError,       # 工具执行错误
    AgentError,              # 智能体错误
    MemoryError,             # 记忆操作错误
    FormatterError,          # 格式化错误
    PipelineError,           # 工作流错误
)
```

### 16.2 异常处理示例

```python
from agentscope.exception import ModelResponseError, ToolExecutionError
from agentscope.agent import ReActAgent
from agentscope.message import Msg
import asyncio

async def robust_agent_call():
    agent = ReActAgent(...)
    
    try:
        # 调用智能体
        response = await agent(Msg("user", "复杂任务", "user"))
        print(f"成功：{response.get_text_content()}")
        
    except ModelResponseError as e:
        # 处理模型响应错误
        print(f"模型错误：{e}")
        print(f"错误详情：{e.details}")
        
    except ToolExecutionError as e:
        # 处理工具执行错误
        print(f"工具执行失败：{e}")
        print(f"工具名称：{e.tool_name}")
        print(f"错误信息：{e.error_message}")
        
    except Exception as e:
        # 其他异常
        print(f"未知错误：{e}")
        import traceback
        traceback.print_exc()
```

### 16.3 重试机制

```python
import asyncio
from agentscope.exception import ModelResponseError

async def retry_agent_call(agent, msg, max_retries=3):
    """带重试的智能体调用"""
    
    for attempt in range(max_retries):
        try:
            response = await agent(msg)
            return response
            
        except ModelResponseError as e:
            if attempt == max_retries - 1:
                raise  # 最后一次重试失败，抛出异常
            
            # 等待后重试
            wait_time = 2 ** attempt  # 指数退避
            print(f"重试 {attempt + 1}/{max_retries}, 等待 {wait_time}秒")
            await asyncio.sleep(wait_time)
    
    return None
```

---

## 17. MCP 集成

MCP（Model Context Protocol）允许智能体调用外部工具和服务。

### 17.1 HTTP 无状态客户端

```python
from agentscope.mcp import HttpStatelessClient
from agentscope.tool import Toolkit
import os
import asyncio

async def mcp_example():
    # 创建 MCP 客户端
    http_client = HttpStatelessClient(
        name="gaode_maps",                    # 客户端名称
        transport="streamable_http",          # 传输方式：streamable_http 或 sse
        url=f"https://mcp.amap.com/mcp?key={os.environ['GAODE_API_KEY']}",
        headers={"Authorization": "Bearer token"},
        timeout=30,                           # 超时时间（秒）
    )
    
    # 列出可用工具
    tools = await http_client.list_tools()
    print(f"可用工具：{[t.name for t in tools]}")
    
    # 获取特定工具作为可调用函数
    maps_geo = await http_client.get_callable_function("maps_geo")
    
    # 直接调用工具
    result = await maps_geo(
        address="天安门广场",
        city="北京",
    )
    print(f"地理编码结果：{result}")
    
    # 注册到工具包
    toolkit = Toolkit()
    await toolkit.register_mcp_client(
        http_client,
        group_name="basic",
        enable_funcs=["maps_geo", "maps_direction"],  # 只启用这些工具
        disable_funcs=None,
        namesake_strategy="rename",  # 处理命名冲突
    )

asyncio.run(mcp_example())
```

### 17.2 精细 MCP 工具控制

```python
from agentscope.mcp import HttpStatelessClient
from agentscope.tool import Toolkit

async def fine_grained_mcp():
    # 初始化 MCP 客户端
    client = HttpStatelessClient(
        name="gaode_mcp",
        transport="streamable_http",
        url=f"https://mcp.amap.com/mcp?key={os.environ['GAODE_API_KEY']}",
    )
    
    # 获取 MCP 工具作为本地可调用函数
    func = await client.get_callable_function(func_name="maps_geo")
    
    # 方式 1: 直接调用
    result = await func(address="天安门广场", city="北京")
    
    # 方式 2: 注册到智能体工具包
    toolkit = Toolkit()
    toolkit.register_tool_function(func)
    
    # 方式 3: 包装为更复杂的工具
    async def enhanced_geo_query(address: str) -> dict:
        geo_result = await func(address=address)
        # 添加额外处理逻辑
        return {
            "raw": geo_result,
            "processed": process(geo_result),
        }
```

---

## 18. 高级应用案例

### 18.1 多智能体辩论系统

```python
from agentscope.pipeline import MsgHub, sequential_pipeline, fanout_pipeline
from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
import asyncio
import os

async def debate_system():
    """多智能体辩论系统"""
    
    # 创建辩论双方
    pro_agent = ReActAgent(
        name="正方",
        sys_prompt="你是辩论的正方，必须支持给定论点。使用有力的论据和证据。",
        model=OpenAIChatModel(model_name="gpt-4", api_key=os.environ["OPENAI_API_KEY"]),
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    con_agent = ReActAgent(
        name="反方",
        sys_prompt="你是辩论的反方，必须反对给定论点。使用有力的论据和证据。",
        model=OpenAIChatModel(model_name="gpt-4", api_key=os.environ["OPENAI_API_KEY"]),
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 创建裁判
    judge_agent = ReActAgent(
        name="裁判",
        sys_prompt="你是辩论裁判。倾听双方论点后，给出公正的评判和理由。",
        model=OpenAIChatModel(model_name="gpt-4", api_key=os.environ["OPENAI_API_KEY"]),
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 辩论主题
    topic = "人工智能应该被严格监管"
    opening_msg = Msg("主持人", f"辩论主题：{topic}", "assistant")
    
    print(f"=== 辩论开始：{topic} ===\n")
    
    # 第一轮：开篇立论
    print("--- 开篇立论 ---")
    pro_opening = await pro_agent(opening_msg)
    print(f"正方：{pro_opening.get_text_content()}\n")
    
    con_opening = await con_agent(pro_opening)
    print(f"反方：{con_opening.get_text_content()}\n")
    
    # 第二轮：自由辩论（3 轮交锋）
    print("--- 自由辩论 ---")
    last_msg = con_opening
    
    for round_num in range(3):
        print(f"\n第{round_num + 1}轮交锋:")
        
        # 正方发言
        pro_response = await pro_agent(last_msg)
        print(f"正方：{pro_response.get_text_content()}\n")
        
        # 反方发言
        con_response = await con_agent(pro_response)
        print(f"反方：{con_response.get_text_content()}\n")
        
        last_msg = con_response
    
    # 总结陈词
    print("--- 总结陈词 ---")
    pro_closing = await pro_agent(Msg("主持人", "请做最后总结", "assistant"))
    print(f"正方总结：{pro_closing.get_text_content()}\n")
    
    con_closing = await con_agent(pro_closing)
    print(f"反方总结：{con_closing.get_text_content()}\n")
    
    # 裁判评判
    print("--- 裁判评判 ---")
    debate_history = [opening_msg, pro_opening, con_opening, pro_closing, con_closing]
    
    judge_input = Msg(
        "主持人",
        f"基于以下辩论历史，请给出评判:\n" + 
        "\n".join([f"{m.name}: {m.get_text_content()}" for m in debate_history]),
        "assistant",
    )
    
    judgment = await judge_agent(judge_input)
    print(f"\n裁判评判:\n{judgment.get_text_content()}")

asyncio.run(debate_system())
```

### 18.2 代码审查工作流

```python
from agentscope.pipeline import sequential_pipeline
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit, execute_python_code
from agentscope.message import Msg
import asyncio

async def code_review_workflow():
    """多阶段代码审查工作流"""
    
    # 阶段 1: 语法检查智能体
    syntax_checker = ReActAgent(
        name="语法检查器",
        sys_prompt="你负责检查代码的语法错误、类型问题和基本规范。只报告问题，不提供修复建议。",
        model=model,
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 阶段 2: 安全检查智能体
    security_checker = ReActAgent(
        name="安全检查器",
        sys_prompt="你负责检查代码的安全漏洞，如 SQL 注入、XSS、硬编码密钥等。",
        model=model,
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 阶段 3: 性能优化智能体
    performance_checker = ReActAgent(
        name="性能检查器",
        sys_prompt="你负责识别代码的性能问题，如低效循环、内存泄漏风险、不必要的计算等。",
        model=model,
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 阶段 4: 修复建议智能体
    fix_suggester = ReActAgent(
        name="修复建议器",
        sys_prompt="基于发现的问题，提供具体的代码修复建议和改写示例。",
        model=model,
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 待审查的代码
    code_to_review = """
def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    return result
"""
    
    # 顺序执行审查流程
    initial_msg = Msg("user", f"请审查这段代码:\n{code_to_review}", "user")
    
    print("=== 代码审查开始 ===\n")
    
    # 阶段 1: 语法检查
    print("--- 语法检查 ---")
    syntax_report = await syntax_checker(initial_msg)
    print(f"语法问题:\n{syntax_report.get_text_content()}\n")
    
    # 阶段 2: 安全检查
    print("--- 安全检查 ---")
    security_report = await security_checker(syntax_report)
    print(f"安全问题:\n{security_report.get_text_content()}\n")
    
    # 阶段 3: 性能检查
    print("--- 性能检查 ---")
    performance_report = await performance_checker(security_report)
    print(f"性能问题:\n{performance_report.get_text_content()}\n")
    
    # 阶段 4: 修复建议
    print("--- 修复建议 ---")
    fix_suggestions = await fix_suggester(performance_report)
    print(f"修复建议:\n{fix_suggestions.get_text_content()}")
    
    print("\n=== 代码审查完成 ===")

asyncio.run(code_review_workflow())
```

### 18.3 研究助手（带 RAG 和工具调用）

```python
from agentscope.rag import TextReader, QdrantStore, KnowledgeBase
from agentscope.embedding import OpenAIEmbedding
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit, execute_python_code
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
import asyncio
import os

async def research_assistant():
    """智能研究助手 - 结合 RAG 和工具调用"""
    
    # 1. 设置知识库
    embedding = OpenAIEmbedding(
        model_name="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    
    # 读取研究论文
    paper_reader = TextReader()
    papers = paper_reader.read("./papers/machine_learning_papers.txt")
    
    # 创建向量存储
    vector_store = QdrantStore(
        collection_name="research_papers",
        path="./qdrant_data",
        embedding=embedding,
    )
    
    # 创建知识库
    knowledge_base = KnowledgeBase(
        store=vector_store,
        readers=[paper_reader],
    )
    await knowledge_base.add_documents(papers)
    
    # 2. 创建工具包
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)
    
    # 3. 创建研究助手智能体
    research_agent = ReActAgent(
        name="研究助手",
        sys_prompt="""你是一个智能研究助手。
        1. 基于知识库中的论文回答问题
        2. 可以执行 Python 代码进行数据分析
        3. 提供准确的引用来源
        4. 如果知识库中没有相关信息，明确说明""",
        model=OpenAIChatModel(
            model_name="gpt-4",
            api_key=os.environ["OPENAI_API_KEY"],
            stream=True,
        ),
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
        toolkit=toolkit,
        knowledge=knowledge_base,
        enable_rewrite_query=True,
    )
    
    # 4. 进行研究查询
    queries = [
        "Transformer 架构的核心创新是什么？",
        "请总结注意力机制的发展历程",
        "写一篇关于 BERT 的 Python 分析代码",
    ]
    
    print("=== 研究助手开始工作 ===\n")
    
    for query in queries:
        print(f"问题：{query}")
        response = await research_agent(Msg("user", query, "user"))
        print(f"回答:\n{response.get_text_content()}\n")
        print("-" * 80 + "\n")

asyncio.run(research_assistant())
```

### 18.4 任务规划与执行系统

```python
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit, execute_python_code, execute_shell_command
from agentscope.message import Msg
from agentscope.pipeline import sequential_pipeline
import asyncio
import json

async def task_planning_system():
    """任务规划与执行系统"""
    
    # 规划器智能体
    planner = ReActAgent(
        name="规划器",
        sys_prompt="""你是一个任务规划专家。
        将复杂任务分解为可执行的子任务。
        输出格式必须是 JSON:
        {
            "subtasks": [
                {"id": 1, "description": "...", "type": "code|shell|research"},
                ...
            ]
        }""",
        model=model,
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 执行器智能体
    executor = ReActAgent(
        name="执行器",
        sys_prompt="你负责执行具体的代码和命令。仔细执行每个步骤并报告结果。",
        model=model,
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
        toolkit=Toolkit(),
    )
    executor.toolkit.register_tool_function(execute_python_code)
    executor.toolkit.register_tool_function(execute_shell_command)
    
    # 验证器智能体
    validator = ReActAgent(
        name="验证器",
        sys_prompt="你负责验证执行结果是否满足任务要求。输出通过/失败及理由。",
        model=model,
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 复杂任务
    complex_task = """
    创建一个数据分析项目：
    1. 生成包含 1000 行模拟销售数据的 CSV 文件
    2. 分析数据并计算关键指标
    3. 创建可视化图表
    4. 生成分析报告
    """
    
    print("=== 任务规划与执行系统 ===\n")
    
    # 步骤 1: 任务规划
    print("--- 任务规划 ---")
    plan_request = Msg("user", f"请规划以下任务:\n{complex_task}", "user")
    plan_response = await planner(plan_request)
    
    # 解析规划（假设返回 JSON）
    try:
        plan_text = plan_response.get_text_content()
        # 提取 JSON 部分
        import re
        json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group())
            print(f"任务分解:\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n")
        else:
            raise ValueError("未找到有效 JSON")
    except Exception as e:
        print(f"规划解析失败：{e}")
        return
    
    # 步骤 2: 执行子任务
    print("--- 执行子任务 ---")
    last_result = None
    
    for subtask in plan["subtasks"]:
        print(f"\n执行子任务 {subtask['id']}: {subtask['description']}")
        
        exec_msg = Msg(
            "planner",
            f"执行任务:\n{subtask['description']}\n类型：{subtask['type']}",
            "assistant",
        )
        
        if last_result:
            exec_msg.content += f"\n上一步结果:\n{last_result.get_text_content()}"
        
        exec_result = await executor(exec_msg)
        print(f"执行结果:\n{exec_result.get_text_content()}")
        
        last_result = exec_result
    
    # 步骤 3: 验证结果
    print("\n--- 验证结果 ---")
    validation_msg = Msg(
        "executor",
        f"原始任务:\n{complex_task}\n\n执行结果:\n{last_result.get_text_content()}",
        "assistant",
    )
    
    validation_result = await validator(validation_msg)
    print(f"验证结果:\n{validation_result.get_text_content()}")

asyncio.run(task_planning_system())
```

---

## 附录 A: 最佳实践

### A.1 性能优化

1. **使用缓存**：为嵌入和频繁调用的工具启用缓存
2. **批量处理**：批量调用嵌入模型而非逐个调用
3. **流式输出**：对用户可见的响应启用流式
4. **并行执行**：使用 `fanout_pipeline` 并行处理独立任务
5. **记忆压缩**：长对话启用记忆压缩避免超出上下文限制

### A.2 成本控制

1. **Token 监控**：使用 Token 计数器跟踪使用情况
2. **模型选择**：根据任务复杂度选择合适的模型
3. **缓存策略**：重复查询使用缓存结果
4. **流式处理**：减少不必要的完整响应生成

### A.3 安全考虑

1. **工具权限**：谨慎授予智能体工具执行权限
2. **输入验证**：对用户输入和工具参数进行验证
3. **输出过滤**：过滤智能体输出中的敏感信息
4. **资源限制**：为工具执行设置超时和资源限制

### A.4 调试技巧

1. **启用日志**：设置 `logging_level="DEBUG"`
2. **链路追踪**：连接 AgentScope Studio 查看执行链路
3. **异常捕获**：在关键调用周围添加异常处理
4. **状态检查**：定期检查记忆和会话状态

---

## 附录 B: 常见问题

### B.1 模型调用失败

**问题**：模型调用返回空响应或超时

**解决**：
- 检查 API 密钥是否正确
- 验证网络连接
- 增加超时时间
- 减少上下文长度
- 检查模型配额

### B.2 记忆泄漏

**问题**：长时间运行后内存占用过高

**解决**：
- 启用记忆压缩
- 定期清理旧对话
- 使用 Redis 等外部存储
- 限制记忆大小

### B.3 工具执行卡住

**问题**：工具调用长时间无响应

**解决**：
- 添加工具执行超时
- 检查工具依赖
- 添加异常处理
- 使用异步工具调用

---

## 附录 C: 资源链接

- **官方文档**: https://doc.agentscope.io
- **GitHub 仓库**: https://github.com/agentscope-ai/agentscope
- **AgentScope Studio**: https://github.com/agentscope-ai/studio
- **示例代码**: https://github.com/agentscope-ai/agentscope/tree/main/examples
- **社区讨论**: https://github.com/agentscope-ai/agentscope/discussions

---

*本指南基于 AgentScope v1.0+ 版本编写，持续更新中。*
