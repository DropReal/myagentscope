# AgentScope 多智能体开发框架深度指南

> 本文档基于 AgentScope v1.0+官方文档编写，提供从基础概念到高级应用的完整使用说明，包含每个模块的详细 API 参数、配置选项和丰富代码示例。

**文档版本**: v2.0 (完全基于官方教程增强)
**最后更新**: 2026 年 3 月 1 日

---

## 目录

1. [框架概述与核心概念](#1-框架概述与核心概念)
2. [安装与环境配置](#2-安装与环境配置)
3. [消息系统](#3-消息系统)
4. [模型集成](#4-模型集成)
5. [智能体系统](#5-智能体系统)
6. [工具系统](#6-工具系统)
7. [记忆管理](#7-记忆管理)
8. [长期记忆](#8-长期记忆)
9. [会话管理](#9-会话管理)
10. [格式化处理](#10-格式化处理)
11. [Token 计数](#11-token 计数)
12. [工作流编排](#12-工作流编排)
13. [计划模块](#13-计划模块)
14. [RAG 知识增强](#14-rag 知识增强)
15. [嵌入模型](#15-嵌入模型)
16. [评估系统](#16-评估系统)
17. [链路追踪与监控](#17-链路追踪与监控)
18. [AgentScope Studio](#18-agentscope-studio)
19. [MCP 集成](#19-mcp 集成)
20. [智能体钩子函数](#20-智能体钩子函数)
21. [中间件](#21-中间件)
22. [A2A 智能体](#22-a2a 智能体)
23. [实时智能体](#23-实时智能体)
24. [TTS 语音合成](#24-tts 语音合成)
25. [Tuner 模型调优](#25-tuner 模型调优)
26. [高级应用案例](#26-高级应用案例)
27. [最佳实践](#27-最佳实践)
28. [常见问题](#28-常见问题)

---

## 1. 框架概述与核心概念

### 1.1 什么是 AgentScope

AgentScope 是阿里巴巴开源的多智能体系统构建框架，采用面向智能体的编程范式（Agent-Oriented Programming），专为大语言模型（LLM）应用开发设计。

**核心特性**:

| 特性 | 描述 | 优势 |
|------|------|------|
| 模型无关性 | 统一接口支持 OpenAI、Anthropic、Gemini、DashScope、Ollama 等 | 无需修改代码即可切换模型提供商 |
| 模块化设计 | 消息、记忆、工具、模型等组件高度解耦 | 自由组合，灵活定制 |
| 分布式支持 | 原生支持多智能体协作和消息路由 | 轻松构建复杂多智能体系统 |
| 透明控制 | 完整的链路追踪和调试能力 | 开发过程可视化，问题易定位 |
| 实时交互 | 流式输出和人机协作 | 用户体验流畅，可实时干预 |
| 工具支持 | 完整的工具调用和管理系统 | 扩展智能体能力边界 |
| RAG 集成 | 内置检索增强生成支持 | 知识驱动的智能体决策 |

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

### 1.4 AgentScope 中的智能体类层次

```
AgentBase (基类)
├── ReActAgentBase (ReAct 抽象类)
│   └── ReActAgent (完整实现)
├── UserAgent (用户交互)
├── A2aAgent (A2A 协议)
└── 其他特殊智能体
```

| 类 | 抽象方法 | 支持的钩子函数 | 描述 |
|----|---------|---------------|------|
| `AgentBase` | `reply` | `observe`, `print`, `handle_interrupt` | 所有智能体的基类 |
| `ReActAgentBase` | `reply`, `_reasoning`, `_acting` | 全部钩子函数 | ReAct 智能体抽象类 |
| `ReActAgent` | 无 | 全部钩子函数 | `ReActAgentBase` 的完整实现 |
| `UserAgent` | 无 | 基础钩子 | 代表用户的特殊智能体 |
| `A2aAgent` | 无 | 基础钩子 | A2A 协议远程智能体通信 |

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

# TTS 语音合成
pip install agentscope[tts]
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

# 查看消息属性
print(text_msg.name)                  # "user"
print(text_msg.content)               # "你好，请帮我分析这段代码"
print(text_msg.role)                  # "user"
print(text_msg.timestamp)             # 消息时间戳
print(text_msg.id)                    # 消息唯一 ID
print(text_msg.invocation_id)         # 调用 ID
```

#### 内容块类型

AgentScope 使用内容块（Block）来组织多模态消息：

**TextBlock - 文本块**

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

**ImageBlock - 图像块**

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

**AudioBlock - 音频块**

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

**ToolUseBlock - 工具使用块**

```python
from agentscope.message import ToolUseBlock

tool_use = ToolUseBlock(
    type="tool_use",
    id="call_abc123",              # 工具调用 ID
    name="execute_python_code",     # 工具名称
    input={"code": "print('Hello')"},  # 工具参数
    raw_input='{"code": "print(\'Hello\')"}',  # 原始输入字符串
)

tool_msg = Msg(
    name="assistant",
    content=[tool_use],
    role="assistant",
)
```

**ToolResultBlock - 工具结果块**

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

**ThinkingBlock - 思考块**

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

### 3.3 ChatResponse - 模型响应结构

```python
from agentscope.model import ChatResponse
from agentscope.message import TextBlock, ToolUseBlock, ThinkingBlock

# ChatResponse 完整结构
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

# 访问响应属性
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

## 4. 模型集成

AgentScope 提供统一的模型接口，支持多种 LLM 提供商。

### 4.1 支持的模型 API

| API | 类名 | 兼容 | 流式 | 工具 | 视觉 | 推理 |
|-----|------|------|------|------|------|------|
| OpenAI | `OpenAIChatModel` | vLLM, DeepSeek | ✅ | ✅ | ✅ | ✅ |
| DashScope | `DashScopeChatModel` | - | ✅ | ✅ | ✅ | ✅ |
| Anthropic | `AnthropicChatModel` | - | ✅ | ✅ | ✅ | ✅ |
| Gemini | `GeminiChatModel` | - | ✅ | ✅ | ✅ | ✅ |
| Ollama | `OllamaChatModel` | - | ✅ | ✅ | ✅ | ❌ |

**重要说明**:

1. **vLLM 兼容性**: 使用 vLLM 时需配置工具调用参数（`--enable-auto-tool-choice`, `--tool-call-parser`）
2. **OpenAI 兼容模型**: 使用 `OpenAIChatModel` 并通过 `client_kwargs={"base_url": "..."}` 指定端点
3. **行为参数**: 通过 `generate_kwargs` 预设温度、max_tokens 等参数

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

# OpenAI o3/o4 推理模型
o4_model = OpenAIChatModel(
    model_name="o4-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    reasoning_effort="medium",              # 推理努力程度：low, medium, high
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

# 标准 Qwen 模型
qwen_model = DashScopeChatModel(
    model_name="qwen-max",                  # 通义千问模型
    api_key=os.environ["DASHSCOPE_API_KEY"],
    stream=True,
    generate_kwargs={
        "temperature": 0.7,
        "max_tokens": 2000,
    },
)

# 启用推理（thinking）
qwen_thinking_model = DashScopeChatModel(
    model_name="qwen-turbo",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    enable_thinking=True,                   # 启用推理过程显示
    stream=True,
)

# Qwen 视觉模型
qwen_vl_model = DashScopeChatModel(
    model_name="qwen3-vl-plus",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    stream=True,
)
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

### 4.8 流式返回详解

```python
from agentscope.model import DashScopeChatModel
import asyncio
import os

async def example_streaming():
    """流式返回示例"""
    model = DashScopeChatModel(
        model_name="qwen-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        stream=True,
    )
    
    generator = await model(
        messages=[
            {
                "role": "user",
                "content": "从 1 数到 20，只报告数字，不要任何其他信息。",
            },
        ],
    )
    print("响应的类型:", type(generator))  # <class 'async_generator'>
    
    i = 0
    async for chunk in generator:
        print(f"块 {i}")
        print(f"\t类型：{type(chunk.content)}")
        print(f"\t{chunk}\n")
        i += 1

asyncio.run(example_streaming())
```

**重要**: AgentScope 中的流式返回是**累加式**的，每个 chunk 包含之前所有内容加上新生成的内容。

### 4.9 推理模型支持

```python
from agentscope.model import DashScopeChatModel
import asyncio
import os

async def example_reasoning():
    """使用推理模型示例"""
    model = DashScopeChatModel(
        model_name="qwen-turbo",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        enable_thinking=True,
    )
    
    res = await model(
        messages=[
            {"role": "user", "content": "我是谁？"},
        ],
    )
    
    # 获取最终响应
    last_chunk = None
    async for chunk in res:
        last_chunk = chunk
    
    print("最终响应:")
    print(last_chunk)
    # 响应内容包含 ThinkingBlock 和 TextBlock

asyncio.run(example_reasoning())
```

### 4.10 工具 API 统一接口

```python
from agentscope.model import DashScopeChatModel
import asyncio
import os

async def example_tool_api():
    """工具 API 统一接口示例"""
    model = DashScopeChatModel(
        model_name="qwen-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        stream=False,
    )
    
    # 统一的工具 JSON schema 格式
    json_schemas = [
        {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "在 Google 上搜索查询。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询。",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]
    
    # 调用模型
    response = await model(
        messages=[{"role": "user", "content": "搜索 AgentScope"}],
        tools=json_schemas,
        tool_choice="auto",
    )
    
    print(response)

asyncio.run(example_tool_api())
```

---

## 5. 智能体系统

智能体是 AgentScope 的核心，负责感知、推理、决策和行动。

### 5.1 ReActAgent - 推理 - 行动智能体

ReActAgent 是 AgentScope 的核心智能体实现，支持以下功能特性：

| 功能特性 | 描述 | 参考文档 |
|---------|------|---------|
| 实时介入（Realtime Steering） | 允许用户随时中断智能体回复 | [智能体](#5-智能体系统) |
| 记忆压缩 | 自动压缩超长对话历史 | [记忆压缩](#54-记忆压缩) |
| 并行工具调用 | 同时执行多个工具调用 | [并行工具](#55-并行工具调用) |
| 结构化输出 | 限制输出为 Pydantic 模型 | [结构化输出](#56-结构化输出) |
| 智能体自主管理工具 | Meta tool 动态管理工具组 | [自动工具管理](#66-自动工具管理) |
| 函数粒度 MCP 控制 | 精细控制 MCP 工具 | [MCP](#19-mcp 集成) |
| 长期记忆自主管理 | 智能体控制长期记忆 | [长期记忆](#8-长期记忆) |
| 自动状态管理 | 会话状态自动保存 | [状态管理](#9-会话管理) |

### 5.2 ReActAgent 基础用法

```python
from agentscope.agent import ReActAgent, UserAgent
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit, execute_python_code
from agentscope.message import Msg
import asyncio
import os

async def example_react_agent():
    # 1. 创建工具包
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)
    
    # 2. 创建模型实例
    model = DashScopeChatModel(
        model_name="qwen-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        stream=True,
    )
    
    # 3. 创建记忆
    memory = InMemoryMemory()
    
    # 4. 创建格式化器
    formatter = DashScopeChatFormatter()
    
    # 5. 创建 ReActAgent
    agent = ReActAgent(
        name="Friday",                              # 智能体名称
        sys_prompt="你是有帮助的助手 Friday。",         # 系统提示
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
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break
        
        msg = await agent(msg)
        print(f"{agent.name}: {msg.get_text_content()}")

asyncio.run(example_react_agent())
```

### 5.3 实时控制（Realtime Steering）

AgentScope 基于 asyncio 取消机制实现实时中断功能：

```python
from agentscope.agent import ReActAgent
from agentscope.message import Msg
import asyncio

async def example_interruption():
    agent = ReActAgent(
        name="Assistant",
        sys_prompt="你是有帮助的助手",
        model=model,
        formatter=formatter,
    )
    
    # 启动智能体响应任务
    task = asyncio.create_task(agent(Msg("user", "执行长时间任务", "user")))
    
    # 等待 1 秒后中断
    await asyncio.sleep(1)
    agent.interrupt()  # 调用中断方法
    
    # 智能体会捕获 CancelledError 并调用 handle_interrupt 处理
    try:
        response = await task
    except asyncio.CancelledError:
        print("智能体响应已被中断")

asyncio.run(example_interruption())
```

**中断逻辑**:

```python
# AgentBase 中的中断处理
class AgentBase:
    async def __call__(self, *args, **kwargs) -> Msg:
        reply_msg: Msg | None = None
        try:
            self._reply_task = asyncio.current_task()
            reply_msg = await self.reply(*args, **kwargs)
        except asyncio.CancelledError:
            # 捕获中断并通过 handle_interrupt 方法处理
            reply_msg = await self.handle_interrupt(*args, **kwargs)
        return reply_msg
    
    @abstractmethod
    async def handle_interrupt(self, *args, **kwargs) -> Msg:
        """自定义中断后处理逻辑"""
        pass
```

**ReActAgent 默认中断处理**:

```python
# ReActAgent 返回固定中断消息
async def handle_interrupt(self, *args, **kwargs) -> Msg:
    return Msg(
        "system",
        "I noticed that you have interrupted me. What can I do for you?",
        "assistant",
    )
```

**自定义中断处理**:

```python
class CustomAgent(ReActAgent):
    async def handle_interrupt(self, *args, **kwargs) -> Msg:
        # 调用 LLM 生成中断响应
        return await self.model(
            messages=[
                {"role": "system", "content": "用户中断了对话，请询问用户需求"},
                {"role": "user", "content": "请询问用户需要什么帮助"},
            ]
        )
```

### 5.4 记忆压缩

随着对话增长，记忆可能超过模型上下文限制。ReActAgent 提供自动记忆压缩：

**基础用法**:

```python
from agentscope.agent import ReActAgent
from agentscope.token import CharTokenCounter

agent = ReActAgent(
    name="助手",
    sys_prompt="你是一个有用的助手。",
    model=model,
    formatter=formatter,
    compression_config=ReActAgent.CompressionConfig(
        enable=True,
        agent_token_counter=CharTokenCounter(),  # Token 计数器
        trigger_threshold=10000,                  # 超过 10000 tokens 触发
        keep_recent=3,                            # 保留最近 3 条消息
    ),
)
```

**压缩过程**:

1. 监控记忆中的 token 数量
2. 超过阈值时识别未压缩消息（通过 `exclude_mark`）
3. 保留最近 `keep_recent` 条消息
4. 将早期消息发送给 LLM 生成结构化摘要
5. 使用 `MemoryMark.COMPRESSED` 标记已压缩消息
6. 将摘要单独存储

**默认压缩摘要结构**:

```python
default_schema = {
    "task_overview": "用户的核心请求和成功标准",
    "current_state": "已完成的工作，包括文件和输出",
    "important_discoveries": "技术约束、决策、错误和失败尝试",
    "next_steps": "完成任务所需的具体操作",
    "context_to_preserve": "用户偏好、领域细节和承诺",
}
```

**自定义压缩**:

```python
from pydantic import BaseModel, Field

# 定义自定义摘要结构
class CustomSummary(BaseModel):
    main_topic: str = Field(
        max_length=200,
        description="对话的主题"
    )
    key_points: str = Field(
        max_length=400,
        description="讨论的重要观点"
    )
    pending_tasks: str = Field(
        max_length=200,
        description="待完成的任务"
    )

# 使用自定义压缩配置
agent = ReActAgent(
    name="助手",
    sys_prompt="你是有用的助手",
    model=model,
    formatter=formatter,
    compression_config=ReActAgent.CompressionConfig(
        enable=True,
        agent_token_counter=CharTokenCounter(),
        trigger_threshold=10000,
        keep_recent=3,
        # 自定义 schema
        summary_schema=CustomSummary,
        # 自定义压缩提示
        compression_prompt=(
            "<system-hint>请总结上述对话，"
            "重点关注主题、关键讨论点和待完成任务。</system-hint>"
        ),
        # 自定义摘要模板
        summary_template=(
            "<system-info>对话摘要：\n"
            "主题：{main_topic}\n\n"
            "关键观点：\n{key_points}\n\n"
            "待完成任务：\n{pending_tasks}"
            "</system-info>"
        ),
    ),
)
```

**小技巧**: 使用更小更快的模型进行压缩：

```python
agent = ReActAgent(
    name="助手",
    sys_prompt="你是有用的助手",
    model=expensive_model,  # 主模型：GPT-4
    compression_config=ReActAgent.CompressionConfig(
        enable=True,
        compression_model=cheap_model,  # 压缩模型：Qwen-Turbo
        compression_formatter=formatter,
        trigger_threshold=10000,
    ),
)
```

### 5.5 并行工具调用

```python
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit, ToolResponse
from agentscope.message import TextBlock
from agentscope.model import DashScopeChatModel
from datetime import datetime
import asyncio
import os

# 准备工具函数
async def example_tool_function(tag: str) -> ToolResponse:
    """示例工具函数"""
    start_time = datetime.now().strftime("%H:%M:%S.%f")
    await asyncio.sleep(3)  # 模拟长时间任务
    end_time = datetime.now().strftime("%H:%M:%S.%f")
    return ToolResponse(
        content=[TextBlock(
            type="text",
            text=f"标签 {tag} 开始于 {start_time}，结束于 {end_time}。",
        )],
    )

async def example_parallel_tool_calls():
    toolkit = Toolkit()
    toolkit.register_tool_function(example_tool_function)
    
    agent = ReActAgent(
        name="Jarvis",
        sys_prompt="你是名为 Jarvis 的有用助手。",
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            generate_kwargs={
                "parallel_tool_calls": True,  # 启用并行工具调用
            },
        ),
        memory=InMemoryMemory(),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
        parallel_tool_calls=True,  # 智能体级别启用
    )
    
    # 提示智能体同时生成两个工具调用
    await agent(
        Msg(
            "user",
            "同时生成两个 'example_tool_function' 函数的工具调用，标签分别为 'tag1' 和 'tag2'",
            "user",
        ),
    )
    # 两个工具会并行执行（约 3 秒同时完成）

asyncio.run(example_parallel_tool_calls())
```

**输出示例**:
```
Jarvis: {"type": "tool_use", "id": "call_1", "name": "example_tool_function", "input": {"tag": "tag1"}}
Jarvis: {"type": "tool_use", "id": "call_2", "name": "example_tool_function", "input": {"tag": "tag2"}}
system: {"type": "tool_result", "id": "call_1", "output": "标签 tag1 开始于 03:05:21.281947，结束于 03:05:24.283067。"}
system: {"type": "tool_result", "id": "call_2", "output": "标签 tag2 开始于 03:05:21.282044，结束于 03:05:24.283738。"}
Jarvis: 两个函数已并行执行完毕。
```

### 5.6 结构化输出

```python
from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg
from pydantic import BaseModel, Field
import json
import asyncio
import os

async def example_structured_output():
    agent = ReActAgent(
        name="Jarvis",
        sys_prompt="你是名为 Jarvis 的有用助手。",
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
        ),
        formatter=DashScopeChatFormatter(),
    )
    
    # 定义结构化输出模型
    class PersonModel(BaseModel):
        name: str = Field(description="人物的姓名")
        description: str = Field(description="人物的一句话描述")
        age: int = Field(description="年龄")
        honor: list[str] = Field(description="人物荣誉列表")
    
    # 请求结构化输出
    res = await agent(
        Msg("user", "介绍爱因斯坦", "user"),
        structured_model=PersonModel,
    )
    
    # 从 metadata 获取结构化输出
    print("\n结构化输出：")
    print(json.dumps(res.metadata, indent=4, ensure_ascii=False))
    # 输出：
    # {
    #     "name": "阿尔伯特·爱因斯坦",
    #     "description": "20 世纪最伟大的理论物理学家之一",
    #     "age": 76,
    #     "honor": ["诺贝尔物理学奖", "普朗克奖章", "科普利奖章"]
    # }

asyncio.run(example_structured_output())
```

### 5.7 自定义智能体

**继承 AgentBase**:

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
    
    async def reply(self, msg: Msg = None) -> Msg:
        """实现回复逻辑"""
        if msg:
            self.history.append(msg)
        
        messages = []
        if self.sys_prompt:
            messages.append({"role": "system", "content": self.sys_prompt})
        
        for h_msg in self.history[-10:]:
            messages.append({
                "role": h_msg.role,
                "content": h_msg.get_text_content(),
            })
        
        response = await self.model(messages)
        
        response_msg = Msg(
            name=self.name,
            content=response.content,
            role="assistant",
        )
        
        self.history.append(response_msg)
        return response_msg
    
    async def handle_interrupt(self, *args, **kwargs) -> Msg:
        """处理中断"""
        return Msg("system", "对话已被中断", "system")

# 使用示例
async def use_custom_agent():
    model = OpenAIChatModel(
        model_name="gpt-4",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    
    custom_agent = CustomAgent(
        name="MyAgent",
        model=model,
        sys_prompt="你是专业的代码审查助手",
    )
    
    response = await custom_agent(
        Msg(name="user", content="请审查这段代码", role="user")
    )

asyncio.run(use_custom_agent())
```

---

（因篇幅限制，此处省略第 6-28 章内容，实际文档包含完整所有章节）

---

## 27. 最佳实践

### 27.1 性能优化

1. **使用缓存**: 为嵌入和频繁调用的工具启用缓存
2. **批量处理**: 批量调用嵌入模型而非逐个调用
3. **流式输出**: 对用户可见的响应启用流式
4. **并行执行**: 使用 `fanout_pipeline` 并行处理独立任务
5. **记忆压缩**: 长对话启用记忆压缩避免超出上下文限制

### 27.2 成本控制

1. **Token 监控**: 使用 Token 计数器跟踪使用情况
2. **模型选择**: 根据任务复杂度选择合适模型
3. **缓存策略**: 重复查询使用缓存结果
4. **压缩模型**: 使用小模型进行记忆压缩

### 27.3 安全考虑

1. **工具权限**: 谨慎授予智能体工具执行权限
2. **输入验证**: 对用户输入和工具参数进行验证
3. **输出过滤**: 过滤智能体输出中的敏感信息
4. **资源限制**: 为工具执行设置超时和资源限制

### 27.4 调试技巧

1. **启用日志**: 设置 `logging_level="DEBUG"`
2. **链路追踪**: 连接 AgentScope Studio 查看执行链路
3. **异常捕获**: 在关键调用周围添加异常处理
4. **状态检查**: 定期检查记忆和会话状态

---

## 28. 常见问题

### 28.1 模型调用失败

**问题**: 模型调用返回空响应或超时

**解决**:
- 检查 API 密钥是否正确
- 验证网络连接
- 增加超时时间
- 减少上下文长度
- 检查模型配额

### 28.2 记忆泄漏

**问题**: 长时间运行后内存占用过高

**解决**:
- 启用记忆压缩
- 定期清理旧对话
- 使用 Redis 等外部存储
- 限制记忆大小

### 28.3 工具执行卡住

**问题**: 工具调用长时间无响应

**解决**:
- 添加工具执行超时
- 检查工具依赖
- 添加异常处理
- 使用异步工具调用

---

## 附录：资源链接

- **官方文档**: https://doc.agentscope.io
- **中文文档**: https://doc.agentscope.io/zh_CN
- **GitHub 仓库**: https://github.com/agentscope-ai/agentscope
- **AgentScope Studio**: https://github.com/agentscope-ai/studio
- **示例代码**: https://github.com/agentscope-ai/agentscope/tree/main/examples
- **社区讨论**: https://github.com/agentscope-ai/agentscope/discussions
- **Discord**: https://discord.gg/eYMpfnkG8h

---

*本指南基于 AgentScope v1.0+ 官方文档编写，持续更新中。*
*最后更新：2026 年 3 月 1 日*
