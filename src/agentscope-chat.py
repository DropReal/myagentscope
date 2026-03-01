from agentscope.model import OpenAIChatModel
from agentscope.model import DashScopeChatModel
import os
import asyncio
from dotenv import load_dotenv
from agentscope.message import Msg
from agentscope.model import ChatResponse

# openai 模型
# 标准 OpenAI
# openai_model = OpenAIChatModel(
#     model_name="gpt-4o",  # 模型名称
#     api_key=os.environ["OPENAI_API_KEY"],  # API 密钥
#     stream=True,  # 启用流式输出
#     generate_kwargs={  # 生成参数
#         "temperature": 0.7,
#         "max_tokens": 2000,
#         "top_p": 0.9,
#     },
# )

# OpenAI 兼容的本地模型（vLLM、DeepSeek 等）
# vllm_model = OpenAIChatModel(
#     model_name="llama-3-70b",
#     api_key="not-needed",  # 本地模型通常不需要密钥
#     client_kwargs={
#         "base_url": "http://localhost:8000/v1",  # vLLM 端点
#     },
#     stream=True,
# )
load_dotenv()  # 加载 .env 文件中的环境变量


# 阿里云 DashScope（通义千问）

qwen_model = DashScopeChatModel(
    model_name="qwen3-max",  # 通义千问模型
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
    response = await qwen_model(messages)  # 调用模型
    # 如果是异步生成器，则逐个获取内容
    async for chunk in response:
        print(chunk.content)  # 打印每个流式输出的片段

async def example_model_usage():
    model = DashScopeChatModel(
        model_name="qwen3-max",  # 通义千问模型
        api_key=os.environ["DASHSCOPE_API_KEY"],
        stream=False,
        generate_kwargs={
            "temperature": 0.7,
            "max_tokens": 2000,
        },
    )
    messages = [{"role": "system", "content": "你是有帮助的助手"},
            {"role": "user", "content": "你是谁"}]
    response = await model(messages)
    print(f"响应 ID: {response.id}")
    print(f"创建时间：{response.created_at}")
    print(f"内容：{response.content}")
    print(f"Token 使用：{response.usage}")


async def example_model_stream_usage():
    # DashScopeChatModel好像不支持流式输出是使用增量方式，每次都是累积方式，incremental_output参数不生效
    model = DashScopeChatModel(
        model_name="qwen3-max",  # 通义千问模型
        api_key=os.environ["DASHSCOPE_API_KEY"],
        stream=True,
        incremental_output=True,
        generate_kwargs={
            "temperature": 0.7,
            "max_tokens": 2000,
        },
    )
    messages = [{"role": "system", "content": "你是有帮助的助手"},
                {"role": "user", "content": "讲一个睡前故事"}]
    stream_response = await model(
        messages=[{"role": "user", "content": "讲个故事"}],
    )
    async for chunk in stream_response:
        print(chunk.content[0]["text"])

async def example_siliconflow_usage():
    model = OpenAIChatModel(
        model_name=os.environ["siliconflow_model"],
        api_key=os.environ["siliconflow_api_key"],  # 本地模型通常不需要密钥
        client_kwargs={
            "base_url": os.environ["siliconflow_base_http_api_url"],  # vLLM 端点
        },
        stream=True,
    )
    messages = [{"role": "system", "content": "你是有帮助的助手"},
            {"role": "user", "content": "你是谁"}]
    response = await model(messages)
    async for chunk in response:
        print(chunk.content[0])
# 使用示例example_qwen
if __name__ == "__main__":
    asyncio.run(example_siliconflow_usage())




