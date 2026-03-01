from agentscope.tool import (
    execute_python_code,      # 执行 Python 代码
    execute_shell_command,    # 执行 Shell 命令
    view_text_file,          # 查看文本文件
    write_text_file,         # 写入（覆盖）文本文件
    insert_text_file,          # 插入文本
    ToolResponse,            # 工具响应
)
import asyncio
import os
from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel
from agentscope.tool import (
    Toolkit,
    execute_shell_command,
    execute_python_code,
    view_text_file,
)

# 内置工具
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

# 自定义工具
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
    print("***********触发工具get_weather：查询天气信息：", location, date)
    WeatherParams(location=location, date=date)
    weather_info = f"{location} {date}的天气：晴，25°C"
    return ToolResponse(
        content=[TextBlock(type="text", text=weather_info)],
    )


# 注册工具
toolkit = Toolkit()
toolkit.register_tool_function(search_web)
toolkit.register_tool_function(get_weather)

# 通过ReActAgent来实现一个查询天气
async def main() -> None:
    agent = ReActAgent(
        name="Friday",
        sys_prompt="You are a helpful assistant named Friday.",
        model=DashScopeChatModel(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            model_name="qwen-max",
            enable_thinking=False,
            stream=True,
        ),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )
    user = UserAgent("User")
    msg = None
    while True:
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break
        msg = await agent(msg)

asyncio.run(main())