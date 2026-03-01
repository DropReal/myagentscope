from agentscope.message import AudioBlock
from agentscope.message import ImageBlock, URLSource, Base64Source
from agentscope.message import Msg
from agentscope.message import TextBlock
from agentscope.message import ToolUseBlock

# 简单文本消息
text_msg = Msg(
    name="user",  # 发送者名称
    content="你好，请帮我分析这段代码",  # 消息内容
    role="user",  # 角色：user, assistant, system
)

# 查看消息
print(text_msg.name)  # "user"
print(text_msg.content)  # "你好，请帮我分析这段代码"
print(text_msg.role)  # "user"
print(text_msg.timestamp)  # 消息时间戳
print(text_msg.id)  # 消息唯一 ID

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

print(msg.content[0].get("text"))

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

# 音频消息
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

# 工具使用块
tool_use = ToolUseBlock(
    type="tool_use",
    id="call_abc123",  # 工具调用 ID
    name="execute_python_code",  # 工具名称
    input={"code": "print('Hello')"},  # 工具参数
)

tool_msg = Msg(
    name="assistant",
    content=[tool_use],
    role="assistant",
)

# 工具结果块
from agentscope.message import ToolResultBlock

tool_result = ToolResultBlock(
    type="tool_result",
    id="call_abc123",  # 对应工具调用的 ID
    name="execute_python_code",  # 工具名称
    output=[TextBlock(type="text", text="Hello")],  # 工具输出
)

result_msg = Msg(
    name="system",
    content=[tool_result],
    role="system",
)

# 操作消息
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
# json_str = msg.to_json()
# msg_from_json = Msg.from_json(json_str)

# 消息历史
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