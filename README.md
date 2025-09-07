# LLM Client

一个基于 httpx 的简洁、强类型的 OpenAI 兼容 API 客户端，支持文本和多模态（文本+图片）聊天。

## 特性

- 🚀 **简洁的API设计**：简单的输入输出接口，易于使用
- 🖼️ **多模态支持**：支持文本和图片输入，图片可以从路径读取或直接传入bytes
- 🔑 **灵活配置**：通过参数或环境变量配置API密钥和基础URL
- 📡 **流式和非流式**：支持实时流式响应和标准响应
- 📊 **详细统计**：返回输入/输出token统计信息
- 🎯 **强类型**：使用Pydantic模型提供完整的IDE补全和类型检查
- 🔌 **OpenAI兼容**：支持所有OpenAI兼容的API服务

## 安装

```bash
uv add git+https://github.com/Eslzzyl/llm-client
```

## 快速开始

### 1. 设置环境变量

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"  # 可选，默认为OpenAI官方API

# Linux/Mac
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选
```

### 2. 基本使用

```python
from llm_client import LLMClient

# 创建客户端
with LLMClient() as client:
    # 简单文本聊天
    response = client.simple_chat(
        text="你好，请介绍一下自己",
        system_prompt="你是一个友善的AI助手"
    )
    
    print(f"回复: {response.content}")
    print(f"Token使用: 输入 {response.input_tokens}, 输出 {response.output_tokens}")
```

### 3. 多模态聊天（文本+图片）

```python
# 支持图片路径
response = client.simple_chat(
    text="这张图片里有什么？",
    images=["path/to/image.jpg"]
)

# 支持图片bytes
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

response = client.simple_chat(
    text="分析这张图片",
    images=[image_bytes]
)
```

### 4. 流式响应

```python
# 流式聊天
for chunk in client.simple_chat_stream(
    text="请写一首关于春天的诗",
    system_prompt="你是一个诗人"
):
    print(chunk.content, end="", flush=True)
```

### 5. 对话历史

```python
from llm_client import Message

# 构建对话
messages = [
    Message.user_text("我想学习Python"),
    Message.assistant("Python是一门很棒的编程语言..."),
    Message.user_text("那我应该从哪里开始？")
]

response = client.chat(
    messages=messages,
    system_prompt="你是一个编程导师"
)
```

## API 参考

### LLMClient

主要的客户端类，提供与OpenAI兼容API的接口。

```python
client = LLMClient(
    api_key="your-key",          # API密钥，可选（从环境变量读取）
    base_url="https://...",      # API基础URL，可选
    model="gpt-3.5-turbo",      # 默认模型
    timeout=60.0                # 请求超时（秒）
)
```

#### 方法

- `simple_chat()` - 简单聊天接口
- `simple_chat_stream()` - 简单流式聊天接口  
- `chat()` - 完整聊天接口
- `close()` - 关闭客户端

### 消息类型

#### Message

表示聊天消息的基础类：

```python
# 创建各种类型的消息
Message.user_text("用户文本")
Message.user_multimodal("文本", images=["path.jpg"])
Message.assistant("助手回复")
Message.system("系统提示")
```

#### 图片处理

```python
# 从路径创建图片
ImageContent.from_path("image.jpg")

# 从bytes创建图片
ImageContent.from_bytes(image_bytes, "image/jpeg")
```

### 响应类型

#### ChatCompletion

非流式响应的结果：

```python
response = client.simple_chat("你好")

# 访问回复内容
print(response.content)

# 访问统计信息
print(response.input_tokens)    # 输入token数
print(response.output_tokens)   # 输出token数  
print(response.total_tokens)    # 总token数

# 原始响应
print(response.choices[0].message.content)
print(response.usage.prompt_tokens)
```

#### StreamChunk

流式响应的每个块：

```python
for chunk in client.simple_chat_stream("你好"):
    print(chunk.content)        # 增量内容
    print(chunk.is_finished)    # 是否完成
    
    if chunk.usage:             # 最后一个chunk包含统计
        print(chunk.usage.total_tokens)
```

## 高级用法

### 自定义参数

```python
response = client.chat(
    messages=[Message.user_text("写一个故事")],
    temperature=0.8,      # 控制随机性
    max_tokens=1000,      # 限制输出长度
    top_p=0.9,           # nucleus sampling
    frequency_penalty=0.5 # 频率惩罚
)
```

### 错误处理

```python
try:
    response = client.simple_chat("你好")
except httpx.HTTPStatusError as e:
    print(f"HTTP错误: {e.response.status_code}")
except httpx.TimeoutException:
    print("请求超时")
except ValueError as e:
    print(f"配置错误: {e}")
```

### 不同的API提供商

```python
# OpenAI
client = LLMClient(
    base_url="https://api.openai.com/v1",
    model="gpt-4"
)

# 其他兼容服务
client = LLMClient(
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat"
)
```

## 依赖

- [httpx](https://www.python-httpx.org/) - 现代HTTP客户端
- [pydantic](https://pydantic-docs.helpmanual.io/) - 数据验证和类型提示
