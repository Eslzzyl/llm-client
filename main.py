#!/usr/bin/env python3
"""
LLM客户端使用示例
"""

import os

from llm_client import LLMClient, Message


def example_simple_text(model: str):
    """简单文本聊天示例"""
    print("=== 简单文本聊天示例 ===")

    # 创建客户端（需要设置环境变量 OPENAI_API_KEY 和 OPENAI_BASE_URL）
    with LLMClient(model=model) as client:
        # 简单聊天
        response = client.simple_chat(
            text="你好，请介绍一下自己", system_prompt="你是一个友善的AI助手"
        )

        print(f"回复: {response.content}")
        print(
            f"使用统计: 输入 {response.input_tokens} tokens, 输出 {response.output_tokens} tokens"
        )


def example_multimodal(model: str):
    """多模态聊天示例（文本+图片）"""
    print("\n=== 多模态聊天示例 ===")

    # 注意：这个示例需要实际的图片文件
    image_path = "test_image.jpg"  # 替换为实际的图片路径

    if os.path.exists(image_path):
        with LLMClient(model=model) as client:
            response = client.simple_chat(
                text="这张图片里有什么？", images=[image_path]
            )

            print(f"回复: {response.content}")
            print(
                f"使用统计: 输入 {response.input_tokens} tokens, 输出 {response.output_tokens} tokens"
            )
    else:
        print(f"图片文件 {image_path} 不存在，跳过多模态示例")


def example_stream(model: str):
    """流式聊天示例"""
    print("\n=== 流式聊天示例 ===")

    with LLMClient(model=model) as client:
        print("AI回复（流式）: ", end="", flush=True)

        total_usage = None
        for chunk in client.simple_chat_stream(
            text="请用50字左右简单介绍一下Python编程语言",
            system_prompt="你是一个编程专家",
        ):
            print(chunk.content, end="", flush=True)

            # 保存最后的使用统计（通常在最后一个chunk中）
            if chunk.usage:
                total_usage = chunk.usage

        print()  # 换行
        if total_usage:
            print(
                f"使用统计: 输入 {total_usage.prompt_tokens} tokens, 输出 {total_usage.completion_tokens} tokens"
            )


def example_conversation(model: str):
    """对话示例"""
    print("\n=== 对话示例 ===")

    with LLMClient(model=model) as client:
        # 构建对话历史
        messages = [
            Message.user_text("我想学习Python，应该从哪里开始？"),
        ]

        # 第一轮对话
        response1 = client.chat(
            messages=messages, system_prompt="你是一个编程导师，善于引导初学者学习"
        )

        print(f"用户: {messages[0].content}")
        print(f"AI: {response1.content}")

        # 添加AI回复到对话历史
        messages.append(Message.assistant(response1.content))

        # 继续对话
        messages.append(Message.user_text("那么安装Python需要注意什么？"))

        response2 = client.chat(messages=messages)

        print(f"用户: {messages[-1].content}")
        print(f"AI: {response2.content}")

        print(
            f"第二轮使用统计: 输入 {response2.input_tokens} tokens, 输出 {response2.output_tokens} tokens"
        )


def main():
    os.environ["OPENAI_API_KEY"] = "eslzzyl"
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:8080/v1"
    """主函数"""
    print("LLM客户端示例")
    print("请确保已设置环境变量:")
    print("- OPENAI_API_KEY: 你的API密钥")
    print("- OPENAI_BASE_URL: API基础URL（可选，默认为OpenAI官方API）")
    print()

    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  环境变量 OPENAI_API_KEY 未设置，示例将无法运行")
        print("请设置环境变量后重试")
        return

    try:
        # 运行各种示例
        MODEL_NAME = "model"
        example_simple_text(MODEL_NAME)
        example_multimodal(MODEL_NAME)
        example_stream(MODEL_NAME)
        example_conversation(MODEL_NAME)

    except Exception as e:
        print(f"❌ 示例运行出错: {e}")
        print("请检查API密钥和网络连接")


if __name__ == "__main__":
    main()
