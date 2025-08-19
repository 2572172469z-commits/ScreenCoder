import base64
import os
import json
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime._exceptions import ArkAPIError

# 1. 设置 API 密钥
api_key = "a9d88314-6dbd-4ded-9f36-5fc19833a3b8"

# 2. 创建客户端
try:
    print("正在创建火山引擎客户端...")
    client = Ark(api_key=api_key)
    print("客户端创建成功")
except Exception as e:
    print(f"创建客户端失败: {str(e)}")
    exit(1)

# 3. 准备测试图像
image_path = r"D:\app\ScreenCoder-main\ScreenCoder-main\data\input\test1.png"
print(f"图像路径: {image_path}")

if not os.path.exists(image_path):
    print(f"错误: 图像文件不存在 - {image_path}")
    exit(1)


def encode_image(image_path):
    """将图像文件编码为base64字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"图像编码失败: {str(e)}")
        exit(1)


try:
    base64_image = encode_image(image_path)
    print(f"成功编码图像 ({len(base64_image)} 字符)")
except Exception as e:
    print(f"图像处理失败: {str(e)}")
    exit(1)

# 4. 构建消息
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            },
            {"type": "text", "text": "请分析这张图片中的UI组件"}
        ]
    }
]

# 5. 发送请求
try:
    print("正在发送请求到火山引擎API...")
    print(f"使用模型: doubao-seed-1-6-250615")

    response = client.chat.completions.create(
        model="doubao-seed-1-6-250615",
        messages=messages,
        max_tokens=2000,
        timeout=120
    )

    # 6. 处理响应 - 更健壮的方式
    print("\nAPI 响应成功！")

    # 检查响应对象的基本属性
    if hasattr(response, 'model'):
        print(f"模型: {response.model}")

    # 尝试获取请求ID（不同SDK版本可能有不同属性名）
    request_id = None
    if hasattr(response, 'request_id'):
        request_id = response.request_id
    elif hasattr(response, 'id'):
        request_id = response.id
    if request_id:
        print(f"请求ID: {request_id}")

    # 打印使用情况（如果可用）
    if hasattr(response, 'usage'):
        usage = response.usage
        usage_info = ""
        if hasattr(usage, 'prompt_tokens'):
            usage_info += f"输入token={usage.prompt_tokens}, "
        if hasattr(usage, 'completion_tokens'):
            usage_info += f"输出token={usage.completion_tokens}, "
        if hasattr(usage, 'total_tokens'):
            usage_info += f"总token={usage.total_tokens}"

        if usage_info:
            print(f"使用情况: {usage_info}")

    # 提取并打印内容
    if hasattr(response, 'choices') and response.choices:
        first_choice = response.choices[0]

        if hasattr(first_choice, 'message') and hasattr(first_choice.message, 'content'):
            content = first_choice.message.content
            print("\n响应内容:")
            print(content)

            # 尝试解析JSON内容（如果响应是JSON格式）
            try:
                parsed_content = json.loads(content)
                print("\n解析后的JSON内容:")
                print(json.dumps(parsed_content, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                # 如果不是JSON格式，继续使用原始文本
                pass
        else:
            print("响应中没有找到内容")
    else:
        print("响应中没有choices")

    # 保存响应到文件（用于调试）
    with open("api_response.txt", "w", encoding="utf-8") as f:
        f.write(f"模型: {getattr(response, 'model', '未知')}\n")
        f.write(f"请求ID: {request_id or '未知'}\n")
        f.write(f"响应内容:\n{content}\n")
    print("\n完整响应已保存到 api_response.txt")

except ArkAPIError as api_error:
    print(f"\n火山引擎API错误: {api_error}")
    print(f"错误代码: {api_error.code}")
    print(f"错误类型: {api_error.type}")
    print(f"错误消息: {api_error.message}")
    if hasattr(api_error, 'request_id'):
        print(f"请求ID: {api_error.request_id}")

except Exception as e:
    print(f"\nAPI 请求失败: {str(e)}")

    # 尝试获取更多错误详情
    if hasattr(e, 'response'):
        print(f"状态码: {e.response.status_code}")
        try:
            error_details = e.response.json()
            print(f"错误详情: {error_details}")
        except:
            print(f"响应文本: {e.response.text}")