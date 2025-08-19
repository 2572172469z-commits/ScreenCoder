import os


def check_api_file(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            print(f"文件内容: '{content}'")
            print(f"长度: {len(content)} 字符")

            if content.startswith("sk-") and len(content) == 40:
                print("✅ 格式正确：OpenAI 兼容密钥")
            elif len(content) == 36 and all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in content):
                print("✅ 格式正确：UUID 原生密钥")
            else:
                print("❌ 格式错误：请检查密钥格式")

    except FileNotFoundError:
        print(f"❌ 文件不存在：{file_path}")


# 替换为您的实际文件路径
check_api_file("D:\\app\\ScreenCoder-main\\doubao_api.txt")