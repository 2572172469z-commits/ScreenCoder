from utils import encode_image, Doubao, Qwen, GPT, Gemini
from PIL import Image
import bs4
from threading import Thread
import time

# 用户自定义说明
user_instruction = {
    "sidebar": "",
    "header": "",
    "navigation": "",
    "main content": ""
}

# 针对每个区域的 prompt
PROMPT_DICT = {
    "sidebar": f"""这是一个container的截图。这是用户给的额外要求：{user_instruction["sidebar"]}请填写一段完整的HTML和tail-wind CSS代码以准确再现给定的容器。请注意所有组块的排版、图标样式、大小、文字信息需要在用户额外条件的基础上与原始截图基本保持一致。以下是供填写的代码：

    <div>
    your code here
    </div>

    只需返回<div>和</div>标签内的代码""",
    "header": f"""这是一个container的截图。这是用户给的额外要求：{user_instruction["header"]}请填写一段完整的HTML和tail-wind CSS代码以准确再现给定的容器。请注意所有组块在boundary box中的相对位置、排版、文字信息、颜色需要在用户额外条件的基础上与原始截图基本保持一致。以下是供填写的代码：

    <div>
    your code here
    </div>

    只需返回<div>和</div>标签内的代码""",
    "navigation": f"""这是一个container的截图。这是用户给的额外要求：{user_instruction["navigation"]}请填写一段完整的HTML和tail-wind CSS代码以准确再现给定的容器。请注意所有组块的在boundary box中的相对位置、文字排版、颜色需要在用户额外条件的基础上与原始截图基本保持一致。请你直接使用原始截图中一致的图标。以下是供填写的代码：

    <div>
    your code here
    </div>

    只需返回<div>和</div>标签内的代码""",
    "main content": f"""这是一个container的截图。这是用户给的额外要求：{user_instruction["main content"]}请填写一段完整的HTML和tail-wind CSS代码以准确再现给定的容器。请使用相同大小的纯灰色图像块替换原始截图中的图像，不需要识别图像中的文字信息。请注意所有组块在boundary box中的相对位置、排版、文字信息、颜色需要在用户额外条件的基础上与原始截图基本保持一致。以下是供填写的代码：

    <div>
    your code here
    </div>

    只需返回<div>和</div>标签内的代码""",
    # 新增通用类型
    "region": """这是一个区域的截图。请填写一段完整的HTML和tail-wind CSS代码以准确再现给定的区域。""",
    "placeholder": """这是一个占位符的截图。请填写一段完整的HTML和tail-wind CSS代码以准确再现该占位符。"""
}

def get_prompt_by_type(type_name):
    if type_name in PROMPT_DICT:
        return PROMPT_DICT[type_name]
    elif type_name.startswith("region"):
        return PROMPT_DICT["region"]
    elif type_name.startswith("placeholder"):
        return PROMPT_DICT["placeholder"]
    else:
        return PROMPT_DICT["main content"]

def generate_code(bbox_tree, img_path, bot):
    img = Image.open(img_path)
    code_dict = {}
    def _generate_code(node):
        if node["children"] == []:
            bbox = node["bbox"]
            cropped_img = img.crop(bbox)
            prompt = get_prompt_by_type(node["type"])
            try:
                code = bot.ask(prompt, encode_image(cropped_img))
                code_dict[node["id"]] = code
            except Exception as e:
                print(f"Error generating code for {node.get('type', 'unknown')}: {str(e)}")
                code_dict[node["id"]] = f"<!-- Error: {str(e)} -->"
        else:
            for child in node["children"]:
                _generate_code(child)
    _generate_code(bbox_tree)
    return code_dict

def generate_code_parallel(bbox_tree, img_path, bot):
    code_dict = {}
    t_list = []
    def _generate_code_with_retry(node, max_retries=3, retry_delay=2):
        try:
            with Image.open(img_path) as img:
                bbox = node.get('bbox', 'div')
                cropped_img = img.crop(bbox)
                prompt = get_prompt_by_type(node["type"])
                for attempt in range(max_retries):
                    try:
                        code = bot.ask(prompt, encode_image(cropped_img))
                        code_dict[node["id"]] = code
                        return
                    except Exception as e:
                        if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                            print(f"Rate limit hit, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            print(f"Error generating code for node {node['id']}: {str(e)}")
                            code_dict[node["id"]] = f"<!-- Error: {str(e)} -->"
                            return
        except Exception as e:
            print(f"Error processing image for node {node['id']}: {str(e)}")
            code_dict[node["id"]] = f"<!-- Error: {str(e)} -->"
    def _generate_code(node):
        if not node.get("children"):
            t = Thread(target=_generate_code_with_retry, args=(node,))
            t.start()
            t_list.append(t)
        else:
            for child in node["children"]:
                _generate_code(child)
    _generate_code(bbox_tree)
    for t in t_list:
        t.join()
    return code_dict

def generate_html(bbox_tree, output_file="output.html", img_path="data/test1.png"):
    html_template_start = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Bounding Boxes Layout</title>
        <style>
            body, html {
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
            }
            .container { 
                position: relative;
                width: 100%;
                height: 100%;
                box-sizing: border-box;
            }
            .box {
                position: absolute;
                box-sizing: border-box;
                overflow: hidden;
            }
            .box > .container {
                display: grid;
                width: 100%;
                height: 100%;
            }
        </style>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container">
    """
    html_template_end = """
        </div>
    </body>
    </html>
    """
    def process_bbox(node, parent_width, parent_height, parent_left, parent_top, img):
        bbox = node['bbox']
        children = node.get('children', [])
        id = node['id']
        left = (bbox[0] - parent_left) / parent_width * 100
        top = (bbox[1] - parent_top) / parent_height * 100
        width = (bbox[2] - bbox[0]) / parent_width * 100
        height = (bbox[3] - bbox[1]) / parent_height * 100
        html = f'''
            <div id="{id}" class="box" style="left: {left}%; top: {top}%; width: {width}%; height: {height}%;">
        '''
        if children:
            html += '''
                <div class="container">
            '''
            current_width = bbox[2] - bbox[0]
            current_height = bbox[3] - bbox[1]
            for child in children:
                html += process_bbox(child, current_width, current_height, bbox[0], bbox[1], img)
            html += '''
                </div>
            '''
        html += '''
            </div>
        '''
        return html
    root_bbox = bbox_tree['bbox']
    root_children = bbox_tree.get('children', [])
    root_width = root_bbox[2]
    root_height = root_bbox[3]
    root_x = root_bbox[0]
    root_y = root_bbox[1]
    html_content = html_template_start
    for child in root_children:
        html_content += process_bbox(child, root_width, root_height, root_x, root_y, img)
    html_content += html_template_end
    soup = bs4.BeautifulSoup(html_content, 'html.parser')
    html_content = soup.prettify()
    with open(output_file, 'w') as f:
        f.write(html_content)

def code_substitution(html_file, code_dict):
    with open(html_file, "r", encoding='utf-8') as f:
        html = f.read()
    soup = bs4.BeautifulSoup(html, 'html.parser')
    for id, code in code_dict.items():
        code = code.replace("```html", "").replace("```", "")
        div = soup.find(id=id)
        if div:
            div.append(bs4.BeautifulSoup(code, 'html.parser'))
    with open(html_file, "w", encoding='utf-8') as f:
        f.write(soup.prettify())

if __name__ == "__main__":
    import json
    import time
    from PIL import Image
    # 读取 bboxes
    boxes_data = json.load(open("data/tmp/test1_bboxes.json"))
    img_path = "data/input/test1.png"
    with Image.open(img_path) as img:
        width, height = img.size
    root = {
        "bbox": [0, 0, width, height],
        "children": []
    }
    # regions
    for region in boxes_data.get('regions', []):
        x1 = int(region['x'] * width)
        y1 = int(region['y'] * height)
        x2 = int((region['x'] + region['w']) * width)
        y2 = int((region['y'] + region['h']) * height)
        child = {
            "bbox": [x1, y1, x2, y2],
            "children": [],
            "type": f"region_{region['id']}"
        }
        root["children"].append(child)
    # placeholders
    for ph in boxes_data.get('placeholders', []):
        x1 = int(ph['x'] * width)
        y1 = int(ph['y'] * height)
        x2 = int((ph['x'] + ph['w']) * width)
        y2 = int((ph['y'] + ph['h']) * height)
        child = {
            "bbox": [x1, y1, x2, y2],
            "children": [],
            "type": f"placeholder_{ph['id']}"
        }
        root["children"].append(child)
    def assign_id(node, id):
        node["id"] = id
        for child in node.get("children", []):
            id = assign_id(child, id+1)
        return id
    assign_id(root, 0)
    generate_html(root, 'data/tmp/test1_layout.html')
    bot = Doubao("doubao_api.txt", model = "doubao-1.5-thinking-vision-pro-250428")
    # bot = Qwen("qwen_api.txt", model="qwen2.5-vl-72b-instruct")
    # bot = GPT("gpt_api.txt", model="gpt-4o")
    # bot = Gemini("gemini_api.txt", model="gemini-1.5-flash-latest")
    code_dict = generate_code_parallel(root, img_path, bot)
    code_substitution('data/tmp/test1_layout.html', code_dict)
    # html_refinement('data/tmp/test1_layout.html', 'data/tmp/test1_layout_refined.html', img_path, bot)