from os.path import join as pjoin
import cv2
import os
import numpy as np
import multiprocessing

print("当前工作目录:", os.getcwd())
# 检查可能的路径
possible_paths = [
    '../data/input/test1.png',  # 上一级的 data 目录
    '../../data/input/test1.png',  # 上两级的 data 目录
    'data/input/test1.png',  # 当前目录下的 data
]

for path in possible_paths:
    if os.path.exists(path):
        print(f"找到文件: {path}")
        break
else:
    print("在所有可能路径中都没找到 test1.png")
def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    if org is None:
        print(f"严重错误：无法读取图片")
        print(f"请求路径: {input_path_img}")
        print(f"绝对路径: {os.path.abspath(input_path_img)}")
        print(f"文件存在: {os.path.exists(input_path_img)}")
        if os.path.exists(input_path_img):
            print("文件存在但OpenCV无法读取，可能文件已损坏或格式不支持")
        return None
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))
def color_tips():
    color_map = {'Text': (0, 0, 255), 'Compo': (0, 255, 0), 'Block': (0, 255, 255), 'Text Content': (255, 0, 255)}
    board = np.zeros((200, 200, 3), dtype=np.uint8)

    board[:50, :, :] = (0, 0, 255)
    board[50:100, :, :] = (0, 255, 0)
    board[100:150, :, :] = (255, 0, 255)
    board[150:200, :, :] = (0, 255, 255)
    cv2.putText(board, 'Text', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, 'Non-text Compo', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, "Compo's Text Content", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, "Block", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    #cv2.imshow('colors', board)


if __name__ == '__main__':
    # Set multiprocessing start method to 'spawn' for macOS compatibility.
    # This must be done at the very beginning of the main block.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # It's OK if it's already set.
    
    # Disable multiprocessing for PaddleOCR to avoid segmentation fault on macOS
    import os
    os.environ['PADDLE_USE_MULTIPROCESSING'] = '0'

    '''
        ele:min-grad: gradient threshold to produce binary map         
        ele:ffl-block: fill-flood threshold
        ele:min-ele-area: minimum area for selected elements 
        ele:merge-contained-ele: if True, merge elements contained in others
        text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
        text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

        Tips:
        1. Larger *min-grad* produces fine-grained binary-map while prone to over-segment element to small pieces
        2. Smaller *min-ele-area* leaves tiny elements while prone to produce noises
        3. If not *merge-contained-ele*, the elements inside others will be recognized, while prone to produce noises
        4. The *max-word-inline-gap* and *max-line-gap* should be dependent on the input image size and resolution

        mobile: {'min-grad':4, 'ffl-block':5, 'min-ele-area':50, 'max-word-inline-gap':6, 'max-line-gap':1}
        web   : {'min-grad':3, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':4, 'max-line-gap':4}
    '''
    key_params = {'min-grad':10, 'ffl-block':5, 'min-ele-area':50,
                  'merge-contained-ele':True, 'merge-line-to-paragraph':False, 'remove-bar':True}

    # set input image path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path_img = os.path.join(project_root, 'data', 'input', 'test1.png')

    print(f"处理图片: {input_path_img}")

    # 检查文件是否存在
    if not os.path.exists(input_path_img):
        print(f"错误：找不到图片文件！")
        print(f"请确认图片存在于: {input_path_img}")
        exit(1)
    output_root = 'data/tmp'

    resized_height = resize_height_by_longest_edge(input_path_img, resize_length=800)
    color_tips()

    is_ip = True
    is_clf = False
    is_ocr = False
    is_merge = False

    if is_ocr:
        import detect_text.text_detection as text
        os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
        text.text_detection(input_path_img, output_root, show=True, method='paddle')

    if is_ip:
        import detect_compo.ip_region_proposal as ip
        os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
        # switch of the classification func
        classifier = None
        if is_clf:
            classifier = {}
            from cnn.CNN import CNN
            # classifier['Image'] = CNN('Image')
            classifier['Elements'] = CNN('Elements')
            # classifier['Noise'] = CNN('Noise')
        ip.compo_detection(input_path_img, output_root, key_params,
                           classifier=classifier, resize_by_height=resized_height, show=False)

    if is_merge:
        import detect_merge.merge as merge
        os.makedirs(pjoin(output_root, 'merge'), exist_ok=True)
        name = input_path_img.split('/')[-1][:-4]
        compo_path = pjoin(output_root, 'ip', str(name) + '.json')
        ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
        merge.merge(input_path_img, compo_path, ocr_path, pjoin(output_root, 'merge'),
                    is_remove_bar=key_params['remove-bar'], is_paragraph=key_params['merge-line-to-paragraph'], show=True)
