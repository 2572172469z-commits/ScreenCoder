import cv2
import os
import requests
import json
from base64 import b64encode
import time


def Google_OCR_makeImageData(imgpath):
    with open(imgpath, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_req = {
            'image': {
                'content': ctxt
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION',
                # 'type': 'TEXT_DETECTION',
                'maxResults': 1
            }]
        }
    return json.dumps({"requests": img_req}).encode()


def ocr_detection_google(imgpath):
    start = time.perf_counter()
    url = 'https://vision.googleapis.com/v1/images:annotate'
    api_key = 'AIzaSyDUc4iOUASJQYkVwSomIArTKhE2C6bHK8U'             # *** Replace with your own Key ***
    imgdata = Google_OCR_makeImageData(imgpath)
    response = requests.post(url,
                             data=imgdata,
                             params={'key': api_key},
                             headers={'Content_Type': 'application/json'})
    # print('*** Text Detection Time Taken:%.3fs ***' % (time.perf_counter() - start))
    print("*** Please replace the Google OCR key at detect_text/ocr.py line 28 with your own (apply in https://cloud.google.com/vision) ***")
    
    response_json = response.json()
    if 'error' in response_json:
        error_msg = response_json['error']
        if 'BILLING_DISABLED' in str(error_msg):
            raise Exception("Google Vision API requires billing to be enabled. Please:\n"
                          "1. Visit https://console.developers.google.com/billing/enable?project=718250946490\n"
                          "2. Enable billing for your project\n"
                          "3. Wait a few minutes for changes to propagate\n"
                          "4. Or use PaddleOCR instead by setting method='paddle'")
        else:
            raise Exception(f"Google Vision API error: {error_msg}")
    
    if 'responses' not in response_json:
        raise Exception(response_json)
    if response_json['responses'] == [{}]:
        # No Text
        return None
    else:
        return response_json['responses'][0]['textAnnotations'][1:]
