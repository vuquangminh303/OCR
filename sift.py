import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from paddleocr import PaddleOCR

def transform_improved(image_src, image_tar=None):
    if image_tar is None:
        image_tar = cv2.imread('anchor.jpg')
        if image_tar is None:
            raise FileNotFoundError("Không thể đọc file hình ảnh đích 'anchor.jpg'")
    
    image_src_gray = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    image_tar_gray = cv2.cvtColor(image_tar, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.SIFT_create()
    kp1, des1 = detector.detectAndCompute(image_tar_gray, mask=None)
    kp2, des2 = detector.detectAndCompute(image_src_gray, mask=None)
    
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return image_src
    
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    try:
        matches = matcher.knnMatch(des1, des2, k=2)
    except cv2.error:
        return image_src
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    min_matches_required = 10
    if len(good_matches) >= min_matches_required:
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None and _is_valid_homography(M):
                h, w = image_tar.shape[:2]
                image_src_aligned = cv2.warpPerspective(image_src, M, (w, h))
                return image_src_aligned
        except cv2.error:
            pass
    return image_src

def _is_valid_homography(H):
    if np.max(np.abs(H)) > 100:
        return False
    if np.linalg.det(H) < 1e-10:
        return False
    if np.linalg.det(H[:2, :2]) <= 0:
        return False
    return True

def get_card_info(info, crop_image):
    card_info = {}
    results = info(crop_image)
    h, w = crop_image.shape[:2]
    class_names = info.names
    for result in results:
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls)
            label = f"{class_names[class_id]}"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            if label not in card_info or confidence > card_info[label]['confidence']:
                card_info[label] = {
                    'coords': (x1, y1, x2, y2),
                    'confidence': confidence
                }
    best_boxes = {}
    for key, value in card_info.items():
        x1, y1, x2, y2 = value['coords']
        best_boxes[key] = crop_image[y1:y2, x1:x2]
    return best_boxes

def process_dob(text):
    text = text[0:2] + '/' + text[2:4] + '/' + text[4:]
    return text
def process_image(input_image):
    info = YOLO('info_detection.pt')
    config = Cfg.load_config_from_name('vgg_transformer')
    viet_detector = Predictor(config)
    paddle_detector = PaddleOCR(use_angle_cls=True, lang='en', rec_char_list='0123456789')
    
    if isinstance(input_image, str):
        image = cv2.imread(input_image)
    else:
        image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    
    crop_image = transform_improved(image)
    if crop_image is None:
        return None, {"error": "Vui lòng chụp lại"}
    
    card_info = get_card_info(info, crop_image)
    results = {}
    for key, value in card_info.items():
        image_rgb = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        if key in ['id', 'dob']:
            paddle_result = paddle_detector.ocr(value, cls=True)
            if paddle_result and paddle_result[0]:
                text = ''.join([line[1][0] for line in paddle_result[0]])
                text = ''.join([char for char in text if char.isdigit()])
                if key == 'dob':
                    text = process_dob(text)
            else:
                text = ''
        else:
            text = viet_detector.predict(image_pil)
        results[key] = text
    
    required_keys = ["id", "dob", "name"]
    if not all(key in results for key in required_keys):
        return None, {"error": "Không trích xuất đủ 3 thông tin. Vui lòng chụp lại"}
    
    crop_image_pil = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
    return crop_image_pil, results

def process_id_card(input_image):
    cropped_img, extracted_info = process_image(input_image)
    if cropped_img is None:
        return None, extracted_info
    return cropped_img, extracted_info

iface = gr.Interface(
    fn=process_id_card,
    inputs=gr.Image(type="pil", label="Upload ảnh căn cước"),
    outputs=[
        gr.Image(label="Ảnh căn cước đã cắt"),
        gr.JSON(label="Thông tin trích xuất")
    ],
    title="Trích xuất thông tin Căn cước Công dân",
    description="Upload ảnh căn cước công dân để trích xuất thông tin ngày sinh, số CMND/CCCD và tên. Nếu không trích xuất đủ 3 thông tin, vui lòng chụp lại."
)

if __name__ == "__main__":
    iface.launch(inbrowser = True)
