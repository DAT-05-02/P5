from typing import Tuple
from ultralytics import YOLO
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def obj_det(img, model: YOLO, conf=0.25, img_size=(640, 640)):
    res = model.predict(source=img, save=False, save_txt=False, imgsz=img_size, conf=conf, device="cpu")
    return res


def load_img(img, xywhn) -> Tuple[float, float, float, float]:
    img_width, img_height = img.size

    x_coordinate = float(xywhn[0][0])
    y_coordinate = float(xywhn[0][1])
    label_width = float(xywhn[0][2])
    label_height = float(xywhn[0][3])

    left = img_width * (x_coordinate - label_width / 2)
    top = img_height * (y_coordinate - label_height / 2)
    right = img_width * (x_coordinate + label_width / 2)
    bottom = img_height * (y_coordinate + label_height / 2)

    fcorners = (left, top, right, bottom)

    return fcorners


def yolo_crop(img: Image.Image, xywhn, rs_size=(416, 416)):
    corners = load_img(img, xywhn)
    img1 = img.crop(corners)
    img1 = img1.resize(rs_size, resample=3)
    return img1
