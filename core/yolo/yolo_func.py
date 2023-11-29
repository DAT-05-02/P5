from typing import Tuple
from ultralytics import YOLO


def obj_det(img, model: YOLO, conf=0.25):
    res = model.predict(source=img, save=False, save_txt=False, imgsz=640, conf=conf, device="cpu")
    return res


def load_img(img, xywhn) -> Tuple[float, float, float, float]:
    img_width, img_height = img.size
    x_coordinate = float(xywhn[0][0])
    y_coordinate = float(xywhn[0][1])
    label_width = float(xywhn[0][2])
    label_height = float(xywhn[0][3])

    label_enlarge = 1.10

    left = img_width * (x_coordinate - label_width / 2 * label_enlarge)
    top = img_height * (y_coordinate - label_height / 2 * label_enlarge)
    right = img_width * (x_coordinate + label_width / 2 * label_enlarge)
    bottom = img_height * (y_coordinate + label_height / 2 * label_enlarge)

    fcorners = (left, top, right, bottom)

    return fcorners


def yolo_crop(img, xywhn):
    corners = load_img(img, xywhn)
    img1 = img.crop(corners)
    return img1
