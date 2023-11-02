from typing import Tuple

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