from core.yolo.load_img import load_img

def yolo_crop(img, xywhn):

    corners = load_img(img, xywhn)
    img1 = img.crop(corners)
    return img1
