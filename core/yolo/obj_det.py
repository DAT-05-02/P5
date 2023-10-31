
def obj_det(img, model):
    res = model.predict(source=img, save=False, imgsz=640, conf=0.25)
    xywhn = res[0].boxes.xywhn

    return xywhn
