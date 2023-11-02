def obj_det(img, model):
    res = model.predict(source=img, save=False, save_txt=False, imgsz=640, conf=0.25)

    return res