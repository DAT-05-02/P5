import os
from ultralytics import YOLO
from PIL import Image

from .obj_det import obj_det
from .yolo_crop import yolo_crop
import time

def run_yolo(model, root_path) -> None:
    start_time = time.time()
    overwrite_count = 0
    delete_count = 0

    for dirpath in os.scandir(root_path):
        imgdir_name = os.path.splitext(f"{dirpath.name}")[0]
        if len(os.listdir(f"{root_path}/{imgdir_name}")) > 0:
            for imgpath in os.scandir(f"{root_path}/{imgdir_name}"):
                img_name = os.path.splitext(f"{imgpath.name}")[0]
                img_type = os.path.splitext(f"{imgpath.name}")[1]
                img = Image.open(f"{root_path}/{imgdir_name}/{img_name}{img_type}")
                res = obj_det(img, model)
                xywhn = res[0].boxes.xywhn
                if xywhn.numel() > 0:
                    img1 = yolo_crop(img, xywhn)
                    img1.save(res[0].path)
                    overwrite_count += 1
                else:
                    print(f"deleting {res[0].path}")
                    os.remove(res[0].path)
                    delete_count += 1

    total_img = overwrite_count + delete_count
    print(f"Deleted {delete_count} out of {total_img} total images.")
    print("--- Took %s seconds ---" % (time.time() - start_time))
