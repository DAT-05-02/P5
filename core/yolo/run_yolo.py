import os
from PIL import Image
import numpy as np

from .yolo_func import obj_det, yolo_crop
import time


def run_yolo(model, df) -> None:
    print("======= RUN YOLO START ======")
    start_time = time.time()
    overwrite_count = 0
    delete_count = 0

    yolo_accepted = [True for x in range(len(df.index))]

    for idx, row in df.iterrows():
        print(row["path"])
        img = Image.fromarray(np.load(row["path"]))
        res = obj_det(img, model)
        xywhn = res[0].boxes.xywhn
        if xywhn.numel() > 0:
            img1 = yolo_crop(img, xywhn)
            img1.save(res[0].path)
            overwrite_count += 1
        else:
            print(f"deleting {res[0].path}")
            #os.remove(res[0].path)
            yolo_accepted[idx] = False
            delete_count += 1

    df["yolo_accepted"] = yolo_accepted

    total_img = overwrite_count + delete_count
    print(f"Deleted {delete_count} out of {total_img} total images.")
    print("--- Took %s seconds ---" % (time.time() - start_time))
    print("======= RUN YOLO DONE ======")

    new_df = df[df["yolo_accepted"] == True] # Drops all rows that doesn't satisfy yolo accepted
    return new_df
