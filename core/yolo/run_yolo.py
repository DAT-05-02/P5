import pandas as pd
from PIL import Image
import numpy as np
from ultralytics import YOLO

from .yolo_func import obj_det, yolo_crop
import time


def run_yolo(df) -> pd.DataFrame:
    if 'yolo_accepted' in df:
        return df

    model = YOLO('yolo/medium250e.pt')
    print("======= RUN YOLO START ======")
    start_time = time.time()
    overwrite_count = 0
    delete_count = 0

    yolo_accepted = np.full(len(df.index), fill_value=np.nan).tolist()

    for idx, row in df.iterrows():
        print(row["path"])
        img = Image.fromarray(np.load(row["path"]))
        res = obj_det(img, model)
        xywhn = res[0].boxes.xywhn
        if xywhn.numel() > 0:
            img1 = yolo_crop(img, xywhn)
            img1.save(res[0].path)
            overwrite_count += 1
            yolo_accepted[idx] = True
        else:
            print(f"deleting {res[0].path}")
            # os.remove(res[0].path)
            yolo_accepted[idx] = False
            delete_count += 1

    df["yolo_accepted"] = yolo_accepted
    df.dropna(subset=['yolo_accepted'], inplace=True)
    df.reset_index(inplace=True, drop=True)

    total_img = overwrite_count + delete_count
    print(f"Deleted {delete_count} out of {total_img} total images.")
    print("--- Took %s seconds ---" % (time.time() - start_time))
    print("======= RUN YOLO DONE ======")

    return df
