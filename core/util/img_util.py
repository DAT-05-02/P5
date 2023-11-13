import random

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt


def show_sample_img_df(df: pd.DataFrame, num_samples=5):
    end = random.randrange(num_samples, len(df.index))
    start = end - num_samples
    print(f"[{start}, {end}]")
    df = df.loc[start: end]
    for idx, x in df.iterrows():
        if x['yolo_accepted'] is True:
            img = Image.fromarray(np.load(x['path'], allow_pickle=True))
            plt.figure(figsize=(6, 8))
            plt.imshow(img)
            plt.show()
