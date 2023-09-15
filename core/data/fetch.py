import pandas as pd
import requests, math
from PIL import Image
from core.data import *


def setup(dataset_path: str, dataset_csv_filename: str):
    if not os.path.exists(dataset_csv_filename):
        return convert_to_csv(read_path=dataset_path, save_path=dataset_csv_filename)
    else:
        return pd.read_csv(dataset_csv_filename, low_memory=False)


def convert_to_csv(read_path, save_path):
    read_file = pd.read_csv(read_path, sep="	", low_memory=False)
    read_file.to_csv(save_path, index=None)
    return read_file


def img_path_from_row(row, index):
    extension = row['identifier'].split(".")[-1]
    return f"{IMG_PATH}{index}.{extension}"


def loading_bar(i, max):
    bl = 50
    i = max - i
    chars = math.ceil(i * (bl / max))
    p = math.ceil((i / max) * 100)
    print('▮' * chars + '▯' * (bl - chars), str(p) + '%', end='\r')


def fetch_images(df: pd.DataFrame, col: str):
    r_count = len(df)
    for index, row in df.iterrows():
        path = img_path_from_row(row, index)
        if not os.path.exists(path):
            # TODO should compress/resize to agreed upon size
            Image.open(requests.get(row[col], stream=True).raw).save(path)
        loading_bar(df.index[-1] - index, r_count)
