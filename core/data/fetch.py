from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import requests
import pandas as pd
import os
from PIL import Image
from core.util.util import timing
from data.feature import lbp, rlbp

IMG_PATH = "image_db/"
MERGE_COLS = ['genericName', 'species', 'family', 'stateProvince', 'gbifID', 'identifier', 'format', 'created',
              'iucnRedListCategory']


def setup_dataset(dataset_path: str, label_path: str, dataset_csv_filename: str, num_rows=None):
    """ Loads a file, converts to csv if none exists, or loads an existing csv into a pd.DateFrame object
    @param label_path: path to label dataset
    @param dataset_path: path to original dataset file
    @param dataset_csv_filename: filename for the csv file
    @param num_rows: number of rows to include

    Returns: pandas.DataFrame object with data
    """
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)
    if not os.path.exists(dataset_csv_filename):
        df1 = pd.read_csv(dataset_path, sep="	", low_memory=False)
        if num_rows:
            df1.drop(df1.index[num_rows:], inplace=True)
        df2 = pd.read_csv(label_path, sep="	", low_memory=False)
        df2.to_csv("occurrence.csv", index=None)
        drop_cols([df1, df2], MERGE_COLS)
        df1 = df1.merge(df2[df2['gbifID'].isin(df1['gbifID'])], on=['gbifID'])
        df1.to_csv(dataset_csv_filename, index=None)
        df = df1
    else:
        df = pd.read_csv(dataset_csv_filename, low_memory=False)
    return df


def drop_cols(dfs, cols):
    for df in list(dfs):
        df.drop(columns=[col for col in df if col not in MERGE_COLS], inplace=True)


def img_path_from_row(row: pd.Series, index: int, column="identifier", extra=None):
    """Generates path for an image based on row and index with optional column, in which the image link is.
    @param row: Series to extract file path from
    @param index: of the row. Series objects don't inherently know which index they are in a DataFrame.
    @param column: (optional) which column the file path is in
    @param extra: (optional) extra keywords in name
    @return: the path to save the image in
    @rtype: str
    """
    extension = row[column].split(".")[-1]
    if len(extension) < 1:
        extension = "jpg"
    out = f"{IMG_PATH}{index}"
    if extra:
        out += extra
    return out + f".{extension}"


def make_square_with_bb(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


@timing
def fetch_images(df: pd.DataFrame, col: str):
    """
    Fetches all image links in a DataFrame column to path defined by :func:`~fetch.img_path_from_row`
    @param df: the DataFrame containing links
    @param col: which column the links are in
    """

    def save_img(row: pd.Series, index) -> Any:
        path = img_path_from_row(row, index, extra=extra)
        if not os.path.exists(path):
            print(path)
        img = Image.open(requests.get(row[col], stream=True).raw)
        with ThreadPoolExecutor(8) as executor:
            [executor.submit(save_img_inner, img, x, False, False) for x in range(1, 13, 2)]
            [executor.submit(save_img_inner, img, x, True, False) for x in range(1, 13, 2)]
            # [executor.submit(save_img_inner, img, x, True, True) for x in range(1, 13, 4)]
    def save_img_inner(img, radius=1, resize=False, resize_first=False) -> Any:
        extra = "_"
        if resize:
            extra += "resize"
        if resize_first:
            extra += "first"
        extra += str(radius)
        path =
        if not os.path.exists(path):
            print(path)
            if resize:
                if resize_first:
                    img = make_square_with_bb(img, 416)
                    img = img.resize((416, 416))
                    img = Image.fromarray(lbp(img, method="ror", radius=radius))
                else:
                    img = Image.fromarray(lbp(img, method="ror", radius=radius))
                    img = make_square_with_bb(img, 416)
                    img = img.resize((416, 416))
            else:
                img = Image.fromarray(lbp(img, method="ror", radius=radius))
            img.save(path)



    r_count = len(df)
    with ThreadPoolExecutor(4) as executor:
        _ = [executor.submit(save_img, row, index) for index, row in df.iterrows()]
