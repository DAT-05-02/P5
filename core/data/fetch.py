from asyncio import wait
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import requests
import pandas as pd
import os
from PIL import Image
from core.util.util import timing
from util.constants import IMGDIR_PATH, MERGE_COLS, BFLY_FAMILY, BFLY_LIFESTAGE, DATASET_PATH


def setup_dataset(raw_dataset_path: str,
                  raw_label_path: str,
                  label_dataset_path: str,
                  dataset_csv_filename: str,
                  num_rows=None,
                  sort=False,
                  bfly=False):
    """ Loads a file, converts to csv if none exists, or loads an existing csv into a pd.DateFrame object
    @param raw_label_path: path to original label dataset file
    @param raw_dataset_path: path to original dataset file
    @param label_dataset_path: path to label csv file
    @param dataset_csv_filename: filename for the csv file
    @param num_rows: number of rows to include
    @param sort: if dataset should be sorted by species
    @param bfly: if dataset should only contain butterflies (no moths)
    Returns: pandas.DataFrame object with data

    """
    if not os.path.exists(IMGDIR_PATH):
        os.makedirs(IMGDIR_PATH)
    if os.path.exists(dataset_csv_filename):
        df = pd.read_csv(dataset_csv_filename, low_memory=False)
        if num_rows and len(df) == num_rows:
            return df

    df: pd.DataFrame = pd.read_csv(raw_dataset_path, sep="	", low_memory=False)
    if os.path.exists(label_dataset_path):
        df_label: pd.DataFrame = pd.read_csv(label_dataset_path, low_memory=False)
    else:
        df_label: pd.DataFrame = pd.read_csv(raw_label_path, sep="	", low_memory=False)
        df_label.to_csv(label_dataset_path, index=False)
    print(df[df.columns])
    drop_cols([df, df_label], MERGE_COLS)
    df = df.merge(df_label[df_label['gbifID'].isin(df['gbifID'])], on=['gbifID'])
    df = df.loc[~df['lifeStage'].isin(BFLY_LIFESTAGE)]
    if bfly:
        df = df.loc[df['family'].isin(BFLY_FAMILY)]
    print(df.shape)
    if sort:
        df.sort_values(by=['species'], inplace=True)
    print(f"found {len(df['species'].unique())} unique species")
    if num_rows:
        df.drop(df.index[num_rows:], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df = df.assign(path="", lbp="", sift="", homsc="")
    df.to_csv(dataset_csv_filename, index=False)
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
    out = f"{IMGDIR_PATH}{index}"
    if extra:
        out += extra
    return out + f".{extension}"


@timing
def fetch_images(df: pd.DataFrame, col: str):
    """
    Fetches all image links in a DataFrame column to path defined by :func:`~fetch.img_path_from_row`
    and assigns the column value to the path saved to.
    @param df: the DataFrame containing links
    @param col: which column the links are in
    """

    def save_img(df: pd.DataFrame, row: pd.Series, index) -> Any:
        path = img_path_from_row(row, index)
        if not os.path.exists(path):
            img = Image.open(requests.get(row[col], stream=True).raw)
            img.save(path)
            print(path)
        df.at[index, 'path'] = path
    # df['path'] = ""
    with ThreadPoolExecutor(50) as executor:
        _ = [executor.submit(save_img, df, row, index) for index, row in df.iterrows()]
        executor.shutdown(wait=True)
    df.to_csv(DATASET_PATH, index=False)
