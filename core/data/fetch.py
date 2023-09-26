from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd
import numpy as np
import os
from PIL import Image
from core.util.util import timing

IMG_PATH = "image_db/"
MERGE_COLS = ['genericName', 'species', 'family', 'stateProvince', 'gbifID','identifier','format','created']


def setup_dataset(dataset_path: str, label_path: str, dataset_csv_filename: str):
    """ Loads a file, converts to csv if none exists, or loads an existing csv into a pd.DateFrame object
    Args:
        label_path: path to label dataset
        dataset_path: path to original dataset file
        dataset_csv_filename: filename for the csv file

    Returns: pandas.DataFrame object with data
    """
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)
    if not os.path.exists(dataset_csv_filename):
        df1 = pd.read_csv(dataset_path, sep="	", low_memory=False)
        df1.drop(columns=['type','references','creator','publisher','license','rightsHolder'], inplace=True)
        df1.drop(df1.index[250:], inplace=True)
        print(df1.shape)
        # df1.dropna(axis=1, inplace=True, thresh=10000)
        df2 = pd.read_csv(label_path, sep="	", low_memory=False)
        df2.drop(columns=[col for col in df2 if col not in MERGE_COLS], inplace=True)
        print(df1.columns)
        print(df2.shape)
        df1 = df1.merge(df2[df2['gbifID'].isin(df1['gbifID'])], on=['gbifID'])
        print(df1)
        print("done")
        df1.to_csv(dataset_csv_filename, index=None)
        df = df1
        df.dropna(axis=1, inplace=True)
    else:
        df = pd.read_csv(dataset_csv_filename, low_memory=False)
    return df


def drop_cols():
    kwargs.get()


def img_path_from_row(row: pd.Series, index: int, column="identifier"):
    """Generates path for an image based on row and index with optional column, in which the image link is.
    @param row: Series to extract file path from
    @param index: of the row. Series objects don't inherently know which index they are in a DataFrame.
    @param column: (optional) which column the file path is in
    @return: the path to save the image in
    @rtype: str
    """
    extension = row[column].split(".")[-1]
    return f"{IMG_PATH}{index}.{extension}"


@timing
def fetch_images(df: pd.DataFrame, col: str):
    """
    Fetches all image links in a DataFrame column to path defined by :func:`~fetch.img_path_from_row`
    @param df: the DataFrame containing links
    @param col: which column the links are in
    """
    def save_img(row, path):
        if not os.path.exists(path):
            Image.open(requests.get(row[col], stream=True).raw).save(path)

    r_count = len(df)
    with ThreadPoolExecutor(r_count) as executor:
        # TODO should compress/resize to agreed upon size
       [executor.submit(save_img(row, img_path_from_row(row, index))) for index, row in df.iterrows()]




