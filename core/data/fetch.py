from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd
import os
from core.data import IMG_PATH
from PIL import Image
from util.util import timing


def setup_dataset(dataset_path: str, dataset_csv_filename: str):
    """ Loads a file, converts to csv if none exists, or loads an exisiting csv into a pd.DateFrame object
    Args:
        dataset_path: path to original dataset file
        dataset_csv_filename: filename for the csv file

    Returns: pandas.DataFrame object with data
    """
    if not os.path.exists(dataset_csv_filename):
        df = pd.read_csv(dataset_path, sep="	", low_memory=False)
        df.to_csv(dataset_csv_filename, index=None)
        return df
    else:
        return pd.read_csv(dataset_csv_filename, low_memory=False)


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




