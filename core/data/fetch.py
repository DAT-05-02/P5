from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd
import os
from PIL import Image
from core.util.util import timing

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


def img_path_from_row(row: pd.Series, index: int, column="identifier"):
    """Generates path for an image based on row and index with optional column, in which the image link is.
    @param row: Series to extract file path from
    @param index: of the row. Series objects don't inherently know which index they are in a DataFrame.
    @param column: (optional) which column the file path is in
    @return: the path to save the image in
    @rtype: str
    """
    extension = row[column].split(".")[-1]
    if len(extension) < 1:
        extension = "jpg"
    return f"{IMG_PATH}{index}.{extension}"


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

    def save_img(row, path):
        if not os.path.exists(path):
            print(path)
            img = Image.open(requests.get(row[col], stream=True).raw)
            img = make_square_with_bb(img, 416)
            img = img.resize((416, 416))
            img.save(path)

    r_count = len(df)
    with ThreadPoolExecutor(r_count) as executor:
        # TODO should compress/resize to agreed upon size
        [executor.submit(save_img(row, img_path_from_row(row, index))) for index, row in df.iterrows()]
