from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image

from core.data import *
from util.util import timing


def setup_dataset(dataset_path: str, dataset_csv_filename: str):
    """ Loads a file, converts to csv if none exists, or loads an exisiting csv into a pd.DateFrame object
    Args:
        dataset_path: path to original dataset file
        dataset_csv_filename: filename for the csv file

    Returns: pandas.DataFrame object with data
    """
    if not os.path.exists(dataset_csv_filename):
        return pd.read_csv(dataset_path, sep="	", low_memory=False).to_csv(dataset_csv_filename, index=None)
    else:
        return pd.read_csv(dataset_csv_filename, low_memory=False)


def img_path_from_row(row, index, row_value="identifier"):
    extension = row[row_value].split(".")[-1]
    return f"{IMG_PATH}{index}.{extension}"


@timing
def fetch_images(df: pd.DataFrame, col: str):
    def save_img(row, path):
        if not os.path.exists(path):
            Image.open(requests.get(row[col], stream=True).raw).save(path)

    r_count = len(df)
    with ThreadPoolExecutor(r_count) as executor:
        # TODO should compress/resize to agreed upon size
       futures = [executor.submit(save_img(row, img_path_from_row(row, index))) for index, row in df.iterrows()]




