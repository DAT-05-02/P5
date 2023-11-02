import logging
import os
import random
import shutil

import pandas as pd
import pytest

from core.data.feature import FeatureExtractor
from core.data.fetch import Database


class FetchTester:

    def __init__(self,
                 save_path="leopidotera-dk/leopidotera-dk.csv",
                 import_path="leopidotera-dk/multimedia.txt",
                 label_path="leopidotera-dk/occurrence.txt",
                 img_path="image-db/",
                 columns=None,
                 label_csv_name="occurrence.csv",
                 csv_name="leopidotera-dk.csv",
                 img_col="identifier"):
        if columns is None:
            columns = ['gbifID', 'identifier', '']
        self.df: pd.DataFrame = pd.DataFrame()
        self.save_path = save_path
        self.label_path = label_path
        self.label_csv_name = label_csv_name
        self.img_path = img_path
        self.import_path = import_path
        self.csv_name = csv_name
        self.img_col = img_col


@pytest.fixture(scope="module")
def fetcher() -> FetchTester:
    return FetchTester()


def test_setup(fetcher: FetchTester):
    if os.path.exists(fetcher.save_path):
        os.remove(fetcher.save_path)
    assert os.path.exists(fetcher.import_path)
    db = Database(raw_dataset_path=fetcher.import_path,
                  raw_label_path=fetcher.label_path,
                  label_dataset_path=fetcher.label_csv_name,
                  dataset_csv_filename=fetcher.save_path,
                  ft_extractor=FeatureExtractor(),
                  num_rows=5,
                  degrees="all",
                  bfly=["all"])
    fetcher.df = db.setup_dataset()
    assert os.path.exists(fetcher.save_path) is True


def random_index(df):
    count = random.randrange(3, 7)
    start = random.randrange(count, len(df))
    logging.warning(f"range:{count}, start: {start}, df.len: {len(df)}")
    return count, start


@pytest.mark.parametrize("n_times", range(5))
def test_fetch_images(fetcher: FetchTester, n_times):
    shutil.rmtree(fetcher.img_path)
    amount, start = random_index(fetcher.df)
    df = fetcher.df.copy()[start:start + amount]
    db = Database(raw_dataset_path=fetcher.import_path,
                  raw_label_path=fetcher.label_path,
                  label_dataset_path=fetcher.label_csv_name,
                  dataset_csv_filename=fetcher.save_path,
                  ft_extractor=FeatureExtractor(),
                  num_rows=50,
                  degrees="all",
                  bfly=["all"])
    db.fetch_images(df, "identifier")
    file_count = sum(len(files) for _, _, files in os.walk(fetcher.img_path))
    assert file_count == db.num_rows
