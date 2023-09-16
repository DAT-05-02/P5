import os

import pytest

from data.fetch import setup


class FetchTester:

    def __init__(self,
                 data_path="leopidotera-dk/leopidotera-dk.csv",
                 save_path="leopidotera-dk/multimedia.txt",
                 img_path="/image_db",
                 csv_name="leopidotera-dk.csv"):
        self.save_path = save_path
        self.img_path = img_path
        self.data_path = data_path
        self.df = setup(dataset_path=save_path, dataset_csv_filename=csv_name)

"""
@pytest.fixture
def fetch():
    return FetchTester()
"""


def test_img_path_from_row(self):
    assert False


def test_fetch_images(self):
    assert False


def test_setup(tester, save_path, file_name):
    os.remove(file_name)
    df = setup(dataset_path=save_path, dataset_csv_filename=file_name)
    assert os.path.exists(file_name) is True
    os.remove(file_name)


def test_all(self):
    test_fetch_images()
    test_img_path_from_row()
