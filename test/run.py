import pytest

from test_data.test_fetch import FetchTester

DATA_PATH = "leopidotera-dk/multimedia.txt"
CSV_PATH = "leopidotera-dk.csv"
IMG_PATH = "image_db"

if __name__ == '__main__':
    pytest.main()
    FetchTester().test_all()
