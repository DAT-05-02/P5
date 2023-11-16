import logging
import os
import json

import numpy as np
import pytest
import pandas as pd
from core.data.feature import FeatureExtractor
from core.data.fetch import Database
from core.util.constants import RAW_DATA_PATH, RAW_LABEL_PATH, LABEL_DATASET_PATH, DATASET_PATH
import shutil

with open('core/util/constants.txt', 'r') as f:
    constants = json.load(f)

@pytest.fixture
def temp_dir(request):
    temp_dir = "temp-test-dir/temp-species-dir"
    os.makedirs(temp_dir, exist_ok=True)
    image = np.full((constants["IMG_SIZE"], constants["IMG_SIZE"], 3), fill_value=1, dtype=np.uint8)
    logging.info(image.shape)
    np.save(f"{temp_dir}/test_0.npy", image)

    def remove_tmp_dir():
        shutil.rmtree(temp_dir)

    request.addfinalizer(remove_tmp_dir)
    return temp_dir


@pytest.fixture(scope="module")
def feature_extractor() -> FeatureExtractor:
    return FeatureExtractor()


@pytest.fixture(scope="module")
def data_frame() -> pd.DataFrame:
    logging.info(os.getcwd())
    db = Database(raw_dataset_path=RAW_DATA_PATH,
                  raw_label_path=RAW_LABEL_PATH,
                  label_dataset_path=LABEL_DATASET_PATH,
                  dataset_csv_filename=DATASET_PATH,
                  ft_extractor=FeatureExtractor(),
                  num_rows=5,
                  degrees="all",
                  bfly=["all"])
    df = db.setup_dataset()
    df['path'] = "temp-test-dir/temp-species-dir/test_0.npy"
    return df


@pytest.mark.parametrize("degree", [0, 90, 180, 270])
def test_rotate_and_save_image_create_rotated_image_files(degree, temp_dir, feature_extractor: FeatureExtractor):
    # Arrange
    name = "test_0.npy"
    path = f"{temp_dir}/{name}"
    ft = feature_extractor

    # Act
    new_path = ft.rotate_and_save_image(path, degree)

    # Assert
    assert os.path.exists(new_path)


def test_augment_image_creates_rotated_and_flipped_files(temp_dir, feature_extractor: FeatureExtractor,
                                                         data_frame: pd.DataFrame):
    # Arrange
    path = temp_dir
    ft = feature_extractor
    row = data_frame.loc[2]

    # Act
    ft.augment_image(row, "all")

    # Assert
    for deg in ["0", "90", "180", "270", "0f", "90f", "180f", "270f"]:
        assert os.path.exists(f"{path}/test_{deg}.npy")


def test_flip_and_save_image_creates_flipped_files(temp_dir, feature_extractor: FeatureExtractor):
    # Arrange
    name = "test_0.npy"
    path = f"{temp_dir}/{name}"
    ft = feature_extractor

    # Act
    new_path = ft.flip_and_save_image(path)

    # Assert
    assert os.path.exists(new_path)


if __name__ == "__main__":
    pytest.main()
