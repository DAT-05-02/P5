import logging
import os
import pytest
import pandas as pd
from PIL import Image
from core.data.feature import FeatureExtractor
from core.data.fetch import setup_dataset
from core.util.constants import RAW_DATA_PATH, RAW_LABEL_PATH, LABEL_DATASET_PATH, DATASET_PATH
import shutil


@pytest.fixture
def temp_dir(request):
    temp_dir = "temp-test-dir/temp-species-dir"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    image = Image.new(mode="RGB", size=(3, 4), color=(255, 0, 255))
    image.save(f"{temp_dir}/test_0.jpg")

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
    df = setup_dataset(raw_dataset_path=RAW_DATA_PATH,
                       raw_label_path=RAW_LABEL_PATH,
                       label_dataset_path=LABEL_DATASET_PATH,
                       dataset_csv_filename=DATASET_PATH,
                       num_rows=5,
                       bfly=["all"])
    df['path'] = "temp-test-dir/temp-species-dir/test_0.jpg"
    return df


@pytest.mark.parametrize("degree", [0, 90, 180, 270])
def test_rotate_and_save_image_create_rotated_image_files(degree, temp_dir, feature_extractor: FeatureExtractor):
    # Arrange
    name = "test_0.jpg"
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
        assert os.path.exists(f"{path}/test_{deg}.jpg")


def test_flip_and_save_image_creates_flipped_files(temp_dir, feature_extractor: FeatureExtractor):
    # Arrange
    name = "test_0.jpg"
    path = f"{temp_dir}/{name}"
    ft = feature_extractor

    # Act
    new_path = ft.flip_and_save_image(path)

    # Assert
    assert os.path.exists(new_path)


if __name__ == "__main__":
    pytest.main()
