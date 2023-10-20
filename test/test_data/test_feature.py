import os
import pytest
from PIL import Image
from core.data.feature import FeatureExtractor
import shutil


@pytest.fixture
def temp_dir(request):
    temp_dir = "temp_test_dir"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    def remove_tmp_dir():
        shutil.rmtree(temp_dir)

    request.addfinalizer(remove_tmp_dir)
    return temp_dir


def test_make_image_degrees_create_rotated_image_files(temp_dir):
    # Arrange
    img = Image.new(mode="RGB", size=(3, 4), color=(255, 255, 0))
    name = "test"
    path = temp_dir

    # Act
    FeatureExtractor.make_image_degrees(img, name, path, ["rotate"])

    # Assert
    for deg in ["0", "90", "180", "270"]:
        assert os.path.exists(f"{path}/{name}_{deg}.jpg")


def test_make_image_degrees_creates_degrees_and_flipped_files(temp_dir):
    # Arrange
    img = Image.new(mode="RGB", size=(3, 4), color=(255, 0, 255))
    name = "test"
    path = temp_dir

    # Act
    FeatureExtractor.make_image_degrees(img, name, path, ["all"])

    # Assert
    for deg in ["0", "90", "180", "270", "0f", "90f", "180f", "270f"]:
        assert os.path.exists(f"{path}/{name}_{deg}.jpg")


def test_make_image_degrees_creates_flipped_files(temp_dir):
    # Arrange
    img = Image.new(mode="RGB", size=(4, 3), color=(255, 0, 255))
    name = "test"
    path = temp_dir

    # Act
    FeatureExtractor.make_image_degrees(img, name, path, ["flip"])

    # Assert
    for deg in ["0", "0f"]:
        assert os.path.exists(f"{path}/{name}_{deg}.jpg")


if __name__ == "__main__":
    pytest.main()
