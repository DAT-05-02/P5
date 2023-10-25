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


def test_rotate_and_save_image_create_rotated_image_files(temp_dir):
    # Arrange
    img = Image.new(mode="RGB", size=(3, 4), color=(255, 255, 0))
    name = "test"
    path = temp_dir

    # Act
    FeatureExtractor.rotate_and_save_image(img, name, path)

    # Assert
    for deg in ["0", "90", "180", "270"]:
        assert os.path.exists(f"{path}/{name}_{deg}.jpg")


def test_create_augmented_images_creates_rotated_and_flipped_files(temp_dir):
    # Arrange
    img = Image.new(mode="RGB", size=(3, 4), color=(255, 0, 255))
    name = "test"
    path = temp_dir

    # Act
    FeatureExtractor.create_augmented_images(temp_dir, ["all"])

    # Assert
    for deg in ["", "_90", "_180", "_270", "f", "_90f", "_180f", "_270f"]:
        print(f"{path}/{name}{deg}.jpg")
        assert os.path.exists(f"{path}/{name}_{deg}.jpg")


def test_flip_and_save_image_creates_flipped_files(temp_dir):
    # Arrange
    img = Image.new(mode="RGB", size=(4, 3), color=(255, 0, 255))
    name = "test"
    path = temp_dir

    # Act
    FeatureExtractor.flip_and_save_image(img, name, path)

    # Assert
    assert os.path.exists(f"{path}/{name}f.jpg")


if __name__ == "__main__":
    pytest.main()
