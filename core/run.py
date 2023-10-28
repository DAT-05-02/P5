import os
import sys

from core.data.fetch import fetch_images, setup_dataset
from core.util.constants import RAW_DATA_PATH, RAW_LABEL_PATH, DATASET_PATH, LABEL_DATASET_PATH, IMGDIR_PATH
from core.data.feature import FeatureExtractor


if __name__ == "__main__":
    sys.path.append(f"{os.getcwd()}{os.sep}core")
    if not os.getcwd().split(os.sep)[-1] == "core":
        os.chdir("core")
    df = setup_dataset(raw_dataset_path=RAW_DATA_PATH,
                       raw_label_path=RAW_LABEL_PATH,
                       label_dataset_path=LABEL_DATASET_PATH,
                       dataset_csv_filename=DATASET_PATH,
                       num_rows=50,
                       bfly=["all"])
    fetch_images(df, "identifier")
    FeatureExtractor.create_augmented_images(IMGDIR_PATH)
    ft_extractor = FeatureExtractor()
    df = ft_extractor.pre_process(df, "sift", radius=2, should_bb=True, should_resize=True)
    # model = Model(df)
    # model.model_compile_fit_evaluate(epochs=50)

