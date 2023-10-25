import os
import sys

from core.data.fetch import fetch_images, setup_dataset
from core.util.constants import RAW_DATA_PATH, RAW_LABEL_PATH, DATASET_PATH, LABEL_DATASET_PATH
from core.data.feature import FeatureExtractor
from core.model.model import Model


if __name__ == "__main__":
    sys.path.append(f"{os.getcwd()}{os.sep}core")
    os.chdir("core")
    df = setup_dataset(raw_dataset_path=RAW_DATA_PATH,
                       raw_label_path=RAW_LABEL_PATH,
                       label_dataset_path=LABEL_DATASET_PATH,
                       dataset_csv_filename=DATASET_PATH,
                       num_rows=2000,
                       bfly=["all"])
    fetch_images(df, "identifier")
    #ft_extractor = FeatureExtractor()
    #df = ft_extractor.pre_process(df, "lbp", radius=7, should_bb=True, should_resize=True)
    model = Model(df)
    model.print_dataset_info()
    model.compile()
    model.split_dataset()
    model.fit(2)
    #model.evaluate()
    #model.evaluate_and_print_predictions()
    model.evaluate_and_show_predictions()