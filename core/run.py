import logging
import os
import sys

from core.data.fetch import Database
from core.util.constants import RAW_DATA_PATH, RAW_LABEL_PATH, DATASET_PATH, LABEL_DATASET_PATH
from core.data.feature import FeatureExtractor
from core.model.model import Model

if __name__ == "__main__":
    sys.path.append(f"{os.getcwd()}{os.sep}core")
    if not os.getcwd().split(os.sep)[-1] == "core":
        os.chdir("core")
    ft_extractor = FeatureExtractor(log_level=logging.DEBUG)
    db = Database(raw_dataset_path=RAW_DATA_PATH,
                  raw_label_path=RAW_LABEL_PATH,
                  label_dataset_path=LABEL_DATASET_PATH,
                  dataset_csv_filename=DATASET_PATH,
                  ft_extractor=ft_extractor,
                  num_rows=50,
                  degrees="all",
                  bfly=["all"])
    df = db.setup_dataset()
    df = db.fetch_images(df, "identifier")
    df = ft_extractor.pre_process(df, "lbp", radius=2)
    model = Model(df)
    # model.load()
    model.print_dataset_info()
    model.compile()
    model.split_dataset()
    # model.fit(5)  # Epochs
    model.save()
    model.evaluate_and_show_predictions()
