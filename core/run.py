import logging
import core

from core.data.fetch import Database
from core.util.constants import (RAW_DATA_PATH, RAW_LABEL_PATH, DATASET_PATH, LABEL_DATASET_PATH, IMGDIR_PATH,
                                 NUM_IMAGES, CROPPED, KERNEL_SIZE, NUM_EPOCHS, LEARNING_RATE)
from core.data.feature import FeatureExtractor
from core.model.model import Model
from core.util.pysetup import PySetup
from core.util.util import setup_argparse

if __name__ == "__main__":
    args = setup_argparse()
    core.util.constants.MODEL_ID = args.MODEL_ID
    core.util.constants.KERNEL_SIZE = args.KERNEL_SIZE
    core.util.constants.LEARNING_RATE = args.LEARNING_RATE
    core.util.constants.NUM_EPOCHS = args.NUM_EPOCHS
    core.util.constants.NUM_IMAGES = args.NUM_IMAGES
    core.util.constants.IMG_SIZE = args.IMG_SIZE
    core.util.constants.CROPPED = args.CROPPED

    ops = PySetup()
    num_rows = NUM_IMAGES
    feature = ""
    ft_extractor = FeatureExtractor(log_level=logging.INFO)
    db = Database(raw_dataset_path=RAW_DATA_PATH,
                  raw_label_path=RAW_LABEL_PATH,
                  label_dataset_path=LABEL_DATASET_PATH,
                  dataset_csv_filename=DATASET_PATH,
                  log_level=logging.DEBUG,
                  ft_extractor=ft_extractor,
                  num_rows=num_rows,
                  crop=CROPPED,
                  minimum_images=None,
                  degrees="none",
                  bfly=["all"])
    df = db.setup_dataset()
    df = ft_extractor.pre_process(df, feature, radius=2)

    df = db.only_accepted(df)
    model = Model(df, IMGDIR_PATH, feature=feature, kernel_size=(KERNEL_SIZE, KERNEL_SIZE))
    # model.load()
    # model.print_dataset_info()
    model.compile(LEARNING_RATE)
    model.split_dataset()
    model.fit(NUM_EPOCHS)
    model.save()
    # model.evaluate_and_show_predictions(num_samples=3)
