import logging
import json

from core.data.fetch import Database
from core.util.constants import RAW_DATA_PATH, RAW_LABEL_PATH, DATASET_PATH, LABEL_DATASET_PATH, IMGDIR_PATH
from core.data.feature import FeatureExtractor
from core.model.model import Model
from core.util.pysetup import PySetup
from core.util.util import setup_argparse

if __name__ == "__main__":
    args = setup_argparse()
    if args is not None:
        json_constants = {
            "MODEL_ID": args.MODEL_ID,
            "KERNEL_SIZE": args.KERNEL_SIZE,
            "LEARNING_RATE": args.LEARNING_RATE,
            "NUM_EPOCHS": args.NUM_EPOCHS,
            "NUM_IMAGES": args.NUM_IMAGES,
            "IMG_SIZE": args.IMG_SIZE,
            "CROPPED": args.CROPPED
        }
        with open('core/util/constants.txt', 'w') as f:
            json.dump(json_constants, f)

    with open('core/util/constants.txt', 'r') as f:
        constants = json.load(f)

    ops = PySetup()
    num_rows = constants["NUM_IMAGES"]
    feature = ""
    ft_extractor = FeatureExtractor(log_level=logging.INFO)
    db = Database(raw_dataset_path=RAW_DATA_PATH,
                  raw_label_path=RAW_LABEL_PATH,
                  label_dataset_path=LABEL_DATASET_PATH,
                  dataset_csv_filename=DATASET_PATH,
                  log_level=logging.DEBUG,
                  ft_extractor=ft_extractor,
                  num_rows=num_rows,
                  crop=constants["CROPPED"],
                  minimum_images=None,
                  degrees="all",
                  bfly=["all"])
    df = db.setup_dataset()
    df = ft_extractor.pre_process(df, feature, radius=2)

    if constants["CROPPED"] == 1:
        df = db.only_accepted(df)

    model = Model(df, IMGDIR_PATH, feature=feature, kernel_size=(constants["KERNEL_SIZE"], constants["KERNEL_SIZE"]))
    #model.load()
    model.compile(constants["LEARNING_RATE"])
    model.split_dataset()
    model.fit(constants["NUM_EPOCHS"])
    model.save()
    model.evaluate()
