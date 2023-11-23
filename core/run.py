import logging

from core.data.fetch import Database
from core.util.constants import RAW_DATA_PATH, RAW_LABEL_PATH, DATASET_PATH, LABEL_DATASET_PATH, IMGDIR_PATH
from core.data.feature import FeatureExtractor
from core.model.model import Model
from core.util.pysetup import PySetup
from core.util.util import ConstantSingleton

if __name__ == "__main__":
    constants = ConstantSingleton()
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
                  minimum_images=False,
                  degrees="none",
                  bfly=["all"])
    df = db.setup_dataset()
    df = ft_extractor.pre_process(df, feature, radius=2)
    df = db.only_accepted(df)
    model = Model(df, IMGDIR_PATH, feature=feature, kernel_size=(constants["KERNEL_SIZE"], constants["KERNEL_SIZE"]))
    # model.load()
    # model.print_dataset_info()
    model.compile(constants["LEARNING_RATE"])
    model.split_dataset()
    model.fit(constants["NUM_EPOCHS"])
    model.save()
    # model.evaluate_and_show_predictions(num_samples=3)
