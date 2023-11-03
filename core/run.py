import logging
from core.data.fetch import fetch_images, setup_dataset
from core.util.constants import RAW_DATA_PATH, RAW_LABEL_PATH, DATASET_PATH, LABEL_DATASET_PATH, IMGDIR_PATH
from core.data.feature import FeatureExtractor
from core.model.model import Model
from ultralytics import YOLO
from yolo import run_yolo
from core.util.pysetup import PySetup

if __name__ == "__main__":
    ops = PySetup()
    df = setup_dataset(raw_dataset_path=RAW_DATA_PATH,
                       raw_label_path=RAW_LABEL_PATH,
                       label_dataset_path=LABEL_DATASET_PATH,
                       dataset_csv_filename=DATASET_PATH,
                       num_rows=20,
                       bfly=["all"])
    fetch_images(df, "identifier")
    model = YOLO("yolo/medium250e.pt")
    run_yolo(model, IMGDIR_PATH)
    ft_extractor = FeatureExtractor(log_level=logging.INFO)
    df = ft_extractor.pre_process(df, "lbp", radius=2, should_bb=True, should_resize=True)
    model = Model(df, path=IMGDIR_PATH)
    # model.load()
    model.print_dataset_info()
    model.compile()
    model.split_dataset()
    model.fit(5)  # Epochs
    model.save()
    model.evaluate_and_show_predictions()
