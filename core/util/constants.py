RAW_DATA_PATH="leopidotera-dk/multimedia.txt"
RAW_LABEL_PATH="leopidotera-dk/occurrence.txt"
DATASET_PATH="leopidotera-dk.csv"
IMGDIR_PATH = "image_db/"
LABEL_DATASET_PATH = "occurrence.csv"
MERGE_COLS = ['genericName', 'species', 'family', 'stateProvince', 'gbifID', 'identifier', 'format', 'created',
              'iucnRedListCategory', 'lifeStage', 'sex']
BFLY_FAMILY = ['Pieridae', 'Papilionidae', 'Lycaenidae', 'Riodinidae', 'Nymphalidae', 'Hesperiidae', 'Hedylidae']
BFLY_LIFESTAGE = ['Pupa', 'Caterpillar', 'Larva']
FEATURE_DIR_PATH = "image"
WINDOWS_COMPLIANT_OS_SEP = "/"
MODEL_CHECKPOINT_PATH: str = "modelcheckpoint/"
FULL_MODEL_CHECKPOINT_PATH: str = MODEL_CHECKPOINT_PATH + "model.ckpt"
