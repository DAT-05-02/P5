RAW_DATA_PATH="leopidotera-dk/multimedia.txt"
RAW_LABEL_PATH="leopidotera-dk/occurrence.txt"
DATASET_PATH="leopidotera-dk.csv"
DIRNAME_DELIM = "-"
IMGDIR_PATH = "image-db/"
LABEL_DATASET_PATH = "occurrence.csv"
MERGE_COLS = ['genericName', 'species', 'family', 'stateProvince', 'gbifID', 'identifier', 'format', 'created',
              'iucnRedListCategory', 'lifeStage', 'sex']
BFLY_FAMILY = ['Pieridae', 'Papilionidae', 'Lycaenidae', 'Riodinidae', 'Nymphalidae', 'Hesperiidae', 'Hedylidae']
BFLY_LIFESTAGE = ['Pupa', 'Caterpillar', 'Larva']
FEATURE_DIR_PATH = "image"
PATH_SEP = "/"
MODEL_CHECKPOINT_PATH: str = "modelcheckpoint/"
FULL_MODEL_CHECKPOINT_PATH: str = MODEL_CHECKPOINT_PATH + "model.ckpt"

RAW_WORLD_DATA_PATH="leopidotera-world/multimedia.txt"
RAW_WORLD_LABEL_PATH="leopidotera-world/occurrence-world.csv"

MODEL_ID = -1
KERNEL_SIZE = -1
LEARNING_RATE = -0.1
NUM_EPOCHS = -1
NUM_IMAGES = -1
IMG_SIZE = -1
CROPPED = False
