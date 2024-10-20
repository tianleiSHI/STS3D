class Config():
    NAME = 'monai'
    FOLD = 1
    FOLD_NUM = 10
    FOLD_SEED = 111
    WEIGHT_DIR = 'weights/'

    MODEL_TYPE = 'custom'
    ENCODER_CHANNELS = []
    ENCODER_NAME = 'resnet10'
    IN_CHANNEL = 1
    OFFSET_CHANNEL = 3
    PRETRAINED = True
    DECODER_CHANNEL = [128, 64, 32, 16]
    WITH_COORDS = False

    SPATIAL_SIZE = (384, 384, 384)
    TRAIN_TYPE = 'seg'
    MAX_INST_NUM = 60
    MAX_SPATIAL_SIZE = 384
    SIGMA = 2

    DATA_ROOT = '' # change to your data root

    TRAIN_DIR = [
        'sts24',]
    VAL_DIR = None

    DATASET_TYPE = 'online'

    MODE = 'multilabel'
    SUFFIX = 'pkl'

    LOAD_FROM = None
    RESUME_FROM = None

    EPOCHS = 300
    LR = 1e-4
    LIMIT = 1.
    BATCH_SIZE = 1
    VERSION = None

    NUM_WORKERS = 2

    LOSS_TYPES = [
        ('bce', 1.),
        # ('focal', 10.),
        ('dice', 1.),
        # ('dice_focal', 1.),
    ]

    GPUS = 1

    ORIG_LABELS = None
    TARGET_LABELS = None
    SHOW = False

    # info
    def get_snapshot(self):
        return {k: self.__getattribute__(k) for k in self.__dir__() if not (k.startswith('__') or k == 'get_snapshot')}

