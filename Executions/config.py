class Config:
    AGG_FREQ = "1h"
    DATASET = "Osnabrück"   # Enschede; Osnabrück
    LAGGED_VALUE = 24
    EXPECTED_MINUTES = 60
    TRAIN_RATIO = 0.725
    VAL_RATIO = 0.125
    RAIN_WEIGHT = 1.0
    EPOCHS = 100
    BATCH_SIZE = 16