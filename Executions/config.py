class Config:
    AGG_FREQ = "1h"
    MODEL_TYPE = "CNNLSTM"  # TCN; CNNLSTM; LSTM


    # DATASET = "Osnabrück"   # Enschede; Osnabrück
    # TRAIN_RATIO = 0.725
    # VAL_RATIO = 0.125

    DATASET = "Enschede"   # Enschede; Osnabrück
    TRAIN_RATIO = 0.725
    VAL_RATIO = 0.125


    LAGGED_VALUE = 24
    EXPECTED_MINUTES = 60



    EPOCHS = 100
    BATCH_SIZE = 16


    RAIN_WEIGHT = 1.0
    ADD_NOISE = False
    NOISE_STD_RATIO = 1.50
    PLOT = False
    TRAIN_RAIN_ONLY=False
    NUMBER_OF_RUNS=5
    GATE = False