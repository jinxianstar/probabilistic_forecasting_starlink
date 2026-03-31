import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
#from Executions.config import Config
import methods.data_utils as du
from methods import models

if __name__ == "__main__":
    df = du.load_data()
    df = du.generate_labels(df)
    df = du.feature_engineering(df)
    df, feature_cols = du.set_mode_features(df, mode="time_weather")

    X, y, r, ri, timestamps = du.make_dataset(df, feature_cols)

    X_tr, y_tr, sw_tr, X_va, y_va, X_te, y_te, masks, X_scaler = du.split_and_scale(
        X, y, r, ri, timestamps
    )


    model = models.train(X_tr, y_tr, sw_tr, X_va, y_va)

    # 原始預測
    y_pred = model.predict(X_te, verbose=0)
    # y_pred_adj = models.apply_rain_residual_correction(
    #     y_pred=y_pred,
    #     y_true=y_te,
    #     rain_mask=masks["test_rain"],
    #     beta=0.8, # 用前一期的高估误差来做动态修正时，修正强度有多大; 前一期预测中心值比真实值高出多少比例
    #     max_adjust=0.5, # 不管怎么修正，最大修正幅度不能超过 50%; 
    #     instant_rain_adjust=0.00, # 只要当前这一期是雨天，先立刻下修 0%; 
    #     first_rain_extra=0.10 # 如果这一期是“刚开始下雨”的第一个点，再额外多下修 8%
    # )
    y_pred_adj = models.apply_rain_intensity_residual_correction(
        y_pred=y_pred,
        y_true=y_te,
        rain_mask=masks["test_rain"],
        rain_intensity=masks["test_rain_intensity"],   # 或 test_rain_amount
        beta_base=0.35,
        beta_rain_gain=0.60,
        instant_base=0.00,
        instant_gain=0.05,
        first_base=0.02,
        first_gain=0.05,
        max_adjust=0.40
    )

    # ========== Evaluation ==========
    eval_before = models.evaluate_from_predictions(
        y_true=y_te,
        y_pred=y_pred,
        rain_mask=masks["test_rain"]
    )

    eval_after = models.evaluate_from_predictions(
        y_true=y_te,
        y_pred=y_pred_adj,
        rain_mask=masks["test_rain"]
    )

    print("\n=== Before correction ===")
    print(eval_before.round(4))

    print("\n=== After correction ===")
    print(eval_after.round(4))

    # 如果你想存檔
    eval_before.round(4).to_csv("eval_before.csv", index=False)
    eval_after.round(4).to_csv("eval_after.csv", index=False)

    # ========== Plot ==========
    models.plot_prediction_from_predictions(
        y_true=y_te,
        y_pred=y_pred,
        timestamps=masks["test_time"],
        rain_mask=masks["test_rain"],
        title="Before Rain Residual Correction",
    )

    models.plot_prediction_from_predictions(
        y_true=y_te,
        y_pred=y_pred_adj,
        timestamps=masks["test_time"],
        rain_mask=masks["test_rain"],
        title="After Rain Residual Correction",
    )