import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
#from Executions.config import Config
import methods.data_utils as du
from methods import models
import itertools



def correction_objective(eval_df):
    row_all = eval_df[eval_df["Subset"] == "all"].iloc[0]
    row_rain = eval_df[eval_df["Subset"] == "rain"].iloc[0]

    loss = 0.7 * row_rain["CRPS_approx"] + 0.3 * row_all["CRPS_approx"]
    loss += 1.5 * abs(row_rain["Cov_90"] - 0.90)
    return loss


def tune_rain_correction_on_validation(models, y_va, y_pred_va, masks_va):
    param_grid = {
        "beta_base":      [0.20, 0.35, 0.50],
        "beta_rain_gain": [0.40, 0.60, 0.80],
        "instant_base":   [0.00, 0.02],
        "instant_gain":   [0.03, 0.05, 0.08],
        "first_base":     [0.00, 0.02, 0.05],
        "first_gain":     [0.03, 0.05, 0.08],
        "max_adjust":     [0.20, 0.30, 0.40],
    }

    keys = list(param_grid.keys())
    best_loss = np.inf
    best_params = None
    best_eval = None

    for values in itertools.product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, values))

        y_pred_adj_va = models.apply_rain_intensity_residual_correction(
            y_pred=y_pred_va,
            y_true=y_va,
            rain_mask=masks_va["rain"],
            rain_intensity=masks_va["rain_intensity"],
            **params
        )

        eval_va = models.evaluate_from_predictions(
            y_true=y_va,
            y_pred=y_pred_adj_va,
            rain_mask=masks_va["rain"]
        )

        loss = correction_objective(eval_va)

        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_eval = eval_va.copy()

    return best_params, best_loss, best_eval


if __name__ == "__main__":
    df = du.load_data()
    df = du.generate_labels(df)
    df = du.feature_engineering(df)
    modes = ["base", "time", "weather", "time_weather"]
    # modes = ["base"]

    # Case I.
    case_i_str = ""
    for i in range (1):
        for mode in modes:
            case_i_str += f"=== Mode: {mode.upper()} ===\n"
            
            # 訓練, 預測
            masks = du.get_masks(df, mode=modes[3])
            current_df, feature_cols = du.set_mode_features(df, mode=mode)

            X, y, r, ri, t = du.make_dataset(current_df, feature_cols)

            X_tr, y_tr, sw_tr, X_va, y_va, X_te, y_te, _masks, X_scaler = du.split_and_scale(
                X, y, r, ri, t
            )
            # print(f"mask: {masks['test_rain']}")
            model = models.train(X_tr, y_tr, sw_tr, X_va, y_va)
            y_pred = model.predict(X_te, verbose=0)


            # ========== Evaluation before correction ==========
            eval = models.evaluate_from_predictions(
                y_true=y_te,
                y_pred=y_pred,
                rain_mask=masks["test_rain"]
            )

            # tuning on validation
            masks_va = {
                "rain": masks["val_rain"],
                "rain_intensity": masks["val_rain_intensity"],
            }
            y_pred_va = model.predict(X_va, verbose=0)

            best_params, best_loss, best_eval_va = tune_rain_correction_on_validation(
                models=models,
                y_va=y_va,
                y_pred_va=y_pred_va,
                masks_va=masks_va
            )

            # print(f"\n[{mode}] best validation params = {best_params}")
            # print(f"[{mode}] best validation loss = {best_loss:.4f}")
            # print(best_eval_va.round(4))

            # y_pred_adj = models.apply_rain_intensity_residual_correction(
            #     y_pred=y_pred,
            #     y_true=y_te,
            #     rain_mask=masks["test_rain"],
            #     rain_intensity=masks["test_rain_intensity"],   # 或 test_rain_amount
            #     beta_base=0.35,
            #     beta_rain_gain=0.60,
            #     instant_base=0.00,
            #     instant_gain=0.05,
            #     first_base=0.02,
            #     first_gain=0.05,
            #     max_adjust=0.40
            # )
            # print("\n=== After correction ===")
            y_pred_adj = models.apply_rain_intensity_residual_correction(
                y_pred=y_pred,
                y_true=y_te,
                rain_mask=masks["test_rain"],
                rain_intensity=masks["test_rain_intensity"],   # 或 test_rain_amount
                beta_base=best_params["beta_base"],
                beta_rain_gain=best_params["beta_rain_gain"],
                instant_base=best_params["instant_base"],
                instant_gain=best_params["instant_gain"],
                first_base=best_params["first_base"],
                first_gain=best_params["first_gain"],
                max_adjust=best_params["max_adjust"]
            )
            eval_before = models.evaluate_from_predictions(y_true=y_te, y_pred=y_pred, rain_mask=masks["test_rain"])
            eval_after = models.evaluate_from_predictions(y_true=y_te, y_pred=y_pred_adj, rain_mask=masks["test_rain"])

            print("\n=== Before correction ===")
            print(eval_before.round(4))
            case_i_str += f"Before correction:\n{eval_before.round(4)}\n\n"
            print("\n=== After correction ===")
            print(eval_after.round(4))
            case_i_str += f"After correction:\n{eval_after.round(4)}\n\n"

            # #     # ========== Plot ==========
            # models.plot_prediction_from_predictions(y_true=y_te,y_pred=y_pred,timestamps=masks["test_time"],rain_mask=masks["test_rain"],title="Before Rain Residual Correction",)

            # models.plot_prediction_from_predictions(y_true=y_te,y_pred=y_pred_adj,timestamps=masks["test_time"],rain_mask=masks["test_rain"],title="After Rain Residual Correction",)


    #     # ========== Plot ==========
    models.plot_prediction_from_predictions(y_true=y_te,y_pred=y_pred,timestamps=masks["test_time"],rain_mask=masks["test_rain"],title="Before Rain Residual Correction",)

    models.plot_prediction_from_predictions(y_true=y_te,y_pred=y_pred_adj,timestamps=masks["test_time"],rain_mask=masks["test_rain"],title="After Rain Residual Correction",)

    print("=== Case I. Original Predictions ===")
    print(case_i_str)

    # y_pred_adj = models.apply_rain_intensity_residual_correction(
    #     y_pred=y_pred,
    #     y_true=y_te,
    #     rain_mask=masks["test_rain"],
    #     rain_intensity=masks["test_rain_intensity"],   # 或 test_rain_amount
    #     beta_base=0.35,
    #     beta_rain_gain=0.60,
    #     instant_base=0.00,
    #     instant_gain=0.05,
    #     first_base=0.02,
    #     first_gain=0.05,
    #     max_adjust=0.40
    # )

    # # ========== Evaluation ==========
    # eval_before = models.evaluate_from_predictions(
    #     y_true=y_te,
    #     y_pred=y_pred,
    #     rain_mask=masks["test_rain"]
    # )

    # eval_after = models.evaluate_from_predictions(
    #     y_true=y_te,
    #     y_pred=y_pred_adj,
    #     rain_mask=masks["test_rain"]
    # )

    # print("\n=== Before correction ===")
    # print(eval_before.round(4))

    # print("\n=== After correction ===")
    # print(eval_after.round(4))

    # # 如果你想存檔
    # eval_before.round(4).to_csv("eval_before.csv", index=False)
    # eval_after.round(4).to_csv("eval_after.csv", index=False)

    # # ========== Plot ==========
    # models.plot_prediction_from_predictions(
    #     y_true=y_te,
    #     y_pred=y_pred,
    #     timestamps=masks["test_time"],
    #     rain_mask=masks["test_rain"],
    #     title="Before Rain Residual Correction",
    # )

    # models.plot_prediction_from_predictions(
    #     y_true=y_te,
    #     y_pred=y_pred_adj,
    #     timestamps=masks["test_time"],
    #     rain_mask=masks["test_rain"],
    #     title="After Rain Residual Correction",
    # )