import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from config import Config
import methods.data_utils as du
from methods import models
from methods import optimization
from methods import corrections

import itertools

def compute_train_rain_intensity_p90(
    ri,
    train_ratio,
    default=1.0,
    quantile=0.9
):
    """
    用 train split 的 rain_intensity 計算 P90。
    只使用 ri > 0 的雨量強度，避免 0 雨量影響分位數。
    """
    if ri is None:
        return default

    ri = np.asarray(ri).reshape(-1)

    tr = int(len(ri) * train_ratio)
    ri_tr = ri[:tr]

    positive = ri_tr[ri_tr > 0]

    if len(positive) == 0:
        return default

    return np.quantile(positive, quantile)


import numpy as np

def audit_make_dataset_v2(df, feature_cols, X, y, r, ri, t, n_check=10):
    """Check rain-free model inputs and separately aligned DIRC rain data."""
    x_cols = [c for c in feature_cols if c not in ["rain_event", "rain_intensity"]]
    if "target_mean" in x_cols:
        raise ValueError("feature_cols must not contain target_mean")
    if X.ndim != 3 or X.shape[-1] != len(x_cols):
        return False
    for k in range(min(n_check, len(X))):
        i = df.index.get_loc(t[k])
        expected_x = df[x_cols].iloc[i - Config.LAGGED_VALUE + 1:i + 1].to_numpy(dtype="float32")
        if not np.allclose(X[k], expected_x):
            return False
        if not np.allclose(y[k], df["target_mean"].iloc[i]):
            return False
        if r is not None and not np.isclose(r[k], df["rain_event"].iloc[i + 1]):
            return False
        if ri is not None and not np.isclose(ri[k], df["rain_intensity"].iloc[i + 1]):
            return False
    return True


def apply_train_rain_only(
    X_tr, y_tr, sw_tr,
    X_va, y_va, sw_va,
    X_te, y_te, sw_te,
    X_va_infer, X_te_infer,
    rain_intensity_va_for_correction,
    rain_intensity_te_for_correction,
    masks,
    r,
):
    n_val = len(X_va)
    n_test = len(X_te)

    r_tr = r[:len(X_tr)]
    r_va = r[len(X_tr):len(X_tr) + n_val]
    r_te = r[-n_test:]

    train_mask = r_tr > 0
    val_mask = r_va > 0
    test_mask = r_te > 0

    X_tr, y_tr, sw_tr = X_tr[train_mask], y_tr[train_mask], sw_tr[train_mask]
    X_va, y_va, sw_va = X_va[val_mask], y_va[val_mask], sw_va[val_mask]
    X_te, y_te, sw_te = X_te[test_mask], y_te[test_mask], sw_te[test_mask]

    X_va_infer = X_va_infer[val_mask]
    X_te_infer = X_te_infer[test_mask]

    rain_intensity_va_for_correction = rain_intensity_va_for_correction[val_mask]
    rain_intensity_te_for_correction = rain_intensity_te_for_correction[test_mask]

    masks["val_rain"] = masks["val_rain"][val_mask]
    masks["test_rain"] = masks["test_rain"][test_mask]
    masks["val_time"] = masks["val_time"][val_mask]
    masks["test_time"] = masks["test_time"][test_mask]
    masks["val_rain_intensity"] = masks["val_rain_intensity"][val_mask]
    masks["test_rain_intensity"] = masks["test_rain_intensity"][test_mask]

    return (
        X_tr, y_tr, sw_tr,
        X_va, y_va, sw_va,
        X_te, y_te, sw_te,
        X_va_infer, X_te_infer,
        rain_intensity_va_for_correction,
        rain_intensity_te_for_correction,
        masks,
    )

if __name__ == "__main__":
    df = du.load_data()
    df = du.generate_labels(df)
    df = du.feature_engineering(df)
    # modes = ["base", "time", "weather", "time_weather"]
    modes = ["time"]

    algorithm = "normal" # "normal", "de", "multistart"
    plot_start = 0
    plot_end = 82


    # Noise

    add_noise_to_rain_intensity = Config.ADD_NOISE
    noise_std_ratio = Config.NOISE_STD_RATIO
    noise_seed = 42

    # Case I.
    case_i_str = ""
    for i in range (Config.NUMBER_OF_RUNS):
        for mode in modes:
            case_i_str += f"=== Mode: {mode.upper()} ===\n"
            
            # 假冒護得參數？為什麼？因為有些特徵沒有 time rain etc.：
            # intensity_p90_train, _ = get_train_rain_intensity_and_masks(df, du, Config)
            
            # 真正獲得資料：
            current_df, feature_cols = du.set_mode_features(df, mode=mode)
            # print(X[1])
            X, y, r, ri, t = du.make_dataset(current_df, feature_cols)
            if audit_make_dataset_v2(
                df=current_df,
                feature_cols=feature_cols,
                X=X,
                y=y,
                r=r,
                ri=ri,
                t=t,
                n_check=20,
            ) == False:
                raise ValueError("❌ make_dataset 產生的資料有問題，請檢查！")
            else:
                print("✅ make_dataset 產生的資料通過檢查")

            intensity_p90_train = compute_train_rain_intensity_p90(
                ri=ri,
                train_ratio=Config.TRAIN_RATIO
            )
            X_tr, y_tr, sw_tr, \
            X_va, y_va, sw_va, \
            X_te, y_te, sw_te, \
            _masks, X_scaler = du.split_and_scale(
                X, y, r, ri, t
            )
            # Rain is auxiliary DIRC data, never a model input.
            X_va_infer = X_va
            X_te_infer = X_te
            rain_intensity_va_for_correction = _masks["val_rain_intensity"]
            rain_intensity_te_for_correction = _masks["test_rain_intensity"]
            if add_noise_to_rain_intensity:
                rain_intensity_va_for_correction = du.inject_correction_intensity_noise(
                    rain_intensity_va_for_correction, _masks["val_rain"],
                    noise_std_ratio=noise_std_ratio, random_state=42,
                )
                rain_intensity_te_for_correction = du.inject_correction_intensity_noise(
                    rain_intensity_te_for_correction, _masks["test_rain"],
                    noise_std_ratio=noise_std_ratio, random_state=43,
                )

            if Config.TRAIN_RAIN_ONLY:
                (
                    X_tr, y_tr, sw_tr,
                    X_va, y_va, sw_va,
                    X_te, y_te, sw_te,
                    X_va_infer, X_te_infer,
                    rain_intensity_va_for_correction,
                    rain_intensity_te_for_correction,
                    _masks,
                ) = apply_train_rain_only(
                    X_tr, y_tr, sw_tr,
                    X_va, y_va, sw_va,
                    X_te, y_te, sw_te,
                    X_va_infer, X_te_infer,
                    rain_intensity_va_for_correction,
                    rain_intensity_te_for_correction,
                    _masks,
                    r,
                )
            # Training 

            # print(f"mask: {masks['test_rain']}")
            model = models.train(
                X_tr, y_tr, sw_tr,
                X_va, y_va, sw_va
            )
            y_pred = model.predict(X_te_infer, verbose=0)


            # ========== Evaluation before correction ==========
            # tuning on validation
            masks_va = {
                "rain": _masks["val_rain"],
                "rain_intensity": rain_intensity_va_for_correction,
            }
            y_pred_va = model.predict(X_va_infer, verbose=0)

            # tuning and optimization via validation

            
            best_params, best_loss, best_eval_va = optimization.tune_rain_correction_on_validation(
                models=models,
                y_va=y_va,
                y_pred_va=y_pred_va,
                masks_va=masks_va,
                mode=algorithm,
                intensity_p90=intensity_p90_train,
                gate=Config.GATE
            )
            if add_noise_to_rain_intensity:
                print("Noise ON")
                print("val clean ri[:5]:", _masks["val_rain_intensity"][:5])
                print("val noisy ri[:5]:", rain_intensity_va_for_correction[:5])
                print("test clean ri[:5]:", _masks["test_rain_intensity"][:5])
                print("test noisy ri[:5]:", rain_intensity_te_for_correction[:5])
            else:
                print("Noise OFF")
                
            print(f"\n[{algorithm}] best validation loss = {best_loss}")
            print(f"\n[{mode}] best validation params = {best_params}")

            # print("\n=== After correction ===")
            y_pred_adj = corrections.apply_rain_intensity_residual_correction(
                y_pred=y_pred,
                y_true=y_te,
                rain_mask=_masks["test_rain"],
                rain_intensity=rain_intensity_te_for_correction,
                beta_base=best_params["beta_base"],
                beta_rain_gain=best_params["beta_rain_gain"],
                instant_base=best_params["instant_base"],
                instant_gain=best_params["instant_gain"],
                first_base=best_params["first_base"],
                first_gain=best_params["first_gain"],
                max_adjust=best_params["max_adjust"],
                intensity_p90=intensity_p90_train
            )
            eval_before = models.evaluate_from_predictions(
                y_true=y_te,
                y_pred=y_pred,
                rain_mask=_masks["test_rain"]
            )

            eval_after = models.evaluate_from_predictions(
                y_true=y_te,
                y_pred=y_pred_adj,
                rain_mask=_masks["test_rain"]
            )
            # print("\n=== Before correction ===")
            # print(eval_before.round(4))
            case_i_str += f"Before correction:\n{eval_before.round(4)}\n\n"
            # print("\n=== After correction ===")
            # print(eval_after.round(4))
            case_i_str += f"After correction:\n{eval_after.round(4)}\n\n"

            if Config.PLOT == True:
            # ========== Plot ==========plot_prediction_from_predictions, plot_prediction_to_pdf
                models.plot_prediction_to_pdf(
                    y_true=y_te,y_pred=y_pred,timestamps=_masks["test_time"], x_axis = "index",
                    rain_mask=_masks["test_rain"],title="Before Rain Residual Correction", start=plot_start, end=plot_end)

                models.plot_prediction_to_pdf(y_true=y_te,y_pred=y_pred_adj, x_axis = "index",
                                                        timestamps=_masks["test_time"],rain_mask=_masks["test_rain"],
                                                            title="After Rain Residual Correction", start=plot_start, end=plot_end)

    print("=== Case I. Original Predictions ===")
    print(case_i_str)