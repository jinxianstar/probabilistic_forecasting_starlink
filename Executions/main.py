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
    rain_cols = ["rain_event", "rain_intensity"]
    x_feature_cols = [c for c in feature_cols if c not in rain_cols]

    if "target_mean" in x_feature_cols:
        raise ValueError("❌ feature_cols 不可以包含 target_mean")

    L = Config.LAGGED_VALUE
    ts = df.index

    data_x = df[x_feature_cols].values.astype("float32")
    data_y = df[["target_mean"]].values.astype("float32")

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("base X feature cols:", x_feature_cols)
    print("X 最後兩欄應該是: rain_event[t+1], rain_intensity[t+1]")
    print("=" * 60)

    expected_n_features = len(x_feature_cols) + 2

    if X.shape[-1] != expected_n_features:
        print("❌ X feature 數量錯")
        print("X.shape[-1]:", X.shape[-1])
        print("expected:", expected_n_features)
        print("len(x_feature_cols):", len(x_feature_cols))
        print("+ 2 rain future cols")
        return False

    for k in range(min(n_check, len(X))):
        i = df.index.get_loc(t[k])
        start_i = i - L + 1
        next_i = i + 1

        expected_x_base = data_x[start_i:i + 1]
        expected_y = data_y[i]

        expected_r = df["rain_event"].iloc[next_i]
        expected_ri = df["rain_intensity"].iloc[next_i]

        ok_x_base = np.allclose(X[k, :, :-2], expected_x_base)
        ok_y = np.allclose(y[k], expected_y)

        # rain 只應該放在最後一個 timestep
        ok_r_last = np.isclose(X[k, -1, -2], expected_r)
        ok_ri_last = np.isclose(X[k, -1, -1], expected_ri)

        # 前 L-1 個 timestep 的 rain 欄位應該是 0
        ok_rain_before_zero = np.allclose(X[k, :-1, -2:], 0.0)

        # r / ri 也要跟 X 裡最後一格一致
        ok_r_array = True if r is None else np.isclose(r[k], expected_r)
        ok_ri_array = True if ri is None else np.isclose(ri[k], expected_ri)

        if not (
            ok_x_base
            and ok_y
            and ok_r_last
            and ok_ri_last
            and ok_rain_before_zero
            and ok_r_array
            and ok_ri_array
        ):
            print(f"❌ sample {k} failed")
            print("t:", ts[i])
            print("X base time:", ts[start_i], "~", ts[i])
            print("rain time:", ts[next_i])
            print("ok_x_base:", ok_x_base)
            print("ok_y:", ok_y)
            print("ok_r_last:", ok_r_last)
            print("ok_ri_last:", ok_ri_last)
            print("ok_rain_before_zero:", ok_rain_before_zero)
            print("ok_r_array:", ok_r_array)
            print("ok_ri_array:", ok_ri_array)
            print("X[k, -1, -2]:", X[k, -1, -2])
            print("expected rain_event[t+1]:", expected_r)
            print("X[k, -1, -1]:", X[k, -1, -1])
            print("expected rain_intensity[t+1]:", expected_ri)
            return False

        print(
            f"✅ sample {k} OK | "
            f"base X: {ts[start_i]} ~ {ts[i]} | "
            f"rain in X: {ts[next_i]}"
        )

    print("✅ 檢查通過")
    print("X[:, :, :-2] = 非 rain features 的 t-L+1 ~ t")
    print("X[:, -1, -2] = rain_event[t+1]")
    print("X[:, -1, -1] = rain_intensity[t+1]")
    print("y = target_mean[t]，假設已經是 t+1")
    print("r / ri = rain_event / rain_intensity 的 t+1")

    return True


if __name__ == "__main__":
    df = du.load_data()
    df = du.generate_labels(df)
    df = du.feature_engineering(df)
    modes = ["base", "time", "weather", "time_weather"]
    modes = ["time_weather"]

    algorithm = "normal" # "normal", "de", "multistart"

    add_noise_to_rain_intensity = False
    noise_std_ratio = 0.2
    plot_start = 0
    plot_end = 82
    # Case I.
    case_i_str = ""
    for i in range (1):
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
            X_tr, y_tr, sw_tr, X_va, y_va, X_te, y_te, _masks, X_scaler = du.split_and_scale(
                X, y, r, ri, t
            )


            rain_intensity_va_for_correction = _masks["val_rain_intensity"]
            rain_intensity_te_for_correction = _masks["test_rain_intensity"]
            # Training 

            # print(f"mask: {masks['test_rain']}")
            model = models.train(X_tr, y_tr, sw_tr, X_va, y_va)
            y_pred = model.predict(X_te, verbose=0)


            # ========== Evaluation before correction ==========
            # tuning on validation
            masks_va = {
                "rain": _masks["val_rain"],
                "rain_intensity": rain_intensity_va_for_correction,
            }
            y_pred_va = model.predict(X_va, verbose=0)

            # tuning and optimization via validation
            best_params, best_loss, best_eval_va = optimization.tune_rain_correction_on_validation(
                models=models,
                y_va=y_va,
                y_pred_va=y_pred_va,
                masks_va=masks_va,
                mode=algorithm,  # "normal", "de", "multistart"
                intensity_p90=intensity_p90_train
            )
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

            # ========== Plot ==========plot_prediction_from_predictions, plot_prediction_to_pdf
            models.plot_prediction_to_pdf(
                y_true=y_te,y_pred=y_pred,timestamps=_masks["test_time"], x_axis = "index",
                rain_mask=_masks["test_rain"],title="Before Rain Residual Correction", start=plot_start, end=plot_end)

            models.plot_prediction_to_pdf(y_true=y_te,y_pred=y_pred_adj, x_axis = "index",
                                                    timestamps=_masks["test_time"],rain_mask=_masks["test_rain"],
                                                        title="After Rain Residual Correction", start=plot_start, end=plot_end)

    print("=== Case I. Original Predictions ===")
    print(case_i_str)