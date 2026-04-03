import tensorflow as tf
import numpy as np
import pandas as pd
from config import Config
import os
import random

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt



def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def multi_quantile_loss(y_true, y_pred):
    qs = tf.constant([0.05, 0.25, 0.75, 0.95], dtype=tf.float32)
    e = y_true - y_pred
    loss = tf.maximum(qs * e, (qs - 1.0) * e)
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))


def interval_metrics(y_true, lower, upper, alpha):
    """
    alpha:
        0.1 -> 90% interval
        0.5 -> 50% interval
    """
    y_true = np.asarray(y_true).reshape(-1)
    lower = np.asarray(lower).reshape(-1)
    upper = np.asarray(upper).reshape(-1)

    # 1) Coverage
    covered = ((y_true >= lower) & (y_true <= upper)).astype(float)
    coverage = covered.mean()

    # 2) Average Width
    width = np.mean(upper - lower)

    # 3) Winkler Score
    winkler = np.where(
        y_true < lower,
        (upper - lower) + (2 / alpha) * (lower - y_true),
        np.where(
            y_true > upper,
            (upper - lower) + (2 / alpha) * (y_true - upper),
            (upper - lower)
        )
    )
    winkler_mean = np.mean(winkler)

    return coverage, width, winkler_mean

def plot_prediction_from_predictions(
    y_true,
    y_pred,
    timestamps=None,
    rain_mask=None,
    title="Prediction",
    n=1000
):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred)

    y_true = y_true[:n]
    y_pred = y_pred[:n]

    # 若沒給 timestamps，改用整數索引
    if timestamps is None:
        x = np.arange(len(y_true))
        use_datetime = False
    else:
        x = pd.to_datetime(timestamps)[:n]
        use_datetime = True

    # 若沒給 rain_mask，就全部設 False
    if rain_mask is None:
        rain_mask = np.zeros(len(y_true), dtype=bool)
    else:
        rain_mask = np.asarray(rain_mask).astype(bool)[:n]

    q05 = y_pred[:, 0]
    q25 = y_pred[:, 1]
    q75 = y_pred[:, 2]
    q95 = y_pred[:, 3]
    median = (q25 + q75) / 2.0

    plt.figure(figsize=(16, 6))

    # 雨天背景（只有有提供 timestamps 時才適合畫時間區塊）
    if use_datetime:
        for i in range(len(x)):
            if rain_mask[i]:
                plt.axvspan(
                    x[i] - pd.Timedelta(minutes=30),
                    x[i] + pd.Timedelta(minutes=30),
                    alpha=0.10
                )
    else:
        for i in range(len(x)):
            if rain_mask[i]:
                plt.axvspan(i - 0.5, i + 0.5, alpha=0.10)

    # prediction interval
    plt.fill_between(x, q05, q95, alpha=0.20, label="90% interval")
    plt.fill_between(x, q25, q75, alpha=0.35, label="50% interval")

    # 中心預測
    plt.plot(x, median, linewidth=2, label="Prediction")

    # 真值點
    plt.scatter(x, y_true, s=14, label="True")

    plt.title(title)
    plt.xlabel("Time" if use_datetime else "Index")
    plt.ylabel("Download Mean")
    plt.legend()

    if use_datetime:
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def evaluate_quantiles(model, X, y_true):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = model.predict(X, verbose=0)

    q05 = y_pred[:, 0]
    q25 = y_pred[:, 1]
    q75 = y_pred[:, 2]
    q95 = y_pred[:, 3]

    # 90% interval: [q05, q95]
    cov_90, width_90, winkler_90 = interval_metrics(
        y_true, q05, q95, alpha=0.1
    )

    # 50% interval: [q25, q75]
    cov_50, width_50, winkler_50 = interval_metrics(
        y_true, q25, q75, alpha=0.5
    )

    result = pd.DataFrame([{
        "Cov_90": cov_90,
        "Width_90": width_90,
        "Winkler_90": winkler_90,
        "Cov_50": cov_50,
        "Width_50": width_50,
        "Winkler_50": winkler_50,
    }])

    return result

# def evaluate_from_predictions(y_true, y_pred, rain_mask=None):
#     y_true = np.asarray(y_true).reshape(-1)
#     y_pred = np.asarray(y_pred)

#     if rain_mask is not None:
#         rain_mask = np.asarray(rain_mask).astype(bool)

#     def _eval_subset(name, mask):
#         y_s = y_true[mask]
#         p_s = y_pred[mask]

#         if len(y_s) == 0:
#             return {
#                 "Subset": name,
#                 "Count": 0,
#                 "Cov_90": np.nan,
#                 "Width_90": np.nan,
#                 "Winkler_90": np.nan,
#                 "Cov_50": np.nan,
#                 "Width_50": np.nan,
#                 "Winkler_50": np.nan,
#             }

#         q05 = p_s[:, 0]
#         q25 = p_s[:, 1]
#         q75 = p_s[:, 2]
#         q95 = p_s[:, 3]

#         cov_90, width_90, winkler_90 = interval_metrics(y_s, q05, q95, alpha=0.1)
#         cov_50, width_50, winkler_50 = interval_metrics(y_s, q25, q75, alpha=0.5)

#         return {
#             "Subset": name,
#             "Count": len(y_s),
#             "Cov_90": cov_90,
#             "Width_90": width_90,
#             "Winkler_90": winkler_90,
#             "Cov_50": cov_50,
#             "Width_50": width_50,
#             "Winkler_50": winkler_50,
#         }

#     rows = [
#         _eval_subset("all", np.ones(len(y_true), dtype=bool)),
#     ]

#     if rain_mask is not None:
#         rows.extend([
#             _eval_subset("rain", rain_mask),
#             _eval_subset("non_rain", ~rain_mask),
#         ])

#     return pd.DataFrame(rows)

def evaluate_from_predictions(y_true, y_pred, rain_mask=None):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred)

    if rain_mask is not None:
        rain_mask = np.asarray(rain_mask).astype(bool)

    def quantile_loss(y, q_pred, tau):
        e = y - q_pred
        return np.mean(np.maximum(tau * e, (tau - 1) * e))

    def approx_crps_from_quantiles(y, p):
        taus = [0.05, 0.25, 0.75, 0.95]
        q_preds = [p[:, 0], p[:, 1], p[:, 2], p[:, 3]]
        losses = [quantile_loss(y, q, tau) for q, tau in zip(q_preds, taus)]
        return np.mean(losses)

    def _eval_subset(name, mask):
        y_s = y_true[mask]
        p_s = y_pred[mask]

        if len(y_s) == 0:
            return {
                "Subset": name,
                "Count": 0,
                "Cov_90": np.nan,
                "Width_90": np.nan,
                "Winkler_90": np.nan,
                "Cov_50": np.nan,
                "Width_50": np.nan,
                "Winkler_50": np.nan,
                "CRPS_approx": np.nan,
            }

        q05 = p_s[:, 0]
        q25 = p_s[:, 1]
        q75 = p_s[:, 2]
        q95 = p_s[:, 3]

        cov_90, width_90, winkler_90 = interval_metrics(y_s, q05, q95, alpha=0.1)
        cov_50, width_50, winkler_50 = interval_metrics(y_s, q25, q75, alpha=0.5)
        crps_approx = approx_crps_from_quantiles(y_s, p_s)

        return {
            "Subset": name,
            "Count": len(y_s),
            "Cov_90": cov_90,
            "Width_90": width_90,
            "Winkler_90": winkler_90,
            "Cov_50": cov_50,
            "Width_50": width_50,
            "Winkler_50": winkler_50,
            "CRPS_approx": crps_approx,
        }

    rows = [
        _eval_subset("all", np.ones(len(y_true), dtype=bool)),
    ]

    if rain_mask is not None:
        rows.extend([
            _eval_subset("rain", rain_mask),
            _eval_subset("non_rain", ~rain_mask),
        ])

    return pd.DataFrame(rows)


# ==========================================
# 5. Model
# ==========================================
def build_model(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp)
    x = layers.Dropout(0.2)(x)
    # x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    # x = layers.MaxPool1D(2)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(4, name="quantiles")(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss=multi_quantile_loss,
        metrics=[multi_quantile_loss]
    )
    return model

# from tensorflow.keras import layers, models, optimizers
# from tcn import TCN

# def build_model(input_shape):
#     inp = layers.Input(shape=input_shape)

#     x = TCN(
#         nb_filters=64,
#         kernel_size=3,
#         dilations=(1, 2, 4, 8),
#         nb_stacks=1,
#         padding="causal",
#         use_skip_connections=True,
#         dropout_rate=0.2,
#         return_sequences=False,
#         activation="relu",
#         name="tcn"
#     )(inp)

#     x = layers.Dense(64, activation="relu")(x)
#     x = layers.Dropout(0.2)(x)
#     out = layers.Dense(4, name="quantiles")(x)

#     model = models.Model(inputs=inp, outputs=out)
#     model.compile(
#         optimizer=optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
#         loss=multi_quantile_loss,
#         metrics=[multi_quantile_loss]
#     )
#     return model

# ==========================================
# 6. Training Runner
# ==========================================
def train(X_train, y_train, sample_weights, X_validation, y_validation):
    model = build_model((X_train.shape[1], X_train.shape[2]))

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        validation_data=(X_validation, y_validation),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )
    return model

def interval_metrics(y_true, lower, upper, alpha):
    y_true = np.asarray(y_true).reshape(-1)
    lower = np.asarray(lower).reshape(-1)
    upper = np.asarray(upper).reshape(-1)

    covered = ((y_true >= lower) & (y_true <= upper)).astype(float)
    coverage = covered.mean() if len(covered) > 0 else np.nan

    width = np.mean(upper - lower) if len(lower) > 0 else np.nan

    winkler = np.where(
        y_true < lower,
        (upper - lower) + (2.0 / alpha) * (lower - y_true),
        np.where(
            y_true > upper,
            (upper - lower) + (2.0 / alpha) * (y_true - upper),
            (upper - lower)
        )
    )
    winkler_mean = np.mean(winkler) if len(winkler) > 0 else np.nan

    return coverage, width, winkler_mean

def apply_rain_residual_correction(
    y_pred,
    y_true,
    rain_mask,
    beta=0.5,
    max_adjust=0.5,
    instant_rain_adjust=0.06,
    first_rain_extra=0.08
):
    """
    y_pred: (N, 4)
    y_true: (N,)
    rain_mask: (N,)

    beta: 用上一期高估比例做動態修正的強度
    max_adjust: 動態修正最大比例
    instant_rain_adjust: 只要這一期是雨天，就先立即下修
    first_rain_extra: 如果是「剛進入雨天」的第一個點，再多下修一點
    """
    y_true = np.asarray(y_true).reshape(-1)
    rain_mask = np.asarray(rain_mask).astype(bool)
    y_pred_adj = np.asarray(y_pred).copy()

    for i in range(len(y_pred_adj)):
        total_adjust = 0.0

        # 1) 只要當前是雨天，先立即下修
        if rain_mask[i]:
            total_adjust += instant_rain_adjust

            # 2) 如果是剛開始下雨的第一個點，再多修一點
            if i > 0 and (not rain_mask[i - 1]):
                total_adjust += first_rain_extra

        # 3) 如果前一期和這一期都雨天，再加上 residual correction
        if i > 0 and rain_mask[i - 1] and rain_mask[i]:
            prev_center = (y_pred_adj[i - 1, 1] + y_pred_adj[i - 1, 2]) / 2.0
            actual_prev = y_true[i - 1]

            over_ratio = max(prev_center - actual_prev, 0.0) / max(abs(prev_center), 1e-6)
            dynamic_adjust = min(beta * over_ratio, max_adjust)

            total_adjust += dynamic_adjust

        # 限制總修正不要太誇張
        total_adjust = min(total_adjust, max_adjust)

        y_pred_adj[i] = y_pred_adj[i] * (1.0 - total_adjust)

    return y_pred_adj

def apply_rain_intensity_residual_correction(
    y_pred,
    y_true,
    rain_mask,
    rain_intensity,   # 建議傳 log1p 後的強度
    beta_base=0.4,    # 基本 residual correction 強度
    beta_rain_gain=0.8,   # 雨越大，beta 再往上加多少
    instant_base=0.00,    # 小雨時的立即下修
    instant_gain=0.12,    # 雨越大，立即下修再增加多少
    first_base=0.04,      # 剛開始下雨的額外修正
    first_gain=0.10,      # 雨越大，first rain 額外修正再增加多少
    max_adjust=0.5,
    intensity_p90=None    # 用來正規化雨勢
):
    y_true = np.asarray(y_true).reshape(-1)
    rain_mask = np.asarray(rain_mask).astype(bool)
    rain_intensity = np.asarray(rain_intensity).reshape(-1)
    y_pred_adj = np.asarray(y_pred).copy()

    # 用高分位數把 rain_intensity 壓到大概 0~1
    # 為什麼要壓？因為我們不希望雨勢強度的絕對值影響修正強度，而是希望它在 0~1 的相對位置決定修正強度
    # 這樣不會有leakage? 需要改training set; 不然有可能distribution leakage.
    if intensity_p90 is None:
        positive = rain_intensity[rain_intensity > 0]
        intensity_p90 = np.quantile(positive, 0.9) if len(positive) > 0 else 1.0

    # 到0.0~1.1.0的比例尺，雨越大越接近1，雨越小越接近0
    intensity_scale = np.clip(rain_intensity / max(intensity_p90, 1e-6), 0.0, 1.0)

    for i in range(len(y_pred_adj)):
        total_adjust = 0.0
        s = intensity_scale[i]   # 當前雨勢強度（0~1）

        if rain_mask[i]:
            # 1) 立即修正：雨越大，下修越多
            instant_adjust = instant_base + instant_gain * s # instant_gain = 第一步修正
            total_adjust += instant_adjust

            # 2) 剛進入雨天第一點：雨越大，多修更多; 
            if i > 0 and (not rain_mask[i - 1]): # first_gain = 第二步修正，第一次修正多
                first_adjust = first_base + first_gain * s
                total_adjust += first_adjust

        # 3) 連續雨天：上一期高估時，再根據當前/前一期雨勢放大修正
        if i > 0 and rain_mask[i - 1] and rain_mask[i]:
            # 75 ~ 25 算中心點
            prev_center = (y_pred_adj[i - 1, 1] + y_pred_adj[i - 1, 2]) / 2.0
            actual_prev = y_true[i - 1]
            # 算出超出了多少？超出比例越大，修正越多
            over_ratio = max(prev_center - actual_prev, 0.0) / max(abs(prev_center), 1e-6)

            s_prev = intensity_scale[i - 1]
            s_pair = max(s, s_prev)   # 或者用 (s + s_prev)/2
            # 只要這兩個相鄰時刻中有一個雨勢很強，就把這段視為強雨情境。

            beta_eff = beta_base + beta_rain_gain * s_pair # beta_rain_gain = 第三步修正，雨越大，residual correction越強
            # # 基本 residual correction 強度 beta_base + 雨勢放大後的額外強度 beta_rain_gain * s_pair
            # beta_rain_gain  雨越大，beta 再往上加多少
            dynamic_adjust = min(beta_eff * over_ratio, max_adjust)
            total_adjust += dynamic_adjust

        total_adjust = min(total_adjust, max_adjust)
        y_pred_adj[i] = y_pred_adj[i] * (1.0 - total_adjust)

    return y_pred_adj