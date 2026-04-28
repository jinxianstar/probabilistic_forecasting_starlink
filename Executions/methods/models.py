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

from datetime import datetime

def plot_prediction_from_predictions(
    y_true,
    y_pred,
    timestamps=None,
    rain_mask=None,
    title="Prediction",
    start=0,
    end=100,
    save_dir=".",
    x_axis="time",   # "time" 或 "index"
):

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred)

    y_true = y_true[start:end]
    y_pred = y_pred[start:end]

    # 決定 x 軸顯示方式
    if x_axis == "time" and timestamps is not None:
        x = pd.to_datetime(timestamps)[start:end]
        use_datetime = True
    else:
        x = np.arange(start, start + len(y_true))
        use_datetime = False

    # 若沒給 rain_mask，就全部設 False
    if rain_mask is None:
        rain_mask = np.zeros(len(y_true), dtype=bool)
    else:
        rain_mask = np.asarray(rain_mask).astype(bool)[start:end]

    q05 = y_pred[:, 0]
    q25 = y_pred[:, 1]
    q75 = y_pred[:, 2]
    q95 = y_pred[:, 3]
    median = (q25 + q75) / 2.0

    plt.figure(figsize=(16, 6))

    # 雨天背景
    rain_labeled = False
    if use_datetime:
        for i in range(len(x)):
            if rain_mask[i]:
                plt.axvspan(
                    x[i] - pd.Timedelta(minutes=30),
                    x[i] + pd.Timedelta(minutes=30),
                    alpha=0.10,
                    label="Rainy period" if not rain_labeled else None
                )
                rain_labeled = True
    else:
        for i in range(len(x)):
            if rain_mask[i]:
                plt.axvspan(
                    x[i] - 0.5,
                    x[i] + 0.5,
                    alpha=0.10,
                    label="Rainy period" if not rain_labeled else None
                )
                rain_labeled = True

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

    # 儲存圖片
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(save_dir, f"{ts}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

    return fig_path


def plot_prediction_from_predictions(
    y_true,
    y_pred,
    timestamps=None,
    rain_mask=None,
    title="Model Prediction Analysis",
    start=0,
    end=100,
    save_dir=".",
    x_axis="time",
):
    # 資料預處理
    y_true = np.asarray(y_true).reshape(-1)[start:end]
    y_pred = np.asarray(y_pred)[start:end]

    if x_axis == "time" and timestamps is not None:
        x = pd.to_datetime(timestamps)[start:end]
        use_datetime = True
    else:
        x = np.arange(0, len(y_true)) # 重置 index 從 0 開始更易讀
        use_datetime = False

    if rain_mask is None:
        rain_mask = np.zeros(len(y_true), dtype=bool)
    else:
        rain_mask = np.asarray(rain_mask).astype(bool)[start:end]

    # 提取分位數
    q05, q25, q75, q95 = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
    median = (q25 + q75) / 2.0

    # 設定繪圖風格
    plt.style.use('seaborn-v0_8-whitegrid') # 使用內建風格讓圖表更乾淨
    plt.figure(figsize=(16, 7))

    # 1. 雨天背景 (改用更明顯的淡藍色，並減少循環次數優化效能)
    rain_indices = np.where(rain_mask)[0]
    if len(rain_indices) > 0:
        first_label = True
        for i in rain_indices:
            width = pd.Timedelta(minutes=30) if use_datetime else 0.5
            plt.axvspan(x[i] - width, x[i] + width, color='skyblue', alpha=0.2, 
                        label="Rainy Period" if first_label else None)
            first_label = False

    # 2. 繪製區間 (使用對比色調)
    plt.fill_between(x, q05, q95, color='royalblue', alpha=0.15, label="90% Prediction Interval")
    plt.fill_between(x, q25, q75, color='royalblue', alpha=0.30, label="50% Prediction Interval")

    # 3. 中心預測線 (深色實線)
    plt.plot(x, median, color='#1f77b4', linewidth=2.5, label="Predicted Median", zorder=3)

    # 4. 真值 (改用帶線的散點，並用顯眼的顏色如橘紅色)
    plt.plot(x, y_true, color='#ff7f0e', marker='o', markersize=4, 
             linewidth=1.5, alpha=0.8, label="Ground Truth", zorder=4)

    # 圖表美化
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Time" if use_datetime else "Index (Steps)", fontsize=12)
    plt.ylabel("Download Mean (Mbps)", fontsize=12)
    
    # 優化圖例
    plt.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
    
    # 優化網格與刻度
    plt.grid(True, linestyle='--', alpha=0.6)
    if use_datetime:
        plt.xticks(rotation=35)

    plt.tight_layout()

    # 儲存與顯示
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(save_dir, f"pred_{ts}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    
    plt.show()
    plt.close()

    return fig_path

def plot_prediction_from_predictions(
    y_true,
    y_pred,
    timestamps=None,
    rain_mask=None,
    title="Model Prediction Analysis",
    start=0,
    end=100,
    save_dir=".",
    x_axis="time",
):
    # --- 1. 資料處理 ---
    y_true_plot = np.asarray(y_true).reshape(-1)[start:end]
    y_pred_plot = np.asarray(y_pred)[start:end]

    if x_axis == "time" and timestamps is not None:
        # 確保 timestamps 是 DatetimeIndex 並選取範圍
        x = pd.to_datetime(timestamps)[start:end]
        use_datetime = True
    else:
        x = np.arange(0, len(y_true_plot))
        use_datetime = False

    # 雨天標記處理
    if rain_mask is None:
        rain_mask_plot = np.zeros(len(y_true_plot), dtype=bool)
    else:
        rain_mask_plot = np.asarray(rain_mask).astype(bool)[start:end]

    # 提取分位數 (假設 y_pred 順序為 q05, q25, q75, q95)
    q05 = y_pred_plot[:, 0]
    q25 = y_pred_plot[:, 1]
    q75 = y_pred_plot[:, 2]
    q95 = y_pred_plot[:, 3]
    median = (q25 + q75) / 2.0

    # --- 2. 畫圖設定 ---
    plt.style.use('seaborn-v0_8-whitegrid') # 清爽的網格風格
    fig, ax = plt.subplots(figsize=(16, 7))

    # [背景] 雨天區塊 (淡藍色)
    rain_indices = np.where(rain_mask_plot)[0]
    if len(rain_indices) > 0:
        first_label = True
        for i in rain_indices:
            # 計算區塊寬度，時間模式下給予正負30分鐘的寬度
            width = pd.Timedelta(minutes=30) if use_datetime else 0.5
            ax.axvspan(x[i] - width, x[i] + width, 
                       color='skyblue', alpha=0.25, 
                       label="Rainy Period" if first_label else None,
                       zorder=1)
            first_label = False

    # [預測層] 信心區塊 (使用漸層藍色，zorder 較低)
    ax.fill_between(x, q05, q95, color='#1f77b4', alpha=0.15, label="90% Prediction Interval", zorder=2)
    ax.fill_between(x, q25, q75, color='#1f77b4', alpha=0.30, label="50% Prediction Interval", zorder=2)
    
    # [預測層] 中位數預測線 (深藍色虛線或細實線)
    ax.plot(x, median, color='#1f77b4', linewidth=1.5, linestyle='--', alpha=0.7, label="Predicted Median", zorder=3)

    # [焦點層] 真值點 (圓點點：高對比橘色 + 白色邊框)
    ax.scatter(
        x, y_true_plot, 
        s=50,                 # 點的大小
        c='#FF4500',          # 鮮艷橘紅
        edgecolors='white',   # 白色邊框
        linewidths=1.2,       # 邊框寬度
        label="True Value", 
        zorder=10,            # 確保在最頂層
        alpha=1
    )

    # --- 3. 圖表美化 ---
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Time" if use_datetime else "Index (Steps)", fontsize=12)
    ax.set_ylabel("Download Mean (Mbps)", fontsize=12)
    
    # 優化座標軸
    if use_datetime:
        plt.xticks(rotation=35)
    
    # 圖例：置於右上角並增加陰影
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
    
    # 網格微調
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    # --- 4. 儲存與輸出 ---
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(save_dir, f"forecast_{ts}.png")
    
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return fig_path

def plot_prediction_to_pdf(
    y_true,
    y_pred,
    timestamps=None,
    rain_mask=None,
    title="Model Prediction Analysis",
    start=0,
    end=100,
    save_dir=".",
    x_axis="time",
    fig_width=20,  # 新增：自定義寬度 (英吋)
    fig_height=8   # 新增：自定義高度 (英吋)
):
    # --- 1. 資料處理 ---
    y_true_plot = np.asarray(y_true).reshape(-1)[start:end]
    y_pred_plot = np.asarray(y_pred)[start:end]

    if x_axis == "time" and timestamps is not None:
        x = pd.to_datetime(timestamps)[start:end]
        use_datetime = True
    else:
        x = np.arange(0, len(y_true_plot))
        use_datetime = False

    if rain_mask is None:
        rain_mask_plot = np.zeros(len(y_true_plot), dtype=bool)
    else:
        rain_mask_plot = np.asarray(rain_mask).astype(bool)[start:end]

    q05, q25, q75, q95 = y_pred_plot[:, 0], y_pred_plot[:, 1], y_pred_plot[:, 2], y_pred_plot[:, 3]
    median = (q25 + q75) / 2.0

    # --- 2. 畫圖設定：調整高寬度 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 這裡的 figsize=(fig_width, fig_height) 控制最終 PDF 的比例
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # [背景] 雨天區塊
    rain_indices = np.where(rain_mask_plot)[0]
    if len(rain_indices) > 0:
        first_label = True
        for i in rain_indices:
            width = pd.Timedelta(minutes=30) if use_datetime else 0.5
            ax.axvspan(x[i] - width, x[i] + width, color='skyblue', alpha=0.2, zorder=1,
                       label="Rainy Period" if first_label else None)
            first_label = False

    # [預測層]
    ax.fill_between(x, q05, q95, color='#1f77b4', alpha=0.15, label="90% Interval", zorder=2)
    ax.fill_between(x, q25, q75, color='#1f77b4', alpha=0.30, label="50% Interval", zorder=2)
    # ax.plot(x, median, color='#1f77b4', linewidth=1.5, linestyle='--', alpha=0.7, zorder=3)

    # [焦點層] 圓點點
    ax.scatter(x, y_true_plot, s=60, c='#FF4500', edgecolors='white', linewidths=1.2, 
               label="True Value", zorder=10)

    # --- 3. 美化 ---
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    if use_datetime:
        plt.xticks(rotation=35)
    ax.legend(loc='upper right', frameon=True, shadow=True)

    plt.tight_layout()

    # --- 4. 儲存為 PDF ---
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 關鍵：副檔名改為 .pdf
    fig_path = os.path.join(save_dir, f"forecast_{ts}.pdf")
    
    # 保存時可以不設定 dpi (PDF 是向量圖，無限放大不失真)
    # 但設定 bbox_inches="tight" 確保邊緣不會被切掉
    plt.savefig(fig_path, format='pdf', bbox_inches="tight")
    
    print(f"檔案已儲存至: {fig_path}")
    plt.show()
    plt.close()

    return fig_path

def plot_prediction_to_pdf(
    y_true,
    y_pred,
    timestamps=None,
    rain_mask=None,
    title="Model Prediction Analysis",
    start=0,
    end=100,
    save_dir=".",
    x_axis="time",
    fig_width=8,
    fig_height=6
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime

    # --- 1. 資料處理 ---
    y_true_plot = np.asarray(y_true).reshape(-1)[start:end]
    y_pred_plot = np.asarray(y_pred)[start:end]

    if x_axis == "time" and timestamps is not None:
        x = pd.to_datetime(timestamps)[start:end]
        use_datetime = True
    else:
        x = np.arange(0, len(y_true_plot))
        use_datetime = False

    if rain_mask is None:
        rain_mask_plot = np.zeros(len(y_true_plot), dtype=bool)
    else:
        rain_mask_plot = np.asarray(rain_mask).astype(bool)[start:end]

    q05, q25, q75, q95 = y_pred_plot[:, 0], y_pred_plot[:, 1], y_pred_plot[:, 2], y_pred_plot[:, 3]
    median = (q25 + q75) / 2.0

    # --- 2. 畫圖設定 ---
    plt.style.use('seaborn-v0_8-whitegrid')

    # 全域字體大小
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 20,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12
    })

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # [背景] 雨天區塊
    rain_indices = np.where(rain_mask_plot)[0]
    if len(rain_indices) > 0:
        first_label = True
        for i in rain_indices:
            width = pd.Timedelta(minutes=30) if use_datetime else 0.5
            ax.axvspan(
                x[i] - width, x[i] + width,
                color='skyblue', alpha=0.2, zorder=1,
                label="Rainy Period" if first_label else None
            )
            first_label = False

    # [預測層]
    ax.fill_between(x, q05, q95, color='#1f77b4', alpha=0.15, label="90% Interval", zorder=2)
    ax.fill_between(x, q25, q75, color='#1f77b4', alpha=0.30, label="50% Interval", zorder=2)
    # ax.plot(x, median, color='#1f77b4', linewidth=2.0, linestyle='--', alpha=0.85, zorder=3, label="Median")

    # [真值]
    ax.scatter(
        x, y_true_plot,
        s=75, c='#FF4500', edgecolors='white', linewidths=1.2,
        label="True Value", zorder=10
    )

    # --- 3. 美化 ---
    ax.set_title(title, fontweight='bold', pad=18)
    ax.set_xlabel("Time" if use_datetime else "Index")
    ax.set_ylabel("Value")

    if use_datetime:
        plt.xticks(rotation=30)

    ax.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    # --- 4. 儲存為 PDF ---
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(save_dir, f"forecast_{ts}.pdf")

    plt.savefig(fig_path, format='pdf', bbox_inches="tight")
    print(f"檔案已儲存至: {fig_path}")

    plt.show()
    plt.close()

    return fig_path


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
