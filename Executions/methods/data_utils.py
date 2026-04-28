import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from config import Config

def load_data():
    df = pd.read_csv(f"./Data/Wetlink/iperf_cleaned_seconds_{Config.DATASET}.csv")
    if "timestamp_start" in df.columns: df.set_index("timestamp_start", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df[["download", "upload"]] / 1e6
    traffic_h = df.resample(Config.AGG_FREQ).agg({
        "download": [lambda x: x.quantile(0.95), lambda x: x.quantile(0.05), "mean", "std"],
        "upload":   ["mean"],
    })
    traffic_h.columns = ["download_q95", "download_q05", "download_mean", "download_std", "upload_mean"]
    traffic_h.replace([np.inf, -np.inf], np.nan, inplace=True)
    traffic_h.dropna(inplace=True)

    wdf = pd.read_csv(f"./Data/Wetlink/analysis_data_{Config.DATASET}.csv")
    if "timestamp_start" in wdf.columns: wdf.set_index("timestamp_start", inplace=True)
    wdf.index = pd.to_datetime(wdf.index)
    wdf.sort_index(inplace=True)

    for c in ["temp", "windspeed", "rain"]:
        if c not in wdf.columns: wdf[c] = np.nan
    weather_h = wdf.resample(Config.AGG_FREQ).agg({
        "temp": "mean", "windspeed": "mean", "rain": "sum"
    })
    weather_h.columns = ["temp_mean", "windspeed_mean", "rain_sum"]
    
    df = traffic_h.join(weather_h, how="left")
    df["temp_mean"] = df["temp_mean"].ffill()#.bfill()
    df["windspeed_mean"] = df["windspeed_mean"].ffill()#.bfill()
    df["rain_sum"] = df["rain_sum"].fillna(0.0)
    
    # 會生成 download_q95, download_q05, download_mean, download_std, upload_mean, temp_mean, windspeed_mean, rain_sum
    # 這些只是特徵，預測的主要還是 download_mean。
    return df

def generate_labels(df):
    df["target_mean"] = df["download_mean"].shift(-1)
    df.dropna(subset=["target_mean"], inplace=True)
    return df

# def feature_engineering(df):
#     df = df.copy()

#     df["hod_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
#     df["hod_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
#     df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
#     df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

#     # 時間特徵 shift.（有沒有都可以）
#     df["hod_sin_next"] = df["hod_sin"].shift(-1)
#     df["hod_cos_next"] = df["hod_cos"].shift(-1)
#     df["dow_sin_next"] = df["dow_sin"].shift(-1)
#     df["dow_cos_next"] = df["dow_cos"].shift(-1)

#     # 天氣特徵 shift.
#     df["temp_next"] = df["temp_mean"].shift(-1).ffill().bfill()
#     df["wind_next"] = df["windspeed_mean"].shift(-1).ffill().bfill()
#     df["rain_next"] = df["rain_sum"].shift(-1).fillna(0.0)

#     df["temp_d1"]  = df["temp_next"] - df["temp_mean"]
#     df["wind_d1"]  = df["wind_next"] - df["windspeed_mean"]

#     df["rain_event"] = (df["rain_next"] > 0.0).astype("float32")
    
#     # intensity, 這樣模型比較容易學，不會被少數極端值主導。
#     df["rain_intensity"] = np.log1p(df["rain_next"].clip(lower=0.0))

#     df.dropna(inplace=True)
#     return df

def feature_engineering(df):
    df = df.copy()

    # ===== Time one-hot =====
    hour = df.index.hour
    dow = df.index.dayofweek

    hour_ohe = pd.get_dummies(hour, prefix="hod", dtype="float32")
    dow_ohe = pd.get_dummies(dow, prefix="dow", dtype="float32")

    # 確保欄位固定完整：24 小時、7 天
    hour_cols = [f"hod_{i}" for i in range(24)]
    dow_cols = [f"dow_{i}" for i in range(7)]

    hour_ohe = hour_ohe.reindex(columns=hour_cols, fill_value=0.0)
    dow_ohe = dow_ohe.reindex(columns=dow_cols, fill_value=0.0)

    hour_ohe.index = df.index
    dow_ohe.index = df.index

    df = pd.concat([df, hour_ohe, dow_ohe], axis=1)

    # ===== 下一時間點 one-hot =====
    for c in hour_cols:
        df[f"{c}"] = df[c]
    for c in dow_cols:
        df[f"{c}"] = df[c]

    # ===== Weather features =====
    # df["temp_next"] = df["temp_mean"].ffill() #.shift(-1).ffill()#.bfill()
    # df["wind_next"] = df["windspeed_mean"].ffill()#.bfill().   .shift(-1).ffill()
    df["rain"] = df["rain_sum"].ffill().fillna(0.0)

    # df["temp_d1"] = df["temp_next"] - df["temp_mean"]
    # df["wind_d1"] = df["wind_next"] - df["windspeed_mean"]

    df["rain_event"] = (df["rain"] > 0.0).astype("float32")
    df["rain_intensity"] = np.log1p(df["rain"].clip(lower=0.0))

    df.dropna(inplace=True)
    return df


def inject_rain_noise_intensity_only(
    df,
    rain_col="rain_next",
    intensity_col="rain_intensity",
    noise_std_ratio=0.05,
    min_positive=1e-6,
    random_state=42
):
    """
    只在 rain_col > 0 時加 Gaussian noise，
    用來模擬 rainfall intensity uncertainty，
    不改變 rain occurrence。
    """
    df_noisy = df.copy()
    rng = np.random.default_rng(random_state)

    rain = df_noisy[rain_col].astype(float).to_numpy()
    rainy_mask = rain > 0.0

    noisy_rain = rain.copy()

    # 僅對 rainy samples 加 noise
    std = noise_std_ratio * np.maximum(rain[rainy_mask], min_positive)
    eps = rng.normal(loc=0.0, scale=std, size=rainy_mask.sum())

    noisy_rain[rainy_mask] = rain[rainy_mask] + eps

    # 為了保持 occurrence 不變，原本有雨的點至少維持為正
    noisy_rain[rainy_mask] = np.maximum(noisy_rain[rainy_mask], min_positive)

    # 更新 rain_next
    df_noisy[rain_col] = noisy_rain

    # 重新由 rain_next 計算 rain_intensity
    df_noisy[intensity_col] = np.log1p(df_noisy[rain_col])

    # 不重算 rain_event，保持原本 event 結構
    return df_noisy

def set_mode_features(df, mode: str):
    base = ["download_mean", "upload_mean"]

    time_feats = [f"hod_{i}_next" for i in range(24)] + [f"dow_{i}_next" for i in range(7)]

    weather_feats = ["rain_event", "rain_intensity"]

    if mode == "base":
        feats = base
    elif mode == "time":
        feats = base + time_feats
    elif mode == "weather":
        feats = base + weather_feats
    elif mode == "time_weather":
        feats = base + time_feats + weather_feats
    else:
        raise ValueError(f"Unknown mode: {mode}")

    missing_cols = [c for c in feats if c not in df.columns]
    feature_cols = [c for c in feats if c in df.columns]

    if "target_mean" not in df.columns:
        raise KeyError("'target_mean' not in df.columns")

    if len(feature_cols) == 0:
        raise ValueError(
            f"No usable features found for mode={mode}. "
            f"Expected one of: {feats}"
        )

    if missing_cols:
        print(f"[WARN] mode={mode}, missing columns skipped: {missing_cols}")

    # 重要：
    # feature_cols 是 model input 用的欄位。
    # aux_cols 是不進 model，但保留下來讓 make_dataset() 產生 rain masks / intensity。
    aux_cols = []
    for c in ["rain_event", "rain_intensity"]:
        if c in df.columns and c not in feature_cols:
            aux_cols.append(c)

    df = df[feature_cols + ["target_mean"] + aux_cols]

    return df, feature_cols

# def make_dataset(df, feature_cols):
#     data_x = df[feature_cols].values.astype("float32")
#     data_y = df[["target_mean"]].values.astype("float32")
#     data_rain = df["rain_event"].values.astype("float32")

#     X_list, y_list, r_list, t_list = [], [], [], []
#     L = Config.LAGGED_VALUE
#     ts = df.index

#     for i in range(L, len(data_x)):
#         time_diff = (ts[i] - ts[i-L]).total_seconds() / 60.0
#         if time_diff <= L * Config.EXPECTED_MINUTES + 15:
#             X_list.append(data_x[i-L:i])
#             y_list.append(data_y[i])
#             r_list.append(data_rain[i])
#             t_list.append(ts[i])

#     return (
#         np.array(X_list),
#         np.array(y_list),
#         np.array(r_list),
#         np.array(t_list)
#     )

# def inject_rain_intensity_noise_after_scaling(
#     X,
#     feature_cols,
#     scaler,
#     noise_std_ratio=0.05,
#     rain_event_col="rain_event",
#     rain_intensity_col="rain_intensity",
#     min_positive=1e-6,
#     random_state=42
# ):
#     """
#     對已 scaled 的 X，在 inverse_transform 後對 rain_intensity 加相對 Gaussian noise，
#     再 transform 回 scaled space。
#     """
#     X_noisy = X.copy()
#     rng = np.random.default_rng(random_state)

#     F = X.shape[-1]
#     X_2d = X.reshape(-1, F)

#     # 回到原始 feature 空間
#     X_orig = scaler.inverse_transform(X_2d)

#     if rain_intensity_col not in feature_cols:
#         raise ValueError(f"{rain_intensity_col} not found in feature_cols")
#     ri_idx = feature_cols.index(rain_intensity_col)

#     if rain_event_col not in feature_cols:
#         raise ValueError(f"{rain_event_col} not found in feature_cols")
#     re_idx = feature_cols.index(rain_event_col)

#     rain_event = X_orig[:, re_idx]
#     rain_intensity = X_orig[:, ri_idx]

#     rainy_mask = rain_event > 0

#     std = noise_std_ratio * np.maximum(np.abs(rain_intensity[rainy_mask]), min_positive)
#     eps = rng.normal(loc=0.0, scale=std, size=rainy_mask.sum())

#     X_orig[rainy_mask, ri_idx] = rain_intensity[rainy_mask] + eps

#     # 再轉回 scaled space
#     X_scaled_back = scaler.transform(X_orig).reshape(X.shape)

#     return X_scaled_back


def inject_rain_intensity_noise_after_scaling(
    X,
    feature_cols,
    scaler,
    noise_std_ratio=0.05,
    rain_event_col="rain_event",
    rain_intensity_col="rain_intensity",
    min_positive=1e-6,
    random_state=42,
    return_correction_intensity=False,
    correction_step=-1,
    clip_nonnegative=True,
):
    """
    對已 scaled 的 X，在 inverse_transform 後對 rain_intensity 加相對 Gaussian noise，
    再 transform 回 scaled space。

    如果 return_correction_intensity=True，額外回傳每個 sample 指定 lag step 的
    noisy 原尺度 rain_intensity，給 correction 使用。

    correction_step=-1 代表取 input window 最後一個時間點。
    """
    rng = np.random.default_rng(random_state)

    original_shape = X.shape
    F = X.shape[-1]
    X_2d = X.reshape(-1, F)

    # 回到原始 feature 空間
    X_orig = scaler.inverse_transform(X_2d)

    if rain_intensity_col not in feature_cols:
        raise ValueError(f"{rain_intensity_col} not found in feature_cols")
    ri_idx = feature_cols.index(rain_intensity_col)

    if rain_event_col not in feature_cols:
        raise ValueError(f"{rain_event_col} not found in feature_cols")
    re_idx = feature_cols.index(rain_event_col)

    rain_event = X_orig[:, re_idx]
    rain_intensity = X_orig[:, ri_idx]

    rainy_mask = rain_event > 0

    std = noise_std_ratio * np.maximum(
        np.abs(rain_intensity[rainy_mask]),
        min_positive
    )
    eps = rng.normal(loc=0.0, scale=std, size=rainy_mask.sum())

    X_orig[rainy_mask, ri_idx] = rain_intensity[rainy_mask] + eps

    if clip_nonnegative:
        X_orig[:, ri_idx] = np.maximum(X_orig[:, ri_idx], 0.0)

    # 先保留 noisy 原尺度版本
    X_orig_noisy_3d = X_orig.reshape(original_shape)

    # 再轉回 scaled space 給 model
    X_scaled_back = scaler.transform(X_orig).reshape(original_shape)

    if return_correction_intensity:
        rain_intensity_for_correction = X_orig_noisy_3d[:, correction_step, ri_idx]
        return X_scaled_back, rain_intensity_for_correction

    return X_scaled_back

# def make_dataset(df, feature_cols):
#     data_x = df[feature_cols].values.astype("float32")
#     data_y = df[["target_mean"]].values.astype("float32")

#     has_rain = "rain_event" in df.columns
#     has_rain_intensity = "rain_intensity" in df.columns

#     data_rain = df["rain_event"].values.astype("float32") if has_rain else None
#     data_rain_intensity = df["rain_intensity"].values.astype("float32") if has_rain_intensity else None

#     X_list, y_list, r_list, ri_list, t_list = [], [], [], [], []
#     L = Config.LAGGED_VALUE
#     ts = df.index

#     for i in range(L, len(data_x)):
#         time_diff = (ts[i] - ts[i - L]).total_seconds() / 60.0
#         if time_diff <= L * Config.EXPECTED_MINUTES + 15:
#             X_list.append(data_x[i - L:i])
#             y_list.append(data_y[i])
#             t_list.append(ts[i])

#             if has_rain:
#                 r_list.append(data_rain[i-1])
#             if has_rain_intensity:
#                 ri_list.append(data_rain_intensity[i-1])

#     X = np.array(X_list)
#     y = np.array(y_list)
#     r = np.array(r_list) if has_rain else None
#     ri = np.array(ri_list) if has_rain_intensity else None
#     t = np.array(t_list)

#     return X, y, r, ri, t

# def make_dataset(df, feature_cols):
#     data_x = df[feature_cols].values.astype("float32")
#     data_y = df[["target_mean"]].values.astype("float32") # 這裡已經是 t+1

#     has_rain = "rain_event" in df.columns
#     has_rain_intensity = "rain_intensity" in df.columns

#     # 注意：在 feature_engineering 中，rain_event 已經是 shift(-1)
#     # 所以 data_rain[i] 代表的是時間點 t+1 的雨
#     data_rain = df["rain_event"].values.astype("float32") if has_rain else None
#     data_rain_intensity = df["rain_intensity"].values.astype("float32") if has_rain_intensity else None

#     X_list, y_list, r_list, ri_list, t_list = [], [], [], [], []
#     L = Config.LAGGED_VALUE
#     ts = df.index

#     # 修正範圍：從 L-1 開始，到 len - 1 結束
#     # 這樣 X_list 會包含到 i，而 y_list 會取到 target_mean[i] (即 i+1 的實際值)
#     for i in range(L - 1, len(data_x)):
#         # 檢查時間連續性
#         time_diff = (ts[i] - ts[i - L + 1]).total_seconds() / 60.0
#         if time_diff <= (L - 1) * Config.EXPECTED_MINUTES: #+ 15:
            
#             # --- 核心修正點 ---
#             # X 取從 i-L+1 到 i (包含第 i 筆特徵)
#             X_list.append(data_x[i - L + 1 : i + 1]) 
            
#             # y 取 target_mean[i]，對應的是時間點 t+1
#             y_list.append(data_y[i])
            
#             # 時間標記為當前時間點 t (發出預測的時間)
#             t_list.append(ts[i])

#             # 雨量資訊取 i，因為已 shift(-1)，所以這代表 t+1 的雨勢
#             if has_rain:
#                 r_list.append(data_rain[i])
#             if has_rain_intensity:
#                 ri_list.append(data_rain_intensity[i])

#     X = np.array(X_list)
#     y = np.array(y_list)
#     r = np.array(r_list) if has_rain else None
#     ri = np.array(ri_list) if has_rain_intensity else None
#     t = np.array(t_list)

#     return X, y, r, ri, t
def make_dataset(df, feature_cols):
    rain_cols = ["rain_event", "rain_intensity"]

    # 一般特徵：排除 rain，因為 rain 要另外用 t+1 加進 X
    x_feature_cols = [c for c in feature_cols if c not in rain_cols]

    if "target_mean" in x_feature_cols:
        raise ValueError("❌ feature_cols 不可以包含 target_mean，這會造成 leakage")

    data_x = df[x_feature_cols].values.astype("float32")
    data_y = df[["target_mean"]].values.astype("float32")  # 已經是 t+1

    has_rain = "rain_event" in df.columns
    has_rain_intensity = "rain_intensity" in df.columns

    data_rain = df["rain_event"].values.astype("float32") if has_rain else None
    data_rain_intensity = (
        df["rain_intensity"].values.astype("float32")
        if has_rain_intensity
        else None
    )

    X_list, y_list, r_list, ri_list, t_list = [], [], [], [], []

    L = Config.LAGGED_VALUE
    ts = df.index

    # 因為 rain 要取 i+1，所以最後只能跑到 len(data_x)-2
    for i in range(L - 1, len(data_x) - 1):
        start_i = i - L + 1
        next_i = i + 1

        hist_time_diff = (ts[i] - ts[start_i]).total_seconds() / 60.0
        next_time_diff = (ts[next_i] - ts[i]).total_seconds() / 60.0

        if (
            hist_time_diff <= (L - 1) * Config.EXPECTED_MINUTES
            and next_time_diff <= Config.EXPECTED_MINUTES
        ):
            # 一般特徵：t-L+1 ~ t
            X_base = data_x[start_i : i + 1]

            # rain：t+1
            rain_future = data_rain[next_i] if has_rain else 0.0
            rain_intensity_future = (
                data_rain_intensity[next_i]
                if has_rain_intensity
                else 0.0
            )

            # rain block: shape = (L, 2)
            # 前 L-1 格補 0，最後一格放 t+1 rain
            rain_block = np.zeros((L, 2), dtype="float32")
            rain_block[-1, 0] = rain_future
            rain_block[-1, 1] = rain_intensity_future

            # X 最後兩欄是 future rain
            X_full = np.concatenate([X_base, rain_block], axis=1)

            X_list.append(X_full)

            # y: target_mean[i]，你前面已經做成 t+1
            y_list.append(data_y[i])

            # t: 預測發出時間 t
            t_list.append(ts[i])

            if has_rain:
                r_list.append(rain_future)

            if has_rain_intensity:
                ri_list.append(rain_intensity_future)

    X = np.array(X_list, dtype="float32")
    y = np.array(y_list, dtype="float32")
    r = np.array(r_list, dtype="float32") if has_rain else None
    ri = np.array(ri_list, dtype="float32") if has_rain_intensity else None
    t = np.array(t_list)

    return X, y, r, ri, t

# def split_and_scale(X, y, r, timestamps):
#     n = len(X)
#     tr = int(n * Config.TRAIN_RATIO)
#     va = int(n * (Config.TRAIN_RATIO + Config.VAL_RATIO))

#     X_tr, y_tr, r_tr, t_tr = X[:tr], y[:tr], r[:tr], timestamps[:tr]
#     X_va, y_va, r_va, t_va = X[tr:va], y[tr:va], r[tr:va], timestamps[tr:va]
#     X_te, y_te, r_te, t_te = X[va:], y[va:], r[va:], timestamps[va:]

#     N, T, F = X_tr.shape
#     X_scaler = RobustScaler()
#     X_scaler.fit(X_tr.reshape(-1, F))

#     scale = lambda A: X_scaler.transform(A.reshape(-1, F)).reshape(A.shape)

#     sw_tr = np.ones(len(r_tr), dtype=np.float32)
#     sw_tr[r_tr > 0] = Config.RAIN_WEIGHT

#     masks = {
#         "val_rain": (r_va > 0).astype(bool),
#         "test_rain": (r_te > 0).astype(bool),
#         "val_time": t_va,
#         "test_time": t_te,
#     }

#     return scale(X_tr), y_tr, sw_tr, scale(X_va), y_va, scale(X_te), y_te, masks, X_scaler

def split_and_scale(X, y, r, ri, timestamps):
    n = len(X)
    tr = int(n * Config.TRAIN_RATIO)
    va = int(n * (Config.TRAIN_RATIO + Config.VAL_RATIO))

    X_tr, y_tr, t_tr = X[:tr], y[:tr], timestamps[:tr]
    X_va, y_va, t_va = X[tr:va], y[tr:va], timestamps[tr:va]
    X_te, y_te, t_te = X[va:], y[va:], timestamps[va:]

    r_tr = r_va = r_te = None
    if r is not None:
        r_tr, r_va, r_te = r[:tr], r[tr:va], r[va:]

    ri_tr = ri_va = ri_te = None
    if ri is not None:
        ri_tr, ri_va, ri_te = ri[:tr], ri[tr:va], ri[va:]

    N, T, F = X_tr.shape
    X_scaler = RobustScaler()
    X_scaler.fit(X_tr.reshape(-1, F))

    def scale(A):
        return X_scaler.transform(A.reshape(-1, F)).reshape(A.shape)

    # sample weight
    if r_tr is not None:
        sw_tr = np.ones(len(r_tr), dtype=np.float32)
        sw_tr[r_tr > 0] = Config.RAIN_WEIGHT
    else:
        sw_tr = np.ones(len(X_tr), dtype=np.float32)

    masks = {
        "val_rain": (r_va > 0).astype(bool) if r_va is not None else None,
        "test_rain": (r_te > 0).astype(bool) if r_te is not None else None,
        "val_rain_intensity": ri_va if ri_va is not None else None,
        "test_rain_intensity": ri_te if ri_te is not None else None,
        "val_time": t_va,
        "test_time": t_te,
    }

    return (
        scale(X_tr), y_tr, sw_tr,
        scale(X_va), y_va,
        scale(X_te), y_te,
        masks, X_scaler
    )


def get_masks(df, mode):
    df, feature_cols = set_mode_features(df, mode=mode)
    
    X, y, r, ri, t = make_dataset(df, feature_cols)

    X_tr, y_tr, sw_tr, X_va, y_va, X_te, y_te, masks, X_scaler = split_and_scale(
        X, y, r, ri, t
    )
    return masks