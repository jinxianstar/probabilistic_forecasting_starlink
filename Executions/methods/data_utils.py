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
    df["temp_mean"] = df["temp_mean"].ffill().bfill()
    df["windspeed_mean"] = df["windspeed_mean"].ffill().bfill()
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
        df[f"{c}_next"] = df[c].shift(-1)
    for c in dow_cols:
        df[f"{c}_next"] = df[c].shift(-1)

    # ===== Weather features =====
    df["temp_next"] = df["temp_mean"].shift(-1).ffill().bfill()
    df["wind_next"] = df["windspeed_mean"].shift(-1).ffill().bfill()
    df["rain_next"] = df["rain_sum"].shift(-1).fillna(0.0)

    df["temp_d1"] = df["temp_next"] - df["temp_mean"]
    df["wind_d1"] = df["wind_next"] - df["windspeed_mean"]

    df["rain_event"] = (df["rain_next"] > 0.0).astype("float32")
    df["rain_intensity"] = np.log1p(df["rain_next"].clip(lower=0.0))

    df.dropna(inplace=True)
    return df

def set_mode_features(df, mode: str):
    # base = ["download_q95", "download_q05", "download_mean", "download_std", "upload_mean"]
    base = ["download_mean", "upload_mean"]

    # time_feats = ["hod_sin_next", "hod_cos_next", "dow_sin_next", "dow_cos_next"]
    time_feats = [f"hod_{i}_next" for i in range(24)] + [f"dow_{i}_next" for i in range(7)]
    # weather_feats = ["temp_next", "temp_d1", "wind_next", "wind_d1", "rain_event", "rain_intensity"]
    # weather_feats = ["temp_next", "temp_d1", "rain_event", "rain_intensity"]
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

    df = df[feature_cols + ["target_mean"]]
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

def make_dataset(df, feature_cols):
    data_x = df[feature_cols].values.astype("float32")
    data_y = df[["target_mean"]].values.astype("float32")

    has_rain = "rain_event" in df.columns
    has_rain_intensity = "rain_intensity" in df.columns

    data_rain = df["rain_event"].values.astype("float32") if has_rain else None
    data_rain_intensity = df["rain_intensity"].values.astype("float32") if has_rain_intensity else None

    X_list, y_list, r_list, ri_list, t_list = [], [], [], [], []
    L = Config.LAGGED_VALUE
    ts = df.index

    for i in range(L, len(data_x)):
        time_diff = (ts[i] - ts[i - L]).total_seconds() / 60.0
        if time_diff <= L * Config.EXPECTED_MINUTES + 15:
            X_list.append(data_x[i - L:i])
            y_list.append(data_y[i])
            t_list.append(ts[i])

            if has_rain:
                r_list.append(data_rain[i])
            if has_rain_intensity:
                ri_list.append(data_rain_intensity[i])

    X = np.array(X_list)
    y = np.array(y_list)
    r = np.array(r_list) if has_rain else None
    ri = np.array(ri_list) if has_rain_intensity else None
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