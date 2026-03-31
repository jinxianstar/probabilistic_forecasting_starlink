import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KernelDensity
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize
import random
import os

# ==========================================
# 1. Config
# ==========================================
class Config:
    #TRAFFIC_PATH = "./Data/Wetlink/iperf_cleaned_seconds_Osnabrück.csv"
    #WEATHER_PATH = "./Data/Wetlink/analysis_data_Osnabrück.csv"
    TRAFFIC_PATH = "./Data/Wetlink/iperf_cleaned_seconds_Enschede.csv"
    WEATHER_PATH = "./Data/Wetlink/analysis_data_Enschede.csv"
    SITE_NAME    = None

    AGG_FREQ = "60min"
    EXPECTED_MINUTES = 60
    INPUT_LEN = 24

    BATCH_SIZE = 32
    EPOCHS = 200
    PATIENCE = 15

    TRIALS_PER_MODE = 6
    RESCUE_TRIALS = 3
    ADAPTIVE_TRIALS = 3

    TRAIN_RATIO = 0.75
    VAL_RATIO   = 0.15

    # 原本固定權重 Rescue
    RAIN_WEIGHT = 10.0

    # 新增：Adaptive Weather-Regime Adaptive Quantile Weighting
    ADAPTIVE_ALPHA_RAIN = 2.5     # 雨勢強度權重
    ADAPTIVE_BETA_RARITY = 1.5    # 稀有 target 權重
    ADAPTIVE_MAX_WEIGHT = 20.0
    KDE_BANDWIDTH = 5.0

    PLOT_CHUNK_LEN = 120
    PLOT_PDF_PATH = "comprehensive_experiment_results_with_adaptive.pdf"

# ==========================================
# 2. Utils & Metrics
# ==========================================
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

def compute_detailed_metrics(y_true, y_pred, model_name="Model", condition="All"):
    if len(y_true) == 0:
        return None

    y = y_true[:, 0]
    y_pred = np.sort(y_pred, axis=1)
    p05, p25, p75, p95 = y_pred.T
    qs = [0.05, 0.25, 0.75, 0.95]

    pinball_sum = sum(
        np.mean(np.maximum(q * (y - y_pred[:, i]), (q - 1.0) * (y - y_pred[:, i])))
        for i, q in enumerate(qs)
    )
    crps = 2.0 * pinball_sum / len(qs)

    cov_90 = np.mean((y >= p05) & (y <= p95))
    w_90 = np.mean(p95 - p05)
    cov_50 = np.mean((y >= p25) & (y <= p75))
    w_50 = np.mean(p75 - p25)

    def calculate_winkler(lower, upper, true_val, alpha):
        score = upper - lower
        mask_below = true_val < lower
        score[mask_below] += (2.0 / alpha) * (lower[mask_below] - true_val[mask_below])
        mask_above = true_val > upper
        score[mask_above] += (2.0 / alpha) * (true_val[mask_above] - upper[mask_above])
        return np.mean(score)

    return {
        "Model": model_name,
        "Condition": condition,
        "CRPS": round(crps, 4),
        "Cov_90": round(cov_90, 4),
        "Width_90": round(w_90, 4),
        "Winkler_90": round(calculate_winkler(p05, p95, y, 0.1), 4),
        "Cov_50": round(cov_50, 4),
        "Width_50": round(w_50, 4),
        "Winkler_50": round(calculate_winkler(p25, p75, y, 0.5), 4),
    }

def evaluate_all_conditions(y_true, y_pred, rain_mask, model_name):
    res_all = compute_detailed_metrics(y_true, y_pred, model_name, "All")
    res_dry = compute_detailed_metrics(y_true[~rain_mask], y_pred[~rain_mask], model_name, "Dry")
    res_rain = compute_detailed_metrics(y_true[rain_mask], y_pred[rain_mask], model_name, "Rain")
    return [r for r in [res_all, res_dry, res_rain] if r is not None]

# ==========================================
# 3. Adaptive Weighting
# ==========================================
def compute_target_rarity(y_train, bandwidth=5.0):
    """
    根據 target_mean 的分布估計 rarity:
    rarity = 1 / p(y)
    再做 robust normalization
    """
    y_train = y_train.reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(y_train)

    logp = kde.score_samples(y_train)
    p = np.exp(logp)
    rarity = 1.0 / (p + 1e-6)

    median_r = np.median(rarity)
    if median_r <= 0:
        median_r = 1.0
    rarity = rarity / median_r
    return rarity.astype(np.float32)

def build_fixed_rain_weights(r_tr, cfg: Config):
    sw_tr = np.ones(len(r_tr), dtype=np.float32)
    sw_tr[r_tr > 0] = cfg.RAIN_WEIGHT
    return sw_tr

def build_adaptive_weather_regime_weights(y_tr, rain_intensity_tr, cfg: Config):
    """
    Adaptive Weather-Regime Adaptive Quantile Weighting (WRAQW)

    w_i = 1 + alpha * rain_intensity_i + beta * rarity(y_i)
    """
    rarity = compute_target_rarity(y_tr[:, 0], bandwidth=cfg.KDE_BANDWIDTH)

    weights = (
        1.0
        + cfg.ADAPTIVE_ALPHA_RAIN * rain_intensity_tr
        + cfg.ADAPTIVE_BETA_RARITY * rarity
    )

    weights = np.clip(weights, 1.0, cfg.ADAPTIVE_MAX_WEIGHT).astype(np.float32)
    return weights

# ==========================================
# 4. Unified Data Pipeline
# ==========================================
class IntervalForecaster:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.df = None
        self.feature_cols = None
        self.X_scaler = RobustScaler()

    def load_and_preprocess(self):
        tdf = pd.read_csv(self.cfg.TRAFFIC_PATH)
        if self.cfg.SITE_NAME and "site_name" in tdf.columns:
            tdf = tdf[tdf["site_name"] == self.cfg.SITE_NAME].copy()

        if "timestamp_start" in tdf.columns:
            tdf.set_index("timestamp_start", inplace=True)
        tdf.index = pd.to_datetime(tdf.index)
        tdf.sort_index(inplace=True)

        tdf = tdf[["download", "upload"]] / 1e6
        traffic_h = tdf.resample(self.cfg.AGG_FREQ).agg({
            "download": [lambda x: x.quantile(0.95), lambda x: x.quantile(0.05), "mean", "std"],
            "upload": ["mean"],
        })
        traffic_h.columns = ["download_q95", "download_q05", "download_mean", "download_std", "upload_mean"]
        traffic_h.replace([np.inf, -np.inf], np.nan, inplace=True)
        traffic_h.dropna(inplace=True)

        wdf = pd.read_csv(self.cfg.WEATHER_PATH)
        if self.cfg.SITE_NAME and "site_name" in wdf.columns:
            wdf = wdf[wdf["site_name"] == self.cfg.SITE_NAME].copy()

        if "timestamp_start" in wdf.columns:
            wdf.set_index("timestamp_start", inplace=True)
        wdf.index = pd.to_datetime(wdf.index)
        wdf.sort_index(inplace=True)

        for c in ["temp", "windspeed", "rain"]:
            if c not in wdf.columns:
                wdf[c] = np.nan

        weather_h = wdf.resample(self.cfg.AGG_FREQ).agg({
            "temp": "mean",
            "windspeed": "mean",
            "rain": "sum",
        })
        weather_h.columns = ["temp_mean", "windspeed_mean", "rain_sum"]

        df = traffic_h.join(weather_h, how="left")
        df["temp_mean"] = df["temp_mean"].ffill().bfill()
        df["windspeed_mean"] = df["windspeed_mean"].ffill().bfill()
        df["rain_sum"] = df["rain_sum"].fillna(0.0)

        self.df = df

    def generate_labels(self):
        self.df["target_mean"] = self.df["download_mean"].shift(-1)
        self.df.dropna(subset=["target_mean"], inplace=True)

    def feature_engineering(self):
        df = self.df.copy()

        df["hod_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hod_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        df["hod_sin_next"] = df["hod_sin"].shift(-1)
        df["hod_cos_next"] = df["hod_cos"].shift(-1)
        df["dow_sin_next"] = df["dow_sin"].shift(-1)
        df["dow_cos_next"] = df["dow_cos"].shift(-1)

        df["temp_next"] = df["temp_mean"].shift(-1).ffill().bfill()
        df["wind_next"] = df["windspeed_mean"].shift(-1).ffill().bfill()
        df["rain_next"] = df["rain_sum"].shift(-1).fillna(0.0)

        df["temp_d1"] = df["temp_next"] - df["temp_mean"]
        df["wind_d1"] = df["wind_next"] - df["windspeed_mean"]

        df["rain_event"] = (df["rain_next"] > 0.0).astype("float32")
        df["rain_intensity"] = np.log1p(df["rain_next"].clip(lower=0.0)).astype("float32")

        df.dropna(inplace=True)
        self.df = df

    def set_mode_features(self, mode: str):
        base = ["download_q95", "download_q05", "download_mean", "download_std", "upload_mean"]
        time_feats = ["hod_sin_next", "hod_cos_next", "dow_sin_next", "dow_cos_next"]
        weather_feats = ["temp_next", "temp_d1", "wind_next", "wind_d1", "rain_event", "rain_intensity"]

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

        self.feature_cols = feats

    def make_dataset(self):
        data_x = self.df[self.feature_cols].values.astype("float32")
        data_y = self.df[["target_mean"]].values.astype("float32")
        data_rain = self.df["rain_event"].values.astype("float32")
        data_rain_intensity = self.df["rain_intensity"].values.astype("float32")

        X_list, y_list, r_list, ri_list = [], [], [], []
        L = self.cfg.INPUT_LEN
        ts = self.df.index

        for i in range(L, len(data_x)):
            time_diff = (ts[i] - ts[i - L]).total_seconds() / 60.0
            if time_diff <= L * self.cfg.EXPECTED_MINUTES + 15:
                X_list.append(data_x[i - L:i])
                y_list.append(data_y[i])
                r_list.append(data_rain[i])
                ri_list.append(data_rain_intensity[i])

        return (
            np.array(X_list),
            np.array(y_list),
            np.array(r_list),
            np.array(ri_list),
        )

    def split_and_scale(self, X, y, r, rain_intensity):
        n = len(X)
        tr = int(n * self.cfg.TRAIN_RATIO)
        va = int(n * (self.cfg.TRAIN_RATIO + self.cfg.VAL_RATIO))

        X_tr, y_tr, r_tr, ri_tr = X[:tr], y[:tr], r[:tr], rain_intensity[:tr]
        X_va, y_va, r_va, ri_va = X[tr:va], y[tr:va], r[tr:va], rain_intensity[tr:va]
        X_te, y_te, r_te, ri_te = X[va:], y[va:], r[va:], rain_intensity[va:]

        N, T, F = X_tr.shape
        self.X_scaler.fit(X_tr.reshape(-1, F))

        def scale(arr):
            return self.X_scaler.transform(arr.reshape(-1, F)).reshape(arr.shape)

        masks = {
            "val_rain": (r_va > 0).astype(bool),
            "test_rain": (r_te > 0).astype(bool),
        }

        aux = {
            "r_tr": r_tr.astype(np.float32),
            "ri_tr": ri_tr.astype(np.float32),
            "r_va": r_va.astype(np.float32),
            "ri_va": ri_va.astype(np.float32),
            "r_te": r_te.astype(np.float32),
            "ri_te": ri_te.astype(np.float32),
        }

        return (
            scale(X_tr), y_tr,
            scale(X_va), y_va,
            scale(X_te), y_te,
            masks, aux
        )

# ==========================================
# 5. Model
# ==========================================
def build_model(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(4, name="quantiles")(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss=multi_quantile_loss
    )
    return model

# ==========================================
# 6. Training Runner
# ==========================================
def run_one_trial(cfg: Config, mode: str, trial_idx: int, weighting_mode: str = "none"):
    """
    weighting_mode:
        - "none"       : 原始 Base
        - "fixed_rain" : 原始 Rescue
        - "adaptive"   : 新增 Adaptive WRAQW
    """
    seed = 42 + trial_idx * 100
    set_seed(seed)

    if weighting_mode == "none":
        tag = "Base"
    elif weighting_mode == "fixed_rain":
        tag = "Rescue"
    elif weighting_mode == "adaptive":
        tag = "AdaptiveWRAQW"
    else:
        raise ValueError(f"Unknown weighting_mode: {weighting_mode}")

    print(f"[Run] Start {mode}_{tag}_t{trial_idx+1} (Seed={seed})...")

    fc = IntervalForecaster(cfg)
    fc.load_and_preprocess()
    fc.generate_labels()
    fc.feature_engineering()
    fc.set_mode_features(mode)

    X, y, r, ri = fc.make_dataset()
    X_tr, y_tr, X_va, y_va, X_te, y_te, masks, aux = fc.split_and_scale(X, y, r, ri)

    if weighting_mode == "none":
        sample_weights = None
    elif weighting_mode == "fixed_rain":
        sample_weights = build_fixed_rain_weights(aux["r_tr"], cfg)
    elif weighting_mode == "adaptive":
        sample_weights = build_adaptive_weather_regime_weights(
            y_tr=y_tr,
            rain_intensity_tr=aux["ri_tr"],
            cfg=cfg
        )

    model = build_model((X_tr.shape[1], X_tr.shape[2]))

    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weights,
        validation_data=(X_va, y_va),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=[callbacks.EarlyStopping(patience=cfg.PATIENCE, restore_best_weights=True)],
        verbose=0
    )

    y_va_pred = np.sort(np.maximum(model.predict(X_va, verbose=0), 0.0), axis=1)
    y_te_pred = np.sort(np.maximum(model.predict(X_te, verbose=0), 0.0), axis=1)

    return {
        "name": f"{mode}_{tag}_t{trial_idx+1}",
        "y_va_true": y_va,
        "y_va_pred": y_va_pred,
        "y_te_true": y_te,
        "y_te_pred": y_te_pred,
        "masks": masks,
    }

# ==========================================
# 7. Ensembles
# ==========================================
def get_ensembles(results_list):
    y_val_true = results_list[0]["y_va_true"][:, 0]
    val_rain = results_list[0]["masks"]["val_rain"]
    test_rain = results_list[0]["masks"]["test_rain"]

    preds_val_stack = np.stack([r["y_va_pred"] for r in results_list], axis=-1)
    preds_test_stack = np.stack([r["y_te_pred"] for r in results_list], axis=-1)

    qs = np.array([0.05, 0.25, 0.75, 0.95])

    def _solve_weights(stack, y):
        n = stack.shape[-1]

        def loss_func(w):
            wp = np.tensordot(stack, w, axes=([2], [0]))
            return sum(
                np.mean(np.maximum(q * (y - wp[:, i]), (q - 1.0) * (y - wp[:, i])))
                for i, q in enumerate(qs)
            )

        res = minimize(
            loss_func,
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
        )
        return res.x

    w_global = _solve_weights(preds_val_stack, y_val_true)
    y_global_pred = np.sort(np.tensordot(preds_test_stack, w_global, axes=([2], [0])), axis=1)

    y_cond_pred = np.zeros_like(y_global_pred)
    if np.sum(val_rain) > 10:
        w_dry = _solve_weights(preds_val_stack[~val_rain], y_val_true[~val_rain])
        w_rain = _solve_weights(preds_val_stack[val_rain], y_val_true[val_rain])

        if np.sum(~test_rain) > 0:
            y_cond_pred[~test_rain] = np.tensordot(preds_test_stack[~test_rain], w_dry, axes=([2], [0]))
        if np.sum(test_rain) > 0:
            y_cond_pred[test_rain] = np.tensordot(preds_test_stack[test_rain], w_rain, axes=([2], [0]))

        y_cond_pred = np.sort(y_cond_pred, axis=1)
    else:
        y_cond_pred = y_global_pred

    return y_global_pred, y_cond_pred

# ==========================================
# 8. Visualization
# ==========================================
def plot_time_series(y_true, y_pred, rain_mask, title="Forecast", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 4))

    x = np.arange(len(y_true))
    y = y_true[:, 0]
    p05, p25, p75, p95 = y_pred.T

    rain_idx = np.where(rain_mask)[0]
    for i in rain_idx:
        ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.3, lw=0)

    ax.fill_between(x, p05, p95, alpha=0.2, label="90% CI")
    ax.fill_between(x, p25, p75, alpha=0.4, label="50% CI")
    ax.plot(x, (p25 + p75) / 2, ls="--", alpha=0.8, label="Pseudo Median")
    ax.scatter(x, y, s=15, alpha=0.9, label="True Value")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Throughput (Mbps)")
    return ax

# ==========================================
# 9. Main
# ==========================================
if __name__ == "__main__":
    cfg = Config()
    all_metrics = []

    print("=" * 80)
    print("PHASE 1 & 2: Original Base Training + Ensemble Comparison")
    print("=" * 80)

    base_results = []
    modes = ["base", "time", "weather", "time_weather"]

    # -------- 原本 Base --------
    for m in modes:
        for t in range(cfg.TRIALS_PER_MODE):
            base_results.append(run_one_trial(cfg, m, t, weighting_mode="none"))

    y_test_true = base_results[0]["y_te_true"]
    test_rain_mask = base_results[0]["masks"]["test_rain"]

    # -------- 原本四種 Base Avg --------
    for m in modes:
        preds = [res["y_te_pred"] for res in base_results if res["name"].startswith(f"{m}_Base")]
        avg_pred = np.sort(np.mean(np.stack(preds, axis=0), axis=0), axis=1)
        all_metrics.extend(evaluate_all_conditions(y_test_true, avg_pred, test_rain_mask, f"0. Base Avg: {m}"))

    # -------- 原本 Ensemble --------
    y_global, y_cond = get_ensembles(base_results)
    all_metrics.extend(evaluate_all_conditions(y_test_true, y_global, test_rain_mask, "1. Global Ensemble"))
    all_metrics.extend(evaluate_all_conditions(y_test_true, y_cond, test_rain_mask, "2. Conditional Ensemble"))

    print("\n" + "=" * 80)
    print(f"PHASE 3: Original Rescue Model (Fixed Rain Weight x{cfg.RAIN_WEIGHT})")
    print("=" * 80)

    # -------- 原本 Rescue --------
    rescue_preds = []
    for t in range(cfg.RESCUE_TRIALS):
        res = run_one_trial(cfg, "time_weather", t, weighting_mode="fixed_rain")
        rescue_preds.append(res["y_te_pred"])

    y_rescue = np.sort(np.mean(np.stack(rescue_preds, axis=0), axis=0), axis=1)
    all_metrics.extend(evaluate_all_conditions(y_test_true, y_rescue, test_rain_mask, "3. Rescue Model (Fixed Weight)"))

    print("\n" + "=" * 80)
    print("PHASE 4: Adaptive Weather-Regime Adaptive Quantile Weighting (WRAQW)")
    print("=" * 80)

    # -------- 新增 Adaptive WRAQW --------
    adaptive_preds = []
    for t in range(cfg.ADAPTIVE_TRIALS):
        res = run_one_trial(cfg, "time_weather", t, weighting_mode="adaptive")
        adaptive_preds.append(res["y_te_pred"])

    y_adaptive = np.sort(np.mean(np.stack(adaptive_preds, axis=0), axis=0), axis=1)
    all_metrics.extend(evaluate_all_conditions(
        y_test_true, y_adaptive, test_rain_mask,
        "4. Adaptive WRAQW Model"
    ))

    # ==========================================
    # 報表
    # ==========================================
    df_all = pd.DataFrame(all_metrics)

    print("\n" + "🌟" * 20 + "\n[1] MASTER SUMMARY (ALL DATA)\n" + "🌟" * 20)
    print(df_all[df_all["Condition"] == "All"][[
        "Model", "Cov_90", "Width_90", "Winkler_90", "Cov_50", "Width_50", "Winkler_50"
    ]].to_string(index=False))

    print("\n" + "☀️ " * 10 + "[2] DRY CONDITION (NO RAIN) " + "☀️ " * 10)
    print(df_all[df_all["Condition"] == "Dry"][[
        "Model", "Cov_90", "Width_90", "Winkler_90", "Cov_50", "Width_50", "Winkler_50"
    ]].to_string(index=False))

    print("\n" + "🌧️ " * 10 + "[3] RAIN CONDITION (THE PROBLEM & SOLUTION) " + "🌧️ " * 10)
    print(df_all[df_all["Condition"] == "Rain"][[
        "Model", "Cov_90", "Width_90", "Winkler_90", "Cov_50", "Width_50", "Winkler_50"
    ]].to_string(index=False))

    # 額外：只看關鍵模型比較
    key_models = [
        "1. Global Ensemble",
        "2. Conditional Ensemble",
        "3. Rescue Model (Fixed Weight)",
        "4. Adaptive WRAQW Model"
    ]

    print("\n" + "=" * 80)
    print("KEY MODEL COMPARISON")
    print("=" * 80)

    for cond in ["All", "Dry", "Rain"]:
        print(f"\n--- {cond} ---")
        print(df_all[
            (df_all["Condition"] == cond) & (df_all["Model"].isin(key_models))
        ][[
            "Model", "Cov_90", "Width_90", "Winkler_90", "Cov_50", "Width_50", "Winkler_50"
        ]].to_string(index=False))

    # 儲存完整結果
    df_all.to_csv("full_comparison_metrics_with_adaptive.csv", index=False)
    print("\n[Saved] full_comparison_metrics_with_adaptive.csv")

    # ==========================================
    # 找最極端雨段畫圖
    # ==========================================
    print("\n[Plot] Searching for the most extreme rain period to display...")
    max_rain_sum = 0
    best_s, best_e = 0, min(cfg.PLOT_CHUNK_LEN, len(y_test_true))

    for s in range(0, len(y_test_true) - cfg.PLOT_CHUNK_LEN, 24):
        e = s + cfg.PLOT_CHUNK_LEN
        rain_sum = np.sum(test_rain_mask[s:e])
        if rain_sum > max_rain_sum:
            max_rain_sum = rain_sum
            best_s, best_e = s, e

    if max_rain_sum > 0:
        fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, sharey=True)

        plot_time_series(
            y_test_true[best_s:best_e],
            y_global[best_s:best_e],
            test_rain_mask[best_s:best_e],
            title=f"Global Ensemble (Steps {best_s}-{best_e})",
            ax=axes[0]
        )
        plot_time_series(
            y_test_true[best_s:best_e],
            y_rescue[best_s:best_e],
            test_rain_mask[best_s:best_e],
            title=f"Fixed Rescue Model (Steps {best_s}-{best_e})",
            ax=axes[1]
        )
        plot_time_series(
            y_test_true[best_s:best_e],
            y_adaptive[best_s:best_e],
            test_rain_mask[best_s:best_e],
            title=f"Adaptive WRAQW Model (Steps {best_s}-{best_e})",
            ax=axes[2]
        )

        plt.tight_layout()
        print("\n>>> Pop-up window displaying! Please close the plot window to finish the program. <<<")
        plt.show()
    else:
        print("\n[Plot] No rain detected in the test set. Skipping the pop-up plot.")

    # ==========================================
    # PDF 匯出
    # ==========================================
    print(f"\n[PDF] Saving all plots to {cfg.PLOT_PDF_PATH} ...")
    with PdfPages(cfg.PLOT_PDF_PATH) as pdf:
        n = len(y_test_true)
        chunk = cfg.PLOT_CHUNK_LEN

        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            if np.sum(test_rain_mask[s:e]) == 0:
                continue

            fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True, sharey=True)

            plot_time_series(
                y_test_true[s:e], y_global[s:e], test_rain_mask[s:e],
                title=f"Global Ensemble (Steps {s}-{e})", ax=axes[0]
            )
            plot_time_series(
                y_test_true[s:e], y_rescue[s:e], test_rain_mask[s:e],
                title=f"Fixed Rescue Model (Steps {s}-{e})", ax=axes[1]
            )
            plot_time_series(
                y_test_true[s:e], y_adaptive[s:e], test_rain_mask[s:e],
                title=f"Adaptive WRAQW Model (Steps {s}-{e})", ax=axes[2]
            )

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print("[Done] Experiment fully completed.")