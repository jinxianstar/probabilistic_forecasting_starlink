import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks
from sklearn.preprocessing import RobustScaler
from matplotlib.backends.backend_pdf import PdfPages
import random
import os

# ==========================================
# 1. Config (統一設定)
# ==========================================
class Config:
    TRAFFIC_PATH = "./Data/Wetlink/iperf_cleaned_seconds_Osnabrück.csv"
    WEATHER_PATH = "./Data/Wetlink/analysis_data_Osnabrück.csv"
    #TRAFFIC_PATH = "./Data/Wetlink/iperf_cleaned_seconds_Enschede.csv"
    #WEATHER_PATH = "./Data/Wetlink/analysis_data_Enschede.csv"

    SITE_NAME    = None 

    AGG_FREQ = "60min"       
    EXPECTED_MINUTES = 60    
    INPUT_LEN = 24           

    BATCH_SIZE = 32
    EPOCHS = 200
    PATIENCE = 15            

    RESCUE_TRIALS = 3        
    TRAIN_RATIO = 0.75
    VAL_RATIO   = 0.10
    
    RAIN_WEIGHT = 10.0      # Rescue 權重

    PLOT_CHUNK_LEN = 120                   
    PLOT_PDF_PATH = "rescue_experiment_results.pdf" 

# ==========================================
# 2. Utils & Metrics
# ==========================================
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
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

    pinball_sum = sum(np.mean(np.maximum(q * (y - y_pred[:, i]), (q - 1.0) * (y - y_pred[:, i]))) for i, q in enumerate(qs))
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
        "Cov_90": round(cov_90, 4), "Width_90": round(w_90, 4), "Winkler_90": round(calculate_winkler(p05, p95, y, 0.1), 4),
        "Cov_50": round(cov_50, 4), "Width_50": round(w_50, 4), "Winkler_50": round(calculate_winkler(p25, p75, y, 0.5), 4)
    }

def evaluate_all_conditions(y_true, y_pred, rain_mask, model_name):
    res_all = compute_detailed_metrics(y_true, y_pred, model_name, "All")
    res_dry = compute_detailed_metrics(y_true[~rain_mask], y_pred[~rain_mask], model_name, "Dry")
    res_rain = compute_detailed_metrics(y_true[rain_mask], y_pred[rain_mask], model_name, "Rain")
    return [r for r in [res_all, res_dry, res_rain] if r is not None]

# ==========================================
# 3. Unified Data Pipeline 
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
        if "timestamp_start" in tdf.columns: tdf.set_index("timestamp_start", inplace=True)
        tdf.index = pd.to_datetime(tdf.index)
        tdf.sort_index(inplace=True)

        tdf = tdf[["download", "upload"]] / 1e6 
        traffic_h = tdf.resample(self.cfg.AGG_FREQ).agg({
            "download": [lambda x: x.quantile(0.95), lambda x: x.quantile(0.05), "mean", "std"],
            "upload":   ["mean"],
        })
        traffic_h.columns = ["download_q95", "download_q05", "download_mean", "download_std", "upload_mean"]
        traffic_h.replace([np.inf, -np.inf], np.nan, inplace=True)
        traffic_h.dropna(inplace=True)

        wdf = pd.read_csv(self.cfg.WEATHER_PATH)
        if self.cfg.SITE_NAME and "site_name" in wdf.columns:
            wdf = wdf[wdf["site_name"] == self.cfg.SITE_NAME].copy()
        if "timestamp_start" in wdf.columns: wdf.set_index("timestamp_start", inplace=True)
        wdf.index = pd.to_datetime(wdf.index)
        wdf.sort_index(inplace=True)

        for c in ["temp", "windspeed", "rain"]:
            if c not in wdf.columns: wdf[c] = np.nan
        weather_h = wdf.resample(self.cfg.AGG_FREQ).agg({
            "temp": "mean", "windspeed": "mean", "rain": "sum"
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

        df["temp_d1"]  = df["temp_next"] - df["temp_mean"]
        df["wind_d1"]  = df["wind_next"] - df["windspeed_mean"]
        
        df["rain_event"] = (df["rain_next"] > 0.0).astype("float32")
        df["rain_intensity"] = np.log1p(df["rain_next"].clip(lower=0.0))

        df.dropna(inplace=True)
        self.df = df

    def set_mode_features(self, mode: str):
        base = ["download_q95", "download_q05", "download_mean", "download_std", "upload_mean"]
        time_feats = ["hod_sin_next", "hod_cos_next", "dow_sin_next", "dow_cos_next"]
        weather_feats = ["temp_next", "temp_d1", "wind_next", "wind_d1", "rain_event", "rain_intensity"]

        if mode == "base": feats = base
        elif mode == "time": feats = base + time_feats
        elif mode == "weather": feats = base + weather_feats
        elif mode == "time_weather": feats = base + time_feats + weather_feats
        else: raise ValueError(f"Unknown mode: {mode}")
        self.feature_cols = feats

    def make_dataset(self):
        data_x = self.df[self.feature_cols].values.astype("float32")
        data_y = self.df[["target_mean"]].values.astype("float32")
        data_rain = self.df["rain_event"].values.astype("float32")
        
        X_list, y_list, r_list = [], [], []
        L = self.cfg.INPUT_LEN
        ts = self.df.index
        
        for i in range(L, len(data_x)):
            time_diff = (ts[i] - ts[i-L]).total_seconds() / 60.0
            if time_diff <= L * self.cfg.EXPECTED_MINUTES + 15:
                X_list.append(data_x[i-L:i])
                y_list.append(data_y[i])
                r_list.append(data_rain[i])

        return np.array(X_list), np.array(y_list), np.array(r_list)

    def split_and_scale(self, X, y, r):
        n = len(X)
        tr = int(n * self.cfg.TRAIN_RATIO)
        va = int(n * (self.cfg.TRAIN_RATIO + self.cfg.VAL_RATIO))

        X_tr, y_tr, r_tr = X[:tr], y[:tr], r[:tr]
        X_va, y_va, r_va = X[tr:va], y[tr:va], r[tr:va]
        X_te, y_te, r_te = X[va:], y[va:], r[va:]

        N, T, F = X_tr.shape
        self.X_scaler.fit(X_tr.reshape(-1, F))
        
        scale = lambda A: self.X_scaler.transform(A.reshape(-1, F)).reshape(A.shape)

        sw_tr = np.ones(len(r_tr), dtype=np.float32)
        sw_tr[r_tr > 0] = self.cfg.RAIN_WEIGHT

        masks = {
            "val_rain": (r_va > 0).astype(bool),
            "test_rain": (r_te > 0).astype(bool)
        }
        return scale(X_tr), y_tr, sw_tr, scale(X_va), y_va, scale(X_te), y_te, masks

# ==========================================
# 4. Model & Training
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
    model.compile(optimizer=optimizers.legacy.Adam(learning_rate=0.001, clipnorm=1.0), loss=multi_quantile_loss)
    return model

def run_one_trial(cfg: Config, mode: str, trial_idx: int, use_weights: bool = False):
    seed = 42 + trial_idx * 100
    set_seed(seed)
    
    tag = "Rescue" if use_weights else "Base"
    print(f"[Run] Start {mode}_{tag}_t{trial_idx+1} (Seed={seed})...")

    fc = IntervalForecaster(cfg)
    fc.load_and_preprocess()
    fc.generate_labels()
    fc.feature_engineering()
    fc.set_mode_features(mode)

    X, y, r = fc.make_dataset()
    X_tr, y_tr, sw_tr, X_va, y_va, X_te, y_te, masks = fc.split_and_scale(X, y, r)

    model = build_model((X_tr.shape[1], X_tr.shape[2]))
    train_weights = sw_tr if use_weights else None
    
    model.fit(
        X_tr, y_tr, sample_weight=train_weights, validation_data=(X_va, y_va),
        epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE,
        callbacks=[callbacks.EarlyStopping(patience=cfg.PATIENCE, restore_best_weights=True)],
        verbose=0 
    )

    y_va_pred = np.sort(np.maximum(model.predict(X_va, verbose=0), 0.0), axis=1)
    y_te_pred = np.sort(np.maximum(model.predict(X_te, verbose=0), 0.0), axis=1)

    return {
        "name": f"{mode}_{tag}_t{trial_idx+1}",
        "y_va_true": y_va, "y_va_pred": y_va_pred,
        "y_te_true": y_te, "y_te_pred": y_te_pred, "masks": masks
    }

# ==========================================
# 5. Visualization Functions
# ==========================================
def plot_time_series(y_true, y_pred, rain_mask, title="Forecast", ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(15, 4))
    x = np.arange(len(y_true))
    y = y_true[:, 0]
    p05, p25, p75, p95 = y_pred.T

    rain_idx = np.where(rain_mask)[0]
    for i in rain_idx:
        ax.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.3, lw=0)
        
    ax.fill_between(x, p05, p95, alpha=0.2, color="blue", label="90% CI")
    ax.fill_between(x, p25, p75, alpha=0.4, color="blue", label="50% CI")
    ax.plot(x, (p25+p75)/2, ls="--", color="navy", alpha=0.8, label="Median")
    ax.scatter(x, y, color="black", s=15, alpha=0.9, label="True Value")

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Throughput (Mbps)")
    return ax

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    cfg = Config()
    all_metrics = []

    print("="*60 + f"\n[START] Rescue Mission (Rain Weight x{cfg.RAIN_WEIGHT})\n" + "="*60)
    rescue_preds = []
    y_test_true = None
    test_rain_mask = None

    for t in range(cfg.RESCUE_TRIALS):
        res = run_one_trial(cfg, "time_weather", t, use_weights=True)
        rescue_preds.append(res['y_te_pred'])
        if t == 0:
            y_test_true = res['y_te_true']
            test_rain_mask = res['masks']['test_rain']
    
    # 計算 Rescue 多次預測的平均值
    y_rescue = np.sort(np.mean(np.stack(rescue_preds, axis=0), axis=0), axis=1)
    all_metrics.extend(evaluate_all_conditions(y_test_true, y_rescue, test_rain_mask, "Rescue Model"))

    # --- 整理報表 ---
    df_all = pd.DataFrame(all_metrics)
    
    print("\n" + "🌟"*20 + "\n[1] MASTER SUMMARY (ALL DATA)\n" + "🌟"*20)
    print(df_all[df_all["Condition"] == "All"][["Model", "Cov_90", "Width_90", "Winkler_90", "Cov_50", "Width_50", "Winkler_50"]].to_string(index=False))

    print("\n" + "☀️ "*10 + "[2] DRY CONDITION (NO RAIN) " + "☀️ "*10)
    print(df_all[df_all["Condition"] == "Dry"][["Model", "Cov_90", "Width_90", "Winkler_90", "Cov_50", "Winkler_50"]].to_string(index=False))

    print("\n" + "🌧️ "*10 + "[3] RAIN CONDITION " + "🌧️ "*10)
    print(df_all[df_all["Condition"] == "Rain"][["Model", "Cov_90", "Width_90", "Winkler_90", "Cov_50", "Winkler_50"]].to_string(index=False))

    # ==========================================
    # 7. 自動尋找最多雨的區段並彈出圖表
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
        fig, ax = plt.subplots(figsize=(14, 4))
        plot_time_series(y_test_true[best_s:best_e], y_rescue[best_s:best_e], test_rain_mask[best_s:best_e], 
                         title=f"Rescue Model (Steps {best_s}-{best_e}) - Note the channels adapting to the grey rain areas!", ax=ax)
        plt.tight_layout()
        print("\n>>> Pop-up window displaying! Please close the plot window to finish the program. <<<")
        plt.show() 
    else:
        print("\n[Plot] No rain detected in the test set. Skipping the pop-up plot.")

    # --- 繪圖輸出 PDF ---
    print(f"\n[PDF] Saving all plots to {cfg.PLOT_PDF_PATH} ...")
    with PdfPages(cfg.PLOT_PDF_PATH) as pdf:
        n = len(y_test_true)
        chunk = cfg.PLOT_CHUNK_LEN
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            if np.sum(test_rain_mask[s:e]) == 0: continue 
            fig, ax = plt.subplots(figsize=(15, 4))
            plot_time_series(y_test_true[s:e], y_rescue[s:e], test_rain_mask[s:e], title=f"Rescue (Steps {s}-{e})", ax=ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
    print("[Done] Experiment fully completed.")