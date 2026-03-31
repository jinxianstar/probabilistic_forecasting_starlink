import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks
from sklearn.preprocessing import RobustScaler
import random
import os

# ==========================================
# 1. Config (統一設定)
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
    EPOCHS = 150 
    PATIENCE = 10            

    TRAIN_RATIO = 0.75
    VAL_RATIO   = 0.10
    
# ==========================================
# 2. Utils & Metrics
# ==========================================
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def multi_quantile_loss(y_true, y_pred):
    qs = tf.constant([0.05, 0.25, 0.75, 0.95], dtype=tf.float32)
    e = y_true - y_pred
    loss = tf.maximum(qs * e, (qs - 1.0) * e)
    # 確保回傳 (batch_size,) 讓模型能正確認識每個點的獨立誤差
    return tf.reduce_sum(loss, axis=-1)

def compute_detailed_metrics(y_true, y_pred, model_name="Model", condition="All"):
    if len(y_true) == 0: return None
        
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
        "Cov_90": round(cov_90, 4), 
        "Width_90": round(w_90, 4), 
        "Winkler_90": round(calculate_winkler(p05, p95, y, 0.1), 4),
        "Cov_50": round(cov_50, 4), 
        "Width_50": round(w_50, 4), 
        "Winkler_50": round(calculate_winkler(p25, p75, y, 0.5), 4)
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
        self.X_scaler = RobustScaler()

    def load_and_preprocess(self):
        tdf = pd.read_csv(self.cfg.TRAFFIC_PATH, index_col="timestamp_start", parse_dates=True).sort_index()
        tdf = tdf[["download", "upload"]] / 1e6 
        traffic_h = tdf.resample(self.cfg.AGG_FREQ).agg({"download": "mean"}).rename(columns={"download": "download_mean"}).dropna()

        wdf = pd.read_csv(self.cfg.WEATHER_PATH, index_col="timestamp_start", parse_dates=True).sort_index()
        weather_h = wdf.resample(self.cfg.AGG_FREQ).agg({"rain": "sum"}).rename(columns={"rain": "rain_sum"})
        
        df = traffic_h.join(weather_h, how="left")
        df["rain_sum"] = df["rain_sum"].fillna(0.0)
        
        df["target_mean"] = df["download_mean"].shift(-1)
        df["rain_event"] = (df["rain_sum"].shift(-1).fillna(0.0) > 0.0).astype("float32")
        df["target_error_proxy"] = df["download_mean"] - df["download_mean"].shift(1)
        df["target_error_proxy"] = df["target_error_proxy"].fillna(0.0)

        self.df = df.dropna()

    def make_dataset(self):
        features = ["download_mean", "rain_sum", "rain_event", "target_error_proxy"]
        data_x = self.df[features].values.astype("float32")
        data_y = self.df[["target_mean"]].values.astype("float32")
        data_rain = self.df["rain_event"].values.astype("float32")
        
        X_list, y_list, r_list = [], [], []
        L = self.cfg.INPUT_LEN
        
        for i in range(L, len(data_x)):
            X_list.append(data_x[i-L:i])
            y_list.append(data_y[i])
            r_list.append(data_rain[i])

        X, y, r = np.array(X_list), np.array(y_list), np.array(r_list)
        
        n = len(X)
        tr, va = int(n * self.cfg.TRAIN_RATIO), int(n * (self.cfg.TRAIN_RATIO + self.cfg.VAL_RATIO))

        X_tr, y_tr, r_tr = X[:tr], y[:tr], r[:tr]
        X_va, y_va, r_va = X[tr:va], y[tr:va], r[tr:va]
        X_te, y_te, r_te = X[va:], y[va:], r[va:]

        N, T, F = X_tr.shape
        self.X_scaler.fit(X_tr.reshape(-1, F))
        scale = lambda A: self.X_scaler.transform(A.reshape(-1, F)).reshape(A.shape)

        masks = {
            "train_rain": (r_tr > 0).astype(bool),
            "val_rain": (r_va > 0).astype(bool),
            "test_rain": (r_te > 0).astype(bool)
        }
        return scale(X_tr), y_tr, scale(X_va), y_va, scale(X_te), y_te, masks

# ==========================================
# 4. Model Architectures
# ==========================================
def build_base_model(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=False)(inp)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(4)(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss=multi_quantile_loss)
    return model

def build_residual_model(input_shape):
    inp = layers.Input(shape=input_shape)
    x1 = layers.LSTM(64, return_sequences=False)(inp)
    base_out = layers.Dense(4)(x1)
    
    x2 = layers.Flatten()(inp)
    x2 = layers.Dense(32, activation="relu")(x2)
    residual_out = layers.Dense(4)(x2)
    
    out = layers.Add()([base_out, residual_out])
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss=multi_quantile_loss)
    return model

# ==========================================
# 5. Main Execution & Comparison
# ==========================================
if __name__ == "__main__":
    set_seed(42)
    cfg = Config()
    
    print("[1/4] Preparing Data...")
    fc = IntervalForecaster(cfg)
    fc.load_and_preprocess()
    X_tr, y_tr, X_va, y_va, X_te, y_te, masks = fc.make_dataset()
    input_shape = (X_tr.shape[1], X_tr.shape[2])
    test_rain_mask = masks['test_rain']
    
    all_metrics = []
    callbacks_list = [callbacks.EarlyStopping(patience=cfg.PATIENCE, restore_best_weights=True)]

    # 1. Base Model
    print("[2/4] Training Base Model...")
    base_model = build_base_model(input_shape)
    base_model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, callbacks=callbacks_list, verbose=0)
    pred_base = np.sort(base_model.predict(X_te, verbose=0), axis=1)
    all_metrics.extend(evaluate_all_conditions(y_te, pred_base, test_rain_mask, "1. Base Model"))

    # 2. Residual Model
    print("[3/4] Training Residual/Two-Stage Model...")
    res_model = build_residual_model(input_shape)
    res_model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, callbacks=callbacks_list, verbose=0)
    pred_res = np.sort(res_model.predict(X_te, verbose=0), axis=1)
    all_metrics.extend(evaluate_all_conditions(y_te, pred_res, test_rain_mask, "2. Residual Model"))

    # 3. MoE Model
    print("[4/4] Training Mixture of Experts (MoE) Models...")
    train_rain_mask = masks['train_rain']
    val_rain_mask = masks['val_rain']
    
    moe_dry = build_base_model(input_shape)
    moe_rain = build_base_model(input_shape)
    
    if np.sum(~train_rain_mask) > 10:
        moe_dry.fit(X_tr[~train_rain_mask], y_tr[~train_rain_mask], validation_data=(X_va[~val_rain_mask], y_va[~val_rain_mask]), 
                    epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, callbacks=callbacks_list, verbose=0)
    
    if np.sum(train_rain_mask) > 10:
        moe_rain.fit(X_tr[train_rain_mask], y_tr[train_rain_mask], validation_data=(X_va[val_rain_mask], y_va[val_rain_mask]), 
                     epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, callbacks=callbacks_list, verbose=0)

    pred_moe_dry = moe_dry.predict(X_te, verbose=0)
    pred_moe_rain = moe_rain.predict(X_te, verbose=0)
    pred_moe = np.zeros_like(pred_moe_dry)
    
    pred_moe[~test_rain_mask] = pred_moe_dry[~test_rain_mask]
    if np.sum(test_rain_mask) > 0:
        pred_moe[test_rain_mask] = pred_moe_rain[test_rain_mask]
        
    pred_moe = np.sort(pred_moe, axis=1)
    all_metrics.extend(evaluate_all_conditions(y_te, pred_moe, test_rain_mask, "3. MoE (Switching)"))

    # ==========================================
    # 6. 輸出最終大亂鬥報表
    # ==========================================
    df_all = pd.DataFrame(all_metrics)
    
    print("\n" + "="*80)
    print("🌧️  ARCHITECTURE SHOWDOWN: RAIN PERFORMANCE  🌧️")
    print("="*80)
    rain_results = df_all[df_all["Condition"] == "Rain"].copy()
    if not rain_results.empty:
        print(rain_results[["Model", "CRPS", "Cov_90", "Width_90", "Winkler_90", "Cov_50", "Width_50", "Winkler_50"]].to_string(index=False))
    else:
        print("No rain in the test set to evaluate!")

    print("\n" + "="*80)
    print("☀️  ARCHITECTURE SHOWDOWN: DRY PERFORMANCE  ☀️")
    print("="*80)
    dry_results = df_all[df_all["Condition"] == "Dry"].copy()
    if not dry_results.empty:
        print(dry_results[["Model", "CRPS", "Cov_90", "Width_90", "Winkler_90", "Cov_50", "Width_50", "Winkler_50"]].to_string(index=False))