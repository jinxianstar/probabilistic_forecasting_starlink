import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
#from Executions.config import Config
import methods.data_utils as du
from methods import models
import itertools
from scipy.optimize import differential_evolution
from methods import corrections


# def correction_objective(eval_df):
#     row_all = eval_df[eval_df["Subset"] == "all"].iloc[0]
#     row_rain = eval_df[eval_df["Subset"] == "rain"].iloc[0]

#     loss = 0.7 * row_rain["CRPS_approx"] + 0.3 * row_all["CRPS_approx"]
#     loss += 1.5 * abs(row_rain["Cov_90"] - 0.90)
#     return loss


# def correction_objective(eval_df):
#     row_all = eval_df[eval_df["Subset"] == "all"].iloc[0]
#     row_rain = eval_df[eval_df["Subset"] == "rain"].iloc[0]

#     rain_crps = row_rain["CRPS_approx"]
#     all_crps = row_all["CRPS_approx"]
#     rain_cov_gap = abs(row_rain["Cov_90"] - 0.90)

#     loss = 1.5 * rain_crps + 0.1 * all_crps
#     loss += 4.0 * rain_cov_gap + 10.0 * rain_cov_gap**2
#     return loss

def correction_objective(eval_df):
    row_all = eval_df[eval_df["Subset"] == "all"].iloc[0]
    row_rain = eval_df[eval_df["Subset"] == "rain"].iloc[0]

    # 1. 抓取所需指標
    rain_crps = row_rain["CRPS_approx"]
    all_crps = row_all["CRPS_approx"]
    
    # 抓取 Winkler Score (越低越好)
    rain_winkler_90 = row_rain["Winkler_90"]
    rain_winkler_50 = row_rain["Winkler_50"]

    # 計算 Coverage 的落差 (作為額外的硬性懲罰)
    rain_cov_90_gap = abs(row_rain["Cov_90"] - 0.90)
    rain_cov_50_gap = abs(row_rain["Cov_50"] - 0.50)

    # 2. 組合 Loss
    # 基礎 CRPS (保持分佈形狀)
    loss = 1.5 * rain_crps + 0.1 * all_crps
    
    # Winkler Score 負責優化區間 (90% 看極端，50% 看核心)
    loss += 0.5 * rain_winkler_90 + 0.5 * rain_winkler_50
    
    # Coverage Gap 負責兜底 (避免模型為了縮減 Winkler 的 Width 而犧牲太多 Cov)
    # 特別是雨天，Under-coverage 通常是我們最不想看到的
    loss += 5.0 * rain_cov_90_gap**2 + 2.0 * rain_cov_50_gap**2 

    return loss

def correction_objective(eval_df):
    row_all = eval_df[eval_df["Subset"] == "all"].iloc[0]
    row_rain = eval_df[eval_df["Subset"] == "rain"].iloc[0]

    # --- 1. 抓取指標 ---
    rain_crps = row_rain["CRPS_approx"]
    
    # 對 Winkler 進行對數處理，避免它在優化初期主導一切
    # 因為 Winkler 的量級通常在 50~150，Log1p 能把它縮到 4~5
    rain_winkler_90 = np.log1p(row_rain["Winkler_90"])
    rain_winkler_50 = np.log1p(row_rain["Winkler_50"])

    # --- 2. 非對稱 Coverage 懲罰 (核心改動) ---
    # 我們只罰「達不到標」的情況 (Under-coverage)
    # 如果超過 0.90 則不罰，讓 Winkler 負責去縮窄寬度
    def under_coverage_penalty(actual, target):
        gap = target - actual
        return max(0, gap) ** 2

    penalty_90 = under_coverage_penalty(row_rain["Cov_90"], 0.90)
    penalty_50 = under_coverage_penalty(row_rain["Cov_50"], 0.50)

    # --- 3. 組合 Loss ---
    # 權重分配邏輯：
    # CRPS 保持分佈形態 (2.0)
    # Winkler 負責區間效率 (1.0)
    # Penalty 給予極高權重 (100+)，確保這是「硬指標」
    
    loss = 2.0 * rain_crps 
    loss += 1.0 * (rain_winkler_90 + 0.5 * rain_winkler_50)
    
    # 這裡的權重要大到能跟 Log(Winkler) 抗衡
    # 如果 penalty 是 0.02 (即 coverage 0.88)，200 * 0.0004 = 0.08
    loss += 300.0 * penalty_90  # 強制拉回 90% 覆蓋
    loss += 100.0 * penalty_50  # 強制拉回 50% 核心

    return loss


def correction_objective(eval_df):
    # 只關注雨天子集
    row_rain = eval_df[eval_df["Subset"] == "rain"].iloc[0]

    # 1. Winkler Score
    w90 = row_rain["Winkler_90"]
    w50 = row_rain["Winkler_50"]

    # 2. Winkler loss：鼓勵區間不要太寬、也不要漏太多
    loss_winkler = np.log1p(w90) + 0.5 * np.log1p(w50)

    # 3. Coverage calibration loss：鼓勵 coverage 接近理論值
    cov90 = row_rain["Cov_90"]
    cov50 = row_rain["Cov_50"]

    target90 = 0.90
    target50 = 0.50

    loss_coverage = (
        (cov90 - target90) ** 2
        + 0.5 * (cov50 - target50) ** 2
    )

    # 4. 權重：控制 coverage 在總目標中的重要性
    lambda_cov = 20

    loss = loss_winkler + lambda_cov * loss_coverage

    return loss

# def correction_objective(eval_df):
#     row_all = eval_df[eval_df["Subset"] == "all"].iloc[0]
#     row_rain = eval_df[eval_df["Subset"] == "rain"].iloc[0]

#     # 1. 抓取所需指標
#     rain_crps = row_rain["CRPS_approx"]
#     all_crps = row_all["CRPS_approx"]
    
#     rain_winkler_90 = row_rain["Winkler_90"]
#     rain_winkler_50 = row_rain["Winkler_50"]

#     rain_cov_90 = row_rain["Cov_90"]
#     rain_cov_50 = row_rain["Cov_50"]

#     # 2. 組合 Loss
#     # 基礎 CRPS (權重稍微拉高，確保模型不會為了調區間而犧牲點預測準度)
#     loss = 2.0 * rain_crps + 0.5 * all_crps
    
#     # 3. 【關鍵 1】對 Winkler 使用 Log1p 縮放
#     # 避免極端大雨造成的單一破萬 Winkler 數值直接把 Loss 算爆，破壞優化方向
#     loss += 1.0 * np.log1p(max(rain_winkler_90, 0))
#     loss += 0.5 * np.log1p(max(rain_winkler_50, 0))

#     # 4. 【關鍵 2】非對稱單向懲罰 (Asymmetric Penalty)
#     # 使用 max(0, 目標 - 實際值)，只在「漏報 (Under-coverage)」時產生懲罰
#     # 也就是說：如果 Cov 是 0.95，這項數值會是 0，完全不懲罰，讓上面的 Winkler 負責把它壓窄
#     cov_90_under = max(0, 0.90 - rain_cov_90)
#     cov_50_under = max(0, 0.50 - rain_cov_50)

#     # 給予漏報極高的權重 (確保底線)
#     loss += 100.0 * (cov_90_under ** 2) 
#     loss += 20.0 * (cov_50_under ** 2)

#     return loss


def tune_rain_correction_on_validation(models, y_va, y_pred_va, masks_va, mode="normal", intensity_p90=None):
    if mode == "normal":
        return tune_rain_correction_on_validation_normal(models, y_va, y_pred_va, masks_va, intensity_p90=intensity_p90)
    elif mode == "de":
        return tune_rain_correction_on_validation_de(models, y_va, y_pred_va, masks_va, intensity_p90=intensity_p90)
    elif mode == "multistart":

        return tune_rain_correction_on_validation_multistart(models, y_va, y_pred_va, masks_va, intensity_p90=intensity_p90)
    else:
        raise ValueError(f"Unknown tuning mode: {mode}")


def tune_rain_correction_on_validation_normal(models, y_va, y_pred_va, masks_va, intensity_p90=None):
    param_grid = {
        "beta_base":      [0.20, 0.35, 0.50, 0.65],
        "beta_rain_gain": [0.40, 0.60, 0.80, 1.00],
        "instant_base":   [0.00, 0.02, 0.05, 0.08],
        "instant_gain":   [0.03, 0.05, 0.08, 0.10],
        "first_base":     [0.00, 0.02, 0.05, 0.08],
        "first_gain":     [0.03, 0.05, 0.08, 0.10],
        "max_adjust":     [0.20, 0.30, 0.40, 0.50],
    }

    keys = list(param_grid.keys())
    best_loss = np.inf
    best_params = None
    best_eval = None

    for values in itertools.product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, values))

        y_pred_adj_va = corrections.apply_rain_intensity_residual_correction(
            y_pred=y_pred_va,
            y_true=y_va,
            rain_mask=masks_va["rain"],
            rain_intensity=masks_va["rain_intensity"],
            intensity_p90=intensity_p90,
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

def tune_rain_correction_on_validation_de(models, y_va, y_pred_va, masks_va, intensity_p90=None):
    keys = [
        "beta_base",
        "beta_rain_gain",
        "instant_base",
        "instant_gain",
        "first_base",
        "first_gain",
        "max_adjust",
    ]

    bounds = [
        (0.0, 1.0),  # beta_base
        (0.0, 1.0),  # beta_rain_gain
        (0.0, 1.0),  # instant_base
        (0.0, 1.0),  # instant_gain
        (0.0, 1.0),  # first_base
        (0.0, 1.0),  # first_gain
        (0.0, 1.0),  # max_adjust
    ]

    # intensity_p90 = masks_va.get("intensity_p90", None)

    def objective(x):
        params = dict(zip(keys, x))

        y_pred_adj_va = corrections.apply_rain_intensity_residual_correction(
            y_pred=y_pred_va,
            y_true=y_va,
            rain_mask=masks_va["rain"],
            rain_intensity=masks_va["rain_intensity"],
            intensity_p90=intensity_p90,
            **params
        )

        eval_va = models.evaluate_from_predictions(
            y_true=y_va,
            y_pred=y_pred_adj_va,
            rain_mask=masks_va["rain"]
        )

        return correction_objective(eval_va)

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        polish=True,
        updating="deferred",
        workers=1,
    )

    best_params = dict(zip(keys, result.x))
    best_loss = result.fun

    y_pred_adj_va = corrections.apply_rain_intensity_residual_correction(
        y_pred=y_pred_va,
        y_true=y_va,
        rain_mask=masks_va["rain"],
        rain_intensity=masks_va["rain_intensity"],
        intensity_p90=intensity_p90,
        **best_params
    )

    best_eval = models.evaluate_from_predictions(
        y_true=y_va,
        y_pred=y_pred_adj_va,
        rain_mask=masks_va["rain"]
    )

    return best_params, best_loss, best_eval


import numpy as np
from scipy.optimize import minimize

def tune_rain_correction_on_validation_multistart(models, y_va, y_pred_va, masks_va, n_starts=20, seed=42, intensity_p90=None):
    rng = np.random.default_rng(seed)

    keys = [
        "beta_base",
        "beta_rain_gain",
        "instant_base",
        "instant_gain",
        "first_base",
        "first_gain",
        "max_adjust",
    ]

    bounds = [(-2.0, 2.0)] * len(keys)

    def objective(x):
        params = dict(zip(keys, x))

        y_pred_adj_va = corrections.apply_rain_intensity_residual_correction(
            y_pred=y_pred_va,
            y_true=y_va,
            rain_mask=masks_va["rain"],
            rain_intensity=masks_va["rain_intensity"],
            intensity_p90=intensity_p90,
            **params
        )

        eval_va = models.evaluate_from_predictions(
            y_true=y_va,
            y_pred=y_pred_adj_va,
            rain_mask=masks_va["rain"]
        )
        return correction_objective(eval_va)

    best_result = None

    for _ in range(n_starts):
        x0 = rng.uniform(-2.0, 2.0, size=len(keys))

        result = minimize(
            objective,
            x0=x0,
            method="Powell",   # 也可以改成 L-BFGS-B
            bounds=bounds,
            options={"maxiter": 1000, "disp": False},
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    best_params = dict(zip(keys, best_result.x))
    best_loss = best_result.fun

    y_pred_adj_va = corrections.apply_rain_intensity_residual_correction(
        y_pred=y_pred_va,
        y_true=y_va,
        rain_mask=masks_va["rain"],
        rain_intensity=masks_va["rain_intensity"],
        intensity_p90=intensity_p90,
        **best_params
    )

    best_eval = models.evaluate_from_predictions(
        y_true=y_va,
        y_pred=y_pred_adj_va,
        rain_mask=masks_va["rain"]
    )

    return best_params, best_loss, best_eval
