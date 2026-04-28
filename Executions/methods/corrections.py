
import numpy as np

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


# def apply_rain_intensity_residual_correction(
#     y_pred,           # 假設 shape 為 (N, 3), 分別是 [lower, center, upper]
#     y_true,
#     rain_mask,
#     rain_intensity,   
#     beta_base=0.4,    
#     beta_rain_gain=0.8,   
#     instant_base=0.00,    
#     instant_gain=0.12,    
#     first_base=0.04,      
#     first_gain=0.10,      
#     max_adjust=0.5,
#     intensity_p90=None    
# ):
#     y_true = np.asarray(y_true).reshape(-1)
#     rain_mask = np.asarray(rain_mask).astype(bool)
#     rain_intensity = np.asarray(rain_intensity).reshape(-1)
#     y_pred_adj = np.asarray(y_pred, dtype=float).copy()

#     # 1. 雨勢正規化 (保留原邏輯)
#     if intensity_p90 is None:
#         positive = rain_intensity[rain_intensity > 0]
#         intensity_p90 = np.quantile(positive, 0.9) if len(positive) > 0 else 1.0

#     intensity_scale = np.clip(rain_intensity / max(intensity_p90, 1e-6), 0.0, 1.0)

#     for i in range(len(y_pred_adj)):
#         if not rain_mask[i]:
#             continue

#         total_adjust = 0.0
#         s = intensity_scale[i]

#         # --- A. 計算修正強度 (完全保留你原本的邏輯與參數) ---
        
#         # 1) 立即修正因子
#         instant_adjust = instant_base + instant_gain * s
#         total_adjust += instant_adjust

#         # 2) 剛進入雨天第一點修正
#         if i > 0 and (not rain_mask[i - 1]):
#             first_adjust = first_base + first_gain * s
#             total_adjust += first_adjust

#         # 3) 連續雨天殘差修正
#         direction = 1.0  # 預設方向：下修
#         if i > 0 and rain_mask[i - 1]:
#             prev_center = (y_pred_adj[i - 1, 0] + y_pred_adj[i - 1, 2]) / 2.0
#             actual_prev = y_true[i - 1]
            
#             # 算出誤差比例與方向
#             # diff > 0 代表高估 (需下修), diff < 0 代表低估 (需上修)
#             diff = prev_center - actual_prev
#             over_ratio = abs(diff) / max(abs(prev_center), 1e-6)
#             direction = np.sign(diff) # 核心改變：追蹤誤差方向

#             s_prev = intensity_scale[i - 1]
#             s_pair = max(s, s_prev)
#             beta_eff = beta_base + beta_rain_gain * s_pair
            
#             dynamic_adjust = min(beta_eff * over_ratio, max_adjust)
#             total_adjust += dynamic_adjust

#         # 限制最大修正率
#         total_adjust = min(total_adjust, max_adjust)

#         # --- B. 套用修正 (核心改動：中心位移法) ---
        
#         # 取得原始區間資訊
#         low, center, upp = y_pred_adj[i, 0], y_pred_adj[i, 1], y_pred_adj[i, 2]
#         original_width = upp - low
        
#         # 計算新的中心點：根據 direction 決定上移或下移
#         # 如果 direction 是 1 (高估)，則 center * (1 - total_adjust)
#         # 如果 direction 是 -1 (低估)，則 center * (1 + total_adjust)
#         new_center = center * (1.0 - (total_adjust * direction))
        
#         # 重新分配上下界，保持「原始寬度」不變
#         y_pred_adj[i, 1] = new_center
#         y_pred_adj[i, 0] = new_center - (original_width / 2.0)
#         y_pred_adj[i, 2] = new_center + (original_width / 2.0)

#     return y_pred_adj



def apply_rain_intensity_residual_correction(
    y_pred,           # Shape (N, 3) -> [Lower, Center, Upper]
    y_true,
    rain_mask,
    rain_intensity,   
    beta_base=0.10,      # Slashing from 0.4 -> 0.1 (Be gentle)
    beta_rain_gain=0.15,   # Slashing from 0.8 -> 0.15
    instant_base=0.00,    
    instant_gain=0.05,    # Slashing from 0.12 -> 0.05
    first_base=0.02,      
    first_gain=0.05,      
    max_adjust=0.15,      # Hard cap at 15% instead of 50%
    intensity_p90=None    
):
    y_true = np.asarray(y_true).reshape(-1)
    rain_mask = np.asarray(rain_mask).astype(bool)
    rain_intensity = np.asarray(rain_intensity).reshape(-1)
    y_pred_adj = np.asarray(y_pred, dtype=float).copy()

    if intensity_p90 is None:
        positive = rain_intensity[rain_intensity > 0]
        intensity_p90 = np.quantile(positive, 0.9) if len(positive) > 0 else 1.0

    intensity_scale = np.clip(rain_intensity / max(intensity_p90, 1e-6), 0.0, 1.0)
    
    # Track a moving average of the error ratio to avoid jitter
    smoothed_error_ratio = 0.0
    ema_alpha = 0.3  # How much to trust the newest error (0.3 = 30%)

    for i in range(len(y_pred_adj)):
        if not rain_mask[i]:
            smoothed_error_ratio = 0.0 # Reset when it stops raining
            continue

        s = intensity_scale[i]
        total_adjust = 0.0

        # 1) Base Rain Bias (Static)
        total_adjust += (instant_base + instant_gain * s)

        # 2) Entry Point Bias
        if i > 0 and (not rain_mask[i - 1]):
            total_adjust += (first_base + first_gain * s)

        # 3) Residual Tracking (Dynamic)
        if i > 0 and rain_mask[i - 1]:
            prev_center = y_pred[i - 1, 1] # Use original y_pred to avoid feedback loops
            actual_prev = y_true[i - 1]
            
            raw_diff = prev_center - actual_prev
            raw_error_ratio = raw_diff / max(abs(prev_center), 1e-6)
            
            # Update EMA: smoothed = (1-a)*old + a*new
            smoothed_error_ratio = (1 - ema_alpha) * smoothed_error_ratio + ema_alpha * raw_error_ratio
            
            s_prev = intensity_scale[i - 1]
            s_pair = max(s, s_prev)
            beta_eff = beta_base + beta_rain_gain * s_pair
            
            # Limit the dynamic part
            dynamic_part = np.clip(beta_eff * smoothed_error_ratio, -max_adjust, max_adjust)
            dynamic_adjust = dynamic_part
            total_adjust += dynamic_adjust
            # Note: total_adjust here can be negative (upward shift) or positive (downward shift)

        # Final safety clamp on the multiplier
        # We use (1 - total_adjust), so if total_adjust is 0.1, we lower the center by 10%
        final_multiplier = 1.0 - np.clip(total_adjust, -max_adjust, max_adjust)

        # --- APPLYING THE CORRECTION ---
        # 1. Lock original width from the RAW input
        low_orig, center_orig, upp_orig = y_pred[i, 0], y_pred[i, 1], y_pred[i, 2]
        original_width = upp_orig - low_orig
        
        # 2. Shift the center
        new_center = center_orig * final_multiplier
        
        # 3. Reconstruct bounds using the original width (Center-anchored)
        y_pred_adj[i, 1] = new_center
        y_pred_adj[i, 0] = new_center - (original_width / 2.0)
        y_pred_adj[i, 2] = new_center + (original_width / 2.0)

    return y_pred_adj
    


def correction_strength(
    s,                 # normalized intensity
    is_first_rain,
    rain_run_length,
    pred_center,
    center_scale,
    gamma=2.0,
    first_bonus=0.08,
    run_gain=0.12,
    level_gain=0.08,
    max_adjust=0.5,
):
    s_eff = s ** gamma
    run_eff = min(rain_run_length / 6.0, 1.0)
    level_eff = np.clip(pred_center / max(center_scale, 1e-6), 0.0, 1.5)

    adj = 0.0
    adj += 0.05 * s_eff
    adj += run_gain * s_eff * run_eff
    adj += level_gain * s_eff * level_eff

    if is_first_rain:
        adj += first_bonus * s_eff

    return min(adj, max_adjust)