import tensorflow as tf
import math

# ============================================================
# Utils
# ============================================================
def _as_col(y):
    y = tf.convert_to_tensor(y)
    if len(y.shape) == 1:
        y = y[:, None]
    return y

def _softplus_pos(x, eps=1e-6):
    return tf.nn.softplus(x) + eps

def _relu_sum(x, axis=1):
    return tf.reduce_sum(tf.nn.relu(x), axis=axis)

# ============================================================
# 1) Quantile / Pinball family
# ============================================================

def pinball_multi_loss(qs=(0.05, 0.25, 0.75, 0.95), reduction="mean_sum"):
    """
    Multi-quantile pinball loss.
    y_true: (B,1), y_pred: (B,Q)
    reduction:
      - "mean_sum": mean over batch of sum over quantiles (你原本的樣子)
      - "mean_mean": mean over batch of mean over quantiles
    """
    qs_tf = tf.constant(list(qs), dtype=tf.float32)

    def loss(y_true, y_pred):
        y = _as_col(y_true)         # (B,1)
        q = tf.convert_to_tensor(y_pred)  # (B,Q)
        e = y - q                   # (B,Q)

        L = tf.maximum(qs_tf * e, (qs_tf - 1.0) * e)  # (B,Q)
        if reduction == "mean_mean":
            return tf.reduce_mean(tf.reduce_mean(L, axis=1))
        else:
            return tf.reduce_mean(tf.reduce_sum(L, axis=1))
    return loss


def pinball_with_order_penalty(
    qs=(0.05, 0.25, 0.75, 0.95),
    lamb=5.0,
    base_reduction="mean_sum",
):
    """
    Pinball + crossing penalty (防 quantile crossing)
    penalty = sum(ReLU(q_i - q_{i+1}))
    """
    base = pinball_multi_loss(qs=qs, reduction=base_reduction)

    def loss(y_true, y_pred):
        pin = base(y_true, y_pred)
        q = tf.convert_to_tensor(y_pred)
        diffs = q[:, :-1] - q[:, 1:]
        pen = tf.reduce_mean(_relu_sum(diffs, axis=1))
        return pin + lamb * pen
    return loss


def quantile_huber_loss(
    qs=(0.05, 0.25, 0.75, 0.95),
    delta=1.0,
    reduction="mean_sum",
):
    """
    Huberized pinball / Quantile Huber loss
    對尖峰/離群更穩
    """
    qs_tf = tf.constant(list(qs), dtype=tf.float32)

    def loss(y_true, y_pred):
        y = _as_col(y_true)
        q = tf.convert_to_tensor(y_pred)
        e = y - q  # (B,Q)
        abs_e = tf.abs(e)

        # huber(e)
        hub = tf.where(
            abs_e <= delta,
            0.5 * tf.square(e),
            delta * (abs_e - 0.5 * delta)
        )

        # asymmetric weighting like pinball
        L = tf.where(e >= 0.0, qs_tf * hub, (1.0 - qs_tf) * hub)

        if reduction == "mean_mean":
            return tf.reduce_mean(tf.reduce_mean(L, axis=1))
        else:
            return tf.reduce_mean(tf.reduce_sum(L, axis=1))
    return loss


def quantile_huber_with_order_penalty(
    qs=(0.05, 0.25, 0.75, 0.95),
    delta=1.0,
    lamb=5.0,
    reduction="mean_sum",
):
    """
    Quantile Huber + crossing penalty
    """
    base = quantile_huber_loss(qs=qs, delta=delta, reduction=reduction)

    def loss(y_true, y_pred):
        L = base(y_true, y_pred)
        q = tf.convert_to_tensor(y_pred)
        diffs = q[:, :-1] - q[:, 1:]
        pen = tf.reduce_mean(_relu_sum(diffs, axis=1))
        return L + lamb * pen
    return loss


# ============================================================
# 2) Interval / WIS family (直接優化區間品質)
# ============================================================

def interval_score_loss(alpha=0.10, lower_idx=0, upper_idx=3):
    """
    Interval score for a (1-alpha) PI [l,u].
    y_pred: quantiles output (B,Q)
    Example:
      alpha=0.10, lower=q0.05 idx=0, upper=q0.95 idx=3  -> 90% PI
    """
    a = float(alpha)

    def loss(y_true, y_pred):
        y = _as_col(y_true)
        q = tf.convert_to_tensor(y_pred)
        l = q[:, lower_idx:lower_idx+1]
        u = q[:, upper_idx:upper_idx+1]

        width = u - l
        below = tf.nn.relu(l - y)
        above = tf.nn.relu(y - u)
        score = width + (2.0 / a) * below + (2.0 / a) * above
        return tf.reduce_mean(score)
    return loss


def wis_loss(
    # 預設對應你的 4 quantiles: [0.05,0.25,0.75,0.95]
    intervals=(
        # (alpha, lower_idx, upper_idx, weight)
        (0.10, 0, 3, 0.7),  # 90% PI
        (0.50, 1, 2, 0.3),  # 50% PI
    )
):
    """
    Weighted Interval Score (WIS) for multiple prediction intervals.
    y_pred: (B,Q) quantiles
    """
    def loss(y_true, y_pred):
        y = _as_col(y_true)
        q = tf.convert_to_tensor(y_pred)

        total = 0.0
        wsum = 0.0
        for alpha, li, ui, w in intervals:
            l = q[:, li:li+1]
            u = q[:, ui:ui+1]
            width = u - l
            below = tf.nn.relu(l - y)
            above = tf.nn.relu(y - u)
            score = width + (2.0 / float(alpha)) * below + (2.0 / float(alpha)) * above
            total += float(w) * tf.reduce_mean(score)
            wsum += float(w)
        return total / max(wsum, 1e-12)
    return loss


def wis_plus_order_penalty(
    intervals=((0.10,0,3,0.7),(0.50,1,2,0.3)),
    lamb=5.0
):
    """
    WIS + crossing penalty
    """
    base = wis_loss(intervals=intervals)

    def loss(y_true, y_pred):
        L = base(y_true, y_pred)
        q = tf.convert_to_tensor(y_pred)
        diffs = q[:, :-1] - q[:, 1:]
        pen = tf.reduce_mean(_relu_sum(diffs, axis=1))
        return L + lamb * pen
    return loss


# ============================================================
# 3) "Soft coverage" / sharpness regularizers (可疊到任何 loss)
# ============================================================

def soft_coverage_penalty(
    # 使用你的 90% 區間（q05,q95）當例子
    lower_idx=0, upper_idx=3,
    target_coverage=0.90,
    temp=0.05,   # 越小越接近 hard indicator，但梯度更尖銳
    weight=1.0
):
    """
    讓 coverage 接近 target_coverage 的可微分正則項。
    coverage ≈ sigmoid((y-l)/t) * sigmoid((u-y)/t)
    penalty = (mean(coverage) - target)^2
    """
    tgt = float(target_coverage)
    t = float(temp)
    w = float(weight)

    def reg(y_true, y_pred):
        y = _as_col(y_true)
        q = tf.convert_to_tensor(y_pred)
        l = q[:, lower_idx:lower_idx+1]
        u = q[:, upper_idx:upper_idx+1]

        inside_soft = tf.sigmoid((y - l)/t) * tf.sigmoid((u - y)/t)
        cov = tf.reduce_mean(inside_soft)
        return w * tf.square(cov - tgt)
    return reg


def width_penalty(lower_idx=0, upper_idx=3, weight=1.0):
    """
    只懲罰區間寬度，讓區間更窄（會跟 coverage 拉扯）
    """
    w = float(weight)

    def reg(y_true, y_pred):
        q = tf.convert_to_tensor(y_pred)
        l = q[:, lower_idx:lower_idx+1]
        u = q[:, upper_idx:upper_idx+1]
        return w * tf.reduce_mean(u - l)
    return reg


def add_regularizers(base_loss_fn, regs):
    """
    base_loss_fn: callable(y_true,y_pred)->scalar
    regs: list of callable(y_true,y_pred)->scalar
    """
    def loss(y_true, y_pred):
        total = base_loss_fn(y_true, y_pred)
        for r in regs:
            total = total + r(y_true, y_pred)
        return total
    return loss


# ============================================================
# 4) Distributional NLL family (直接學分佈，比 quantile 更「概率」)
# ============================================================

def gaussian_nll():
    """
    Gaussian NLL
    y_pred: (B,2) = [mu, raw_sigma]
    sigma = softplus(raw_sigma)
    """
    log2pi = tf.constant(math.log(2.0 * math.pi), dtype=tf.float32)

    def loss(y_true, y_pred):
        y = _as_col(y_true)
        mu = y_pred[:, 0:1]
        sigma = _softplus_pos(y_pred[:, 1:2])
        z = (y - mu) / sigma
        nll = 0.5 * (tf.square(z) + 2.0 * tf.math.log(sigma) + log2pi)
        return tf.reduce_mean(nll)
    return loss


def laplace_nll():
    """
    Laplace NLL (比 Gaussian 更 robust，厚尾一些)
    y_pred: (B,2) = [mu, raw_b]
    b = softplus(raw_b)
    """
    def loss(y_true, y_pred):
        y = _as_col(y_true)
        mu = y_pred[:, 0:1]
        b = _softplus_pos(y_pred[:, 1:2])
        nll = tf.math.log(2.0*b) + tf.abs(y - mu) / b
        return tf.reduce_mean(nll)
    return loss


def student_t_nll():
    """
    Student-t NLL (厚尾很適合 throughput)
    y_pred: (B,3) = [mu, raw_sigma, raw_nu]
    sigma = softplus(raw_sigma)
    nu = 2 + softplus(raw_nu)   # >2 to have finite variance
    """
    log_pi = tf.constant(math.log(math.pi), dtype=tf.float32)

    def loss(y_true, y_pred):
        y = _as_col(y_true)
        mu = y_pred[:, 0:1]
        sigma = _softplus_pos(y_pred[:, 1:2])
        nu = 2.0 + _softplus_pos(y_pred[:, 2:3])  # >2

        z = (y - mu) / sigma

        term1 = tf.math.lgamma((nu + 1.0) / 2.0) - tf.math.lgamma(nu / 2.0)
        term2 = -0.5 * (tf.math.log(nu) + log_pi) - tf.math.log(sigma)
        term3 = - (nu + 1.0) / 2.0 * tf.math.log1p(tf.square(z) / nu)

        logp = term1 + term2 + term3
        return -tf.reduce_mean(logp)
    return loss


def gamma_nll():
    """
    Gamma NLL (目標為正值且偏態時可試)
    y_pred: (B,2) = [raw_k, raw_theta]
    k = softplus(raw_k)     (shape)
    theta = softplus(raw_theta)  (scale)
    logpdf = (k-1)log y - y/theta - k log theta - lgamma(k)
    """
    def loss(y_true, y_pred):
        y = _as_col(y_true)
        y = tf.maximum(y, 1e-6)
        k = _softplus_pos(y_pred[:, 0:1])
        theta = _softplus_pos(y_pred[:, 1:2])

        logp = (k - 1.0) * tf.math.log(y) - y / theta - k * tf.math.log(theta) - tf.math.lgamma(k)
        return -tf.reduce_mean(logp)
    return loss


def tweedie_deviance(p=1.5, eps=1e-6):
    """
    Tweedie deviance (1<p<2 常用於正值 + 偶爾接近0 的連續量)
    需要 y_pred 輸出 mean(mu) 且 mu>0（你可用 softplus）
    這裡用 deviance（非完整 NLL），實務也常用
    """
    p = float(p)
    assert 1.0 < p < 2.0, "Tweedie p typically in (1,2)"

    def loss(y_true, y_pred):
        y = _as_col(y_true)
        mu = tf.maximum(_as_col(y_pred), eps)
        y = tf.maximum(y, 0.0)

        # deviance up to constant
        term = (tf.pow(y, 2.0 - p) / ((1.0 - p) * (2.0 - p))
                - y * tf.pow(mu, 1.0 - p) / (1.0 - p)
                + tf.pow(mu, 2.0 - p) / (2.0 - p))
        return tf.reduce_mean(2.0 * term)
    return loss


# ============================================================
# 5) Mixture Density (MDN) Gaussian NLL (更彈性，但參數多)
# ============================================================

def mdn_gaussian_nll(K=3):
    """
    Mixture of K Gaussians NLL
    y_pred: (B, 3K) = [logits( K ), mu( K ), raw_sigma( K )] concat
      - logits: mixture weights before softmax
      - mu: means
      - raw_sigma: -> sigma = softplus
    """
    log2pi = tf.constant(math.log(2.0 * math.pi), dtype=tf.float32)
    K = int(K)

    def loss(y_true, y_pred):
        y = _as_col(y_true)  # (B,1)
        y_pred = tf.convert_to_tensor(y_pred)

        logits = y_pred[:, :K]                     # (B,K)
        mus = y_pred[:, K:2*K]                     # (B,K)
        raw_sigmas = y_pred[:, 2*K:3*K]            # (B,K)
        sigmas = _softplus_pos(raw_sigmas)         # (B,K)

        # log pi_k
        log_pi_k = tf.nn.log_softmax(logits, axis=1)  # (B,K)

        # log N(y | mu_k, sigma_k)
        yK = tf.repeat(y, repeats=K, axis=1)        # (B,K)
        z = (yK - mus) / sigmas
        log_norm = -0.5 * (tf.square(z) + 2.0 * tf.math.log(sigmas) + log2pi)  # (B,K)

        # log sum_k pi_k * N_k  (use log-sum-exp)
        log_mix = tf.reduce_logsumexp(log_pi_k + log_norm, axis=1)  # (B,)
        return -tf.reduce_mean(log_mix)
    return loss


# ============================================================
# 6) Mega combos (你可以當模板自己改權重)
# ============================================================

def mega_loss_example(
    # 你的 4 quantiles
    qs=(0.05,0.25,0.75,0.95),
    a=1.0,   # pinball weight
    b=0.3,   # WIS weight
    c=5.0,   # order penalty weight
):
    base_pin = pinball_multi_loss(qs=qs, reduction="mean_sum")
    base_wis = wis_loss(intervals=((0.10,0,3,0.7),(0.50,1,2,0.3)))

    def loss(y_true, y_pred):
        pin = base_pin(y_true, y_pred)
        wis = base_wis(y_true, y_pred)

        q = tf.convert_to_tensor(y_pred)
        diffs = q[:, :-1] - q[:, 1:]
        cross = tf.reduce_mean(_relu_sum(diffs, axis=1))

        return a*pin + b*wis + c*cross
    return loss
