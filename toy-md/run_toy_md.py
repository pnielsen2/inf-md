import json
import os
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import argparse


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def Q_of_a(a, params):
    return -0.5 * params.qA * (a - params.mu_Q) ** 2


def base_policy_density(a, params):
    return gaussian_pdf(a, params.mu_a, params.sigma_a)


def base_policy_logpdf(a, params):
    return gaussian_logpdf(a, params.mu_a, params.sigma_a)


def cond_sprime_given_a_logpdf(sprime, a, s: float, params):
    mean = params.alpha * a + params.d * s + params.mu_eps
    return gaussian_logpdf(sprime, mean, params.sigma_sprime)


def tilt_action_closed_form(params) -> Tuple[float, float]:
    # pi_beta(a|s) ∝ q(a|s) * exp(beta * Q(a))
    # With Gaussian q and quadratic Q we have a Gaussian with updated precision and mean
    tau_q = 1.0 / (params.sigma_a ** 2)
    tau_r = params.lambda_scale * params.qA
    tau_tilt = tau_q + tau_r
    mu_tilt = (tau_q * params.mu_a + tau_r * params.mu_Q) / tau_tilt
    sigma_tilt = np.sqrt(1.0 / tau_tilt)
    return mu_tilt, sigma_tilt


def sample_base_joint(N: int, s: float, params) -> Tuple[np.ndarray, np.ndarray]:
    a = np.random.normal(params.mu_a, params.sigma_a, size=N)
    sprime_mean = params.alpha * a + params.d * s + params.mu_eps
    sprime = np.random.normal(sprime_mean, params.sigma_sprime)
    return a, sprime


def sample_tilted_joint_exact(N: int, s: float, params) -> Tuple[np.ndarray, np.ndarray]:
    mu_tilt, sigma_tilt = tilt_action_closed_form(params)
    a = np.random.normal(mu_tilt, sigma_tilt, size=N)
    sprime_mean = params.alpha * a + params.d * s + params.mu_eps
    sprime = np.random.normal(sprime_mean, params.sigma_sprime)
    return a, sprime


def importance_weights_from_base(a, params):
    # w ∝ exp(beta Q(a))
    w = np.exp(params.lambda_scale * Q_of_a(a, params))
    w /= np.maximum(w.mean(), 1e-300)
    return w


def weighted_mean_and_var(x, w) -> Tuple[float, float]:
    w = np.asarray(w)
    w = np.maximum(w, 0)
    w_sum = w.sum()
    if w_sum == 0:
        return float('nan'), float('nan')
    w_norm = w / w_sum
    mean = np.sum(w_norm * x)
    var = np.sum(w_norm * (x - mean) ** 2)
    return mean, var


def bivariate_logpdf_base(a, sprime, s: float, params):
    return base_policy_logpdf(a, params) + cond_sprime_given_a_logpdf(sprime, a, s, params)


def bivariate_logpdf_tilt(a, sprime, s: float, params):
    mu_tilt, sigma_tilt = tilt_action_closed_form(params)
    log_pi_a = gaussian_logpdf(a, mu_tilt, sigma_tilt)
    return log_pi_a + cond_sprime_given_a_logpdf(sprime, a, s, params)


def grid_logratio(a_grid, sprime_grid, s: float, params):
    A, S = np.meshgrid(a_grid, sprime_grid, indexing='xy')
    log_tilt = bivariate_logpdf_tilt(A, S, s, params)
    log_base = bivariate_logpdf_base(A, S, s, params)
    return log_tilt - log_base


def ks_test_conditionals(a_samples, sprime_samples,
                         a_samples_tilt, sprime_samples_tilt,
                         a0: float, width: float, s: float, params) -> Dict[str, float]:
    # Select bands |a - a0| <= width in each set
    idx_base = np.where(np.abs(a_samples - a0) <= width)[0]
    idx_tilt = np.where(np.abs(a_samples_tilt - a0) <= width)[0]
    s_base = sprime_samples[idx_base]
    s_tilt = sprime_samples_tilt[idx_tilt]

    # Analytic conditional
    mu = params.alpha * a0 + params.d * s + params.mu_eps
    scale = params.sigma_sprime

    # Sample from the analytic conditional to compare via KS in a controlled way
    M = max(len(s_base), len(s_tilt), 200)
    s_analytic = np.random.normal(mu, scale, size=M)

    # KS tests (two-sample statistic only)
    ks_base = ks_2samp_statistic(s_base, s_analytic) if len(s_base) > 10 else np.nan
    ks_tilt = ks_2samp_statistic(s_tilt, s_analytic) if len(s_tilt) > 10 else np.nan

    return {
        'a0': float(a0),
        'count_base': int(len(s_base)),
        'count_tilt': int(len(s_tilt)),
        'ks_base_vs_analytic': float(ks_base) if not np.isnan(ks_base) else float('nan'),
        'ks_tilt_vs_analytic': float(ks_tilt) if not np.isnan(ks_tilt) else float('nan'),
        'mu_cond': float(mu),
        'sigma_cond': float(scale),
    }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_joint_heat(ax, a, sprime, title: str):
    h = ax.hist2d(a, sprime, bins=120, cmap='viridis', density=True)
    ax.set_xlabel("action a")
    ax.set_ylabel("next state s'")
    ax.set_title(title)
    plt.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)


def plot_action_marginals(ax, params, a_base, w, a_tilt_exact):
    a_grid = np.linspace(-4*params.sigma_a + params.mu_a + min(-2, params.mu_Q - 3),
                         4*params.sigma_a + params.mu_a + max(2, params.mu_Q + 3), 400)
    mu_tilt, sigma_tilt = tilt_action_closed_form(params)

    ax.plot(a_grid, gaussian_pdf(a_grid, params.mu_a, params.sigma_a), label='base q(a)', color='C0')
    ax.plot(a_grid, gaussian_pdf(a_grid, mu_tilt, sigma_tilt), label='tilt pi_beta(a) exact', color='C3')

    # Weighted histogram from base samples
    bins = 60
    ax.hist(a_base, bins=bins, density=True, weights=w / w.mean(), alpha=0.35, label='tilt via importance (reweighted)')

    # Histogram of exact tilted samples
    ax.hist(a_tilt_exact, bins=bins, density=True, alpha=0.25, label='tilt exact (samples)')

    ax.set_xlabel("action a")
    ax.set_ylabel("density")
    ax.set_title("Action marginals: base vs tilted")
    ax.legend()


def plot_logratio(ax, a_grid, sprime_grid, s: float, params):
    LR = grid_logratio(a_grid, sprime_grid, s, params)
    im = ax.imshow(LR.T, origin='lower', aspect='auto',
                   extent=[a_grid.min(), a_grid.max(), sprime_grid.min(), sprime_grid.max()],
                   cmap='coolwarm')
    ax.set_xlabel("action a")
    ax.set_ylabel("next state s'")
    ax.set_title("log(tilt/base) = log pi_beta(a) - log q(a) (vertical stripes)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_conditionals(axs, a0_list, width, a_base, sprime_base, a_tilt, sprime_tilt, s, params):
    for ax, a0 in zip(axs, a0_list):
        idx_b = np.where(np.abs(a_base - a0) <= width)[0]
        idx_t = np.where(np.abs(a_tilt - a0) <= width)[0]
        s_b = sprime_base[idx_b]
        s_t = sprime_tilt[idx_t]

        mu = params.alpha * a0 + params.d * s + params.mu_eps
        x = np.linspace(mu - 4*params.sigma_sprime, mu + 4*params.sigma_sprime, 400)
        y = gaussian_pdf(x, mu, params.sigma_sprime)

        if len(s_b) > 0:
            ax.hist(s_b, bins=30, density=True, alpha=0.4, label="base | a≈{:.2f}".format(a0))
        if len(s_t) > 0:
            ax.hist(s_t, bins=30, density=True, alpha=0.4, label="tilt | a≈{:.2f}".format(a0))
        ax.plot(x, y, 'k-', lw=2, label='analytic N(mu, sigma)')
        ax.set_title("Conditionals at a≈{:.2f}".format(a0))
        ax.set_xlabel("s'")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)


# ---- Utility statistical helpers (no SciPy) ----
def gaussian_pdf(x, mu, sigma):
    x = np.asarray(x)
    sigma = max(float(sigma), 1e-12)
    coef = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    z = (x - mu) / sigma
    return coef * np.exp(-0.5 * z * z)


def gaussian_logpdf(x, mu, sigma):
    x = np.asarray(x)
    sigma = max(float(sigma), 1e-12)
    return -0.5 * np.log(2.0 * np.pi * sigma * sigma) - 0.5 * ((x - mu) / sigma) ** 2


# ---- Energy-based marginal target for a at level t ----
def energy_a(a, lvl, eps: float = 0.0, omega: float = 2.0):
    tau = 1.0 / max(lvl['var_a'], 1e-12)
    mu = lvl['m_a']
    return 0.5 * tau * (a - mu) ** 2 + eps * np.sin(omega * a)


def grad_energy_a(a, lvl, eps: float = 0.0, omega: float = 2.0):
    tau = 1.0 / max(lvl['var_a'], 1e-12)
    mu = lvl['m_a']
    return tau * (a - mu) + eps * omega * np.cos(omega * a)


def Q_hat(a, lvl, params):
    # Quadratic surrogate at level t: Q_t(a) = -0.5 * (A a^2 - 2 b a) + const
    A = lvl['A']; b = lvl['b']
    return -0.5 * (A * a * a - 2.0 * b * a)


def log_unnorm_pi_t(a, lvl, params, sps=None):
    # Unnormalized log-target for the marginal π_t(a) ∝ q_t(a) exp{β_t Q_hat(a)}
    return -energy_a(a, lvl, eps=0.0, omega=2.0) + lvl['beta_t'] * Q_hat(a, lvl, params)


def log_unnorm_joint_t(a: float, sps: np.ndarray, lvl, params, tempered: bool = False):
    # Unnormalized joint log-target for π_t(a, s') ∝ q_t(a) [∏_i q_t(s'_i|a)] exp{β_t Q_t(a)}
    rho, c, var_cond = lvl['rho'], lvl['c'], lvl['var_cond']
    resid = sps - (rho * a + c)
    if tempered:
        ll_s = -0.5 / max(var_cond, 1e-12) * (np.mean(resid * resid) if resid.size else 0.0)
    else:
        ll_s = -0.5 / max(var_cond, 1e-12) * np.sum(resid * resid)
    return -energy_a(a, lvl, eps=0.0, omega=2.0) + ll_s + lvl['beta_t'] * Q_hat(a, lvl, params)


def to_u(a: float, sps: np.ndarray, lvl):
    # Whiten s': u = (s' - (rho a + c)) / sigma, with sigma^2 = var_cond
    sigma = np.sqrt(max(lvl['var_cond'], 1e-12))
    u = (sps - (lvl['rho'] * a + lvl['c'])) / sigma
    return u, sigma


def from_u(a: float, u: np.ndarray, lvl, sigma: float):
    # Map back to s' from whitened u
    return lvl['rho'] * a + lvl['c'] + sigma * u


def ks_2samp_statistic(x, y):
    x = np.sort(np.asarray(x))
    y = np.sort(np.asarray(y))
    n1 = x.size
    n2 = y.size
    if n1 == 0 or n2 == 0:
        return np.nan
    data_all = np.concatenate((x, y))
    cdf1 = np.searchsorted(x, data_all, side='right') / n1
    cdf2 = np.searchsorted(y, data_all, side='right') / n2
    d = np.max(np.abs(cdf1 - cdf2))
    return float(d)


# ---- Joint base Gaussian and VP schedule helpers ----
def joint_base_mean_cov(params):
    mu_a = params.mu_a
    mu_s = params.alpha * params.mu_a + params.d * 0.0 + params.mu_eps
    var_a = params.sigma_a ** 2
    cov_as = params.alpha * var_a
    var_s = (params.alpha ** 2) * var_a + (params.sigma_sprime ** 2)
    mu0 = np.array([mu_a, mu_s])
    C0 = np.array([[var_a, cov_as], [cov_as, var_s]])
    return mu0, C0


def vp_alpha_bar_grid(params):
    K = params.K
    alphas = np.linspace(params.alpha_bar_max, params.alpha_bar_min, K)
    return alphas


def q_t_gaussian_params(alpha_bar: float, params):
    mu0, C0 = joint_base_mean_cov(params)
    m_t = np.sqrt(alpha_bar) * mu0
    C_t = alpha_bar * C0 + (1.0 - alpha_bar) * np.eye(2)
    return m_t, C_t


def score_q_t(x, m_t, C_t):
    iC = np.linalg.inv(C_t)
    return -iC.dot((x - m_t))


def marginal_action_params_from_joint(m_t, C_t):
    m_a = m_t[0]
    var_a = C_t[0, 0]
    return float(m_a), float(var_a)


def conditional_sprime_given_a_params(m_t, C_t):
    m_a = m_t[0]
    m_s = m_t[1]
    Caa = C_t[0, 0]
    Css = C_t[1, 1]
    Cas = C_t[0, 1]
    rho = Cas / max(Caa, 1e-12)
    var_cond = Css - Cas * Cas / max(Caa, 1e-12)
    c = m_s - rho * m_a
    return float(rho), float(c), float(var_cond)


def reward_proxy(a, s, params):
    return -0.5 * (params.r_wa * (a - params.r_a_star) ** 2 + params.r_ws * (s - params.r_s_star) ** 2)


def Qt_and_grad(alpha_bar: float, params):
    m_t, C_t = q_t_gaussian_params(alpha_bar, params)
    rho, c, var_cond = conditional_sprime_given_a_params(m_t, C_t)
    A = params.r_wa + params.r_ws * (rho ** 2)
    b = params.r_wa * params.r_a_star + params.r_ws * rho * (params.r_s_star - c)
    def Q(a):
        return -0.5 * (A * a * a - 2.0 * b * a) + const_part
    const_part = -0.5 * (params.r_ws * var_cond + params.r_wa * params.r_a_star ** 2 + params.r_ws * (params.r_s_star - c) ** 2)
    def dQ(a):
        return -A * a + b
    return A, b, const_part, Q, dQ, (rho, c, var_cond)


def beta_fraction(alpha_bar: float, params) -> float:
    # progress r: 0 at alpha_min (noisiest), 1 at alpha=1 (clean)
    r = (alpha_bar - params.alpha_bar_min) / max(1.0 - params.alpha_bar_min, 1e-12)
    r = max(0.0, min(1.0, r))
    sched = params.beta_schedule
    if sched == 'linear':
        return r
    elif sched == 'late_constant':
        return 1.0 if (params.window_alpha_low <= alpha_bar <= params.window_alpha_high) else 0.0
    elif sched == 'late_ramp':
        if alpha_bar <= params.window_alpha_low:
            return 0.0
        if alpha_bar >= params.window_alpha_high:
            return 1.0
        num = (alpha_bar - params.window_alpha_low)
        den = max(params.window_alpha_high - params.window_alpha_low, 1e-12)
        return max(0.0, min(1.0, (num / den) ** params.ramp_p))
    elif sched == 'tfg_plus_final_mala':
        # emulate σ^p shape that decays to 0 at clean: use 1-r
        return max(0.0, min(1.0, (1.0 - r) ** params.tfg_p))
    elif sched == 'sigma_power':
        sig = np.sqrt(max(1.0 - alpha_bar, 0.0))
        return sig ** getattr(params, 'sigma_p', 2.0)
    else:
        return r


def pi_beta_t_action_params(alpha_bar: float, params):
    m_t, C_t = q_t_gaussian_params(alpha_bar, params)
    m_a, var_a = marginal_action_params_from_joint(m_t, C_t)
    tau_q = 1.0 / max(var_a, 1e-12)
    A, b, _, _, _, _ = Qt_and_grad(alpha_bar, params)
    # schedule for beta (fraction in [0,1]), scaled by lambda
    frac = beta_fraction(alpha_bar, params)
    beta_t = params.lambda_scale * frac
    tau_r = beta_t * A
    tau_tilt = tau_q + tau_r
    mu_tilt = (tau_q * m_a + beta_t * b) / max(tau_tilt, 1e-12)
    sigma_tilt = np.sqrt(1.0 / max(tau_tilt, 1e-12))
    return mu_tilt, sigma_tilt, beta_t


def tilted_clean_joint_params(params):
    mu_tilt, sigma_tilt = clean_poe_tilt_from_Q(params)
    c = params.d * 0.0 + params.mu_eps
    m = np.array([mu_tilt, params.alpha * mu_tilt + c])
    vA = sigma_tilt ** 2
    C = np.array([[vA, params.alpha * vA],
                  [params.alpha * vA, (params.alpha ** 2) * vA + (params.sigma_sprime ** 2)]])
    return m, C


def sample_grad_Q(a: float, sps: np.ndarray, params) -> float:
    # Monte Carlo gradient of E[R(a, s')], using current s' population.
    # R(a,s') = -0.5 * (r_wa (a - r_a*)^2 + r_ws (s' - r_s*)^2)
    # d/da R = -r_wa (a - r_a*). Independent of s', but we keep the structure for generality.
    return -params.r_wa * (a - params.r_a_star)


def sample_grad_Q_with_jvp(a: float, sps: np.ndarray, rho: float, params) -> float:
    # JVP-style gradient using s' population: d/da R(a, s') ≈ E[-r_wa (a-a*) + (-r_ws (s'-s*))*∂s'/∂a]
    # For a well-trained denoiser in this toy, ∂s'/∂a ≈ rho (conditional mean coefficient)
    term_direct = -params.r_wa * (a - params.r_a_star)
    term_chain = -params.r_ws * (np.mean(sps) - params.r_s_star) * rho
    return term_direct + term_chain


def clean_poe_tilt_from_Q(params):
    # Reference MD target at t=0 using Q_0(a) from Qt_and_grad(alpha_bar=1)
    alpha_bar = 1.0
    m_t, C_t = q_t_gaussian_params(alpha_bar, params)
    m_a, var_a = marginal_action_params_from_joint(m_t, C_t)
    A, b, *_ = Qt_and_grad(alpha_bar, params)  # Q0(a) = -0.5(A a^2 - 2 b a) + const
    tau_q = 1.0 / max(var_a, 1e-12)
    tau_r = params.lambda_scale * A
    tau = tau_q + tau_r
    mu = (tau_q * m_a + params.lambda_scale * b) / max(tau, 1e-12)
    sigma = np.sqrt(1.0 / max(tau, 1e-12))
    return mu, sigma


def gaussian_cdf(x, mu, sigma):
    z = (x - mu) / (sigma * np.sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


def erf(z):
    # vectorized wrapper around math.erf
    return np.vectorize(math.erf)(z)


def kl_hist_vs_gaussian(samples: np.ndarray, mu: float, sigma: float, bins: int = 100):
    # histogram-based KL(p||q), where q is Gaussian. q bin masses via CDF differences.
    counts, edges = np.histogram(samples, bins=bins, density=False)
    N = np.sum(counts)
    if N == 0:
        return np.nan
    p = counts.astype(np.float64) / float(N)
    # q_i = ∫_bin N(mu,sigma^2) dx = CDF(edge_{i+1}) - CDF(edge_i)
    cdf_edges = gaussian_cdf(edges, mu, sigma)
    q = np.maximum(cdf_edges[1:] - cdf_edges[:-1], 1e-12)
    p = np.maximum(p, 1e-12)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def log_pi_action(a, params):
    # log πβ(a) = log q(a) + β Q(a)
    return base_policy_logpdf(a, params) + params.lambda_scale * Q_of_a(a, params)


def grad_log_pi_action(a, params):
    # ∇ log q(a) for Gaussian N(μ_a, σ_a^2): -(a-μ_a)/σ_a^2
    # ∇ Q(a) for -0.5 qA (a-μ_Q)^2: -qA (a-μ_Q)
    grad_log_q = -(a - params.mu_a) / (params.sigma_a ** 2)
    grad_Q = -params.qA * (a - params.mu_Q)
    return grad_log_q + params.lambda_scale * grad_Q


def sample_conditional_sprime(a, s: float, params):
    mean = params.alpha * a + params.d * s + params.mu_eps
    return np.random.normal(mean, params.sigma_sprime, size=np.shape(a))


def run_independence_mh_actions(num_samples: int, burn: int, s: float, params):
    # Independence MH with proposal q(a) and target πβ(a) ∝ q(a) exp{β Q(a)}
    # Acceptance α = min(1, exp(β[Q(a')-Q(a)]))
    total = burn + num_samples
    a_curr = np.random.normal(params.mu_a, params.sigma_a)
    q_curr = Q_of_a(a_curr, params)
    a_samples = np.empty(num_samples)
    sprime_samples = np.empty(num_samples)
    accepts = 0
    out_idx = 0
    for t in range(total):
        a_prop = np.random.normal(params.mu_a, params.sigma_a)
        q_prop = Q_of_a(a_prop, params)
        log_alpha = params.lambda_scale * (q_prop - q_curr)
        if np.log(np.random.rand()) < min(0.0, log_alpha):
            a_curr = a_prop
            q_curr = q_prop
            accepts += 1 if t >= burn else 0
        if t >= burn:
            a_samples[out_idx] = a_curr
            sprime_samples[out_idx] = np.random.normal(params.alpha * a_curr + params.d * s + params.mu_eps,
                                                       params.sigma_sprime)
            out_idx += 1
    acc_rate = accepts / max(1, num_samples)
    return a_samples, sprime_samples, acc_rate


def run_ula_actions(num_samples: int, burn: int, step_size: float, s: float, params):
    total = burn + num_samples
    a = np.random.normal(params.mu_a, params.sigma_a)
    a_samples = np.empty(num_samples)
    sprime_samples = np.empty(num_samples)
    out_idx = 0
    sqrt2eta = np.sqrt(2.0 * step_size)
    for t in range(total):
        grad = grad_log_pi_action(a, params)
        a = a + step_size * grad + sqrt2eta * np.random.randn()
        if t >= burn:
            a_samples[out_idx] = a
            sprime_samples[out_idx] = np.random.normal(params.alpha * a + params.d * s + params.mu_eps,
                                                       params.sigma_sprime)
            out_idx += 1
    return a_samples, sprime_samples


def run_mala_actions(num_samples: int, burn: int, step_size: float, s: float, params):
    total = burn + num_samples
    a = np.random.normal(params.mu_a, params.sigma_a)
    a_samples = np.empty(num_samples)
    sprime_samples = np.empty(num_samples)
    out_idx = 0
    accepts = 0
    sd = np.sqrt(2.0 * step_size)
    for t in range(total):
        grad = grad_log_pi_action(a, params)
        mean_prop = a + step_size * grad
        a_prop = np.random.normal(mean_prop, sd)
        # Compute MH log acceptance
        log_pi_prop = log_pi_action(a_prop, params)
        log_pi_curr = log_pi_action(a, params)
        # Reverse/forward proposal densities
        grad_prop = grad_log_pi_action(a_prop, params)
        mean_rev = a_prop + step_size * grad_prop
        log_q_forward = gaussian_logpdf(a_prop, mean_prop, sd)
        log_q_reverse = gaussian_logpdf(a, mean_rev, sd)
        log_alpha = (log_pi_prop - log_pi_curr) + (log_q_reverse - log_q_forward)
        if np.log(np.random.rand()) < min(0.0, log_alpha):
            a = a_prop
            if t >= burn:
                accepts += 1
        if t >= burn:
            a_samples[out_idx] = a
            sprime_samples[out_idx] = np.random.normal(params.alpha * a + params.d * s + params.mu_eps,
                                                       params.sigma_sprime)
            out_idx += 1
    acc_rate = accepts / max(1, num_samples)
    return a_samples, sprime_samples, acc_rate


def run_gibbs_joint(num_samples: int, burn: int, s: float, params):
    """
    Gibbs sampler for the tilted joint π(a,s'|s) = π(a|s) q(s'|a,s).
    Steps:
      s' ~ q(s'|a,s)
      a  ~ π(a|s') ∝ π(a|s) q(s'|a,s), which is Gaussian since both are Gaussian in a.
    """
    mu_tilt, sigma_tilt = tilt_action_closed_form(params)
    tau_tilt = 1.0 / (sigma_tilt ** 2)
    total = burn + num_samples
    # Initialize a
    a = np.random.normal(mu_tilt, sigma_tilt)
    a_samples = np.empty(num_samples)
    sprime_samples = np.empty(num_samples)
    out_idx = 0
    c = params.d * s + params.mu_eps
    for t in range(total):
        # s' | a
        sprime = np.random.normal(params.alpha * a + c, params.sigma_sprime)
        # a | s' is Gaussian with precision tau = tau_tilt + alpha^2/sigma_s'^2
        tau = tau_tilt + (params.alpha ** 2) / (params.sigma_sprime ** 2)
        mu_num = tau_tilt * mu_tilt + params.alpha * (sprime - c) / (params.sigma_sprime ** 2)
        mu_post = mu_num / tau
        sigma_post = np.sqrt(1.0 / tau)
        a = np.random.normal(mu_post, sigma_post)
        if t >= burn:
            a_samples[out_idx] = a
            sprime_samples[out_idx] = sprime
            out_idx += 1
    return a_samples, sprime_samples


def main():
    # Argparse-only configuration (defaults inlined)
    parser = argparse.ArgumentParser(description='Toy MD with diffusion PC visualizations')
    # 1) Mirror-descent guidance (most tuned / impactful)
    parser.add_argument('--beta-schedule', dest='beta_schedule', type=str,
                        choices=['linear','late_constant','late_ramp','tfg_plus_final_mala','sigma_power'],
                        default='linear',
                        help='Guidance schedule for per-level MD strength: linear 0→1; late_constant active only in [window_alpha_low,window_alpha_high]; late_ramp ramps to 1 within window; tfg_plus_final_mala decays during PC and uses a final clean MALA to enforce endpoint.')
    parser.add_argument('--lambda-scale', dest='lambda_scale', type=float, default=1.5,
                        help='Overall MD strength at the clean endpoint. Multiplies the schedule fraction. Default preserves previous β=1.5 behavior.')
    parser.add_argument('--window-alpha-low', dest='window_alpha_low', type=float, default=0.0,
                        help='Lower ᾱ bound for late_* schedules. Guidance inactive below.')
    parser.add_argument('--window-alpha-high', dest='window_alpha_high', type=float, default=1.0,
                        help='Upper ᾱ bound for late_* schedules. Guidance saturates by this level.')
    parser.add_argument('--ramp-p', dest='ramp_p', type=float, default=2.0,
                        help='Exponent for late_ramp schedule (e.g., 1=linear, 2=quadratic).')
    parser.add_argument('--tfg-p', dest='tfg_p', type=float, default=2.0,
                        help='Exponent for tfg_plus_final_mala fractional shape during PC (decays to 0 at clean).')
    parser.add_argument('--sigma-p', dest='sigma_p', type=float, default=2.0,
                        help='Exponent for sigma_power schedule: uses (sqrt(1-ᾱ_t))^p.')
    parser.add_argument('--final-mala-step', dest='final_mala_step', type=float, default=0.1,
                        help='One-step MALA step size for the clean-end correction when using tfg_plus_final_mala.')
    # 2) PC sampler/mixing controls
    parser.add_argument('--refresh-L', dest='refresh_L', type=int, default=3,
                        help='Number of s′ refresh ULA steps per PC level. Larger improves mixing but costs more denoiser evals.')
    parser.add_argument('--action-steps-per-level', dest='action_steps_per_level', type=int, default=1,
                        help='Number of consecutive action updates per noise level before state refreshes. For pc_joint_mala, this is the number of joint-MALA steps per level.')
    parser.add_argument('--ca', type=float, default=0.08,
                        help='Action step-size coefficient for PC (multiplied by (1-ᾱ)).')
    parser.add_argument('--cs', type=float, default=0.08,
                        help='State s′ refresh step-size coefficient for PC (multiplied by (1-ᾱ)).')
    parser.add_argument('--K', type=int, default=40,
                        help='Number of VP levels (PC steps).')
    parser.add_argument('--alpha-bar-min', dest='alpha_bar_min', type=float, default=0.05, help='Minimum ᾱ in VP grid (noisiest).')
    parser.add_argument('--alpha-bar-max', dest='alpha_bar_max', type=float, default=1.0, help='Maximum ᾱ in VP grid (clean).')
    # 3) Evaluation / diagnostics controls
    parser.add_argument('--kl-runs', type=int, default=7500,
                        help='Number of runs (actions) used to estimate the final action marginal and KL.')
    parser.add_argument('--kl-bins', type=int, default=80,
                        help='Histogram bins used by the KL estimator.')
    parser.add_argument('--sweep-max-refresh', type=int, default=10,
                        help='Max refresh_L value for the KL-vs-refresh sweep (0..max).')
    parser.add_argument('--sweep-runs', type=int, default=1500,
                        help='Runs per setting in the KL-vs-refresh sweep (per method).')
    parser.add_argument('--samplers', type=str, default='pc_ula,pc_mala',
                        help='Comma-separated list of samplers to run: choose from pc_ula, pc_mala, pc_joint_mala')
    # 4) Animation controls
    parser.add_argument('--seeds-ula', type=str, default='0,1,2,3',
                        help='Comma-separated RNG seeds for the 2x2 ULA animation panels.')
    parser.add_argument('--seeds-mala', type=str, default='4,5,6,7',
                        help='Comma-separated RNG seeds for the 2x2 MALA animation panels.')
    parser.add_argument('--points-per-seed', type=int, default=20,
                        help='Number of s′ particles visualized per panel (per seed).')
    # 5) Model / environment (less frequently tuned)
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Linear dynamics coefficient: s′|a,s ~ N(α a + d s + μ_ε, σ_s′^2).')
    parser.add_argument('--mu-a', dest='mu_a', type=float, default=0.0,
                        help='Base action prior mean μ_a for q(a|s).')
    parser.add_argument('--sigma-a', dest='sigma_a', type=float, default=1.0,
                        help='Base action prior std σ_a for q(a|s).')
    parser.add_argument('--sigma-sprime', dest='sigma_sprime', type=float, default=0.5,
                        help='Std σ_s′ for the conditional dynamics q(s′|a,s).')
    parser.add_argument('--d', type=float, default=0.0, help='Linear dependence of s′ on s in q(s′|a,s).')
    parser.add_argument('--mu-eps', dest='mu_eps', type=float, default=0.0, help='Additive mean in q(s′|a,s).')
    parser.add_argument('--qA', dest='qA', type=float, default=1.0, help='Quadratic coefficient in action-only Q(a).')
    parser.add_argument('--mu-Q', dest='mu_Q', type=float, default=1.5, help='Target mean in action-only Q(a).')
    parser.add_argument('--r-wa', dest='r_wa', type=float, default=1.0,
                        help='Reward weight on (a - r_a*)^2 in the quadratic proxy R.')
    parser.add_argument('--r-ws', dest='r_ws', type=float, default=1.0,
                        help='Reward weight on (s′ - r_s*)^2 in the quadratic proxy R.')
    parser.add_argument('--r-a-star', dest='r_a_star', type=float, default=1.5,
                        help='Reward target r_a* in the quadratic proxy R.')
    parser.add_argument('--r-s-star', dest='r_s_star', type=float, default=1.0,
                        help='Reward target r_s* in the quadratic proxy R.')
    parser.add_argument('--s', type=float, default=0.0,
                        help='Current state s (held fixed throughout).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Global RNG seed for reproducibility.')
    # VE visualization scale used by animations
    parser.add_argument('--sigma-ve-max', dest='sigma_ve_max', type=float, default=3.0, help='VE σ_max for visualization init/noisy overlays.')
    parser.add_argument('--sigma-ve-min', dest='sigma_ve_min', type=float, default=0.01, help='VE σ_min for visualization overlays.')
    args = parser.parse_args()

    # Apply overrides: use args Namespace directly as params
    set_seed(args.seed)
    params = args

    # Derived controls
    def parse_seeds(s):
        try:
            return tuple(int(x.strip()) for x in s.split(','))
        except Exception:
            return (0,1,2,3)
    seeds_ula = parse_seeds(args.seeds_ula)
    seeds_mala = parse_seeds(args.seeds_mala)
    def parse_methods(s: str):
        try:
            items = [t.strip() for t in s.split(',') if t.strip()]
        except Exception:
            return ['pc_ula', 'pc_mala']
        allowed = {'pc_ula', 'pc_mala', 'pc_joint_mala'}
        out = [m for m in items if m in allowed]
        return out if len(out)>0 else ['pc_ula', 'pc_mala']
    samplers = parse_methods(args.samplers)
    points_per_seed = args.points_per_seed
    kl_runs = args.kl_runs
    kl_bins = args.kl_bins
    sweep_max_refresh = args.sweep_max_refresh
    sweep_runs = args.sweep_runs

    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    ensure_dir(out_dir)

    # Current state s
    s = args.s

    # [Removed legacy static plots and demos: action_marginals_*.png, conditional_* , joint_* , logratio_heatmap.png, posterior_mean_demo_*, reward_proxy_clean.png]

    # ---- Build per-level cache (shared across animation and samplers) ----
    def build_level_cache(params):
        cache = []
        alpha_bars = vp_alpha_bar_grid(params)
        for alpha_bar in alpha_bars[::-1]:  # from noisy to clean
            m_t, C_t = q_t_gaussian_params(alpha_bar, params)
            m_a, var_a = marginal_action_params_from_joint(m_t, C_t)
            A, b, _, _, dQ, (rho, c, var_cond) = Qt_and_grad(alpha_bar, params)
            frac = beta_fraction(alpha_bar, params)
            beta_t = params.lambda_scale * frac
            tau_q = 1.0 / max(var_a, 1e-12)
            tau_r = beta_t * A
            tau_tilt = tau_q + tau_r
            mu_tilt = (tau_q * m_a + beta_t * b) / max(tau_tilt, 1e-12)
            sigma_tilt = np.sqrt(1.0 / max(tau_tilt, 1e-12))
            # per-level base step sizes
            eta_a = params.ca * max(var_a, 1e-12)
            cache.append({
                'alpha_bar': alpha_bar,
                'm_t': m_t, 'C_t': C_t,
                'm_a': m_a, 'var_a': var_a,
                'A': A, 'b': b, 'rho': rho, 'c': c, 'var_cond': var_cond,
                'beta_t': beta_t, 'mu_tilt': mu_tilt, 'sigma_tilt': sigma_tilt,
                'eta_a': eta_a,
            })
        return cache

    levels = build_level_cache(params)

    # ---- PC animations (PoE action guidance) ----
    def make_pc_animation(method_name='pc_ula', seeds=(0,1,2,3), points_per_seed=16, fname='anim_pc_ula.gif'):
        alpha_bars = vp_alpha_bar_grid(params)
        grids = np.linspace(-3, 3, 120)

        # Static clean densities (precompute once)
        mu0, C0 = joint_base_mean_cov(params)
        iC0 = np.linalg.inv(C0)
        m_tilt_clean, C_tilt_clean = tilted_clean_joint_params(params)
        iCt = np.linalg.inv(C_tilt_clean)

        # Precompute grids and static overlays
        Agrid, Sgrid = np.meshgrid(grids, grids, indexing='xy')
        X = np.stack([Agrid.ravel(), Sgrid.ravel()], axis=1)
        diff0 = X - mu0
        log_q0 = -0.5 * np.sum(diff0.dot(iC0) * diff0, axis=1)
        q0_img = np.exp(log_q0 - np.max(log_q0)).reshape(Agrid.shape)
        difft = X - m_tilt_clean
        log_tilt_clean = -0.5 * np.sum(difft.dot(iCt) * difft, axis=1)
        tilt_clean_img = np.exp(log_tilt_clean - np.max(log_tilt_clean)).reshape(Agrid.shape)

        # VE visualization schedule (noisy overlays)
        sigmas = np.linspace(params.sigma_ve_max, params.sigma_ve_min, params.K)

        # init per seed: same action, diverse s' (VE high-noise init)
        rngs = [np.random.RandomState(int(s0)) for s0 in seeds]
        groups = []
        a0 = 0.0
        for r in rngs:
            sps = r.normal(0.0, params.sigma_ve_max, size=points_per_seed)
            groups.append({'a': a0, 'sps': sps, 'rng': r})

        fig, axes = plt.subplots(2, 2, figsize=(9, 8))
        plt.close(fig)

        from PIL import Image
        imgs = []
        for k, (alpha_bar, lvl) in enumerate([(d['alpha_bar'], d) for d in levels]):  # from noisy to clean
            m_a = lvl['m_a']; var_a = lvl['var_a']
            beta_t = lvl['beta_t']
            mu_tilt = lvl['mu_tilt']; sigma_tilt = lvl['sigma_tilt']
            rho = lvl['rho']; c = lvl['c']; var_cond = lvl['var_cond']
            eta_a = lvl['eta_a']

            # Per-level tilted joint (green contours) for current level t
            var_a_lvl = sigma_tilt ** 2
            m_lvl = np.array([mu_tilt, rho * mu_tilt + c])
            C_lvl = np.array([[var_a_lvl, rho * var_a_lvl],
                              [rho * var_a_lvl, (rho ** 2) * var_a_lvl + var_cond]])
            iCn = np.linalg.inv(C_lvl)
            diffn = X - m_lvl
            log_tilt_level = -0.5 * np.sum(diffn.dot(iCn) * diffn, axis=1)
            tilt_level_img = np.exp(log_tilt_level - np.max(log_tilt_level)).reshape(Agrid.shape)
            # reward proxy
            R_img = reward_proxy(Agrid, Sgrid, params)

            # update groups
            for g in groups:
                if method_name == 'pc_joint_mala':
                    # Run multiple joint-MALA steps per level in whitened (a,u)
                    for _ in range(params.action_steps_per_level):
                        eta_a_j = eta_a / (1.0 + beta_t)
                        eta_u_j = params.ca
                        if eta_a_j <= 1e-12 or eta_u_j <= 1e-12:
                            continue
                        u, sigma = to_u(g['a'], g['sps'], lvl)
                        grad_a = -grad_energy_a(g['a'], lvl, eps=0.0, omega=2.0) + beta_t * (-lvl['A'] * g['a'] + lvl['b'])
                        grad_u = -u
                        mean_a = g['a'] + eta_a_j * grad_a
                        mean_u = u + eta_u_j * grad_u
                        sd_a = np.sqrt(2.0 * eta_a_j)
                        sd_u = np.sqrt(2.0 * eta_u_j)
                        a_prop = mean_a + sd_a * g['rng'].randn()
                        u_prop = mean_u + sd_u * g['rng'].randn(*u.shape)
                        logp_curr = -energy_a(g['a'], lvl, eps=0.0, omega=2.0) + beta_t * Q_hat(g['a'], lvl, params) - 0.5 * np.sum(u * u)
                        logp_prop = -energy_a(a_prop, lvl, eps=0.0, omega=2.0) + beta_t * Q_hat(a_prop, lvl, params) - 0.5 * np.sum(u_prop * u_prop)
                        grad_a_prop = -grad_energy_a(a_prop, lvl, eps=0.0, omega=2.0) + beta_t * (-lvl['A'] * a_prop + lvl['b'])
                        grad_u_prop = -u_prop
                        mean_rev_a = a_prop + eta_a_j * grad_a_prop
                        mean_rev_u = u_prop + eta_u_j * grad_u_prop
                        logq_f = gaussian_logpdf(a_prop, mean_a, sd_a) + np.sum(gaussian_logpdf(u_prop, mean_u, sd_u))
                        logq_r = gaussian_logpdf(g['a'], mean_rev_a, sd_a) + np.sum(gaussian_logpdf(u, mean_rev_u, sd_u))
                        log_alpha = (logp_prop - logp_curr) + (logq_r - logq_f)
                        if np.log(np.random.rand()) < min(0.0, log_alpha):
                            g['a'] = a_prop
                            g['sps'] = from_u(a_prop, u_prop, lvl, sigma)
                else:
                    # 1) action steps first
                    for _ in range(params.action_steps_per_level):
                        grad_log_q_a = -grad_energy_a(g['a'], lvl, eps=0.0, omega=2.0)
                        dQ_da = -lvl['A'] * g['a'] + lvl['b']
                        drift_a = grad_log_q_a + beta_t * dQ_da
                        if method_name == 'pc_ula':
                            g['a'] = g['a'] + eta_a * drift_a + np.sqrt(2.0 * eta_a) * g['rng'].randn()
                        elif method_name == 'pc_mala':
                            if eta_a <= 1e-12:
                                continue
                            mean_prop = g['a'] + eta_a * drift_a
                            sd_prop = np.sqrt(2.0 * eta_a)
                            a_prop = mean_prop + sd_prop * g['rng'].randn()
                            logp_curr = log_unnorm_pi_t(g['a'], lvl, params)
                            logp_prop = log_unnorm_pi_t(a_prop, lvl, params)
                            grad_prop = -grad_energy_a(a_prop, lvl, eps=0.0, omega=2.0) + beta_t * (-lvl['A'] * a_prop + lvl['b'])
                            mean_rev = a_prop + eta_a * grad_prop
                            logq_f = gaussian_logpdf(a_prop, mean_prop, sd_prop)
                            logq_r = gaussian_logpdf(g['a'], mean_rev, sd_prop)
                            log_alpha = (logp_prop - logp_curr) + (logq_r - logq_f)
                            if np.log(np.random.rand()) < min(0.0, log_alpha):
                                g['a'] = a_prop
                    # 2) then s' refresh with one-shot OU equilibrium (noise-level determined)
                    for _ in range(params.refresh_L):
                        mu = rho * g['a'] + c
                        g['sps'] = mu + np.sqrt(max(var_cond, 1e-12)) * g['rng'].randn(*g['sps'].shape)

            # draw frame
            fig, axes = plt.subplots(2, 2, figsize=(9, 8))
            legend_handles = [
                Line2D([0],[0], color='0.4', lw=2.0, alpha=0.6, label='clean base (fill+lines)'),
                Line2D([0],[0], color='red', lw=2.0, alpha=0.6, label='clean tilted (fill+lines)'),
                Line2D([0],[0], color='green', lw=1.5, label='tilted joint (level) contours'),
                Line2D([0],[0], color='blue', lw=1.5, linestyle=':', label='reward (dotted)'),
            ]
            for ax, g, title in zip(axes.ravel(), groups, [f'seed {s}' for s in seeds]):
                # static clean base and clean tilted (slightly harder: fill + thin lines)
                ax.contourf(Agrid, Sgrid, q0_img, levels=15, cmap='Greys', alpha=0.35)
                ax.contour(Agrid, Sgrid, q0_img, levels=8, colors='0.4', linewidths=0.5, alpha=0.4)
                ax.contourf(Agrid, Sgrid, tilt_clean_img, levels=15, cmap='Reds', alpha=0.35)
                ax.contour(Agrid, Sgrid, tilt_clean_img, levels=8, colors='red', linewidths=0.5, alpha=0.4)
                # dynamic per-level tilted joint (hard contours)
                ax.contour(Agrid, Sgrid, tilt_level_img, levels=10, colors='green', linewidths=1.2)
                # reward proxy (dotted blue)
                ax.contour(Agrid, Sgrid, R_img, levels=10, colors='blue', linestyles=':', alpha=0.7)
                a_line = np.full_like(g['sps'], g['a'])
                ax.scatter(a_line, g['sps'], s=10, c='k')
                ax.set_xlim(grids.min(), grids.max())
                ax.set_ylim(grids.min(), grids.max())
                ax.set_title(title)
                ax.set_xlabel('a_t')
                ax.set_ylabel("s'_t")
                ax.legend(handles=legend_handles, loc='upper right', fontsize=7)
            label_m = 'Joint-MALA' if method_name == 'pc_joint_mala' else ('MALA-on-marginal(a)' if method_name == 'pc_mala' else 'ULA')
            fig.suptitle(f'{label_m} | step {k+1}/{params.K} | alpha_bar={alpha_bar:.2f}, beta_t={beta_t:.2f}')
            fig.tight_layout(rect=[0,0,1,0.97])
            # render frame to image and close to avoid too many open figures
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = buf.reshape(h, w, 3)
            imgs.append(Image.fromarray(img))
            plt.close(fig)

        # Optional final MALA correction at t=0 for TFG-like schedule that decays to 0 at clean
        if params.beta_schedule == 'tfg_plus_final_mala':
            mu_clean, sigma_clean = clean_poe_tilt_from_Q(params)
            step = params.final_mala_step
            sd = np.sqrt(2.0 * step)
            for g in groups:
                grad = - (g['a'] - mu_clean) / max(sigma_clean**2, 1e-12)
                mean_prop = g['a'] + step * grad
                a_prop = mean_prop + sd * g['rng'].randn()
                # target is N(mu_clean, sigma_clean)
                logp_prop = gaussian_logpdf(a_prop, mu_clean, sigma_clean)
                logp_curr = gaussian_logpdf(g['a'], mu_clean, sigma_clean)
                grad_prop = - (a_prop - mu_clean) / max(sigma_clean**2, 1e-12)
                mean_rev = a_prop + step * grad_prop
                logq_f = gaussian_logpdf(a_prop, mean_prop, sd)
                logq_r = gaussian_logpdf(g['a'], mean_rev, sd)
                if np.log(np.random.rand()) < min(0.0, (logp_prop - logp_curr) + (logq_r - logq_f)):
                    g['a'] = a_prop
            # draw a final frame after correction
            fig, axes = plt.subplots(2, 2, figsize=(9, 8))
            legend_handles = [
                Line2D([0],[0], color='0.4', lw=2.0, alpha=0.6, label='clean base (fill+lines)'),
                Line2D([0],[0], color='red', lw=2.0, alpha=0.6, label='clean tilted (fill+lines)'),
                Line2D([0],[0], color='green', lw=1.5, label='tilted joint (level) contours'),
                Line2D([0],[0], color='blue', lw=1.5, linestyle=':', label='reward (dotted)'),
            ]
            for ax, g, title in zip(axes.ravel(), groups, [f'seed {s}' for s in seeds]):
                ax.contourf(Agrid, Sgrid, q0_img, levels=15, cmap='Greys', alpha=0.35)
                ax.contour(Agrid, Sgrid, q0_img, levels=8, colors='0.4', linewidths=0.5, alpha=0.4)
                ax.contourf(Agrid, Sgrid, tilt_clean_img, levels=15, cmap='Reds', alpha=0.35)
                ax.contour(Agrid, Sgrid, tilt_clean_img, levels=8, colors='red', linewidths=0.5, alpha=0.4)
                ax.contour(Agrid, Sgrid, tilt_level_img, levels=10, colors='green', linewidths=1.2)
                ax.contour(Agrid, Sgrid, R_img, levels=10, colors='blue', linestyles=':', alpha=0.7)
                a_line = np.full_like(g['sps'], g['a'])
                ax.scatter(a_line, g['sps'], s=10, c='k')
                ax.set_xlim(grids.min(), grids.max())
                ax.set_ylim(grids.min(), grids.max())
                ax.set_title(title + ' | final MALA')
                ax.set_xlabel('a_t')
                ax.set_ylabel("s'_t")
                ax.legend(handles=legend_handles, loc='upper right', fontsize=7)
            fig.suptitle(f'{method_name} | final MALA correction at clean | beta_clean={params.lambda_scale:.2f}')
            fig.tight_layout(rect=[0,0,1,0.97])
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = buf.reshape(h, w, 3)
            imgs.append(Image.fromarray(img))
            plt.close(fig)

        if len(imgs) > 0:
            imgs[0].save(os.path.join(out_dir, fname), save_all=True, append_images=imgs[1:], duration=120, loop=0)

        return groups  # return final groups for downstream diagnostics

    if 'pc_ula' in samplers:
        groups_ula = make_pc_animation('pc_ula', seeds=seeds_ula, points_per_seed=points_per_seed, fname='anim_pc_ula.gif')
    if 'pc_mala' in samplers:
        groups_mala = make_pc_animation('pc_mala', seeds=seeds_mala, points_per_seed=points_per_seed, fname='anim_pc_mala.gif')
    if 'pc_joint_mala' in samplers:
        groups_joint = make_pc_animation('pc_joint_mala', seeds=seeds_mala, points_per_seed=points_per_seed, fname='anim_pc_joint_mala.gif')

    # Non-plotting PC sampler to collect many final actions
    def pc_sample_final_actions(method_name='pc_ula', runs=2000, points_per_seed=20):
        alpha_bars = vp_alpha_bar_grid(params)
        sigmas = np.linspace(params.sigma_ve_max, params.sigma_ve_min, params.K)
        out = np.empty(runs)
        sps_all = []
        denoiser_evals_total = 0
        q_evals_total = 0
        # ESJD and acceptance/ESS tracking
        esjd_sum = 0.0
        esjd_count = 0  # number of proposals counted (steps); for ULA all accepted; for MALA include rejects
        mala_accepts = 0
        mala_props = 0
        proposals_count = 0
        # lag-1 stats for ESS across steps
        sum_curr = 0.0; sum_curr2 = 0.0
        sum_prev = 0.0; sum_prev2 = 0.0
        sum_xy = 0.0; n_pairs = 0
        # diagnostics of s' statistics across runs
        means_sps = []
        vars_sps = []
        # MALA acceptance adaptation per level
        K_levels = len(levels)
        log_eta = np.full(K_levels, np.log(max(params.ca, 1e-8)), dtype=float)
        log_eta_s = np.zeros(K_levels, dtype=float)
        acc_target = 0.57
        kappa = 0.01
        eta_cap_factor = 10.0
        for rseed in range(runs):
            rng = np.random.RandomState(rseed)
            # init
            a = 0.0
            sps = rng.normal(0.0, params.sigma_ve_max, size=points_per_seed)
            for k, lvl in enumerate(levels):
                m_a = lvl['m_a']; var_a = lvl['var_a']
                rho = lvl['rho']; c = lvl['c']; var_cond = lvl['var_cond']
                beta_t = lvl['beta_t']
                mu_tilt = lvl['mu_tilt']; sigma_tilt = lvl['sigma_tilt']
                # ULA base step for this level
                eta_a_ula = params.ca * max(var_a, 1e-12)
                # s' refresh: exact OU step for non-joint methods; joint-MALA updates s' inside the joint kernel
                if method_name != 'pc_joint_mala':
                    for _ in range(params.refresh_L):
                        mu = rho * a + c
                        sps = mu + np.sqrt(max(var_cond, 1e-12)) * rng.randn(*sps.shape)
                        denoiser_evals_total += sps.size
                # action update (EBM base energy + JVP reward gradient)
                grad_log_q_a = -grad_energy_a(a, lvl, eps=0.0, omega=2.0)
                grad_Q = sample_grad_Q_with_jvp(a, sps, rho, params)
                drift_a = grad_log_q_a + beta_t * grad_Q
                if method_name == 'pc_ula':
                    # one Q eval per (a,s') contribution
                    q_evals_total += sps.size
                    eta_a = eta_a_ula
                    a_new = a + eta_a * drift_a + np.sqrt(2.0 * eta_a) * rng.randn()
                    # ESJD (always accepted)
                    esjd_sum += (a_new - a) ** 2
                    esjd_count += 1
                    proposals_count += 1
                    # ESS lag-1 stats
                    sum_curr += a_new; sum_curr2 += a_new * a_new
                    sum_prev += a;     sum_prev2 += a * a
                    sum_xy += a_new * a; n_pairs += 1
                    a = a_new
                elif method_name == 'pc_mala':
                    eta_a = max(var_a, 1e-12) * np.exp(log_eta[k])
                    eta_a = float(np.clip(eta_a, 1e-10, eta_cap_factor * max(var_a, 1e-12)))
                    if eta_a <= 1e-12:
                        pass
                    else:
                        # use analytic dQ/da to match acceptance target
                        dQ_da = -lvl['A'] * a + lvl['b']
                        drift_a = -grad_energy_a(a, lvl, eps=0.0, omega=2.0) + beta_t * dQ_da
                        mean_prop = a + eta_a * drift_a
                        sd_prop = np.sqrt(2.0 * eta_a)
                        a_prop = mean_prop + sd_prop * rng.randn()
                        # unnormalized marginal target log π_t(a) ∝ exp{-E(a)+β_t Q_hat(a)}
                        logp_curr = log_unnorm_pi_t(a, lvl, params)
                        logp_prop = log_unnorm_pi_t(a_prop, lvl, params)
                        # two Q evals per step (current and proposal), each across all s' contributions
                        q_evals_total += 2 * sps.size
                        grad_prop = -grad_energy_a(a_prop, lvl, eps=0.0, omega=2.0) + beta_t * (-lvl['A'] * a_prop + lvl['b'])
                        mean_rev = a_prop + eta_a * grad_prop
                        logq_f = gaussian_logpdf(a_prop, mean_prop, sd_prop)
                        logq_r = gaussian_logpdf(a, mean_rev, sd_prop)
                        mala_props += 1
                        accepted = np.log(np.random.rand()) < min(0.0, (logp_prop - logp_curr) + (logq_r - logq_f))
                        # ESJD counts proposals, with zero contribution on rejects
                        esjd_sum += ((a_prop - a) ** 2) * (1.0 if accepted else 0.0)
                        esjd_count += 1
                        proposals_count += 1
                        # Robbins-Monro acceptance adaptation per level
                        log_eta[k] += kappa * ((1.0 if accepted else 0.0) - acc_target)
                        if accepted:
                            # accept
                            mala_accepts += 1
                            # ESS lag-1 stats
                            sum_curr += a_prop; sum_curr2 += a_prop * a_prop
                            sum_prev += a;      sum_prev2 += a * a
                            sum_xy += a_prop * a; n_pairs += 1
                            a = a_prop
                        else:
                            # rejected => chain stays; contributes to lag-1 stats with identical pair
                            sum_curr += a; sum_curr2 += a * a
                            sum_prev += a; sum_prev2 += a * a
                            sum_xy += a * a; n_pairs += 1
                elif method_name == 'pc_joint_mala':
                    # Whitened joint MALA in (a,u): log pi(a,u) = -E(a) + beta_t Q_hat(a) - 0.5||u||^2
                    eta_a = max(var_a, 1e-12) * np.exp(log_eta[k])
                    eta_a = float(np.clip(eta_a, 1e-10, eta_cap_factor * max(var_a, 1e-12)))
                    eta_a = eta_a / (1.0 + beta_t)
                    eta_u_base = params.ca
                    eta_u = eta_u_base * np.exp(log_eta_s[k])
                    if eta_a > 1e-12 and eta_u > 1e-12:
                        u, sigma = to_u(a, sps, lvl)
                        # Gradients
                        grad_a = -grad_energy_a(a, lvl, eps=0.0, omega=2.0) + beta_t * (-lvl['A'] * a + lvl['b'])
                        grad_u = -u
                        # Forward proposal x' ~ N(x + M grad, 2M)
                        mean_a = a + eta_a * grad_a
                        mean_u = u + eta_u * grad_u
                        sd_a = np.sqrt(2.0 * eta_a)
                        sd_u = np.sqrt(2.0 * eta_u)
                        a_prop = mean_a + sd_a * rng.randn()
                        u_prop = mean_u + sd_u * rng.randn(*u.shape)
                        # Targets in whitened space
                        logp_curr = -energy_a(a, lvl, eps=0.0, omega=2.0) + beta_t * Q_hat(a, lvl, params) - 0.5 * np.sum(u * u)
                        logp_prop = -energy_a(a_prop, lvl, eps=0.0, omega=2.0) + beta_t * Q_hat(a_prop, lvl, params) - 0.5 * np.sum(u_prop * u_prop)
                        # Reverse proposal uses same drift at x'
                        grad_a_prop = -grad_energy_a(a_prop, lvl, eps=0.0, omega=2.0) + beta_t * (-lvl['A'] * a_prop + lvl['b'])
                        grad_u_prop = -u_prop
                        mean_rev_a = a_prop + eta_a * grad_a_prop
                        mean_rev_u = u_prop + eta_u * grad_u_prop
                        # Proposal densities
                        logq_f = gaussian_logpdf(a_prop, mean_a, sd_a) + np.sum(gaussian_logpdf(u_prop, mean_u, sd_u))
                        logq_r = gaussian_logpdf(a, mean_rev_a, sd_a) + np.sum(gaussian_logpdf(u, mean_rev_u, sd_u))
                        mala_props += 1
                        # Cost accounting: no conditional grads needed in whitened coords; Q grad analytic
                        q_evals_total += 2
                        # MH accept
                        accepted = np.log(np.random.rand()) < min(0.0, (logp_prop - logp_curr) + (logq_r - logq_f))
                        esjd_sum += ((a_prop - a) ** 2) * (1.0 if accepted else 0.0)
                        esjd_count += 1
                        proposals_count += 1
                        # Adapt both blocks
                        signal = (1.0 if accepted else 0.0) - acc_target
                        log_eta[k] += kappa * signal
                        log_eta_s[k] += kappa * signal
                        if accepted:
                            mala_accepts += 1
                            sum_curr += a_prop; sum_curr2 += a_prop * a_prop
                            sum_prev += a;      sum_prev2 += a * a
                            sum_xy += a_prop * a; n_pairs += 1
                            a, u = a_prop, u_prop
                            sps = from_u(a, u, lvl, sigma)
                        else:
                            sum_curr += a; sum_curr2 += a * a
                            sum_prev += a; sum_prev2 += a * a
                            sum_xy += a * a; n_pairs += 1
                else:
                    raise ValueError(f"Unknown method_name: {method_name}")
            # final correction if schedule decays to 0 at clean
            if params.beta_schedule == 'tfg_plus_final_mala':
                mu_clean, sigma_clean = clean_poe_tilt_from_Q(params)
                step = params.final_mala_step
                sd = np.sqrt(2.0 * step)
                grad = - (a - mu_clean) / max(sigma_clean**2, 1e-12)
                mean_prop = a + step * grad
                a_prop = mean_prop + sd * rng.randn()
                logp_prop = gaussian_logpdf(a_prop, mu_clean, sigma_clean)
                logp_curr = gaussian_logpdf(a, mu_clean, sigma_clean)
                grad_prop = - (a_prop - mu_clean) / max(sigma_clean**2, 1e-12)
                mean_rev = a_prop + step * grad_prop
                logq_f = gaussian_logpdf(a_prop, mean_prop, sd)
                logq_r = gaussian_logpdf(a, mean_rev, sd)
                if np.log(np.random.rand()) < min(0.0, (logp_prop - logp_curr) + (logq_r - logq_f)):
                    # accepted jump at clean step
                    esjd_sum += (a_prop - a) ** 2
                    esjd_count += 1
                    if method_name == 'pc_mala':
                        mala_accepts += 1
                        mala_props += 1
                    a = a_prop
                else:
                    if method_name == 'pc_mala':
                        mala_props += 1
            # --- final s' refresh so s' matches the final a ---
            final_lvl = levels[-1]
            rho_f, c_f, var_cond_f = final_lvl['rho'], final_lvl['c'], final_lvl['var_cond']
            if method_name != 'pc_joint_mala':
                for _ in range(params.refresh_L):
                    mu_f = rho_f * a + c_f
                    sps = mu_f + np.sqrt(max(var_cond_f, 1e-12)) * rng.randn(*sps.shape)
                    denoiser_evals_total += sps.size
            # diagnostics per run
            means_sps.append(float(np.mean(sps)))
            vars_sps.append(float(np.var(sps)))
            out[rseed] = a
            sps_all.append(sps.copy())
        sps_all = np.concatenate(sps_all, axis=0) if len(sps_all)>0 else np.array([])
        # compute ESJD per-step and acceptance
        esjd = esjd_sum / max(1, esjd_count)
        # ESJD per unit cost (model evals): use denoiser+Q totals
        total_cost = denoiser_evals_total + q_evals_total
        esjd_per_cost = esjd_sum / max(1, total_cost)
        acc_rate = (mala_accepts / max(1, mala_props)) if method_name in ('pc_mala','pc_joint_mala') else float('nan')
        # approximate lag-1 ESS across all steps
        if n_pairs > 0:
            mean_c = sum_curr / n_pairs
            mean_p = sum_prev / n_pairs
            var_c = max(sum_curr2 / n_pairs - mean_c * mean_c, 1e-12)
            var_p = max(sum_prev2 / n_pairs - mean_p * mean_p, 1e-12)
            rho1 = (sum_xy / n_pairs - mean_c * mean_p) / np.sqrt(var_c * var_p)
            rho1 = float(np.clip(rho1, -0.999, 0.999))
            n_eff = n_pairs * (1.0 - rho1) / (1.0 + rho1)
        else:
            rho1 = float('nan'); n_eff = float('nan')
        var_mean_sps = float(np.var(means_sps)) if len(means_sps)>0 else float('nan')
        within_var_sps_avg = float(np.mean(vars_sps)) if len(vars_sps)>0 else float('nan')
        return out, sps_all, denoiser_evals_total, q_evals_total, esjd, acc_rate, n_eff, esjd_per_cost, var_mean_sps, within_var_sps_avg

    # ---- Final action marginal vs clean tilted (per PC method) ----
    def plot_action_match(final_as: np.ndarray, method_name: str, kl_val: float, denoiser_evals_total: int, q_evals_total: int, esjd: float, acc_rate: float):
        mu_tilt, sigma_tilt = clean_poe_tilt_from_Q(params)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.hist(final_as, bins=40, density=True, alpha=0.6, label=f'{method_name} final a')
        x = np.linspace(mu_tilt - 4*sigma_tilt, mu_tilt + 4*sigma_tilt, 400)
        ax.plot(x, gaussian_pdf(x, mu_tilt, sigma_tilt), 'r-', lw=2, label='clean tilted π(a)')
        ax.set_title(f'Final action marginal vs clean tilted ({method_name})')
        ax.set_xlabel('a')
        ax.set_ylabel('density')
        ax.legend(fontsize=8)
        n_actions = max(1, len(final_as))
        den_per = denoiser_evals_total / n_actions
        q_per = q_evals_total / n_actions
        acc_txt = f"acc≈{acc_rate:.3f}" if not np.isnan(acc_rate) else "acc≈—"
        ax.text(0.02, 0.98, f'KL(p||π)≈{kl_val:.4f}\n denoiser evals/action: {den_per:.2f}\n Q evals/action: {q_per:.2f}\n ESJD/step: {esjd:.3f}\n {acc_txt}', transform=ax.transAxes,
                ha='left', va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, lw=0.0))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'action_match_{method_name}.png'), dpi=200)
        plt.close(fig)

    def plot_state_match(final_sps: np.ndarray, method_name: str, s_current: float, kl_val: float, denoiser_evals_total: int, q_evals_total: int, esjd: float, acc_rate: float):
        mu_a_clean, sigma_a_clean = clean_poe_tilt_from_Q(params)
        mu_s = params.alpha * mu_a_clean + params.d * s_current + params.mu_eps
        sigma_s = np.sqrt((params.alpha ** 2) * (sigma_a_clean ** 2) + (params.sigma_sprime ** 2))
        fig, ax = plt.subplots(figsize=(5,4))
        ax.hist(final_sps, bins=40, density=True, alpha=0.6, label=f'{method_name} final s\'')
        x = np.linspace(mu_s - 4*sigma_s, mu_s + 4*sigma_s, 400)
        ax.plot(x, gaussian_pdf(x, mu_s, sigma_s), 'r-', lw=2, label='clean tilted π(s\')')
        ax.set_title(f'Final state marginal vs clean tilted ({method_name})')
        ax.set_xlabel("s\'")
        ax.set_ylabel('density')
        ax.legend(fontsize=8)
        n_actions = max(1, len(final_sps))
        den_per = denoiser_evals_total / n_actions
        q_per = q_evals_total / n_actions
        acc_txt = f"acc≈{acc_rate:.3f}" if not np.isnan(acc_rate) else "acc≈—"
        ax.text(0.02, 0.98, f'KL(p||π)≈{kl_val:.4f}\n denoiser evals/action: {den_per:.2f}\n Q evals/action: {q_per:.2f}\n ESJD/step: {esjd:.3f}\n {acc_txt}', transform=ax.transAxes,
                ha='left', va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, lw=0.0))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'state_match_{method_name}.png'), dpi=200)
        plt.close(fig)

    # Use customizable number of samples for histogram/KL estimation
    mu_tilt, sigma_tilt = clean_poe_tilt_from_Q(params)
    mu_a_clean, sigma_a_clean = mu_tilt, sigma_tilt
    mu_s_clean = params.alpha * mu_a_clean + params.d * s + params.mu_eps
    sigma_s_clean = np.sqrt((params.alpha ** 2) * (sigma_a_clean ** 2) + (params.sigma_sprime ** 2))
    results = {}
    for method in samplers:
        fa, fs, den, q, esjd, acc, ess, esjdpc, var_mean_sps, within_var_sps_avg = pc_sample_final_actions(method, runs=kl_runs, points_per_seed=points_per_seed)
        results[method] = dict(fa=fa, fs=fs, den=den, q=q, esjd=esjd, acc=acc, ess=ess, esjdpc=esjdpc,
                               var_mean_sps=var_mean_sps, within_var_sps_avg=within_var_sps_avg)
        # Action plots/KL
        kl_a = kl_hist_vs_gaussian(fa, mu_tilt, sigma_tilt, bins=kl_bins)
        results[method]['kl_action'] = float(kl_a)
        plot_action_match(fa, method, kl_a, den, q, esjd, acc)
        # State plots/KL
        kl_s = kl_hist_vs_gaussian(fs, mu_s_clean, sigma_s_clean, bins=kl_bins)
        results[method]['kl_state'] = float(kl_s)
        plot_state_match(fs, method, s, kl_s, den, q, esjd, acc)
        # GT state KL
        sps_gt = np.random.normal(params.alpha * fa + params.d * s + params.mu_eps, params.sigma_sprime)
        kl_s_gt = kl_hist_vs_gaussian(sps_gt, mu_s_clean, sigma_s_clean, bins=kl_bins)
        results[method]['kl_state_gt'] = float(kl_s_gt)
        plot_state_match(sps_gt, f'{method}_gt', s, kl_s_gt, den, q, esjd, acc)

    # ---- KL vs number of state refreshes per step (0..max) ----
    def action_kl_vs_refresh_plot(methods, max_refresh=10, runs_sweep=1500, points_per_seed=20):
        refresh_vals = list(range(0, max_refresh + 1))
        orig_refresh = params.refresh_L
        series = {m: {'vals': [], 'se': []} for m in methods}
        mu_tilt_loc, sigma_tilt_loc = clean_poe_tilt_from_Q(params)
        samples_cache = {}  # (method, r) -> (fa, fs)

        def boot_se(samples, B=30):
            if len(samples) == 0:
                return np.nan
            vals = []
            n = len(samples)
            for _ in range(B):
                idx = np.random.randint(0, n, size=n)
                vals.append(kl_hist_vs_gaussian(samples[idx], mu_tilt_loc, sigma_tilt_loc, bins=80))
            return float(np.std(np.array(vals), ddof=1))

        for r in refresh_vals:
            params.refresh_L = r
            for m in methods:
                fa, fs, *_ = pc_sample_final_actions(m, runs=runs_sweep, points_per_seed=points_per_seed)
                samples_cache[(m, r)] = (fa, fs)
                kl_val = kl_hist_vs_gaussian(fa, mu_tilt_loc, sigma_tilt_loc, bins=80)
                series[m]['vals'].append(kl_val)
                series[m]['se'].append(boot_se(fa))
        params.refresh_L = orig_refresh
        fig, ax = plt.subplots(figsize=(6,4))
        colors = ['C0','C1','C2','C3','C4']
        for i, m in enumerate(methods):
            vals = np.array(series[m]['vals']); se = np.array(series[m]['se'])
            ax.plot(refresh_vals, vals, marker='o', color=colors[i % len(colors)], label=m)
            ax.fill_between(refresh_vals, vals - se, vals + se, color=colors[i % len(colors)], alpha=0.2)
        ax.set_xlabel("state refreshes per step (refresh_L)")
        ax.set_ylabel("Action KL(p||π)")
        ax.set_yscale('log')
        ax.set_title("Action KL vs refresh_L")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'action_kl_vs_refresh.png'), dpi=200)
        plt.close(fig)
        return samples_cache

    def state_kl_vs_refresh_plot(methods, max_refresh=10, runs_sweep=1500, points_per_seed=20, samples_cache=None):
        refresh_vals = list(range(0, max_refresh + 1))
        orig_refresh = params.refresh_L
        series = {m: {'vals': [], 'se': []} for m in methods}
        mu_tilt_loc, sigma_tilt_loc = clean_poe_tilt_from_Q(params)
        mu_s_loc = params.alpha * mu_tilt_loc + params.d * s + params.mu_eps
        sigma_s_loc = np.sqrt((params.alpha ** 2) * (sigma_tilt_loc ** 2) + (params.sigma_sprime ** 2))

        def boot_se(samples, B=30):
            if len(samples) == 0:
                return np.nan
            vals = []
            n = len(samples)
            for _ in range(B):
                idx = np.random.randint(0, n, size=n)
                vals.append(kl_hist_vs_gaussian(samples[idx], mu_s_loc, sigma_s_loc, bins=80))
            return float(np.std(np.array(vals), ddof=1))

        for r in refresh_vals:
            params.refresh_L = r
            for m in methods:
                if samples_cache is not None and (m, r) in samples_cache:
                    _, fs = samples_cache[(m, r)]
                else:
                    _, fs, *_ = pc_sample_final_actions(m, runs=runs_sweep, points_per_seed=points_per_seed)
                kl_val = kl_hist_vs_gaussian(fs, mu_s_loc, sigma_s_loc, bins=80)
                series[m]['vals'].append(kl_val)
                series[m]['se'].append(boot_se(fs))
        params.refresh_L = orig_refresh
        fig, ax = plt.subplots(figsize=(6,4))
        colors = ['C0','C1','C2','C3','C4']
        for i, m in enumerate(methods):
            vals = np.array(series[m]['vals']); se = np.array(series[m]['se'])
            ax.plot(refresh_vals, vals, marker='o', color=colors[i % len(colors)], label=m)
            ax.fill_between(refresh_vals, vals - se, vals + se, color=colors[i % len(colors)], alpha=0.2)
        ax.set_xlabel("state refreshes per step (refresh_L)")
        ax.set_ylabel("State KL(p||π)")
        ax.set_yscale('log')
        ax.set_title("State KL vs refresh_L")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'state_kl_vs_refresh.png'), dpi=200)
        plt.close(fig)

    samples_cache = action_kl_vs_refresh_plot(samplers, max_refresh=sweep_max_refresh, runs_sweep=sweep_runs, points_per_seed=points_per_seed)
    state_kl_vs_refresh_plot(samplers, max_refresh=sweep_max_refresh, runs_sweep=sweep_runs, points_per_seed=points_per_seed, samples_cache=samples_cache)

    # Print a short summary to stdout for convenience
    # Build and save metrics summary
    metrics = {
        'config': {
            'beta_schedule': params.beta_schedule,
            'lambda_scale': params.lambda_scale,
            'K': params.K,
            'refresh_L': params.refresh_L,
            'action_steps_per_level': params.action_steps_per_level,
            'points_per_seed': points_per_seed,
            'samplers': samplers,
        },
    }
    # fill metrics dynamically for selected samplers
    metrics['kl'] = {}
    metrics['evals_per_action'] = {}
    metrics['mixing'] = {}
    metrics['diagnostics'] = {'sigma_sprime2': float(params.sigma_sprime ** 2)}
    for method, res in results.items():
        n_actions = max(1, len(res['fa']))
        # kl
        if 'kl_action' in res:
            key = method
            metrics['kl'][key] = float(res['kl_action'])
        if 'kl_state' in res:
            key = f"state_{method}"
            metrics['kl'][key] = float(res['kl_state'])
        if 'kl_state_gt' in res:
            key = f"state_{method}_gt"
            metrics['kl'][key] = float(res['kl_state_gt'])
        # evals per action
        metrics['evals_per_action'][f'denoiser_{method}'] = res['den'] / n_actions
        metrics['evals_per_action'][f'Q_{method}'] = res['q'] / n_actions
        # mixing
        metrics['mixing'][f'esjd_{method}'] = float(res['esjd'])
        if method in ('pc_mala','pc_joint_mala'):
            metrics['mixing'][f'acc_{method}'] = float(res['acc']) if not np.isnan(res['acc']) else None
        metrics['mixing'][f'ess_steps_{method}'] = float(res['ess']) if not np.isnan(res['ess']) else None
        metrics['mixing'][f'esjd_per_cost_{method}'] = float(res['esjdpc'])
        # diagnostics
        metrics['diagnostics'][f'var_mean_sps_{method}'] = float(res['var_mean_sps'])
        metrics['diagnostics'][f'alpha2_var_a_{method}'] = float((params.alpha ** 2) * np.var(res['fa']) if len(res['fa'])>0 else float('nan'))
        metrics['diagnostics'][f'within_var_sps_avg_{method}'] = float(res['within_var_sps_avg'])
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved outputs to:", out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
