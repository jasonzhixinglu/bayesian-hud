"""
Use case 2: Archetype-based multi-stat Bayesian filtering.

Maintains a posterior over player archetypes (Fish, TAG, LAG) updated jointly
on observed VPIP, PFR, and 3B%, then uses that posterior to produce
archetype-weighted shrinkage estimates that outperform a single-population prior.

Model
-----
theta_hat[j] | archetype=k  ~  N(mu[k,j],  sigma[k,j]**2 + s_hat[j]**2)
  where s_hat[j] = sqrt(theta_hat[j]*(1-theta_hat[j]) / n_opp[j])

P(k | theta_hat) ∝ pi[k] * prod_j N(theta_hat[j]; mu[k,j], total_var[k,j])
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy import stats

from .archetypes import (
    STAT_NAMES,
    STAT_OPP_RATES,
    ARCHETYPE_NAMES,
    ARCHETYPE_COLORS,
    get_archetype_params,
)
from .single_stat import bayesian_estimate, simulate_population, _population_priors


# ---------------------------------------------------------------------------
# Core inference functions
# ---------------------------------------------------------------------------

def archetype_posterior(
    theta_hat: np.ndarray,
    n_opp: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    pi: np.ndarray,
) -> np.ndarray:
    """
    Posterior probability of each archetype given observed stats.

    Parameters
    ----------
    theta_hat : (3,)   observed sample averages for [VPIP, PFR, 3B%]
    n_opp     : (3,)   number of opportunities per stat
    mu        : (K, 3) archetype prior means
    sigma     : (K, 3) archetype prior stds
    pi        : (K,)   mixture weights

    Returns
    -------
    posterior : (K,) array, sums to 1
    """
    theta_hat = np.asarray(theta_hat, dtype=float)  # (3,)
    n_opp     = np.asarray(n_opp,     dtype=float)  # (3,)

    # Sampling noise per stat
    s_hat = np.sqrt(
        theta_hat * (1.0 - theta_hat) / np.maximum(n_opp, 1)
    ).clip(min=1e-9)                                 # (3,)

    # Total variance per archetype per stat: sigma[k,j]^2 + s_hat[j]^2
    total_std = np.sqrt(sigma ** 2 + s_hat[None, :] ** 2)  # (K, 3)

    # Log-likelihood: sum over stats of log N(theta_hat[j]; mu[k,j], total_std[k,j])
    log_lk = np.sum(
        stats.norm.logpdf(theta_hat[None, :], loc=mu, scale=total_std),
        axis=1,
    )  # (K,)

    # Combine with log-prior and normalise
    log_unnorm = np.log(pi) + log_lk                # (K,)
    log_post   = log_unnorm - logsumexp(log_unnorm)  # (K,)
    return np.exp(log_post)                          # (K,)


def archetype_weighted_estimate(
    theta_hat: np.ndarray,
    n_opp: np.ndarray,
    posterior: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Archetype-posterior-weighted Bayesian shrinkage estimate for each stat.

    For each stat j and archetype k, computes the per-archetype shrinkage
    estimate, then takes the posterior-weighted average over archetypes.

    Parameters
    ----------
    theta_hat : (3,)   observed sample averages
    n_opp     : (3,)   opportunities per stat
    posterior : (K,)   archetype posterior weights
    mu        : (K, 3) archetype prior means
    sigma     : (K, 3) archetype prior stds

    Returns
    -------
    theta_weighted : (3,) posterior-averaged estimates
    """
    K = len(posterior)
    theta_b_all = np.zeros((K, 3))

    for k in range(K):
        for j in range(3):
            tb, _, _ = bayesian_estimate(
                theta_hat[j], mu[k, j], sigma[k, j], n_opp[j]
            )
            theta_b_all[k, j] = float(tb)

    # Posterior-weighted average: (K,) @ (K, 3) → (3,)
    return posterior @ theta_b_all


# ---------------------------------------------------------------------------
# Population simulation
# ---------------------------------------------------------------------------

def simulate_archetype_population(
    total_hands: int,
    n_players: int,
    seed: int | None = None,
) -> dict:
    """
    Simulate a mixed population drawn from the 3 archetypes.

    Parameters
    ----------
    total_hands : int   Total hands observed per player.
    n_players   : int   Number of players to simulate.
    seed        : int   RNG seed.

    Returns
    -------
    dict with keys:
        true_archetype : (n_players,)    int index of true archetype (0=Fish,1=TAG,2=LAG)
        theta_true     : (n_players, 3)  true rates per stat
        theta_hat      : (n_players, 3)  raw sample estimates
        theta_b_single : (n_players, 3)  single-stat Bayesian (population prior)
        theta_b_arch   : (n_players, 3)  archetype-weighted Bayesian estimates
        posteriors     : (n_players, K)  archetype posteriors
        n_opp          : (n_players, 3)  opportunities per stat
    """
    rng = np.random.default_rng(seed)
    mu, sigma, pi = get_archetype_params()       # (K,3), (K,3), (K,)
    opp_rates     = np.array(STAT_OPP_RATES)     # (3,)
    mu_pop, sigma_pop = _population_priors()      # (3,), (3,)
    K = len(pi)

    # Sample archetype index per player
    true_archetype = rng.choice(K, size=n_players, p=pi)  # (n_players,)

    # Allocate output arrays
    theta_true     = np.zeros((n_players, 3))
    theta_hat      = np.zeros((n_players, 3))
    theta_b_single = np.zeros((n_players, 3))
    theta_b_arch   = np.zeros((n_players, 3))
    posteriors     = np.zeros((n_players, K))
    n_opp_arr      = np.zeros((n_players, 3), dtype=int)

    for i in range(n_players):
        k = true_archetype[i]

        # True rate per stat: N(mu[k,j], sigma[k,j]) clipped to [0.01, 0.99]
        theta_true[i] = rng.normal(mu[k], sigma[k]).clip(0.01, 0.99)

        # Opportunities per stat: Binomial(total_hands, opp_rate), min 1
        n_opp_i = rng.binomial(total_hands, opp_rates).clip(min=1)
        n_opp_arr[i] = n_opp_i

        # Observed successes and Laplace-smoothed rates
        successes = rng.binomial(n_opp_i, theta_true[i])
        theta_hat[i] = (successes + 0.5) / (n_opp_i + 1.0)

        # Single-stat Bayesian estimate (population-weighted prior per stat)
        for j in range(3):
            tb, _, _ = bayesian_estimate(
                theta_hat[i, j], mu_pop[j], sigma_pop[j], n_opp_i[j]
            )
            theta_b_single[i, j] = float(tb)

        # Archetype posterior and archetype-weighted estimate
        post = archetype_posterior(theta_hat[i], n_opp_i, mu, sigma, pi)
        posteriors[i]    = post
        theta_b_arch[i]  = archetype_weighted_estimate(
            theta_hat[i], n_opp_i, post, mu, sigma
        )

    return {
        "true_archetype": true_archetype,
        "theta_true":     theta_true,
        "theta_hat":      theta_hat,
        "theta_b_single": theta_b_single,
        "theta_b_arch":   theta_b_arch,
        "posteriors":     posteriors,
        "n_opp":          n_opp_arr,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_archetype_posteriors(
    total_hands: int = 50,
    n_players: int = 2000,
    seed: int = 42,
) -> plt.Figure:
    """
    3×3 grid: rows = true archetype, cols = posterior P(archetype | data).

    Each cell shows the distribution of the posterior probability for that
    column's archetype, restricted to players whose true archetype is the row.

    Returns
    -------
    fig : matplotlib Figure
    """
    sim = simulate_archetype_population(total_hands, n_players, seed)
    true_arch = sim["true_archetype"]   # (n_players,)
    posteriors = sim["posteriors"]       # (n_players, K)
    K = len(ARCHETYPE_NAMES)

    fig, axes = plt.subplots(K, K, figsize=(12, 9), sharey=False)
    fig.suptitle(
        f"Archetype posterior distributions  |  {total_hands} hands/player, "
        f"n={n_players:,}",
        fontsize=13,
    )

    for row, true_name in enumerate(ARCHETYPE_NAMES):
        mask = true_arch == row
        for col, col_name in enumerate(ARCHETYPE_NAMES):
            ax = axes[row, col]
            vals = posteriors[mask, col]
            ax.hist(vals, bins=30, color=ARCHETYPE_COLORS[col], alpha=0.75,
                    edgecolor="white", linewidth=0.4)
            ax.axvline(vals.mean(), color="black", linestyle="--",
                       linewidth=1.2, label=f"mean={vals.mean():.2f}")
            ax.set_xlim(0, 1)
            ax.legend(fontsize=8)
            if col == 0:
                ax.set_ylabel(f"True: {true_name}\nCount", fontsize=9)
            if row == K - 1:
                ax.set_xlabel(f"P({col_name} | data)", fontsize=9)
            if row == 0:
                ax.set_title(f"P({col_name})", fontsize=10)

    plt.tight_layout()
    return fig




def plot_population_scatter() -> plt.Figure:
    """
    2D scatter of VPIP vs PFR for a simulated archetype population.

    Simulates 800 players (200 hands each) from the archetype mixture,
    colours points by true archetype.

    Returns
    -------
    fig : matplotlib Figure
    """
    sim = simulate_archetype_population(total_hands=200, n_players=800, seed=1)
    theta_true = sim["theta_true"]      # (800, 3)
    true_arch  = sim["true_archetype"]  # (800,)

    fig, ax = plt.subplots(figsize=(8, 6))

    for k, (name, color) in enumerate(zip(ARCHETYPE_NAMES, ARCHETYPE_COLORS)):
        mask = true_arch == k
        ax.scatter(
            theta_true[mask, 0], theta_true[mask, 1],
            color=color, alpha=0.45, s=15, label=name,
        )

    ax.set_xlabel("VPIP")
    ax.set_ylabel("PFR")
    ax.set_title("Population scatter: VPIP vs PFR by archetype")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_variance_decomposition() -> plt.Figure:
    """
    Stacked bar chart showing within- vs between-archetype variance
    as fractions of total variance for each stat.

    Computed analytically from archetype parameters (no simulation).

    Returns
    -------
    fig : matplotlib Figure
    """
    mu, sigma, pi = get_archetype_params()

    mu_pop      = pi @ mu                                             # (3,)
    between_var = np.sum(pi[:, None] * (mu - mu_pop[None, :]) ** 2, axis=0)  # (3,)
    within_var  = np.sum(pi[:, None] * sigma ** 2, axis=0)           # (3,)
    total_var   = between_var + within_var                            # (3,)

    between_frac = between_var / total_var
    within_frac  = within_var  / total_var

    x     = np.arange(len(STAT_NAMES))
    width = 0.5

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(x, within_frac,  width, label="Within-archetype",
           color="#aec6e8", alpha=0.9)
    ax.bar(x, between_frac, width, bottom=within_frac,
           label="Between-archetype", color="#2980b9", alpha=0.9)

    for i in range(len(STAT_NAMES)):
        ratio = between_frac[i]
        ax.text(i, 1.02, f"B/(B+W)={ratio:.2f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(STAT_NAMES)
    ax.set_ylabel("Fraction of total variance")
    ax.set_title("Variance decomposition: within vs between archetype")
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
    plt.tight_layout()
    return fig


def plot_correlation_structure() -> plt.Figure:
    """
    Single global correlation heatmap of theta_true across all archetypes.

    Simulates 2000 players (500 hands each) to obtain clean theta_true values.

    Returns
    -------
    fig : matplotlib Figure
    """
    sim = simulate_archetype_population(total_hands=500, n_players=2000, seed=2)
    theta_true = sim["theta_true"]      # (2000, 3)

    corr = np.corrcoef(theta_true.T)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(STAT_NAMES, fontsize=10)
    ax.set_yticklabels(STAT_NAMES, fontsize=10)
    ax.set_title("Global stat correlations (true rates)", fontsize=11)
    for i in range(3):
        for j in range(3):
            text_color = "white" if abs(corr[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}",
                    ha="center", va="center", fontsize=11, color=text_color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# New simulation helper
# ---------------------------------------------------------------------------

def simulate_archetype_draws(
    N: int,
    n_draws: int = 100,
    seed: int | None = None,
) -> dict:
    """
    For each archetype k, draw n_draws players and compute posterior and
    estimates analytically without simulating individual hands.

    For each draw d and archetype k:
      theta_true[d,j] ~ N(mu[k,j], sigma[k,j]) clipped [0.01, 0.99]
      n_opp[j] = round(N * opp_rate[j]), min 1  (deterministic)
      theta_hat[d,j] ~ N(theta_true[d,j],
                          sqrt(theta_true[d,j]*(1-theta_true[d,j])/n_opp[j]))
                       clipped [0.01, 0.99]
      archetype_posterior, archetype_weighted_estimate, and
      single-stat Bayesian estimate via _population_priors() are all computed.

    Parameters
    ----------
    N       : int   Total hands (used to compute n_opp = round(N * opp_rate))
    n_draws : int   Number of draws per archetype
    seed    : int   RNG seed

    Returns
    -------
    dict keyed by archetype name, each value a dict with:
      'theta_true'     : (n_draws, 3)
      'theta_hat'      : (n_draws, 3)
      'theta_b_single' : (n_draws, 3)
      'theta_b_arch'   : (n_draws, 3)
      'posteriors'     : (n_draws, 3)  archetype posteriors
    """
    rng = np.random.default_rng(seed)
    mu, sigma, pi = get_archetype_params()          # (K,3), (K,3), (K,)
    mu_pop, sigma_pop = _population_priors()         # (3,), (3,)
    opp_rates = np.array(STAT_OPP_RATES)            # (3,)
    K = len(ARCHETYPE_NAMES)

    # Deterministic opportunity counts for this N
    n_opp = np.maximum(np.round(N * opp_rates).astype(int), 1)  # (3,)

    result = {}
    for k, name in enumerate(ARCHETYPE_NAMES):
        theta_true     = np.zeros((n_draws, 3))
        theta_hat      = np.zeros((n_draws, 3))
        theta_b_single = np.zeros((n_draws, 3))
        theta_b_arch   = np.zeros((n_draws, 3))
        posteriors     = np.zeros((n_draws, K))

        for d in range(n_draws):
            # Draw true rate from archetype k
            tt = rng.normal(mu[k], sigma[k]).clip(0.01, 0.99)  # (3,)
            theta_true[d] = tt

            # Sampling noise and noisy observation
            s = np.sqrt(tt * (1.0 - tt) / n_opp).clip(1e-9, None)  # (3,)
            th = rng.normal(tt, s).clip(0.01, 0.99)                  # (3,)
            theta_hat[d] = th

            # Single-stat Bayesian estimate using population priors
            for j in range(3):
                tb, _, _ = bayesian_estimate(th[j], mu_pop[j], sigma_pop[j], n_opp[j])
                theta_b_single[d, j] = float(tb)

            # Archetype posterior and archetype-weighted estimate
            post = archetype_posterior(th, n_opp, mu, sigma, pi)
            posteriors[d] = post
            theta_b_arch[d] = archetype_weighted_estimate(th, n_opp, post, mu, sigma)

        result[name] = {
            "theta_true":     theta_true,
            "theta_hat":      theta_hat,
            "theta_b_single": theta_b_single,
            "theta_b_arch":   theta_b_arch,
            "posteriors":     posteriors,
        }

    return result


# ---------------------------------------------------------------------------
# New plots
# ---------------------------------------------------------------------------

def plot_archetype_posteriors_line(
    n_values: list | None = None,
    n_draws: int = 100,
    seed: int = 42,
) -> plt.Figure:
    """
    For each N in n_values and each true archetype, show the mean posterior
    probability P(archetype | data) averaged across n_draws players.

    N=0 means prior only — posterior = pi for all draws.

    Layout: 1×3 subplots, one per true archetype.
    Each subplot: line plot with x=n_values, 3 lines (one per archetype),
    colored by ARCHETYPE_COLORS.
    Legend on last subplot only, placed outside.

    Returns
    -------
    fig : matplotlib Figure
    """
    if n_values is None:
        n_values = [0, 100, 250, 500, 1000]

    mu, sigma, pi = get_archetype_params()
    K = len(ARCHETYPE_NAMES)

    # Build mean posteriors: shape (len(n_values), K_true, K_posterior)
    mean_posteriors = np.zeros((len(n_values), K, K))
    for i, N in enumerate(n_values):
        if N == 0:
            for k in range(K):
                mean_posteriors[i, k, :] = pi
        else:
            draws = simulate_archetype_draws(N, n_draws=n_draws, seed=seed)
            for k, name in enumerate(ARCHETYPE_NAMES):
                mean_posteriors[i, k, :] = draws[name]["posteriors"].mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for k_true, (ax, true_name) in enumerate(zip(axes, ARCHETYPE_NAMES)):
        for k_arch, (arch_name, color) in enumerate(zip(ARCHETYPE_NAMES, ARCHETYPE_COLORS)):
            ax.plot(
                n_values, mean_posteriors[:, k_true, k_arch],
                color=color, marker="o", linewidth=2, markersize=5, label=arch_name,
            )

        ax.set_xlabel("N (total hands)")
        ax.set_ylabel("Mean P(archetype | data)")
        ax.set_ylim(0, 1)
        ax.set_title(f"True archetype: {true_name}")

    # Legend on last subplot only, outside chart
    axes[-1].legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    plt.tight_layout()
    return fig


def plot_rmse_by_archetype(
    n_values: list | None = None,
    n_draws: int = 100,
    seed: int = 42,
) -> plt.Figure:
    """
    For each true archetype, show RMSE vs N for three methods:
    raw, single-stat Bayes, archetype Bayes.
    RMSE averaged across n_draws and across all 3 stats.

    Layout: 1×3 subplots, one per true archetype.
    Lines: red=raw, blue=single-stat Bayes, green=archetype Bayes.
    Legend on last subplot only, outside chart.

    Returns
    -------
    fig : matplotlib Figure
    """
    if n_values is None:
        n_values = [100, 250, 500, 1000]

    K = len(ARCHETYPE_NAMES)
    rmse_raw    = np.zeros((len(n_values), K))
    rmse_single = np.zeros((len(n_values), K))
    rmse_arch   = np.zeros((len(n_values), K))

    for i, N in enumerate(n_values):
        draws = simulate_archetype_draws(N, n_draws=n_draws, seed=seed)
        for k, name in enumerate(ARCHETYPE_NAMES):
            d = draws[name]
            rmse_raw[i, k]    = np.sqrt(np.mean((d["theta_hat"]      - d["theta_true"]) ** 2))
            rmse_single[i, k] = np.sqrt(np.mean((d["theta_b_single"] - d["theta_true"]) ** 2))
            rmse_arch[i, k]   = np.sqrt(np.mean((d["theta_b_arch"]   - d["theta_true"]) ** 2))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("RMSE by archetype (avg across stats)", fontsize=13)

    for k, (ax, name) in enumerate(zip(axes, ARCHETYPE_NAMES)):
        ax.plot(n_values, rmse_raw[:, k],    color="red",   marker="o", linewidth=2,
                markersize=5, label="MLE")
        ax.plot(n_values, rmse_single[:, k], color="blue",  marker="o", linewidth=2,
                markersize=5, label="Single-stat Bayes")
        ax.plot(n_values, rmse_arch[:, k],   color="green", marker="o", linewidth=2,
                markersize=5, label="Archetype Bayes")
        ax.set_xlabel("N (total hands)")
        ax.set_ylabel("RMSE (avg across stats)")
        ax.set_title(f"True archetype: {name}")

    # Legend on last subplot only, outside chart
    axes[-1].legend(
        bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0
    )

    plt.tight_layout()
    return fig


def plot_rmse_unconditional(
    n_values: list | None = None,
    n_draws: int = 100,
    seed: int = 42,
) -> plt.Figure:
    """
    Population-weighted RMSE vs N for raw, single-stat Bayes, archetype Bayes.

    For each N, computes RMSE per archetype from simulate_archetype_draws,
    then takes pi-weighted average across archetypes and across all 3 stats.

    Single panel figure. Legend outside chart.

    Returns
    -------
    fig : matplotlib Figure
    """
    if n_values is None:
        n_values = [100, 250, 500, 1000]

    _, _, pi = get_archetype_params()
    K = len(ARCHETYPE_NAMES)
    rmse_raw    = np.zeros((len(n_values), K))
    rmse_single = np.zeros((len(n_values), K))
    rmse_arch   = np.zeros((len(n_values), K))

    for i, N in enumerate(n_values):
        draws = simulate_archetype_draws(N, n_draws=n_draws, seed=seed)
        for k, name in enumerate(ARCHETYPE_NAMES):
            d = draws[name]
            rmse_raw[i, k]    = np.sqrt(np.mean((d["theta_hat"]      - d["theta_true"]) ** 2))
            rmse_single[i, k] = np.sqrt(np.mean((d["theta_b_single"] - d["theta_true"]) ** 2))
            rmse_arch[i, k]   = np.sqrt(np.mean((d["theta_b_arch"]   - d["theta_true"]) ** 2))

    # pi-weighted average across archetypes
    rmse_raw_w    = rmse_raw    @ pi
    rmse_single_w = rmse_single @ pi
    rmse_arch_w   = rmse_arch   @ pi

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_values, rmse_raw_w,    color="red",   marker="o", linewidth=2,
            markersize=5, label="MLE")
    ax.plot(n_values, rmse_single_w, color="blue",  marker="o", linewidth=2,
            markersize=5, label="Single-stat Bayes")
    ax.plot(n_values, rmse_arch_w,   color="green", marker="o", linewidth=2,
            markersize=5, label="Archetype Bayes")
    ax.set_xlabel("N (total hands)")
    ax.set_ylabel("RMSE (population-weighted avg)")
    ax.set_title("RMSE vs sample size — population average")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    plt.tight_layout()
    return fig


_STAT_COLORS = ["#e67e22", "#8e44ad", "#16a085"]  # orange, purple, teal


def plot_stat_informativeness(
    n_values: list | None = None,
    n_draws: int = 100,
    seed: int = 42,
) -> plt.Figure:
    """
    For each stat j, compute the expected KL divergence between the
    archetype posterior using only that stat versus the prior,
    averaged across n_draws players from each archetype.

    KL divergence from prior pi to posterior p:
      KL(p || pi) = sum_k p[k] * log(p[k] / pi[k])

    Layout: 1×3 subplots, one per true archetype.
    x=n_values, 3 lines one per stat (VPIP orange, PFR purple, 3B teal).
    Legend on last subplot only, outside chart.

    Returns
    -------
    fig : matplotlib Figure
    """
    if n_values is None:
        n_values = [100, 250, 500, 1000]

    rng = np.random.default_rng(seed)
    mu, sigma, pi = get_archetype_params()      # (K, 3), (K, 3), (K,)
    opp_rates = np.array(STAT_OPP_RATES)        # (3,)
    K = len(ARCHETYPE_NAMES)
    J = len(STAT_NAMES)

    # mean_kl[i_N, k_true, j_stat]
    mean_kl = np.zeros((len(n_values), K, J))

    for i, N in enumerate(n_values):
        n_opp_full = np.maximum(np.round(N * opp_rates).astype(int), 1)  # (3,)

        for k in range(K):
            kl_draws = np.zeros((n_draws, J))

            for d in range(n_draws):
                # Draw true rate from archetype k (all stats, but we use each independently)
                tt = rng.normal(mu[k], sigma[k]).clip(0.01, 0.99)  # (3,)

                for j in range(J):
                    n_opp_j = n_opp_full[j]
                    s = np.sqrt(tt[j] * (1.0 - tt[j]) / n_opp_j).clip(1e-9, None)
                    th_j = float(np.clip(rng.normal(tt[j], s), 0.01, 0.99))

                    # Posterior using only stat j: pass 1-element slices
                    post = archetype_posterior(
                        np.array([th_j]),
                        np.array([n_opp_j], dtype=float),
                        mu[:, j:j+1],
                        sigma[:, j:j+1],
                        pi,
                    )  # (K,)

                    # KL(post || pi)
                    kl = np.sum(post * np.log(post / pi))
                    kl_draws[d, j] = kl

            mean_kl[i, k, :] = kl_draws.mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Per-stat KL divergence from prior (single-stat posterior)", fontsize=13)

    for k, (ax, true_name) in enumerate(zip(axes, ARCHETYPE_NAMES)):
        for j, (stat_name, color) in enumerate(zip(STAT_NAMES, _STAT_COLORS)):
            ax.plot(n_values, mean_kl[:, k, j], color=color, marker="o",
                    linewidth=2, markersize=5, label=stat_name)
        ax.set_xlabel("N (total hands)")
        ax.set_ylabel("Mean KL divergence from prior")
        ax.set_title(f"True archetype: {true_name}")

    axes[-1].legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    plt.tight_layout()
    return fig
