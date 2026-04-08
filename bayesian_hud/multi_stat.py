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

        # Observed successes and raw rates
        successes = rng.binomial(n_opp_i, theta_true[i])
        theta_hat[i] = successes / n_opp_i

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


def plot_estimate_improvement(
    total_hands: int = 50,
    n_players: int = 2000,
    seed: int = 42,
) -> plt.Figure:
    """
    Bar chart of RMSE for raw, single-stat Bayes, and archetype Bayes per stat.

    Returns
    -------
    fig : matplotlib Figure
    """
    sim        = simulate_archetype_population(total_hands, n_players, seed)
    theta_true = sim["theta_true"]      # (n_players, 3)
    theta_hat  = sim["theta_hat"]       # (n_players, 3)
    theta_b_s  = sim["theta_b_single"]  # (n_players, 3)
    theta_b_a  = sim["theta_b_arch"]    # (n_players, 3)

    rmse = lambda pred: np.sqrt(np.mean((pred - theta_true) ** 2, axis=0))  # (3,)

    rmse_raw   = rmse(theta_hat)
    rmse_single = rmse(theta_b_s)
    rmse_arch  = rmse(theta_b_a)

    x      = np.arange(len(STAT_NAMES))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_raw    = ax.bar(x - width, rmse_raw,    width, label="Raw",
                         color="#e74c3c", alpha=0.85)
    bars_single = ax.bar(x,         rmse_single, width, label="Single-stat Bayes",
                         color="#3498db", alpha=0.85)
    bars_arch   = ax.bar(x + width, rmse_arch,   width, label="Archetype Bayes",
                         color="#2ecc71", alpha=0.85)

    for bars in (bars_raw, bars_single, bars_arch):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.0003,
                f"{h:.4f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(STAT_NAMES)
    ax.set_ylabel("RMSE")
    ax.set_title(
        f"Estimation RMSE by method  |  {total_hands} hands/player, "
        f"n={n_players:,}"
    )
    ax.legend()
    ax.set_ylim(0, rmse_raw.max() * 1.25)

    plt.tight_layout()
    return fig
