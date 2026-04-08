"""
Use case 1: Single-stat Bayesian filtering.

Implements shrinkage-based Bayesian estimation for proportion stats
(VPIP, PFR, 3B%) using a Gaussian prior over the true rate.

Estimator
---------
Given a player with n opportunities and observed rate theta_hat:
  s_hat   = sqrt(theta_hat * (1 - theta_hat) / n)   # sampling std
  w       = sigma**2 / (sigma**2 + s_hat**2)          # data weight
  theta_b = mu + w * (theta_hat - mu)                # shrinkage estimate

When n is small, s_hat is large → w is small → estimate shrinks toward mu.
As n → ∞, s_hat → 0 → w → 1 → estimate converges to theta_hat.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .archetypes import (
    STAT_NAMES,
    STAT_OPP_RATES,
    ARCHETYPE_NAMES,
    ARCHETYPE_COLORS,
    get_archetype_params,
)


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

def bayesian_estimate(
    theta_hat: np.ndarray | float,
    mu: float,
    sigma: float,
    n: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gaussian shrinkage estimator for a proportion stat.

    Parameters
    ----------
    theta_hat : scalar or array
        Observed sample average (successes / opportunities).
    mu : float
        Prior mean (population average for this stat).
    sigma : float
        Prior std (population spread for this stat).
    n : scalar or array
        Number of opportunities observed (same shape as theta_hat).

    Returns
    -------
    theta_b : ndarray
        Bayesian (shrinkage) estimate.
    s_hat : ndarray
        Estimated sampling std of theta_hat.
    w : ndarray
        Data weight in [0, 1].  0 = all prior, 1 = all data.
    """
    theta_hat = np.asarray(theta_hat, dtype=float)
    n         = np.asarray(n,         dtype=float)

    s_hat   = np.sqrt(theta_hat * (1.0 - theta_hat) / np.maximum(n, 1))
    s_hat   = np.clip(s_hat, 1e-9, None)
    w       = sigma**2 / (sigma**2 + s_hat**2)
    theta_b = mu + w * (theta_hat - mu)

    return theta_b, s_hat, w


# ---------------------------------------------------------------------------
# Population simulation
# ---------------------------------------------------------------------------

def simulate_population(
    mu: float,
    sigma: float,
    opp_rate: float,
    total_hands: int,
    n_players: int,
    seed: int | None = None,
) -> dict:
    """
    Simulate a population of players and compute estimates for one stat.

    Parameters
    ----------
    mu : float
        Prior mean of the true rate for this stat.
    sigma : float
        Prior std of the true rate for this stat.
    opp_rate : float
        Fraction of hands in which the stat is "in play".
    total_hands : int
        Total hands observed per player.
    n_players : int
        Number of players to simulate.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        theta_true : (n_players,) true rates drawn from the prior
        theta_hat  : (n_players,) raw sample estimates
        theta_b    : (n_players,) Bayesian shrinkage estimates
        s_hat      : (n_players,) sampling stds
        w          : (n_players,) data weights
        n_opp      : (n_players,) number of opportunities per player
    """
    rng = np.random.default_rng(seed)

    # Opportunities per player: Binomial(total_hands, opp_rate), minimum 1
    n_opp = rng.binomial(total_hands, opp_rate, size=n_players).clip(min=1)

    # True rates: Normal(mu, sigma) clipped to [0.01, 0.99]
    theta_true = rng.normal(mu, sigma, size=n_players).clip(0.01, 0.99)

    # Observed successes and Laplace-smoothed estimates
    successes = rng.binomial(n_opp, theta_true)
    theta_hat = (successes + 0.5) / (n_opp + 1.0)

    theta_b, s_hat, w = bayesian_estimate(theta_hat, mu, sigma, n_opp)

    return {
        "theta_true": theta_true,
        "theta_hat":  theta_hat,
        "theta_b":    theta_b,
        "s_hat":      s_hat,
        "w":          w,
        "n_opp":      n_opp,
    }


# ---------------------------------------------------------------------------
# Shrinkage weight curve
# ---------------------------------------------------------------------------

def shrinkage_weight_curve(
    mu: float,
    sigma: float,
    n_grid: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Compute the shrinkage weight w as a function of opportunity count n.

    Uses theta = mu as a representative value for s_hat (i.e., evaluates
    the curve at the prior mean, giving the "typical" weight trajectory).

    Parameters
    ----------
    mu : float
        Prior mean (used to compute s_hat at a representative point).
    sigma : float
        Prior std.
    n_grid : ndarray
        Array of opportunity counts at which to evaluate w.

    Returns
    -------
    w : ndarray
        Data weights corresponding to n_grid.
    crossover_n : float
        Opportunity count where w = 0.5, i.e. mu*(1-mu) / sigma**2.
    """
    _, _, w = bayesian_estimate(
        theta_hat=np.full_like(n_grid, mu, dtype=float),
        mu=mu,
        sigma=sigma,
        n=n_grid,
    )
    crossover_n = mu * (1.0 - mu) / sigma ** 2
    return w, crossover_n


# ---------------------------------------------------------------------------
# Population-weighted prior parameters
# ---------------------------------------------------------------------------

def _population_priors() -> tuple[np.ndarray, np.ndarray]:
    """
    Compute population-weighted mu and sigma for each stat.

    Returns
    -------
    mu_pop    : (3,) weighted average of archetype means
    sigma_pop : (3,) weighted average of archetype stds
    """
    mu, sigma, pi = get_archetype_params()
    mu_pop    = pi @ mu     # (3,)
    sigma_pop = pi @ sigma  # (3,)
    return mu_pop, sigma_pop


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_estimation_comparison(
    total_hands: int = 100,
    n_players: int = 2000,
    seed: int = 42,
) -> plt.Figure:
    """
    Compare raw vs Bayesian estimates against true values for each stat.

    Runs simulate_population for VPIP, PFR, and 3B% using population-
    weighted mu and sigma, then creates a 1×3 scatter plot.

    Parameters
    ----------
    total_hands : int
        Hands observed per simulated player.
    n_players : int
        Number of players per stat simulation.
    seed : int
        RNG seed.

    Returns
    -------
    fig : matplotlib Figure
    """
    mu_pop, sigma_pop = _population_priors()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Raw vs Bayesian estimates  |  {total_hands} opportunities/player, "
        f"n={n_players:,} players",
        fontsize=13,
    )

    for ax, stat_name, mu, sigma, opp_rate in zip(
        axes, STAT_NAMES, mu_pop, sigma_pop, STAT_OPP_RATES
    ):
        sim = simulate_population(mu, sigma, opp_rate, total_hands, n_players, seed=seed)
        true  = sim["theta_true"]
        hat   = sim["theta_hat"]
        bayes = sim["theta_b"]
        n_opp = sim["n_opp"]
        w     = sim["w"]

        rmse_raw   = np.sqrt(np.mean((hat   - true) ** 2))
        rmse_bayes = np.sqrt(np.mean((bayes - true) ** 2))
        avg_opp    = n_opp.mean()
        avg_w      = w.mean()

        lim = [
            min(true.min(), hat.min(), bayes.min()) - 0.02,
            max(true.max(), hat.max(), bayes.max()) + 0.02,
        ]

        ax.scatter(true, hat,   alpha=0.3, s=8, color="#e74c3c",
                   label=f"Raw   RMSE={rmse_raw:.4f}")
        ax.scatter(true, bayes, alpha=0.3, s=8, color="#3498db",
                   label=f"Bayes RMSE={rmse_bayes:.4f}")
        ax.plot(lim, lim, "k--", linewidth=1, label="45° line")

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("True θ")
        ax.set_ylabel("Estimated θ")
        ax.set_title(
            f"{stat_name}\n"
            f"avg opps={avg_opp:.1f}  avg w={avg_w:.2f}"
        )
        ax.legend(fontsize=8, markerscale=2)

    plt.tight_layout()
    return fig


def plot_shrinkage_curves() -> plt.Figure:
    """
    Plot shrinkage weight w vs opportunity count for each stat.

    Uses population-weighted mu and sigma. Marks the crossover point
    (w = 0.5) on each curve.

    Returns
    -------
    fig : matplotlib Figure
    """
    mu_pop, sigma_pop = _population_priors()
    n_grid = np.linspace(1, 500, 1000)

    fig, ax = plt.subplots(figsize=(9, 5))

    for stat_name, mu, sigma, color in zip(
        STAT_NAMES, mu_pop, sigma_pop, ARCHETYPE_COLORS
    ):
        w, crossover_n = shrinkage_weight_curve(mu, sigma, n_grid)

        ax.plot(n_grid, w, color=color, linewidth=2, label=stat_name)

        # Mark crossover point
        w_cross, _, _ = bayesian_estimate(mu, mu, sigma, crossover_n)
        ax.scatter(
            [crossover_n], [0.5],
            color=color, s=80, zorder=5,
            marker="D",
        )
        ax.annotate(
            f"n={crossover_n:.0f}",
            xy=(crossover_n, 0.5),
            xytext=(crossover_n + 10, 0.5 - 0.04),
            fontsize=8,
            color=color,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Opportunities observed")
    ax.set_ylabel("Data weight w")
    ax.set_title("Shrinkage weight vs opportunities  (w=0.5 crossover marked)")
    ax.legend()
    ax.set_xlim(0, n_grid[-1])
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    return fig
