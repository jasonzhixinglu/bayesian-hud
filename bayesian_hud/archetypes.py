"""
Archetype definitions for the Bayesian HUD.

Three player archetypes are defined over three stats [VPIP, PFR, 3B%]:
  Fish  — loose-passive calling station
  TAG   — tight-aggressive regular
  LAG   — loose-aggressive aggressive regular

Each archetype carries:
  - A Gaussian prior: mu_k, sigma_k  (independent per stat)
  - A mixture weight pi_k            (population share)
  - Action probabilities at each BTN-vs-BB decision node (use case 3)
"""

from __future__ import annotations

from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

STAT_NAMES: list[str] = ["VPIP", "PFR", "3B%"]

# Fraction of hands in which the stat is "in play" (opportunity rate).
# Used to convert observed hand counts to effective trial counts.
STAT_OPP_RATES: list[float] = [1.0, 0.20, 0.15]

ARCHETYPE_NAMES: list[str] = ["Fish", "TAG", "LAG"]

ARCHETYPE_COLORS: list[str] = ["#e74c3c", "#3498db", "#2ecc71"]

# ---------------------------------------------------------------------------
# Archetype definitions
# ---------------------------------------------------------------------------

ARCHETYPES: Dict[str, dict] = {
    "Fish": {
        # Gaussian prior over [VPIP, PFR, 3B%]
        "mu":    [0.45, 0.10, 0.03],
        "sigma": [0.08, 0.06, 0.03],
        # Population share
        "pi": 0.40,
        # BTN-vs-BB decision-node action probabilities
        "action_probs": {
            "preflop": {
                "fold":   0.05,
                "call":   0.85,
                "threbet": 0.10,
            },
            "flop_donk": {
                "donk":  0.10,
                "check": 0.90,
            },
            "flop_vs_cbet": {
                "fold":  0.25,
                "call":  0.65,
                "raise": 0.10,
            },
            "turn_vs_barrel": {
                "fold":  0.30,
                "call":  0.60,
                "raise": 0.10,
            },
            "turn_donk": {
                "donk":  0.10,
                "check": 0.90,
            },
        },
    },
    "TAG": {
        "mu":    [0.22, 0.18, 0.08],
        "sigma": [0.04, 0.04, 0.02],
        "pi": 0.45,
        "action_probs": {
            "preflop": {
                "fold":   0.55,
                "call":   0.35,
                "threbet": 0.10,
            },
            "flop_donk": {
                "donk":  0.15,
                "check": 0.85,
            },
            "flop_vs_cbet": {
                "fold":  0.55,
                "call":  0.30,
                "raise": 0.15,
            },
            "turn_vs_barrel": {
                "fold":  0.55,
                "call":  0.25,
                "raise": 0.20,
            },
            "turn_donk": {
                "donk":  0.15,
                "check": 0.85,
            },
        },
    },
    "LAG": {
        "mu":    [0.35, 0.28, 0.16],
        "sigma": [0.05, 0.05, 0.03],
        "pi": 0.15,
        "action_probs": {
            "preflop": {
                "fold":   0.30,
                "call":   0.45,
                "threbet": 0.25,
            },
            "flop_donk": {
                "donk":  0.25,
                "check": 0.75,
            },
            "flop_vs_cbet": {
                "fold":  0.30,
                "call":  0.35,
                "raise": 0.35,
            },
            "turn_vs_barrel": {
                "fold":  0.25,
                "call":  0.35,
                "raise": 0.40,
            },
            "turn_donk": {
                "donk":  0.30,
                "check": 0.70,
            },
        },
    },
}

# Convenience: mixture weights in ARCHETYPE_NAMES order
MIXTURE_WEIGHTS: Dict[str, float] = {
    name: ARCHETYPES[name]["pi"] for name in ARCHETYPE_NAMES
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_archetype_params() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return stacked parameter arrays in ARCHETYPE_NAMES order [Fish, TAG, LAG].

    Returns
    -------
    mu    : ndarray, shape (3, 3)  — mu_k    for each archetype (rows) and stat (cols)
    sigma : ndarray, shape (3, 3)  — sigma_k for each archetype (rows) and stat (cols)
    pi    : ndarray, shape (3,)    — mixture weights (sum to 1)
    """
    mu    = np.array([ARCHETYPES[n]["mu"]    for n in ARCHETYPE_NAMES], dtype=float)
    sigma = np.array([ARCHETYPES[n]["sigma"] for n in ARCHETYPE_NAMES], dtype=float)
    pi    = np.array([ARCHETYPES[n]["pi"]    for n in ARCHETYPE_NAMES], dtype=float)
    return mu, sigma, pi
