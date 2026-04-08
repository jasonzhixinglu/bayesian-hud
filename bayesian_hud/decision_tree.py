"""
Use case 3: Sequential within-hand Bayesian updating (BTN vs BB SRP).

At each decision node we observe the villain's action and update our
posterior over archetypes using the per-archetype action probabilities
stored in ARCHETYPES[name]['action_probs'].

Tree structure (we are BTN, villain is BB):
  PREFLOP        → fold* / threbet* / call
  FLOP_DONK      → donk* / check  (villain leads or checks to our cbet)
  FLOP_VS_CBET   → fold* / raise* / call
  TURN_DONK      → donk* / check
  TURN_VS_BARREL → fold* / call* / raise*   (* = terminal for our purposes)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

from .archetypes import (
    ARCHETYPES,
    ARCHETYPE_NAMES,
    ARCHETYPE_COLORS,
    get_archetype_params,
)

# Ordered archetype names / short labels for annotation
_K = len(ARCHETYPE_NAMES)
_SHORT = ["F", "T", "L"]   # Fish, TAG, LAG

# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def update_posterior(
    prior: np.ndarray,
    action_node: str,
    action: str,
    mu: Optional[np.ndarray] = None,    # unused; kept for API symmetry
    sigma: Optional[np.ndarray] = None,
    pi: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Update archetype posterior after observing one villain action.

    Parameters
    ----------
    prior       : (K,) current archetype posterior
    action_node : node name in {'preflop', 'flop_donk', 'flop_vs_cbet',
                                'turn_donk', 'turn_vs_barrel'}
    action      : observed action string (must exist under that node)
    mu, sigma, pi : unused; present for API symmetry with other modules

    Returns
    -------
    posterior : (K,) updated and normalised posterior
    """
    prior = np.asarray(prior, dtype=float)

    likelihoods = np.array([
        ARCHETYPES[name]["action_probs"][action_node][action]
        for name in ARCHETYPE_NAMES
    ])

    unnorm = prior * likelihoods
    total  = unnorm.sum()
    if total == 0:
        raise ValueError(
            f"Zero likelihood at node='{action_node}', action='{action}'. "
            "Check action_probs in archetypes.py."
        )
    return unnorm / total


def trace_path(
    action_sequence: list[tuple[str, str]],
    prior: Optional[np.ndarray] = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Apply sequential Bayesian updates along a sequence of (node, action) pairs.

    Parameters
    ----------
    action_sequence : list of (node_name, action) tuples
    prior           : (K,) initial distribution; defaults to mixture weights pi

    Returns
    -------
    posteriors_history : list of (K,) arrays, length = len(action_sequence) + 1
                         (index 0 is the initial prior, index i is after step i)
    final_posterior    : (K,) posterior after all updates
    """
    _, _, pi = get_archetype_params()
    current = np.asarray(prior, dtype=float) if prior is not None else pi.copy()

    history = [current.copy()]
    for node, action in action_sequence:
        current = update_posterior(current, node, action)
        history.append(current.copy())

    return history, current


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_DEFAULT_PATHS = [
    (
        "Path A — Fish (passive call-down)",
        [
            ("preflop",          "call"),
            ("flop_donk",        "check"),
            ("flop_vs_cbet",     "call"),
            ("turn_donk",        "check"),
            ("turn_vs_barrel",   "call"),
            ("river_donk",       "check"),
            ("river_vs_barrel",  "call"),
        ],
    ),
    (
        "Path B — TAG (identified at terminal fold)",
        [
            ("preflop",          "call"),
            ("flop_donk",        "check"),
            ("flop_vs_cbet",     "call"),
            ("turn_donk",        "check"),
            ("turn_vs_barrel",   "call"),
            ("river_donk",       "check"),
            ("river_vs_barrel",  "fold"),
        ],
    ),
    (
        "Path C — LAG (check-raise flop, barrel turn)",
        [
            ("preflop",                   "call"),
            ("flop_donk",                 "check"),
            ("flop_vs_cbet",              "raise"),
            ("flop_checkraise_vs_call",   "bet"),
        ],
    ),
]


def plot_posterior_evolution(
    paths: Optional[list] = None,
) -> plt.Figure:
    """
    2×2 subplot showing archetype posterior evolution along 4 hand paths.

    Parameters
    ----------
    paths : list of (title, action_sequence) tuples, or None for defaults.

    Returns
    -------
    fig : matplotlib Figure
    """
    if paths is None:
        paths = _DEFAULT_PATHS

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Archetype posterior evolution — BTN vs BB",
        fontsize=14,
    )

    for ax, (title, seq) in zip(axes.flat, paths):
        history, final = trace_path(seq)

        steps   = list(range(len(history)))
        x_ticks = ["prior"] + [f"{node}\n({act})" for node, act in seq]

        for k, (name, color) in enumerate(zip(ARCHETYPE_NAMES, ARCHETYPE_COLORS)):
            vals = [h[k] for h in history]
            ax.plot(steps, vals, color=color, linewidth=2.2, marker="o",
                    markersize=5, label=name)
            ax.fill_between(steps, vals, alpha=0.08, color=color)

        ax.set_xticks(steps)
        ax.set_xticklabels(x_ticks, fontsize=7.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("P(archetype | history)")
        ax.set_title(
            f"{title}\n"
            + "  ".join(
                f"P({n})={p:.2f}"
                for n, p in zip(ARCHETYPE_NAMES, final)
            ),
            fontsize=9.5,
        )
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Decision tree visualisation
# ---------------------------------------------------------------------------

# Tree definition: list of (node, action, child_node_or_terminal)
# Layout helper: assigns (x, y) positions.
#   x = depth level, y = vertical slot (spread evenly).

_TREE_EDGES = [
    # (parent_node, action, child_node_or_terminal_label)
    ("PREFLOP",                 "fold",   "TERMINAL"),
    ("PREFLOP",                 "threbet","TERMINAL"),
    ("PREFLOP",                 "call",   "FLOP_DONK"),
    ("FLOP_DONK",               "donk",   "TERMINAL"),
    ("FLOP_DONK",               "check",  "FLOP_VS_CBET"),
    ("FLOP_VS_CBET",            "fold",   "TERMINAL"),
    ("FLOP_VS_CBET",            "raise",  "FLOP_CHECKRAISE_VS_CALL"),
    ("FLOP_VS_CBET",            "call",   "TURN_DONK"),
    ("FLOP_CHECKRAISE_VS_CALL", "check",  "TERMINAL"),
    ("FLOP_CHECKRAISE_VS_CALL", "bet",    "TERMINAL"),
    ("TURN_DONK",               "donk",   "TERMINAL"),
    ("TURN_DONK",               "check",  "TURN_VS_BARREL"),
    ("TURN_VS_BARREL",          "fold",   "TERMINAL"),
    ("TURN_VS_BARREL",          "raise",  "TERMINAL"),
    ("TURN_VS_BARREL",          "call",   "RIVER_DONK"),
    ("RIVER_DONK",              "donk",   "TERMINAL"),
    ("RIVER_DONK",              "check",  "RIVER_VS_BARREL"),
    ("RIVER_VS_BARREL",         "fold",   "TERMINAL"),
    ("RIVER_VS_BARREL",         "call",   "TERMINAL"),
    ("RIVER_VS_BARREL",         "raise",  "TERMINAL"),
]

# Map each non-terminal node to the action_probs key
_NODE_KEY = {
    "PREFLOP":                 "preflop",
    "FLOP_DONK":               "flop_donk",
    "FLOP_VS_CBET":            "flop_vs_cbet",
    "FLOP_CHECKRAISE_VS_CALL": "flop_checkraise_vs_call",
    "TURN_DONK":               "turn_donk",
    "TURN_VS_BARREL":          "turn_vs_barrel",
    "RIVER_DONK":              "river_donk",
    "RIVER_VS_BARREL":         "river_vs_barrel",
}


def _action_prob_label(node: str, action: str) -> str:
    """'F:0.05 T:0.55 L:0.30' style annotation."""
    key = _NODE_KEY[node]
    probs = [
        ARCHETYPES[name]["action_probs"][key][action]
        for name in ARCHETYPE_NAMES
    ]
    return "  ".join(f"{s}:{p:.2f}" for s, p in zip(_SHORT, probs))


def plot_path_tree() -> plt.Figure:
    """
    Matplotlib figure visualising the full BTN-vs-BB decision tree.

    Each edge is annotated with the action name and per-archetype
    probabilities in 'F:x.xx T:x.xx L:x.xx' format.

    Returns
    -------
    fig : matplotlib Figure
    """
    # ------------------------------------------------------------------
    # Assign (x, y) coordinates to every node occurrence.
    # We lay the tree out left-to-right by depth, spreading children
    # vertically.  Terminal nodes reuse a unique label per occurrence.
    # ------------------------------------------------------------------

    # BFS to assign positions
    # Each entry: (node_label, x, y, unique_id)
    # We may have multiple TERMINAL leaves; give each a unique id.

    node_pos: dict[str, tuple[float, float]] = {}   # unique_id → (x, y)
    edges_plot: list[tuple[str, str, str, str]] = [] # (uid_from, uid_to, action, node)

    # We use a counter to uniquify terminal nodes
    _term_counter = [0]

    def _uid(label: str) -> str:
        if label == "TERMINAL":
            _term_counter[0] += 1
            return f"TERMINAL_{_term_counter[0]}"
        return label

    # Build a tree structure: parent_uid → list of (action, child_uid)
    children: dict[str, list[tuple[str, str]]] = {}
    root_uid = "PREFLOP"

    def _build(parent_label: str, parent_uid: str):
        children[parent_uid] = []
        for p_node, action, child_label in _TREE_EDGES:
            if p_node != parent_label:
                continue
            child_uid = _uid(child_label)
            children[parent_uid].append((action, child_uid))
            edges_plot.append((parent_uid, child_uid, action, parent_label))
            if child_label != "TERMINAL":
                _build(child_label, child_uid)

    _build("PREFLOP", root_uid)

    # Assign vertical positions via a leaf-counting DFS
    leaf_counter = [0]

    def _assign_y(uid: str) -> float:
        """Return the y-centroid for this node."""
        if not children.get(uid):  # leaf
            y = leaf_counter[0]
            leaf_counter[0] += 1
            return float(y)
        child_ys = []
        for _, c_uid in children[uid]:
            child_ys.append(_assign_y(c_uid))
        return float(np.mean(child_ys))

    # x = depth
    depth: dict[str, int] = {"PREFLOP": 0}
    _y_map: dict[str, float] = {}

    def _assign_depth(uid: str, d: int):
        depth[uid] = d
        for _, c_uid in children.get(uid, []):
            _assign_depth(c_uid, d + 1)

    _assign_depth(root_uid, 0)
    _assign_y(root_uid)

    # Re-do y with correct DFS order
    leaf_counter[0] = 0

    def _compute_positions(uid: str) -> float:
        if not children.get(uid):
            y = float(leaf_counter[0])
            leaf_counter[0] += 1
            node_pos[uid] = (depth[uid], y)
            return y
        child_ys = [_compute_positions(c_uid) for _, c_uid in children[uid]]
        y = float(np.mean(child_ys))
        node_pos[uid] = (depth[uid], y)
        return y

    _compute_positions(root_uid)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    n_leaves = leaf_counter[0]
    fig_h = max(8, n_leaves * 0.75)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-1, n_leaves)
    ax.axis("off")
    ax.set_title("BTN vs BB Decision Tree  (F=Fish, T=TAG, L=LAG)", fontsize=12)

    node_labels = {
        "PREFLOP":                 "PREFLOP",
        "FLOP_DONK":               "FLOP\n(OOP)",
        "FLOP_VS_CBET":            "FLOP\nvs CBET",
        "FLOP_CHECKRAISE_VS_CALL": "FLOP\nCR vs CALL",
        "TURN_DONK":               "TURN\n(OOP)",
        "TURN_VS_BARREL":          "TURN\nvs BARREL",
        "RIVER_DONK":              "RIVER\n(OOP)",
        "RIVER_VS_BARREL":         "RIVER\nvs BARREL",
    }

    # Draw edges first (under nodes)
    for uid_from, uid_to, action, parent_node in edges_plot:
        x0, y0 = node_pos[uid_from]
        x1, y1 = node_pos[uid_to]

        ax.annotate(
            "",
            xy=(x1 - 0.18, y1),
            xytext=(x0 + 0.18, y0),
            arrowprops=dict(arrowstyle="-|>", color="#555555",
                            lw=1.2, mutation_scale=12),
        )

        # Edge label: action + prob annotation
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        prob_str = _action_prob_label(parent_node, action)
        label = f"{action}\n{prob_str}"
        ax.text(
            mx, my + 0.08, label,
            ha="center", va="bottom", fontsize=6.5,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

    # Draw nodes
    for uid, (x, y) in node_pos.items():
        if uid.startswith("TERMINAL"):
            ax.plot(x, y, marker="x", markersize=10, color="#c0392b", linewidth=2)
            ax.text(x + 0.05, y, "end", ha="left", va="center",
                    fontsize=7, color="#c0392b")
        else:
            label = node_labels.get(uid, uid)
            ax.text(
                x, y, label,
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="#dfe6f0",
                          ec="#2c3e50", linewidth=1.5),
            )

    # Legend for archetype colours (reuse ARCHETYPE_COLORS)
    handles = [
        mpatches.Patch(color=c, label=f"{s} = {n}")
        for s, n, c in zip(_SHORT, ARCHETYPE_NAMES, ARCHETYPE_COLORS)
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    return fig
