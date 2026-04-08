# Bayesian HUD

Bayesian inference applied to poker player statistics, demonstrating three progressively richer use cases.

## Use cases

| # | Module | Description |
|---|--------|-------------|
| 1 | `single_stat.py` | Beta-Binomial conjugate filter for a single stat (e.g. VPIP). Updates a prior as hands are observed and exposes the posterior mean and credible interval. |
| 2 | `multi_stat.py` | Archetype-based multi-stat filter. Maintains a discrete posterior over five player archetypes (nit, TAG, LAG, fish, maniac) updated jointly on VPIP, PFR, WTSD, and AF using Beta-Binomial and Gamma-Poisson marginal likelihoods. |
| 3 | `decision_tree.py` | Sequential within-hand updater. Tracks a posterior over villain hand-strength buckets (air → set+) and refines it at every street/action using lookup likelihood tables. |

## Project layout

```
bayesian_hud/
    __init__.py
    archetypes.py       # archetype definitions and Beta/Gamma priors
    single_stat.py      # use case 1
    multi_stat.py       # use case 2
    decision_tree.py    # use case 3

notebooks/
    bayesian_hud.ipynb  # worked examples for all three use cases

requirements.txt
README.md
```

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/bayesian_hud.ipynb
```

Or use the library directly:

```python
from bayesian_hud import SingleStatFilter, MultiStatFilter, DecisionTreeUpdater, Bucket

# Use case 1 — VPIP filter starting from a TAG-ish prior
f = SingleStatFilter(alpha0=6, beta0=19)
f.update(successes=18, trials=60)
print(f)  # posterior mean + 95% CI

# Use case 2 — archetype classifier
mf = MultiStatFilter()
mf.update(vpip=(18, 60), pfr=(14, 60), wtsd=(12, 40), af=(80, 60))
print(mf.most_likely, mf.posterior)

# Use case 3 — within-hand range narrowing
dt = DecisionTreeUpdater()
dt.observe("preflop", "3bet").observe("flop", "bet_large").observe("turn", "raise")
print(dt.most_likely_bucket, dt.posterior)
```
