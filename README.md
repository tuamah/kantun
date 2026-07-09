# Kantun

A lightweight, **model-agnostic** hyperparameter tuning library.

Built as a companion to [KANBoost](https://github.com/tuamah/kanboost), but with **zero
hard dependency on it** — Kantun works with any estimator that follows a
scikit-learn-like `fit`/`predict`/`predict_proba` interface. Use it with
KANBoost, with `RandomForestClassifier`, with your own custom model —
your choice.

## Why a separate package?

Not everyone using KANBoost needs hyperparameter search, and not
everyone doing hyperparameter search needs KANBoost. Splitting them
keeps each library's dependency footprint minimal and lets Kantun be
useful on its own.

## Install

```bash
pip install kantun
```

To also tune KANBoost models, install it separately:

```bash
pip install kanboost
```

## Quickstart

```python
from kantun import KantunSearch
from kanboost import KANBoostClassifier

param_space = {
    "n_estimators": [30, 60, 100],
    "learning_rate": [0.1, 0.2, 0.3],
    "kan_hidden": [3, 4, 6],
    "kan_grid": [2, 3],
}

search = KantunSearch(
    KANBoostClassifier,
    param_space,
    n_iter=10,
    cv=3,
    scoring="auc",
)
search.fit(X, y)

print(search.best_params_, search.best_score_)
best_model = search.best_estimator_          # ready to use
results_df = search.results_dataframe()       # sorted leaderboard
```

### Works with any sklearn-style estimator, not just KANBoost

```python
from sklearn.ensemble import RandomForestClassifier
from kantun import KantunSearch

search = KantunSearch(
    RandomForestClassifier,
    {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
    n_iter=5, cv=3, scoring="f1",
    use_eval_set=False,   # RandomForestClassifier.fit() has no eval_set kwarg
)
search.fit(X, y)
```

## How it decides whether to use early stopping

`KantunSearch` inspects the target model class's `fit()` signature. If
it finds an `eval_set` parameter (as KANBoost's estimators do), it
automatically passes `eval_set=(X_val, y_val)` on each fold so early
stopping kicks in during the search itself. You can override this with
`use_eval_set=True/False` explicitly.

## Supported scoring

- Classification: `"auc"` (default), `"f1"`, `"accuracy"`, or a callable
  `scorer(y_true, y_pred, y_prob, labels) -> float` (higher is better).
- Regression: `"neg_mse"` (default), `"neg_mae"`, or a callable
  `scorer(y_true, y_pred) -> float` (higher is better).

## Continuous parameter ranges

`param_distributions` values are usually lists, but for
`search_type="random"`/`"halving"` (not `"grid"`, which must enumerate)
a value can instead be a callable `sampler(rng) -> value`, where `rng`
is a `random.Random` seeded from `random_state`:

```python
param_space = {
    "n_estimators": [30, 60, 100],
    "kan_lr": lambda rng: 10 ** rng.uniform(-3, -1),   # log-uniform
    "learning_rate": lambda rng: rng.uniform(0.01, 0.3),  # linear-uniform
}
```

## Time budget

`time_budget_s` caps wall-clock search time. It's checked between combos
(in batches of `n_jobs`) or between halving rungs -- never mid-fit, so a
combo already dispatched always finishes, and at least one combo/rung
always runs. If the budget is hit before halving's final full-data rung,
the best candidate from the last completed rung is promoted so `fit()`
still succeeds.

```python
search = KantunSearch(KANBoostClassifier, param_space, time_budget_s=600)
```

## Skipping the final refit

By default `fit()` refits `best_params_` on the full dataset
(`best_estimator_`). Pass `refit=False` to skip that -- useful when
`model_cls` is expensive to train and you only need `best_params_`/
`cv_results_`.

## Search types

- `search_type="random"` (default): samples `n_iter` random combinations
- `search_type="grid"`: tries every combination in `param_distributions`
- `search_type="halving"`: successive halving -- starts every candidate
  on a small, stratified subsample of each fold's *training* data (held-out
  validation data is always full and untouched), keeps the top
  `1/halving_factor` by score, and grows the training subsample by
  `halving_factor` each round until a round trains on the full data.
  Training-set size is the resource halved (not, say, `n_estimators`),
  since it's the only resource meaningful for *any* estimator -- kantun
  tunes arbitrary sklearn-compatible models, not just KANBoost. Useful
  when you have more candidates than you can afford to fully evaluate.

  ```python
  search = KantunSearch(
      KANBoostClassifier, param_space, search_type="halving",
      n_iter=20, cv=3, halving_factor=3, min_resource=50,
  )
  ```

## Speeding up an expensive search

Two independent knobs, useful together or separately, both aimed at
the case kantun was built for: tuning an estimator that's slow to
fit per combination (like KANBoost, ~10-20x a tree ensemble):

- **`n_jobs`**: evaluate multiple param combos concurrently (threads,
  not processes -- safe for CUDA device selection, and PyTorch releases
  the GIL during tensor ops so real overlap still happens).
  ```python
  search = KantunSearch(KANBoostClassifier, param_space, n_jobs=4)
  ```
- **`prune=True`**: abandon a combo after its *first* CV fold if that
  fold's score already falls more than `prune_margin` standard
  deviations (of the current best combo's own fold spread) below the
  running best -- skips the remaining `cv - 1` folds for combos that
  are essentially never going to become the best. A pruned combo's
  single-fold score is recorded in `cv_results_` (`"pruned": True`) but
  never becomes `best_params_`/`best_score_`. Off by default; the first
  combo evaluated is never pruned (there's nothing to compare against
  yet).
  ```python
  search = KantunSearch(KANBoostClassifier, param_space, prune=True, prune_margin=1.0)
  ```

## License

MIT — see [`LICENSE`](./LICENSE).
