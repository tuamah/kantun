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
pip install -r requirements.txt
pip install -e .
```

To also tune KANBoost models, install it separately:

```bash
pip install -e ../kanboost_project   # local, or: pip install kanboost (once published to PyPI)
# repo: https://github.com/tuamah/kanboost
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

- Classification: `"auc"` (default), `"f1"`, `"accuracy"`
- Regression: `"neg_mse"` (default), `"neg_mae"`

## Search types

- `search_type="random"` (default): samples `n_iter` random combinations
- `search_type="grid"`: tries every combination in `param_distributions`

## License

MIT — see [`LICENSE`](./LICENSE).
