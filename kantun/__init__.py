"""
Kantun: a lightweight, model-agnostic hyperparameter tuning library.

Designed first for KANBoost (KANBoostClassifier / KANBoostRegressor),
but usable with *any* estimator that exposes a scikit-learn-like
interface:

    model.fit(X, y)                      # required
    model.fit(X, y, eval_set=(X_val, y_val))   # optional, used if the
                                                # estimator supports it
    model.predict(X)                     # required
    model.predict_proba(X)               # required only for classifiers

Kantun does NOT import kanboost. It has zero hard dependency on it --
install kanboost separately if that's what you're tuning. This keeps
kantun usable for tuning other custom estimators too.
"""

from .tuner import KantunSearch

__version__ = "0.0.1"
__all__ = ["KantunSearch"]
