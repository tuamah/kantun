"""
Tests for kantun. Some tests optionally use kanboost if it's installed
(to verify real-world compatibility); others use only sklearn to prove
kantun has no hard dependency on kanboost.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from kantun import KantunSearch


def test_search_with_sklearn_classifier():
    X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    search = KantunSearch(
        RandomForestClassifier,
        {"n_estimators": [10, 20], "max_depth": [3, 5]},
        n_iter=3, cv=2, scoring="f1", use_eval_set=False, random_state=0, verbose=False,
    )
    search.fit(X_df, y)

    assert search.best_params_ is not None
    assert search.best_estimator_ is not None
    assert -1 <= search.best_score_ <= 1  # f1 in [0,1] but allow slack


def test_search_with_sklearn_regressor():
    X, y = make_regression(n_samples=300, n_features=4, noise=0.1, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])

    search = KantunSearch(
        RandomForestRegressor,
        {"n_estimators": [10, 20]},
        n_iter=2, cv=2, scoring="neg_mse", use_eval_set=False, random_state=0, verbose=False,
    )
    search.fit(X_df, y)
    assert search.best_estimator_ is not None


def test_results_dataframe_sorted_best_first():
    X, y = make_classification(n_samples=200, n_features=4, random_state=1)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    search = KantunSearch(
        RandomForestClassifier,
        {"n_estimators": [10, 30, 60]},
        search_type="grid", cv=2, scoring="auc", use_eval_set=False,
        random_state=0, verbose=False,
    )
    search.fit(X_df, y)
    df = search.results_dataframe()
    assert len(df) == 3
    assert list(df["mean_score"]) == sorted(df["mean_score"], reverse=True)


def test_kanboost_compatibility_if_available():
    try:
        from kanboost import KANBoostClassifier
    except ImportError:
        print("kanboost not installed -- skipping compatibility test")
        return

    X, y = make_classification(n_samples=200, n_features=5, random_state=2)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    search = KantunSearch(
        KANBoostClassifier,
        {"n_estimators": [5], "kan_steps": [3], "kan_hidden": [3]},
        n_iter=1, cv=2, scoring="auc", random_state=0, verbose=False,
    )
    search.fit(X_df, y)
    assert search.best_estimator_ is not None
    assert search.use_eval_set is True  # auto-detected from KANBoost's fit signature


def test_search_multiclass_sklearn():
    """No kanboost dependency: proves the multiclass scoring fix is
    generic, not special-cased for KANBoostClassifier."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    for scoring in ("auc", "f1", "accuracy"):
        search = KantunSearch(
            RandomForestClassifier, {"n_estimators": [10, 20]},
            n_iter=2, cv=2, scoring=scoring, use_eval_set=False,
            random_state=0, verbose=False,
        )
        search.fit(X_df, y)
        assert search.best_params_ is not None
        assert not any(r.get("error") for r in search.cv_results_)


def test_kanboost_multiclass_if_available():
    try:
        from kanboost import KANBoostClassifier
    except ImportError:
        print("kanboost not installed -- skipping multiclass compatibility test")
        return

    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=1,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    search = KantunSearch(
        KANBoostClassifier,
        {"n_estimators": [5], "kan_steps": [3], "kan_hidden": [3]},
        fixed_params={"early_stopping_rounds": None},
        n_iter=1, cv=2, scoring="auc", random_state=0, verbose=False,
    )
    search.fit(X_df, y)
    assert search.best_estimator_ is not None
    assert search.best_score_ > -np.inf


def test_search_all_folds_fail_raises():
    class _AlwaysBroken:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            raise RuntimeError("intentional failure")

        def predict(self, X):
            raise RuntimeError("intentional failure")

    X, y = make_regression(n_samples=50, n_features=3, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])

    search = KantunSearch(
        _AlwaysBroken, {"x": [1]}, task="regression",
        n_iter=1, cv=2, use_eval_set=False, verbose=False,
    )
    try:
        search.fit(X_df, y)
        raise AssertionError("all-folds-failed case did not raise")
    except RuntimeError as e:
        assert "intentional failure" in str(e)


if __name__ == "__main__":
    test_search_with_sklearn_classifier()
    test_search_with_sklearn_regressor()
    test_results_dataframe_sorted_best_first()
    test_kanboost_compatibility_if_available()
    test_search_multiclass_sklearn()
    test_kanboost_multiclass_if_available()
    test_search_all_folds_fail_raises()
    print("All kantun tests passed.")
