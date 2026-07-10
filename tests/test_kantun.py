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

from kantun import KantunSearch, select_smallest_within_epsilon


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


def test_parallel_matches_sequential():
    """n_jobs>1 must produce identical best_params_/best_score_ and the
    same set of per-combo scores as n_jobs=1, since candidates are
    generated up front from a single seeded RNG and splits are fixed
    before dispatch -- only wall-clock changes, not results."""
    X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    kwargs = dict(
        param_distributions={"n_estimators": [10, 20], "max_depth": [3, 5]},
        fixed_params={"random_state": 0},  # fix the *model's* own randomness too
        n_iter=4, cv=2, scoring="f1", use_eval_set=False, random_state=0, verbose=False,
    )
    seq = KantunSearch(RandomForestClassifier, **kwargs, n_jobs=1)
    seq.fit(X_df, y)
    par = KantunSearch(RandomForestClassifier, **kwargs, n_jobs=2)
    par.fit(X_df, y)

    assert seq.best_params_ == par.best_params_
    assert abs(seq.best_score_ - par.best_score_) < 1e-9
    assert sorted(r["mean_score"] for r in seq.cv_results_) == \
           sorted(r["mean_score"] for r in par.cv_results_)


def test_prune_skips_bad_combo():
    X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    search = KantunSearch(
        RandomForestClassifier,
        {"n_estimators": [1, 50], "max_depth": [1, 10]},
        fixed_params={"random_state": 0},
        search_type="grid", cv=5, scoring="f1", use_eval_set=False,
        random_state=0, verbose=False, prune=True, prune_margin=0.5,
    )
    search.fit(X_df, y)

    assert any(r.get("pruned") for r in search.cv_results_)
    # a pruned row must never win best_params_/best_score_ (only a
    # single fold's score, not comparable to a full CV mean)
    for r in search.cv_results_:
        if r.get("pruned"):
            assert search.best_params_ != r["params"] or search.best_score_ != r["mean_score"]


def test_prune_first_combo_never_pruned():
    X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    search = KantunSearch(
        RandomForestClassifier, {"n_estimators": [1]}, fixed_params={"random_state": 0},
        search_type="grid", cv=3, scoring="f1", use_eval_set=False,
        random_state=0, verbose=False, prune=True,
    )
    search.fit(X_df, y)
    assert not search.cv_results_[0].get("pruned", False)


def test_prune_default_off_results_unchanged():
    X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    kwargs = dict(
        param_distributions={"n_estimators": [1, 50], "max_depth": [1, 10]},
        fixed_params={"random_state": 0},
        search_type="grid", cv=5, scoring="f1", use_eval_set=False,
        random_state=0, verbose=False,
    )
    without_prune = KantunSearch(RandomForestClassifier, **kwargs)
    without_prune.fit(X_df, y)
    assert not any(r.get("pruned") for r in without_prune.cv_results_)
    assert all(r["std_score"] is not None for r in without_prune.cv_results_ if r["mean_score"] is not None)


def test_halving_runs_and_finds_reasonable_params():
    X, y = make_classification(n_samples=600, n_features=8, n_informative=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])

    halving = KantunSearch(
        RandomForestClassifier, {"n_estimators": [5, 20, 50, 100], "max_depth": [2, 4, 8, None]},
        fixed_params={"random_state": 0}, search_type="halving", n_iter=8, cv=3,
        scoring="auc", use_eval_set=False, random_state=0, verbose=False,
        halving_factor=2, min_resource=20,
    )
    halving.fit(X_df, y)
    assert halving.best_estimator_ is not None

    grid = KantunSearch(
        RandomForestClassifier, {"n_estimators": [5, 20, 50, 100], "max_depth": [2, 4, 8, None]},
        fixed_params={"random_state": 0}, search_type="grid", cv=3,
        scoring="auc", use_eval_set=False, random_state=0, verbose=False,
    )
    grid.fit(X_df, y)
    assert abs(halving.best_score_ - grid.best_score_) < 0.1


def test_halving_cv_results_have_rungs():
    X, y = make_classification(n_samples=400, n_features=6, n_informative=4, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])

    search = KantunSearch(
        RandomForestClassifier, {"n_estimators": [5, 50], "max_depth": [2, None]},
        fixed_params={"random_state": 0}, search_type="halving", n_iter=4, cv=3,
        scoring="auc", use_eval_set=False, random_state=0, verbose=False,
        halving_factor=2, min_resource=10,
    )
    search.fit(X_df, y)
    assert all("rung" in r and "n_resources" in r for r in search.cv_results_)
    rungs = sorted(set(r["rung"] for r in search.cv_results_))
    assert rungs == list(range(len(rungs)))  # 0, 1, 2, ... with no gaps
    resources_by_rung = {r["rung"]: r["n_resources"] for r in search.cv_results_}
    assert list(resources_by_rung.values()) == sorted(resources_by_rung.values())


def test_halving_best_score_only_from_full_data_rung():
    """Regression test: an earlier version updated best_score_/best_params_
    from every rung, including subsampled ones -- a low-resource score
    isn't comparable to a full CV mean, any more than a single pruned
    fold's score is. best_params_/best_score_ must only ever come from
    the final, full-training-data rung."""
    X, y = make_classification(n_samples=307, n_features=8, n_informative=5, random_state=7)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])

    search = KantunSearch(
        RandomForestClassifier, {"n_estimators": [5, 20, 50, 100], "max_depth": [2, 4, 8, None]},
        fixed_params={"random_state": 0}, search_type="halving", n_iter=8, cv=3,
        scoring="auc", use_eval_set=False, random_state=0, verbose=False,
        halving_factor=2, min_resource=20,
    )
    search.fit(X_df, y)

    max_n_resources = max(r["n_resources"] for r in search.cv_results_)
    best_rows = [
        r for r in search.cv_results_
        if r["params"] == search.best_params_ and r["mean_score"] == search.best_score_
    ]
    assert best_rows
    for r in best_rows:
        assert r["n_resources"] == max_n_resources


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


def test_callable_param_distribution_samples_continuous_values():
    X, y = make_classification(n_samples=250, n_features=4, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    search = KantunSearch(
        RandomForestClassifier,
        {
            "n_estimators": lambda rng: rng.randint(10, 50),
            "max_depth": [3, 5],
        },
        n_iter=6, cv=2, scoring="f1", use_eval_set=False, random_state=0, verbose=False,
    )
    search.fit(X_df, y)

    n_estimators_seen = {r["params"]["n_estimators"] for r in search.cv_results_}
    assert all(10 <= v <= 50 for v in n_estimators_seen)
    assert len(n_estimators_seen) > 1  # actually sampled a range, not a constant
    assert search.best_estimator_ is not None


def test_callable_param_distribution_rejected_for_grid():
    X, y = make_classification(n_samples=100, n_features=4, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    search = KantunSearch(
        RandomForestClassifier,
        {"n_estimators": lambda rng: rng.randint(10, 50)},
        search_type="grid", cv=2, use_eval_set=False, random_state=0, verbose=False,
    )
    try:
        search.fit(X_df, y)
        raise AssertionError("callable param_distributions value was not rejected for grid search")
    except ValueError:
        pass


def test_callable_scoring():
    from sklearn.metrics import f1_score

    def custom_scorer(y_true, y_pred, y_prob, labels):
        return f1_score(y_true, y_pred, average="binary", pos_label=labels[-1], zero_division=0)

    X, y = make_classification(n_samples=250, n_features=4, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    search = KantunSearch(
        RandomForestClassifier,
        {"n_estimators": [10, 30]},
        cv=2, scoring=custom_scorer, use_eval_set=False, random_state=0, verbose=False,
    )
    search.fit(X_df, y)
    assert 0 <= search.best_score_ <= 1


def test_time_budget_stops_search_early_but_still_yields_best_params():
    X, y = make_classification(n_samples=400, n_features=6, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])

    search = KantunSearch(
        RandomForestClassifier,
        {"n_estimators": [10, 20, 30, 40, 50]},
        n_iter=5, cv=3, scoring="f1", use_eval_set=False, random_state=0, verbose=False,
        time_budget_s=1e-6,  # effectively zero -- must still complete >=1 combo
    )
    search.fit(X_df, y)
    assert search.best_params_ is not None
    assert len(search.cv_results_) < 5


def test_time_budget_with_halving_promotes_last_rung_best():
    X, y = make_classification(n_samples=600, n_features=6, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])

    search = KantunSearch(
        RandomForestClassifier,
        {"n_estimators": [10, 20, 30, 40]},
        search_type="halving", cv=3, scoring="f1", use_eval_set=False,
        random_state=0, verbose=False, time_budget_s=1e-6, min_resource=20,
    )
    search.fit(X_df, y)
    assert search.best_params_ is not None


def test_refit_false_skips_best_estimator():
    X, y = make_classification(n_samples=200, n_features=4, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    search = KantunSearch(
        RandomForestClassifier,
        {"n_estimators": [10, 20]},
        cv=2, scoring="f1", use_eval_set=False, random_state=0, verbose=False, refit=False,
    )
    search.fit(X_df, y)
    assert search.best_params_ is not None
    assert search.best_estimator_ is None


def test_select_smallest_within_epsilon_picks_lowest_complexity_within_tolerance():
    cv_results = [
        {"params": {"n_estimators": 5}, "mean_score": 0.90, "std_score": 0.01, "seconds": 0.1},
        {"params": {"n_estimators": 10}, "mean_score": 0.95, "std_score": 0.01, "seconds": 0.1},
        {"params": {"n_estimators": 50}, "mean_score": 0.96, "std_score": 0.01, "seconds": 0.1},  # best
        {"params": {"n_estimators": 100}, "mean_score": 0.955, "std_score": 0.01, "seconds": 0.1},
    ]
    result = select_smallest_within_epsilon(
        cv_results, complexity_fn=lambda p: p["n_estimators"], epsilon=0.02,
    )
    # 0.95 and 0.96 and 0.955 all qualify (>= 0.96 - 0.02 = 0.94); smallest n_estimators among those is 10
    assert result["params"] == {"n_estimators": 10}
    assert result["complexity"] == 10
    assert result["mean_score"] == 0.95


def test_select_smallest_within_epsilon_narrow_tolerance_forces_best():
    cv_results = [
        {"params": {"n_estimators": 5}, "mean_score": 0.90, "std_score": 0.01, "seconds": 0.1},
        {"params": {"n_estimators": 50}, "mean_score": 0.96, "std_score": 0.01, "seconds": 0.1},
    ]
    result = select_smallest_within_epsilon(
        cv_results, complexity_fn=lambda p: p["n_estimators"], epsilon=0.001,
    )
    assert result["params"] == {"n_estimators": 50}


def test_select_smallest_within_epsilon_skips_errored_and_none_rows():
    cv_results = [
        {"params": {"n_estimators": 5}, "mean_score": None, "std_score": None, "seconds": 0.1},
        {"params": {"n_estimators": 10}, "mean_score": 0.5, "std_score": None,
         "seconds": 0.1, "error": "boom"},
        {"params": {"n_estimators": 20}, "mean_score": 0.93, "std_score": 0.01, "seconds": 0.1},
    ]
    result = select_smallest_within_epsilon(
        cv_results, complexity_fn=lambda p: p["n_estimators"], epsilon=0.05,
    )
    assert result["params"] == {"n_estimators": 20}


def test_select_smallest_within_epsilon_raises_when_nothing_valid():
    cv_results = [
        {"params": {"n_estimators": 5}, "mean_score": None, "std_score": None, "seconds": 0.1},
        {"params": {"n_estimators": 10}, "mean_score": 0.5, "error": "boom", "seconds": 0.1},
    ]
    try:
        select_smallest_within_epsilon(cv_results, complexity_fn=lambda p: p["n_estimators"])
        raise AssertionError("no valid rows was not rejected")
    except ValueError:
        pass


def test_select_smallest_within_epsilon_integrates_with_real_grid_search():
    # A dataset where more trees genuinely stop mattering past a point --
    # verifies the helper picks a smaller real KantunSearch grid combo,
    # not just a hand-built cv_results_ list.
    X, y = make_classification(
        n_samples=300, n_features=5, n_informative=3, random_state=0,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    search = KantunSearch(
        RandomForestClassifier,
        {"n_estimators": [3, 10, 50, 100]},
        cv=3, scoring="f1", search_type="grid", use_eval_set=False,
        random_state=0, verbose=False,
    )
    search.fit(X_df, y)

    result = select_smallest_within_epsilon(
        search.cv_results_, complexity_fn=lambda p: p["n_estimators"], epsilon=0.1,
    )
    # the selected combo must itself be one of the real searched combos,
    # and no smaller n_estimators outside the qualifying set should have won
    assert result["params"]["n_estimators"] in (3, 10, 50, 100)
    assert result["mean_score"] >= search.best_score_ - 0.1


if __name__ == "__main__":
    test_search_with_sklearn_classifier()
    test_search_with_sklearn_regressor()
    test_results_dataframe_sorted_best_first()
    test_kanboost_compatibility_if_available()
    test_search_multiclass_sklearn()
    test_kanboost_multiclass_if_available()
    test_parallel_matches_sequential()
    test_prune_skips_bad_combo()
    test_prune_first_combo_never_pruned()
    test_prune_default_off_results_unchanged()
    test_halving_runs_and_finds_reasonable_params()
    test_halving_cv_results_have_rungs()
    test_halving_best_score_only_from_full_data_rung()
    test_search_all_folds_fail_raises()
    test_callable_param_distribution_samples_continuous_values()
    test_callable_param_distribution_rejected_for_grid()
    test_callable_scoring()
    test_time_budget_stops_search_early_but_still_yields_best_params()
    test_time_budget_with_halving_promotes_last_rung_best()
    test_refit_false_skips_best_estimator()
    test_select_smallest_within_epsilon_picks_lowest_complexity_within_tolerance()
    test_select_smallest_within_epsilon_narrow_tolerance_forces_best()
    test_select_smallest_within_epsilon_skips_errored_and_none_rows()
    test_select_smallest_within_epsilon_raises_when_nothing_valid()
    test_select_smallest_within_epsilon_integrates_with_real_grid_search()
    print("All kantun tests passed.")
