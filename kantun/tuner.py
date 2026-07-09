"""
kantun.tuner -- the core search engine.

Deliberately dependency-light: only numpy, pandas, scikit-learn's
splitters/metrics, and joblib (already a transitive scikit-learn
dependency). No Optuna/Ray requirement.
"""

import inspect
import itertools
import random
import threading
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
)


def _auc_score(y_true, y_pred, y_prob, labels):
    if len(labels) <= 2:
        return roc_auc_score(y_true, y_prob)
    return roc_auc_score(y_true, y_prob, multi_class="ovr", labels=labels)


def _f1_score(y_true, y_pred, y_prob, labels):
    average = "binary" if len(labels) <= 2 else "macro"
    return f1_score(y_true, y_pred, average=average, pos_label=labels[-1] if average == "binary" else 1, zero_division=0)


def _accuracy_score(y_true, y_pred, y_prob, labels):
    return accuracy_score(y_true, y_pred)


_CLASSIFICATION_SCORERS = {
    "auc": _auc_score,
    "f1": _f1_score,
    "accuracy": _accuracy_score,
}
_REGRESSION_SCORERS = {
    "neg_mse": lambda y_true, y_pred: -mean_squared_error(y_true, y_pred),
    "neg_mae": lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred),
}


def _supports_eval_set(model_cls) -> bool:
    try:
        sig = inspect.signature(model_cls.fit)
        return "eval_set" in sig.parameters
    except (TypeError, ValueError):
        return False


def _is_classifier(model_cls) -> bool:
    return hasattr(model_cls, "predict_proba")


def _iloc(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


class KantunSearch:
    """
    Hyperparameter search, compatible with KANBoost out of the box and
    with any estimator exposing fit/predict[/predict_proba].

    Parameters
    ----------
    model_cls : class
        Any estimator class with a scikit-learn-like fit/predict API.
    param_distributions : dict
        Mapping from constructor kwarg name -> either a list of candidate
        values, or (for `search_type="random"`/`"halving"` only) a
        callable `sampler(rng) -> value` for continuous ranges, where
        `rng` is a `random.Random` instance seeded from `random_state`
        (e.g. `"kan_lr": lambda rng: 10 ** rng.uniform(-3, -1)` for
        log-uniform sampling). `search_type="grid"` requires plain lists
        (a callable can't be enumerated).
    task : str, optional
        "classification" or "regression". Auto-detected from
        `hasattr(model_cls, "predict_proba")` if not given.
    n_iter : int
        Number of random parameter combinations to try (search_type="random"
        or "halving").
    cv : int
        Number of cross-validation folds.
    scoring : str or callable
        classification: "auc" (default), "f1", "accuracy", or a callable
        `scorer(y_true, y_pred, y_prob, labels) -> float` (higher is
        better).
        regression: "neg_mse" (default), "neg_mae", or a callable
        `scorer(y_true, y_pred) -> float` (higher is better).
    search_type : str
        "random" (default), "grid", or "halving" (successive halving --
        see below).
    fixed_params : dict
        Extra kwargs passed to every model instantiation (not tuned).
    use_eval_set : bool or None
        Whether to pass eval_set=(X_val, y_val) into .fit() for early
        stopping. If None (default), auto-detected from the model's
        fit() signature.
    n_jobs : int
        Number of combos to evaluate concurrently (default 1 =
        sequential, matching prior behavior exactly). Uses threads, not
        processes: safe for CUDA device selection and avoids pickling
        arbitrary estimator classes, and PyTorch releases the GIL during
        tensor ops so real overlap still happens for a KANBoost-style
        estimator.
    prune : bool
        If True, abandon a combo after its first fold if that fold's
        score is already far below the current best (see
        `_evaluate_combo`'s docstring for the exact rule). Off by
        default -- it trades a small amount of accuracy in the reported
        mean_score for pruned combos (which never becomes `best_score_`
        anyway) for a large wall-clock saving on expensive estimators.
    prune_margin : float
        How many (best combo's own) standard deviations below the
        current best a fold-1 score must fall to be pruned. Larger =
        more conservative (prunes less).
    halving_factor : int
        Only used when `search_type="halving"`: candidates surviving
        each rung are cut by this factor, and each rung's training data
        is (roughly) this factor closer to the full dataset.
    min_resource : int
        Only used when `search_type="halving"`: the smallest per-class
        training-set size used at the first rung.
    time_budget_s : float, optional
        Wall-clock budget in seconds. Checked between combos
        (`search_type="random"`/`"grid"`, in batches of `n_jobs`) or
        between rungs (`search_type="halving"`) -- never mid-fit, so a
        combo already dispatched always finishes. At least one
        combo/rung always runs regardless of budget. For "halving", if
        the budget is hit before the final full-data rung, the best
        candidate from the last completed rung is promoted to
        `best_params_` so `fit()` still succeeds.
    refit : bool, default=True
        If True (default), refit `best_params_` on the full dataset and
        store it as `best_estimator_`. Set False to skip that final fit
        (e.g. when `model_cls` is expensive to train and you only need
        `best_params_`/`cv_results_`) -- `best_estimator_` stays None.
    """

    def __init__(
        self,
        model_cls,
        param_distributions: dict,
        task: str | None = None,
        n_iter: int = 10,
        cv: int = 3,
        scoring=None,
        search_type: str = "random",
        fixed_params: dict | None = None,
        use_eval_set: bool | None = None,
        random_state: int = 42,
        verbose: bool = True,
        n_jobs: int = 1,
        prune: bool = False,
        prune_margin: float = 1.0,
        halving_factor: int = 3,
        min_resource: int = 8,
        time_budget_s: float | None = None,
        refit: bool = True,
    ):
        self.model_cls = model_cls
        self.param_distributions = param_distributions
        self.task = task or ("classification" if _is_classifier(model_cls) else "regression")
        self.n_iter = n_iter
        self.cv = cv
        self.search_type = search_type
        self.fixed_params = fixed_params or {}
        self.use_eval_set = (
            use_eval_set if use_eval_set is not None else _supports_eval_set(model_cls)
        )
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.prune = prune
        self.prune_margin = prune_margin
        self.halving_factor = halving_factor
        self.min_resource = min_resource
        self.time_budget_s = time_budget_s
        self.refit = refit

        if scoring is None:
            self.scoring = "auc" if self.task == "classification" else "neg_mse"
        else:
            self.scoring = scoring

        self.cv_results_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_std_ = None
        self.best_estimator_ = None
        self._best_lock = threading.Lock()

    # ------------------------------------------------------------------
    def _candidate_params(self):
        keys = list(self.param_distributions.keys())
        values = list(self.param_distributions.values())

        if self.search_type == "grid":
            for k, v in zip(keys, values):
                if callable(v):
                    raise ValueError(
                        f"param_distributions[{k!r}] is a callable sampler, which "
                        "needs an enumerable list of values for search_type='grid'. "
                        "Use search_type='random' or 'halving' for continuous "
                        "(callable) parameter distributions."
                    )
            for combo in itertools.product(*values):
                yield dict(zip(keys, combo))
        else:
            rng = random.Random(self.random_state)
            for _ in range(self.n_iter):
                yield {
                    k: (v(rng) if callable(v) else rng.choice(v))
                    for k, v in self.param_distributions.items()
                }

    def _fit_one(self, model, X_train, y_train, X_val, y_val):
        if self.use_eval_set:
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            model.fit(X_train, y_train)
        return model

    def _score(self, model, X_val, y_val):
        if self.task == "classification":
            proba = model.predict_proba(X_val)
            is_binary = proba.ndim == 1 or proba.shape[1] <= 2
            y_prob = proba[:, 1] if (proba.ndim == 2 and is_binary) else proba
            # model.predict() (not argmax) so labels stay correct even when
            # classes_ isn't {0, 1, ..., K-1} -- e.g. KANBoostClassifier.
            y_pred = model.predict(X_val)
            labels = getattr(model, "classes_", None)
            if labels is None:
                labels = np.unique(y_val)
            scorer = self.scoring if callable(self.scoring) else _CLASSIFICATION_SCORERS[self.scoring]
            return scorer(y_val, y_pred, y_prob, labels)
        else:
            y_pred = model.predict(X_val)
            scorer = self.scoring if callable(self.scoring) else _REGRESSION_SCORERS[self.scoring]
            return scorer(y_val, y_pred)

    def _subsample_train(self, X_train, y_train, n_resources):
        """Stratified (classification) subsample of a fold's *train*
        portion down to `n_resources` rows -- never touches validation
        indices, so held-out scoring is always on full, untouched data
        even at the smallest halving rung."""
        n_available = len(y_train)
        n_classes = len(np.unique(y_train)) if self.task == "classification" else 1
        # Fold sizes aren't perfectly equal (StratifiedKFold rounds), so a
        # rung's resource count computed from one fold can land just
        # *below* another fold's n_available -- close enough that the
        # discarded remainder is smaller than n_classes, which
        # train_test_split's stratification rejects outright. Treat
        # "not enough left over to stratify the discarded part" the same
        # as "at or above capacity": use the full fold untouched.
        if n_resources >= n_available or (n_available - n_resources) < n_classes:
            return X_train, y_train
        stratify = y_train if self.task == "classification" else None
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train, train_size=n_resources,
            stratify=stratify, random_state=self.random_state,
        )
        return X_sub, y_sub

    def _evaluate_combo(self, i, n_total, params, X, y, splits, n_resources=None, count_towards_best=True):
        """Evaluate one param combo across precomputed `splits` (a list
        of (train_idx, val_idx) pairs, fixed up front so parallel
        execution can't change which folds a combo sees). Returns a
        cv_results_-style row dict and, if `count_towards_best` (False
        for every halving rung except the final full-data one -- a
        subsampled rung's score isn't comparable to a full CV mean, any
        more than a single pruned fold's is), thread-safely updates
        best_score_/best_params_/best_std_ if this combo's mean_score
        beats the current best (guarded by `self._best_lock` since
        threaded n_jobs>1 means combos can finish in any order and share
        that state).

        Pruning (`self.prune=True`): after the *first* fold's score,
        compare against the current running best (read under the same
        lock). If this combo is already more than `prune_margin`
        standard deviations (of the *best* combo's own fold spread, or a
        std floor of 0.01 if the best has no recorded spread yet) below
        the best's mean, stop evaluating further folds -- a combo that
        bad on fold 1 essentially never recovers to become the new best,
        and expensive estimators (this is the whole point) benefit far
        more from skipping k-1 remaining folds than from the small risk
        of it later. The very first combo overall is never pruned
        (there is nothing to compare against yet), and pruned combos
        never update best_score_/best_params_ -- a single fold's score
        isn't comparable to a full CV mean.
        """
        fold_scores = []
        fold_errors = []
        pruned = False
        t0 = time.time()

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = _iloc(X, train_idx), _iloc(X, val_idx)
            y_train, y_val = y[train_idx], y[val_idx]

            if n_resources is not None:
                X_train, y_train = self._subsample_train(X_train, y_train, n_resources)

            try:
                model = self.model_cls(**params, **self.fixed_params)
                model = self._fit_one(model, X_train, y_train, X_val, y_val)
                fold_scores.append(self._score(model, X_val, y_val))
            except Exception as exc:  # noqa: BLE001 - deliberate: one bad
                # combo shouldn't kill the whole search
                fold_errors.append(str(exc))

            if self.prune and fold_i == 0 and fold_scores:
                with self._best_lock:
                    current_best = self.best_score_
                    best_std = self.best_std_
                if current_best > -np.inf:
                    std_floor = max(best_std or 0.0, 0.01)
                    threshold = current_best - self.prune_margin * std_floor
                    if fold_scores[0] < threshold:
                        pruned = True
                        break

        elapsed = time.time() - t0

        if pruned:
            row = {
                "params": params, "mean_score": fold_scores[0],
                "std_score": None, "seconds": elapsed, "pruned": True,
            }
            if self.verbose:
                print(f"[kantun] ({i+1}/{n_total}) {params} -> pruned after fold 1 "
                      f"({self.scoring}={fold_scores[0]:.4f}) [{elapsed:.1f}s]")
        elif not fold_scores:
            row = {
                "params": params, "mean_score": None, "std_score": None,
                "seconds": elapsed, "error": fold_errors[0],
            }
            if self.verbose:
                print(f"[kantun] ({i+1}/{n_total}) {params} "
                      f"-> FAILED on all folds: {fold_errors[0]}")
        else:
            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))
            row = {
                "params": params, "mean_score": mean_score,
                "std_score": std_score, "seconds": elapsed,
            }
            if self.verbose:
                print(f"[kantun] ({i+1}/{n_total}) {params} "
                      f"-> {self.scoring}={mean_score:.4f} (+/- {std_score:.4f}) "
                      f"[{elapsed:.1f}s]")

            if count_towards_best:
                with self._best_lock:
                    if mean_score > self.best_score_:
                        self.best_score_ = mean_score
                        self.best_params_ = params
                        self.best_std_ = std_score

        if n_resources is not None:
            row["n_resources"] = n_resources
        return row

    # ------------------------------------------------------------------
    def _budget_exceeded(self, start_time, rows) -> bool:
        if self.time_budget_s is None or not rows:
            return False
        if time.time() - start_time < self.time_budget_s:
            return False
        if self.verbose:
            print(f"[kantun] time_budget_s={self.time_budget_s} reached "
                  f"after {len(rows)} combo(s); stopping search")
        return True

    def _run_flat_search(self, candidates, X, y, splits):
        n = len(candidates)
        start = time.time()
        rows = []

        if self.n_jobs == 1:
            for i, params in enumerate(candidates):
                if self._budget_exceeded(start, rows):
                    break
                rows.append(self._evaluate_combo(i, n, params, X, y, splits))
            return rows

        for batch_start in range(0, n, self.n_jobs):
            if self._budget_exceeded(start, rows):
                break
            batch = list(enumerate(candidates))[batch_start:batch_start + self.n_jobs]
            rows.extend(Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._evaluate_combo)(i, n, params, X, y, splits)
                for i, params in batch
            ))
        return rows

    def _run_halving_search(self, candidates, X, y, splits):
        """Successive halving: start with all candidates at a small
        per-fold training-set size, keep the top `1/halving_factor` by
        mean score, and grow the training-set size by `halving_factor`
        each round, until a rung trains on the full data. Training-set
        size (not e.g. `n_estimators`) is the resource halved, since
        it's the only one meaningful for *any* sklearn-compatible
        estimator -- kantun tunes arbitrary models, not just KANBoost.
        """
        n_train_full = len(splits[0][0])
        n_classes = len(np.unique(y)) if self.task == "classification" else 1
        n_resources = min(self.min_resource * max(n_classes, 1), n_train_full)

        rung = 0
        rows = []
        start = time.time()
        last_scored = []
        while True:
            if self._budget_exceeded(start, rows):
                if self.best_params_ is None and last_scored:
                    best_row, best_mean = max(last_scored, key=lambda pair: pair[1])
                    if self.verbose:
                        print(f"[kantun] promoting best candidate from rung {rung - 1} "
                              "(reached before the final full-data rung)")
                    with self._best_lock:
                        self.best_score_ = best_mean
                        self.best_params_ = best_row["params"]
                        self.best_std_ = best_row["std_score"]
                break

            n = len(candidates)
            is_final_rung = n_resources >= n_train_full
            if self.n_jobs == 1:
                rung_rows = [
                    self._evaluate_combo(i, n, params, X, y, splits, n_resources=n_resources,
                                          count_towards_best=is_final_rung)
                    for i, params in enumerate(candidates)
                ]
            else:
                rung_rows = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                    delayed(self._evaluate_combo)(i, n, params, X, y, splits, n_resources=n_resources,
                                                   count_towards_best=is_final_rung)
                    for i, params in enumerate(candidates)
                )
            for r in rung_rows:
                r["rung"] = rung
            rows.extend(rung_rows)

            if is_final_rung:
                break

            scored = [(r, r["mean_score"]) for r in rung_rows if r["mean_score"] is not None]
            if not scored:
                break  # every candidate at this rung failed; nothing to promote
            last_scored = scored
            scored.sort(key=lambda pair: pair[1], reverse=True)
            n_survivors = max(1, len(scored) // self.halving_factor)
            candidates = [r["params"] for r, _ in scored[:n_survivors]]

            rung += 1
            n_resources = min(n_resources * self.halving_factor, n_train_full)

        return rows

    # ------------------------------------------------------------------
    def fit(self, X, y):
        y = np.asarray(y)
        splitter_cls = StratifiedKFold if self.task == "classification" else KFold
        splitter = splitter_cls(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        split_iter = (
            splitter.split(X, y) if self.task == "classification" else splitter.split(X)
        )
        splits = list(split_iter)

        candidates = list(self._candidate_params())
        for params in candidates:
            overlap = set(params) & set(self.fixed_params)
            if overlap:
                raise ValueError(
                    f"Parameter(s) {sorted(overlap)} appear in both "
                    "param_distributions and fixed_params -- remove the "
                    "duplicate from one of them."
                )

        if self.verbose:
            print(
                f"[kantun] {len(candidates)} param combos x {self.cv} folds "
                f"= {len(candidates) * self.cv} fits ({self.task}, scoring={self.scoring}, "
                f"search_type={self.search_type}, n_jobs={self.n_jobs})"
            )

        if self.search_type == "halving":
            self.cv_results_ = self._run_halving_search(candidates, X, y, splits)
        else:
            self.cv_results_ = self._run_flat_search(candidates, X, y, splits)

        if self.best_params_ is None:
            first_error = next(
                (r["error"] for r in self.cv_results_ if r.get("error")), "unknown error"
            )
            raise RuntimeError(
                f"Every parameter combination failed on every fold; first error: {first_error}"
            )

        if self.refit:
            # Refit best params on the full dataset for immediate use.
            final_params = {**self.best_params_, **self.fixed_params}
            self.best_estimator_ = self.model_cls(**final_params)
            self.best_estimator_.fit(X, y)

        return self

    def results_dataframe(self):
        """Return cv_results_ as a pandas DataFrame, sorted best-first."""
        import pandas as pd

        rows = []
        for r in self.cv_results_:
            row = {**r["params"], "mean_score": r["mean_score"],
                   "std_score": r["std_score"], "seconds": r["seconds"]}
            if "pruned" in r:
                row["pruned"] = r["pruned"]
            if "rung" in r:
                row["rung"] = r["rung"]
            if "n_resources" in r:
                row["n_resources"] = r["n_resources"]
            rows.append(row)
        df = pd.DataFrame(rows).sort_values(
            "mean_score", ascending=False, na_position="last"
        ).reset_index(drop=True)
        return df
