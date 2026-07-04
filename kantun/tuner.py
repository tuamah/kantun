"""
kantun.tuner -- the core search engine.

Deliberately dependency-light: only numpy, pandas, and scikit-learn's
splitters/metrics. No Optuna/Ray requirement.
"""

import inspect
import itertools
import random
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
)


_CLASSIFICATION_SCORERS = {
    "auc": lambda y_true, y_pred, y_prob: roc_auc_score(y_true, y_prob),
    "f1": lambda y_true, y_pred, y_prob: f1_score(y_true, y_pred, zero_division=0),
    "accuracy": lambda y_true, y_pred, y_prob: accuracy_score(y_true, y_pred),
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


class KantunSearch:
    """
    Hyperparameter search, compatible with KANBoost out of the box and
    with any estimator exposing fit/predict[/predict_proba].

    Parameters
    ----------
    model_cls : class
        Any estimator class with a scikit-learn-like fit/predict API.
    param_distributions : dict
        Mapping from constructor kwarg name -> list of candidate values.
    task : str, optional
        "classification" or "regression". Auto-detected from
        `hasattr(model_cls, "predict_proba")` if not given.
    n_iter : int
        Number of random parameter combinations to try (search_type="random").
    cv : int
        Number of cross-validation folds.
    scoring : str
        classification: "auc" (default), "f1", "accuracy"
        regression: "neg_mse" (default), "neg_mae"
    search_type : str
        "random" (default) or "grid".
    fixed_params : dict
        Extra kwargs passed to every model instantiation (not tuned).
    use_eval_set : bool or None
        Whether to pass eval_set=(X_val, y_val) into .fit() for early
        stopping. If None (default), auto-detected from the model's
        fit() signature.
    """

    def __init__(
        self,
        model_cls,
        param_distributions: dict,
        task: str | None = None,
        n_iter: int = 10,
        cv: int = 3,
        scoring: str | None = None,
        search_type: str = "random",
        fixed_params: dict | None = None,
        use_eval_set: bool | None = None,
        random_state: int = 42,
        verbose: bool = True,
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

        if scoring is None:
            self.scoring = "auc" if self.task == "classification" else "neg_mse"
        else:
            self.scoring = scoring

        self.cv_results_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None

    # ------------------------------------------------------------------
    def _candidate_params(self):
        keys = list(self.param_distributions.keys())
        values = list(self.param_distributions.values())

        if self.search_type == "grid":
            for combo in itertools.product(*values):
                yield dict(zip(keys, combo))
        else:
            rng = random.Random(self.random_state)
            for _ in range(self.n_iter):
                yield {k: rng.choice(v) for k, v in self.param_distributions.items()}

    def _fit_one(self, model, X_train, y_train, X_val, y_val):
        if self.use_eval_set:
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            model.fit(X_train, y_train)
        return model

    def _score(self, model, X_val, y_val):
        if self.task == "classification":
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            scorer = _CLASSIFICATION_SCORERS[self.scoring]
            return scorer(y_val, y_pred, y_prob)
        else:
            y_pred = model.predict(X_val)
            scorer = _REGRESSION_SCORERS[self.scoring]
            return scorer(y_val, y_pred)

    # ------------------------------------------------------------------
    def fit(self, X, y):
        y = np.asarray(y)
        splitter_cls = StratifiedKFold if self.task == "classification" else KFold
        splitter = splitter_cls(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        candidates = list(self._candidate_params())
        if self.verbose:
            print(
                f"[kantun] {len(candidates)} param combos x {self.cv} folds "
                f"= {len(candidates) * self.cv} fits ({self.task}, scoring={self.scoring})"
            )

        for i, params in enumerate(candidates):
            overlap = set(params) & set(self.fixed_params)
            if overlap:
                raise ValueError(
                    f"Parameter(s) {sorted(overlap)} appear in both "
                    "param_distributions and fixed_params -- remove the "
                    "duplicate from one of them."
                )

            fold_scores = []
            fold_errors = []
            t0 = time.time()

            split_iter = (
                splitter.split(X, y) if self.task == "classification" else splitter.split(X)
            )
            for train_idx, val_idx in split_iter:
                X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
                X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                try:
                    model = self.model_cls(**params, **self.fixed_params)
                    model = self._fit_one(model, X_train, y_train, X_val, y_val)
                    fold_scores.append(self._score(model, X_val, y_val))
                except Exception as exc:  # noqa: BLE001 - deliberate: one bad
                    # combo shouldn't kill the whole search
                    fold_errors.append(str(exc))

            if not fold_scores:
                if self.verbose:
                    print(f"[kantun] ({i+1}/{len(candidates)}) {params} "
                          f"-> FAILED on all folds: {fold_errors[0]}")
                self.cv_results_.append({
                    "params": params, "mean_score": None,
                    "std_score": None, "seconds": time.time() - t0,
                    "error": fold_errors[0],
                })
                continue

            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))
            elapsed = time.time() - t0

            self.cv_results_.append({
                "params": params, "mean_score": mean_score,
                "std_score": std_score, "seconds": elapsed,
            })

            if self.verbose:
                print(
                    f"[kantun] ({i+1}/{len(candidates)}) {params} "
                    f"-> {self.scoring}={mean_score:.4f} (+/- {std_score:.4f}) "
                    f"[{elapsed:.1f}s]"
                )

            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params

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
            rows.append(row)
        df = pd.DataFrame(rows).sort_values(
            "mean_score", ascending=False, na_position="last"
        ).reset_index(drop=True)
        return df
