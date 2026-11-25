"""
model_zoo.py
Shared pieces for 08_stress_test.py and 09_neural_net.py.

* Re-uses the preprocessing pipeline, features and metrics from 07_models.py, so every
  model (old and new) sees EXACTLY the same winsorizing / imputing / scaling / encoding,
  all fit on the training rows only.
* Adds a neural network: a small multilayer perceptron (sklearn MLP) averaged over several
  random seeds. Averaging matters here because the signal is weak and a single MLP's
  result depends heavily on its random initialisation.
* fit_eval(train, test, k) fits every model for one return window and returns result rows
  in the same format as data/model_results.csv.
"""
import importlib
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)

m07 = importlib.import_module("07_models")      # file name starts with a digit -> importlib
NUM, CAT, pipe = m07.NUM, m07.CAT, m07.pipe
RANDOM_STATE, EMBARGO_DAYS = m07.RANDOM_STATE, m07.EMBARGO_DAYS

NN_NAME = "Neural Net (64-32)"
NN_PARAMS = dict(hidden_layer_sizes=(64, 32), alpha=1e-2)       # pre-specified main config
MIN_TRAIN, MIN_TEST = 200, 50


# --------------------------------------------------------------------------- NN wrappers
class MLPEnsembleRegressor(RegressorMixin, BaseEstimator):
    """Average of `n_seeds` MLPs. The target is standardised internally (returns are ~0.03
    in scale, which makes raw-target MLP training badly conditioned)."""
    def __init__(self, hidden_layer_sizes=(64, 32), alpha=1e-2, n_seeds=5,
                 max_iter=300, random_state=RANDOM_STATE):
        self.hidden_layer_sizes, self.alpha, self.n_seeds = hidden_layer_sizes, alpha, n_seeds
        self.max_iter, self.random_state = max_iter, random_state

    def fit(self, X, y):
        y = np.asarray(y, float)
        self.mu_, self.sd_ = y.mean(), (y.std() or 1.0)
        z = (y - self.mu_) / self.sd_
        self.models_ = [
            MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                         max_iter=self.max_iter, random_state=self.random_state + s).fit(X, z)
            for s in range(self.n_seeds)]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0) * self.sd_ + self.mu_


class MLPEnsembleClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden_layer_sizes=(64, 32), alpha=1e-2, n_seeds=5,
                 max_iter=300, random_state=RANDOM_STATE):
        self.hidden_layer_sizes, self.alpha, self.n_seeds = hidden_layer_sizes, alpha, n_seeds
        self.max_iter, self.random_state = max_iter, random_state

    def fit(self, X, y):
        self.models_ = [
            MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                          early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                          max_iter=self.max_iter, random_state=self.random_state + s).fit(X, y)
            for s in range(self.n_seeds)]
        self.classes_ = self.models_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# --------------------------------------------------------------------------- evaluation
def _X(d):
    return d[NUM + CAT].assign(quarter=d["quarter"].astype(str))


def _clf_metrics(y, pred, score):
    both = y.nunique() > 1
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "auc": float(roc_auc_score(y, score)) if both else np.nan}


def fit_eval(tr, te, k, meta=None, classic=True, nn_cfgs=None):
    """Fit all models on `tr`, score on `te`, for the k-day window. Returns a list of dicts.
    classic=False -> only the neural net(s). nn_cfgs: {name: MLP kwargs}; default = main NN."""
    car, lab = f"car_{k}d", f"positive_car_{k}d"
    tr, te = tr.dropna(subset=[car]), te.dropna(subset=[car])
    meta = {**(meta or {}), "window": f"{k}d", "n_train": len(tr), "n_test": len(te)}
    if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
        print(f"    {k}d: too little data (train {len(tr)}, test {len(te)}), skipped")
        return []
    nn_cfgs = {NN_NAME: NN_PARAMS} if nn_cfgs is None else nn_cfgs
    Xtr, Xte, rows = _X(tr), _X(te), []

    # ---- regression (target clipped at train-only 1/99 percentiles, as in 07) ----
    lo, hi = tr[car].quantile([0.01, 0.99])
    ytr, yte = tr[car].clip(lo, hi), te[car]
    regs = {}
    if classic:
        regs = {"Baseline (train mean)": DummyRegressor(strategy="mean"),
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(
                    n_estimators=300, min_samples_leaf=25, max_features=0.5,
                    n_jobs=-1, random_state=RANDOM_STATE)}
    regs.update({n: MLPEnsembleRegressor(**p) for n, p in nn_cfgs.items()})
    for name, m in regs.items():
        pred = pipe(m).fit(Xtr, ytr).predict(Xte)
        rows.append({**meta, "task": "regression", "model": name,
                     **m07.reg_metrics(yte, pred, ytr.mean())})

    # ---- classification ----
    ytr_c, yte_c = tr[lab].astype(int), te[lab].astype(int)
    base_acc = max(yte_c.mean(), 1 - yte_c.mean())
    clfs = {}
    if classic:
        clfs = {"Baseline (majority class)": DummyClassifier(strategy="prior"),
                "Logistic Regression": LogisticRegression(max_iter=2000),
                "Random Forest": RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=25, max_features="sqrt",
                    n_jobs=-1, random_state=RANDOM_STATE)}
    clfs.update({n: MLPEnsembleClassifier(**p) for n, p in nn_cfgs.items()})
    for name, m in clfs.items():
        p = pipe(m).fit(Xtr, ytr_c)
        rows.append({**meta, "task": "classification", "model": name,
                     **_clf_metrics(yte_c, p.predict(Xte), p.predict_proba(Xte)[:, 1]),
                     "majority_acc": base_acc})
    if classic:
        rule = (te["surprise"] > 0).astype(int)
        rows.append({**meta, "task": "classification", "model": "Rule: beat = up",
                     **_clf_metrics(yte_c, rule, te["surprise_dir"]), "majority_acc": base_acc})
    return rows