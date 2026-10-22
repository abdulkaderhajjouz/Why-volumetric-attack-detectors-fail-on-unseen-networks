"""
run_lodo.py -- Leave-One-Network-Out (LONO) evaluation of volumetric attack detectors.

For every target network T and seed s:
  * train on the remaining three networks (each environment class-balanced, capped),
  * evaluate on T, which was never seen during training (different network AND,
    except for UNSW<->BOTIOT, a different flow exporter).
Writes one row per (target, seed, view, model) to results/lono.csv
"""
import argparse, os, time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import xgboost as xgb
from cfs import ENVS, CANON_COLS, load_env, featurize, BaselineRelative, GlobalStandardize
from ergb import EnvBoost

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def clean(d):
    """Remove label-conflicting duplicates, then exact duplicates (per network)."""
    key = CANON_COLS
    nlab = d.groupby(key, observed=True)["y"].transform("nunique")
    d = d[nlab == 1]
    return d.drop_duplicates(subset=key + ["y"]).reset_index(drop=True)


def load_all():
    return {e: clean(load_env(e)) for e in ENVS}


def sample_class_balanced(d, cap, rng):
    parts = []
    for c in (0, 1):
        dc = d[d.y == c]
        parts.append(dc.sample(n=min(cap, len(dc)), random_state=rng.randint(1 << 30)))
    return pd.concat(parts)


def split_env(d, n_ref, cap, rng, contam=0.0):
    """Return (train/test pool, reference window) for one network.
    Reference window = n_ref flows believed benign; a fraction `contam` are actually attacks."""
    n_att = int(round(contam * n_ref))
    ben = d[d.y == 0].sample(frac=1.0, random_state=rng.randint(1 << 30))
    att = d[d.y == 1].sample(frac=1.0, random_state=rng.randint(1 << 30))
    ref = pd.concat([ben.iloc[:n_ref - n_att], att.iloc[:n_att]])
    rest = pd.concat([ben.iloc[n_ref - n_att:], att.iloc[n_att:]])
    pool = sample_class_balanced(rest, cap, rng)
    return pool, ref


def build_views(pools, refs, alls, envs, views):
    """Return dict view -> {env: X}. Normalizers are fitted per network."""
    out = {}
    for v in views:
        out[v] = {}
        for e in envs:
            if v == "naive":
                out[v][e] = featurize(pools[e], "naive")
            elif v == "cfs":
                out[v][e] = featurize(pools[e], "canon")
            elif v == "cfs_gz":   # per-network z-score on ALL unlabeled traffic (common practice)
                gz = GlobalStandardize().fit(featurize(alls[e], "canon"))
                out[v][e] = gz.transform(featurize(pools[e], "canon"))
            elif v == "cfs_bre":  # proposed: baseline-relative encoding
                bre = BaselineRelative().fit(featurize(refs[e], "canon"))
                out[v][e] = bre.transform(featurize(pools[e], "canon"))
    return out


def models(seed, which):
    M = {
        "LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "RF": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=1,
                                             random_state=seed),
        "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=300,
                                     early_stopping=True, random_state=seed)),
        "XGBoost": lambda: xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                             subsample=0.8, colsample_bytree=0.8, n_jobs=1,
                                             random_state=seed, verbosity=0),
        "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                               n_jobs=1, random_state=seed, verbose=-1),
        "GB-pooled": lambda: EnvBoost(criterion="pooled", random_state=seed),
        "Era-mean": lambda: EnvBoost(criterion="era_mean", random_state=seed),
        "Era-min": lambda: EnvBoost(criterion="era_min", random_state=seed),
        "Era-dir": lambda: EnvBoost(criterion="era_dir", random_state=seed),
        "SL-WEG": lambda: EnvBoost(criterion="slweg", rho=1.0, random_state=seed),
    }
    return {k: M[k] for k in which}


def metrics(y, s, thr=0.5):
    fpr, tpr, _ = roc_curve(y, s)
    pred = (s >= thr).astype(int)
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    return dict(auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s),
                f1=f1_score(y, pred), tpr05=tp / max(tp + fn, 1), fpr05=fp / max(fp + tn, 1),
                bacc=0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
                tpr_at_fpr1=float(np.interp(0.01, fpr, tpr)))


def one_run(data, target, seed, views, which, cap, n_ref, contam=0.0, tag="main"):
    rng = np.random.RandomState(seed * 1000 + ENVS.index(target))
    pools, refs = {}, {}
    for e in ENVS:
        c = contam if e == target else 0.0
        pools[e], refs[e] = split_env(data[e], n_ref, cap, rng, contam=c)
    V = build_views(pools, refs, data, ENVS, views)
    src = [e for e in ENVS if e != target]
    rows = []
    for v in views:
        Xtr = pd.concat([V[v][e] for e in src]).values
        ytr = np.concatenate([pools[e].y.values for e in src])
        etr = np.concatenate([np.full(len(pools[e]), i) for i, e in enumerate(src)])
        # equal mass per environment for off-the-shelf learners as well
        w = np.concatenate([np.full(len(pools[e]), 1.0 / len(pools[e])) for e in src])
        w = w * len(w) / w.sum()
        Xte, yte = V[v][target].values, pools[target].y.values
        for name, mk in models(seed, which).items():
            if name.startswith(("GB-", "Era", "SL-")) and v in ("naive", "cfs_gz"):
                continue
            t0 = time.time()
            m = mk()
            if isinstance(m, EnvBoost):
                m.fit(Xtr, ytr, etr)
            elif name in ("LR", "MLP"):
                if name == "LR":
                    m.fit(Xtr, ytr, logisticregression__sample_weight=w)
                else:
                    m.fit(Xtr, ytr)   # sklearn MLP has no sample_weight; data is near balanced
            else:
                m.fit(Xtr, ytr, sample_weight=w)
            s = m.predict_proba(Xte)[:, 1]
            r = dict(tag=tag, target=target, seed=seed, view=v, model=name, contam=contam,
                     n_ref=n_ref, n_test=len(yte), fit_s=time.time() - t0)
            r.update(metrics(yte, s))
            rows.append(r)
    return rows


def in_distribution(data, seed, cap):
    """Reference: train/test within the same network (random 70/30)."""
    rows = []
    rng = np.random.RandomState(seed)
    for e in ENVS:
        pool = sample_class_balanced(data[e], cap, rng)
        X = featurize(pool, "canon").values; y = pool.y.values
        idx = rng.permutation(len(y)); k = int(0.7 * len(y))
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1, random_state=seed, n_jobs=1)
        m.fit(X[idx[:k]], y[idx[:k]])
        r = dict(tag="in_dist", target=e, seed=seed, view="cfs", model="LightGBM")
        r.update(metrics(y[idx[k:]], m.predict_proba(X[idx[k:]])[:, 1]))
        rows.append(r)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--n_ref", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="lono.csv")
    ap.add_argument("--models", default="LR,RF,MLP,XGBoost,LightGBM,GB-pooled,Era-mean,Era-min,Era-dir,SL-WEG")
    ap.add_argument("--views", default="naive,cfs,cfs_gz,cfs_bre")
    a = ap.parse_args()
    data = load_all()
    stats = pd.DataFrame([dict(env=e, n=len(d), benign=int((d.y == 0).sum()), attack=int(d.y.sum()))
                          for e, d in data.items()])
    stats.to_csv(f"{OUT}/env_stats_after_cleaning.csv", index=False)
    print(stats)
    jobs = [(t, s) for s in range(a.seeds) for t in ENVS]
    t0 = time.time()
    res = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(one_run)(data, t, s, a.views.split(","), a.models.split(","), a.cap, a.n_ref)
        for t, s in jobs)
    rows = [r for rr in res for r in rr]
    for s in range(a.seeds):
        rows += in_distribution(data, s, a.cap)
    pd.DataFrame(rows).to_csv(f"{OUT}/{a.out}", index=False)
    print(f"done in {time.time()-t0:.0f}s")
