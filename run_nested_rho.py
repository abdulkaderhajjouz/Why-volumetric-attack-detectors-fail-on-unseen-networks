"""
run_nested_rho.py -- SL-WEG with rho selected WITHOUT any target information.

Training-domain validation (Gulrajani & Lopez-Paz, 2021): for each outer LONO fold
(target T, seed s), each candidate rho is scored by leave-one-source-out validation
inside the three source networks (train on two sources, validate on the third, rotate).
The rho with the best mean inner AUROC is then used to train on all three sources and
is evaluated once on T. rho = 0 (pooled gain) is part of the grid, so the procedure can
always fall back to standard boosting.
"""
import time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import run_lodo as R
from ergb import EnvBoost

GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
VIEW = "cfs_bre"


def fold(data, target, seed):
    rng = np.random.RandomState(seed * 1000 + R.ENVS.index(target))   # same draws as run_lodo
    pools, refs = {}, {}
    for e in R.ENVS:
        pools[e], refs[e] = R.split_env(data[e], 1000, 6000, rng)
    V = R.build_views(pools, refs, data, R.ENVS, [VIEW])[VIEW]
    src = [e for e in R.ENVS if e != target]
    inner = {}
    for rho in GRID:
        scores = []
        for val in src:
            tr = [e for e in src if e != val]
            X = pd.concat([V[e] for e in tr]).values
            y = np.concatenate([pools[e].y.values for e in tr])
            env = np.concatenate([np.full(len(pools[e]), i) for i, e in enumerate(tr)])
            m = EnvBoost(criterion="slweg", rho=rho, random_state=seed).fit(X, y, env)
            scores.append(roc_auc_score(pools[val].y.values, m.predict_proba(V[val].values)[:, 1]))
        inner[rho] = float(np.mean(scores))
    best = max(GRID, key=lambda r: (inner[r], -r))
    X = pd.concat([V[e] for e in src]).values
    y = np.concatenate([pools[e].y.values for e in src])
    env = np.concatenate([np.full(len(pools[e]), i) for i, e in enumerate(src)])
    m = EnvBoost(criterion="slweg", rho=best, random_state=seed).fit(X, y, env)
    s = m.predict_proba(V[target].values)[:, 1]
    r = dict(tag="main", target=target, seed=seed, view=VIEW, model="SL-WEG (ρ by source val.)",
             rho_selected=best, **{f"inner_rho_{g}": inner[g] for g in GRID})
    r.update(R.metrics(pools[target].y.values, s))
    return r


if __name__ == "__main__":
    data = R.load_all()
    t0 = time.time()
    rows = Parallel(n_jobs=2, verbose=5)(delayed(fold)(data, t, s) for s in range(5) for t in R.ENVS)
    pd.DataFrame(rows).to_csv(f"{R.OUT}/nested_rho.csv", index=False)
    print(f"done nested in {time.time()-t0:.0f}s")
