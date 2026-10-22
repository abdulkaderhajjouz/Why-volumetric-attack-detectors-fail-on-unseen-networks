"""synthetic.py -- (i) controlled shortcut experiment; (ii) split activity of SL-WEG on real data."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from ergb import EnvBoost
import run_lodo as R

rows = []
for seed in range(5):
    rng = np.random.RandomState(seed)

    def make(n, e, flip):
        y = rng.randint(0, 2, n)
        x0 = y + rng.randn(n) * 1.0                      # invariant but noisy mechanism
        x1 = (y if not flip else 1 - y) + rng.randn(n) * 0.3  # sharp shortcut, flips in env 2
        return np.c_[x0, x1, rng.randn(n)], y, np.full(n, e)
    parts = [make(4000, 0, False), make(4000, 1, False), make(1000, 2, True)]
    X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    env = np.concatenate([p[2] for p in parts])
    Xt, yt, _ = make(5000, 9, True)
    for c in ["pooled", "era_mean", "era_min", "era_dir", "slweg"]:
        m = EnvBoost(n_estimators=100, max_depth=4, criterion=c, random_state=seed).fit(X, y, env)
        g = m.feature_gain(); tot = sum(g.values())
        rows.append(dict(seed=seed, model=c, test_auroc=roc_auc_score(yt, m.predict_proba(Xt)[:, 1]),
                         shortcut_split_share=g.get(1, 0) / tot))
    l = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=seed).fit(X, y)
    rows.append(dict(seed=seed, model="LightGBM", test_auroc=roc_auc_score(yt, l.predict_proba(Xt)[:, 1]),
                     shortcut_split_share=np.nan))
s = pd.DataFrame(rows).groupby("model")[["test_auroc", "shortcut_split_share"]].agg(["mean", "std"]).round(3)
print(s); s.to_csv(f"{R.OUT}/table_synthetic.csv")

# split activity on real data: average internal nodes per tree
data = R.load_all()
act = []
for t in R.ENVS:
    rng = np.random.RandomState(t.__hash__() % 1000)
    pools, refs = {}, {}
    for e in R.ENVS:
        pools[e], refs[e] = R.split_env(data[e], 1000, 6000, np.random.RandomState(1 + R.ENVS.index(e)))
    V = R.build_views(pools, refs, data, R.ENVS, ["cfs_bre"])["cfs_bre"]
    src = [e for e in R.ENVS if e != t]
    X = pd.concat([V[e] for e in src]).values; y = np.concatenate([pools[e].y for e in src])
    env = np.concatenate([np.full(len(pools[e]), i) for i, e in enumerate(src)])
    for c in ["pooled", "era_min", "slweg"]:
        m = EnvBoost(criterion=c, random_state=0).fit(X, y, env)
        nodes = np.mean([(tr[0] >= 0).sum() for tr in m.trees_])
        act.append(dict(target=t, criterion=c, splits_per_tree=nodes,
                        source_train_auroc=roc_auc_score(y, m.predict_proba(X)[:, 1])))
a = pd.DataFrame(act); print(a.groupby("criterion")[["splits_per_tree", "source_train_auroc"]].mean().round(3))
a.to_csv(f"{R.OUT}/table_split_activity.csv", index=False)
