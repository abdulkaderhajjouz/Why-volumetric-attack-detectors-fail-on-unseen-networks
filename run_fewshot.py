"""
run_fewshot.py -- label-efficient adaptation to an unseen network.

The operator of the new (target) network labels k flows per class. Compared strategies
(all on CFS+BRE features, LightGBM-equivalent capacity):
  Target-only     : LightGBM trained on the k-shot target labels only
  Source-only     : zero-shot model trained on the three source networks
  Pooled          : sources + target shots, target given the mass of one environment
  Warm-start      : source LightGBM, boosting continued on target shots (init_score)
  GB-pooled+T     : EnvBoost (pooled gain), target shots as a 4th environment
  SL-WEG+T        : EnvBoost with Shared-Leaf Worst-Environment Gain, target shots as a
                    4th environment => every split must also not harm the target network
                    under the shared leaves ("target veto").
"""
import time, sys
import numpy as np, pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed
import run_lodo as R
from ergb import EnvBoost

KS = [5, 10, 25, 50, 100, 250]
SEEDS = 3
LGB = dict(n_estimators=200, num_leaves=31, learning_rate=0.1, min_child_samples=5,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=1, verbose=-1)


def run(data, target, seed, view):
    rng = np.random.RandomState(seed * 1000 + R.ENVS.index(target) + 7)
    pools, refs = {}, {}
    for e in R.ENVS:
        pools[e], refs[e] = R.split_env(data[e], 1000, 6000, rng)
    V = R.build_views(pools, refs, data, R.ENVS, [view])[view]
    src = [e for e in R.ENVS if e != target]
    Xs = pd.concat([V[e] for e in src]).values
    ys = np.concatenate([pools[e].y.values for e in src])
    es = np.concatenate([np.full(len(pools[e]), i) for i, e in enumerate(src)])
    ws = np.concatenate([np.full(len(pools[e]), 1.0 / len(pools[e])) for e in src])
    XT, yT = V[target].values, pools[target].y.values
    src_lgb = lgb.LGBMClassifier(random_state=seed, **LGB).fit(Xs, ys, sample_weight=ws * len(ws))
    rows = []
    for k in KS:
        idx_l = np.concatenate([rng.choice(np.where(yT == c)[0], k, replace=False) for c in (0, 1)])
        mask = np.ones(len(yT), bool); mask[idx_l] = False
        Xl, yl, Xte, yte = XT[idx_l], yT[idx_l], XT[mask], yT[mask]
        preds = {}
        t = lgb.LGBMClassifier(random_state=seed, **{**LGB, "n_estimators": 100,
                                                      "min_child_samples": 2, "subsample": 1.0})
        preds["Target-only"] = t.fit(Xl, yl).predict_proba(Xte)[:, 1]
        preds["Source-only"] = src_lgb.predict_proba(Xte)[:, 1]
        Xp = np.vstack([Xs, Xl]); yp = np.concatenate([ys, yl])
        wp = np.concatenate([ws, np.full(len(yl), 1.0 / len(yl))]); wp = wp * len(wp) / wp.sum()
        preds["Pooled"] = lgb.LGBMClassifier(random_state=seed, **LGB).fit(
            Xp, yp, sample_weight=wp).predict_proba(Xte)[:, 1]
        init_l = src_lgb.predict(Xl, raw_score=True)
        ws_m = lgb.LGBMClassifier(random_state=seed, **{**LGB, "n_estimators": 50,
                                                         "learning_rate": 0.05, "min_child_samples": 2,
                                                         "subsample": 1.0})
        ws_m.fit(Xl, yl, init_score=init_l)
        raw = src_lgb.predict(Xte, raw_score=True) + ws_m.predict(Xte, raw_score=True)
        preds["Warm-start"] = 1 / (1 + np.exp(-raw))
        ep = np.concatenate([es, np.full(len(yl), len(src))])
        for name, crit in [("GB-pooled+T", "pooled"), ("SL-WEG+T", "slweg")]:
            m = EnvBoost(criterion=crit, random_state=seed, min_env_frac=0.0).fit(Xp, yp, ep)
            preds[name] = m.predict_proba(Xte)[:, 1]
        for name, s in preds.items():
            r = dict(target=target, seed=seed, view=view, k=k, model=name)
            r.update(R.metrics(yte, s))
            rows.append(r)
    return rows


if __name__ == "__main__":
    view = sys.argv[1] if len(sys.argv) > 1 else "cfs_bre"
    data = R.load_all()
    t0 = time.time()
    res = Parallel(n_jobs=2, verbose=5)(delayed(run)(data, t, s, view)
                                        for s in range(SEEDS) for t in R.ENVS)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(f"{R.OUT}/fewshot_{view}.csv", index=False)
    print(f"done fewshot in {time.time()-t0:.0f}s")
