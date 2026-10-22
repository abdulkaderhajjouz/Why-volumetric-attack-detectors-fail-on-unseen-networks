"""
run_ablation.py -- robustness and sensitivity studies (all leave-one-network-out).

  A) reference-window contamination: fraction of attack flows hidden in the target's
     "benign" reference window  c in {0, .05, .10, .20, .40}
  B) reference-window size: n_ref in {50, 100, 250, 500, 1000, 2000}
  C) SL-WEG mixing weight rho in {0, .25, .5, .75, 1}  (rho=0 == pooled gain)
"""
import sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import run_lodo as R
from ergb import EnvBoost

SEEDS = 3


def study_contam(data):
    jobs = [(t, s, c) for c in [0.0, 0.05, 0.10, 0.20, 0.40] for s in range(SEEDS) for t in R.ENVS]
    res = Parallel(n_jobs=2)(delayed(R.one_run)(data, t, s, ["cfs_bre"], ["LightGBM", "SL-WEG"],
                                                6000, 1000, contam=c, tag="contam")
                             for t, s, c in jobs)
    return [r for rr in res for r in rr]


def study_nref(data):
    jobs = [(t, s, n) for n in [50, 100, 250, 500, 1000, 2000] for s in range(SEEDS) for t in R.ENVS]
    res = Parallel(n_jobs=2)(delayed(R.one_run)(data, t, s, ["cfs_bre"], ["LightGBM", "SL-WEG"],
                                                6000, n, tag="nref")
                             for t, s, n in jobs)
    return [r for rr in res for r in rr]


def study_rho(data):
    rows = []
    orig = R.models

    def run(t, s, rho):
        R.models = lambda seed, which: {"SL-WEG": lambda: EnvBoost(criterion="slweg", rho=rho,
                                                                   random_state=seed)}
        out = R.one_run(data, t, s, ["cfs", "cfs_bre"], ["SL-WEG"], 6000, 1000, tag="rho")
        for r in out:
            r["rho"] = rho
        return out

    res = Parallel(n_jobs=2)(delayed(run)(t, s, rho) for rho in [0.0, 0.25, 0.5, 0.75, 1.0]
                             for s in range(SEEDS) for t in R.ENVS)
    R.models = orig
    return [r for rr in res for r in rr]


if __name__ == "__main__":
    which = sys.argv[1]
    data = R.load_all()
    t0 = time.time()
    rows = {"contam": study_contam, "nref": study_nref, "rho": study_rho}[which](data)
    pd.DataFrame(rows).to_csv(f"{R.OUT}/ablation_{which}.csv", index=False)
    print(f"done {which} in {time.time()-t0:.0f}s")
