"""diagnostics.py -- per-network signature analysis (univariate AUROC & direction consistency)."""
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_lodo as R
from cfs import featurize

FIG = os.path.join(R.OUT, "figs"); os.makedirs(FIG, exist_ok=True)
data = R.load_all()
U = {}
for e, d in data.items():
    X = featurize(d, "canon"); y = d.y.values
    U[e] = {c: roc_auc_score(y, X[c]) for c in X.columns if X[c].nunique() > 1}
U = pd.DataFrame(U)[R.ENVS]
U = U.drop(index=[i for i in U.index if U.loc[i].sub(0.5).abs().max() < 0.02])  # drop dead features
U.round(3).to_csv(f"{R.OUT}/table_univariate_auroc.csv")

# Direction consistency index per feature: |mean sign(AUROC-0.5)| over networks (1 = same direction everywhere)
sgn = np.sign(U - 0.5)
dci = sgn.mean(1).abs()
strength = (U - 0.5).abs().mean(1)
summ = pd.DataFrame({"DCI": dci, "mean_|AUROC-0.5|": strength}).round(3)
print(summ.sort_values("DCI"))
summ.to_csv(f"{R.OUT}/table_direction_consistency.csv")
print("features with DCI==1:", list(dci[dci == 1].index))

# Source-target agreement: for each held-out target, fraction of informative features whose
# direction in the target matches the majority direction of the three source networks
rows = []
for t in R.ENVS:
    src = [e for e in R.ENVS if e != t]
    maj = np.sign(sgn[src].sum(1))
    inf = (U[t] - 0.5).abs() > 0.05
    agree = ((maj == sgn[t]) & inf & (maj != 0)).sum() / max((inf & (maj != 0)).sum(), 1)
    # weighted directional correlation
    rho = np.corrcoef(U[src].mean(1) - 0.5, U[t] - 0.5)[0, 1]
    rows.append(dict(target=t, direction_agreement=agree, signature_correlation=rho))
st = pd.DataFrame(rows).round(3); print(st); st.to_csv(f"{R.OUT}/table_source_target_agreement.csv", index=False)

fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow((U - 0.5).values, cmap="RdBu_r", vmin=-0.45, vmax=0.45, aspect="auto")
ax.set_xticks(range(len(R.ENVS))); ax.set_xticklabels(["UNSW-NB15", "Bot-IoT", "ToN-IoT", "CIC-IDS2017"], fontsize=8)
ax.set_yticks(range(len(U))); NAMES = {"log_dur": "duration", "log_fpk": "fwd packets", "log_bpk": "bwd packets", "log_fby": "fwd bytes", "log_bby": "bwd bytes", "log_fsize": "fwd mean packet size", "log_bsize": "bwd mean packet size", "log_pps": "packet rate", "log_bps": "byte rate", "pk_asym": "packet asymmetry", "by_asym": "byte asymmetry", "no_resp": "no response", "is_tcp": "TCP", "is_udp": "UDP"}
ax.set_yticklabels([NAMES.get(i, i) for i in U.index], fontsize=8)
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        ax.text(j, i, f"{U.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(U.values[i, j] - 0.5) > 0.3 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label("univariate AUROC − 0.5", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_signature_heatmap.png", dpi=300); plt.close(fig)
