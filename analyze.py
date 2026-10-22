"""analyze.py -- aggregate results into the tables/figures used in the manuscript."""
import os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = os.path.join(RES, "figs"); os.makedirs(FIG, exist_ok=True)
ENVS = ["UNSW", "BOTIOT", "TONIOT", "CIC17"]
NICE = {"UNSW": "UNSW-NB15", "BOTIOT": "Bot-IoT", "TONIOT": "ToN-IoT", "CIC17": "CIC-IDS2017"}

d = pd.read_csv(f"{RES}/lono_main.csv")
lono = d[d.tag == "main"]
ind = d[d.tag == "in_dist"]

out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

# ---- Table: in-distribution vs out-of-network gap
p("== In-distribution (LightGBM, CFS) per network: mean AUROC / F1")
p(ind.groupby("target")[["auroc", "auprc", "f1"]].mean().round(4).to_string())

# ---- Main table: mean over targets x seeds, per view/model
met = ["auroc", "auprc", "f1", "bacc", "tpr_at_fpr1"]
p("\n== LONO mean (over 4 targets x seeds)")
tab = lono.groupby(["view", "model"])[met].mean().round(4)
p(tab.to_string())
tab.to_csv(f"{RES}/table_main_mean.csv")

p("\n== LONO per target AUROC (mean +- sd over seeds)")
pt = lono.groupby(["view", "model", "target"]).auroc.agg(["mean", "std"]).round(3)
pt["s"] = pt["mean"].map("{:.3f}".format) + "±" + pt["std"].map("{:.3f}".format)
wide = pt["s"].unstack("target")[ENVS]
p(wide.to_string()); wide.to_csv(f"{RES}/table_per_target_auroc.csv")

pt2 = lono.groupby(["view", "model", "target"]).f1.mean().unstack("target")[ENVS].round(3)
p("\n== LONO per target F1"); p(pt2.to_string()); pt2.to_csv(f"{RES}/table_per_target_f1.csv")

# ---- Paired Wilcoxon tests over (target, seed) units
def paired(a, b, metric="auroc"):
    A = lono[(lono.view == a[0]) & (lono.model == a[1])].set_index(["target", "seed"])[metric]
    B = lono[(lono.view == b[0]) & (lono.model == b[1])].set_index(["target", "seed"])[metric]
    j = A.to_frame("a").join(B.to_frame("b"), how="inner")
    st = wilcoxon(j.a, j.b) if (j.a - j.b).abs().sum() > 0 else None
    return len(j), (j.a - j.b).mean(), (j.a > j.b).mean(), (st.pvalue if st else np.nan)

p("\n== Paired comparisons (n units, mean diff, win rate, Wilcoxon p)")
comps = [
    (("cfs", "LightGBM"), ("naive", "LightGBM")),
    (("cfs", "XGBoost"), ("naive", "XGBoost")),
    (("cfs_bre", "LightGBM"), ("cfs", "LightGBM")),
    (("cfs_bre", "LightGBM"), ("cfs_gz", "LightGBM")),
    (("cfs_bre", "XGBoost"), ("cfs_gz", "XGBoost")),
    (("cfs_bre", "SL-WEG"), ("cfs_bre", "GB-pooled")),
    (("cfs_bre", "SL-WEG"), ("cfs_bre", "Era-mean")),
    (("cfs_bre", "SL-WEG"), ("cfs_bre", "Era-min")),
    (("cfs_bre", "SL-WEG"), ("cfs_bre", "Era-dir")),
    (("cfs_bre", "SL-WEG"), ("cfs_bre", "LightGBM")),
    (("cfs_bre", "SL-WEG"), ("cfs_bre", "XGBoost")),
    (("cfs_bre", "SL-WEG"), ("naive", "LightGBM")),
    (("cfs", "SL-WEG"), ("cfs", "GB-pooled")),
]
rows = []
for a, b in comps:
    for m in ["auroc", "f1"]:
        n, md, wr, pv = paired(a, b, m)
        rows.append(dict(A="/".join(a), B="/".join(b), metric=m, n=n, mean_diff=md, win_rate=wr, p=pv))
cmp = pd.DataFrame(rows)
p(cmp.round(4).to_string()); cmp.to_csv(f"{RES}/table_paired_tests.csv", index=False)

# ---- Figure: pipeline-stage gains (naive -> CFS -> CFS+BRE -> +SL-WEG)
stages = [("naive", "LightGBM", "Naive schema\nLightGBM"), ("cfs", "LightGBM", "CFS\nLightGBM"),
          ("cfs_gz", "LightGBM", "CFS + z-score\nLightGBM"), ("cfs_bre", "LightGBM", "CFS + BRE\nLightGBM"),
          ("cfs_bre", "SL-WEG*", "CFS + BRE\nSL-WEG (source-val. ρ)")]
nested = pd.read_csv(f"{RES}/nested_rho.csv")
fig, ax = plt.subplots(figsize=(8.5, 3.6))
cols = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
x = np.arange(len(stages)); wdt = 0.2
for i, t in enumerate(ENVS):
    vals = [(nested[nested.target == t].auroc.mean() if m == "SL-WEG*" else
             lono[(lono.view == v) & (lono.model == m) & (lono.target == t)].auroc.mean()) for v, m, _ in stages]
    ax.bar(x + (i - 1.5) * wdt, vals, wdt, label=f"target: {NICE[t]}", color=cols[i])
ax.axhline(0.5, color="grey", lw=0.8, ls="--")
ax.set_xticks(x); ax.set_xticklabels([s[2] for s in stages], fontsize=8)
ax.set_ylabel("AUROC on unseen network"); ax.set_ylim(0, 1.0)
ax.legend(fontsize=7, ncol=4, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_stages.png", dpi=300); plt.close(fig)

# ---- ablations if present
if os.path.exists(f"{RES}/ablation_contam.csv"):
    a = pd.read_csv(f"{RES}/ablation_contam.csv")
    g = a.groupby(["contam", "model"])[["auroc", "f1"]].mean().round(4)
    p("\n== Contamination of reference window"); p(g.to_string()); g.to_csv(f"{RES}/table_contam.csv")
if os.path.exists(f"{RES}/ablation_nref.csv"):
    a = pd.read_csv(f"{RES}/ablation_nref.csv")
    g = a.groupby(["n_ref", "model"])[["auroc", "f1"]].mean().round(4)
    p("\n== Reference window size"); p(g.to_string()); g.to_csv(f"{RES}/table_nref.csv")
if os.path.exists(f"{RES}/ablation_rho.csv"):
    a = pd.read_csv(f"{RES}/ablation_rho.csv")
    g = a.groupby(["view", "rho"])[["auroc", "f1"]].mean().round(4)
    p("\n== rho sensitivity"); p(g.to_string()); g.to_csv(f"{RES}/table_rho.csv")
if os.path.exists(f"{RES}/ablation_contam.csv") and os.path.exists(f"{RES}/ablation_nref.csv"):
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.0))
    a = pd.read_csv(f"{RES}/ablation_contam.csv")
    for m, c in [("LightGBM", "#4C72B0"), ("SL-WEG", "#C44E52")]:
        s = a[a.model == m].groupby("contam").auroc.agg(["mean", "std"])
        axs[0].errorbar(s.index * 100, s["mean"], yerr=s["std"], marker="o", label=m, color=c, capsize=3)
    axs[0].set_xlabel("Attack flows in reference window (%)"); axs[0].set_ylabel("AUROC (mean of 4 targets)")
    b = pd.read_csv(f"{RES}/ablation_nref.csv")
    for m, c in [("LightGBM", "#4C72B0"), ("SL-WEG", "#C44E52")]:
        s = b[b.model == m].groupby("n_ref").auroc.agg(["mean", "std"])
        axs[1].errorbar(s.index, s["mean"], yerr=s["std"], marker="o", label=m, color=c, capsize=3)
    axs[1].set_xscale("log"); axs[1].set_xlabel("Reference window size (flows)")
    for ax in axs:
        ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_bre_robustness.png", dpi=300); plt.close(fig)

if os.path.exists(f"{RES}/fewshot_cfs_bre.csv"):
    f = pd.read_csv(f"{RES}/fewshot_cfs_bre.csv")
    order = ["Source-only", "Target-only", "Pooled", "Warm-start", "GB-pooled+T", "SL-WEG+T"]
    g = f.groupby(["k", "model"]).auroc.mean().unstack()[order].round(3)
    p("\n== Few-shot AUROC (mean of 4 targets x 3 seeds)"); p(g.to_string()); g.to_csv(f"{RES}/table_fewshot_auroc.csv")
    g2 = f.groupby(["k", "model"]).f1.mean().unstack()[order].round(3)
    p("\n== Few-shot F1"); p(g2.to_string()); g2.to_csv(f"{RES}/table_fewshot_f1.csv")
    g3 = f.groupby(["k", "target", "model"]).auroc.mean().unstack()[order].round(3)
    g3.to_csv(f"{RES}/table_fewshot_per_target.csv")
    p("\n== Few-shot paired tests (Wilcoxon over target x seed)")
    for k in sorted(f.k.unique()):
        for a_, b_ in [("Pooled", "Target-only"), ("Warm-start", "Target-only"), ("SL-WEG+T", "GB-pooled+T")]:
            A = f[(f.k == k) & (f.model == a_)].set_index(["target", "seed"]).auroc
            B = f[(f.k == k) & (f.model == b_)].set_index(["target", "seed"]).auroc
            j = A.to_frame("a").join(B.to_frame("b"))
            pv = wilcoxon(j.a, j.b).pvalue
            p(f"k={k:3d} {a_:12s} vs {b_:12s} diff={(j.a-j.b).mean():+.4f} win={(j.a>j.b).mean():.2f} p={pv:.4f}")
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    sty = {"Source-only": ("#999999", "--"), "Target-only": ("#4C72B0", "-"), "Pooled": ("#55A868", "-"),
           "Warm-start": ("#DD8452", "-"), "GB-pooled+T": ("#8172B2", ":"), "SL-WEG+T": ("#C44E52", "-.")}
    for m in order:
        s = f[f.model == m].groupby("k").auroc.mean()
        ax.plot(s.index, s.values, marker="o", ms=3, label=m, color=sty[m][0], ls=sty[m][1])
    ax.set_xscale("log"); ax.set_xlabel("Labelled target flows per class (k)")
    ax.set_ylabel("AUROC on target network"); ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=7, ncol=2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_fewshot.png", dpi=300); plt.close(fig)

open(f"{RES}/summary.txt", "w").write("\n".join(out))
