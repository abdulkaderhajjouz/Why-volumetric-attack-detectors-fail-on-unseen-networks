"""
cfs.py -- Canonical Flow Semantics (CFS) for cross-exporter volumetric attack detection.

Loads four public flow datasets produced by three different flow exporters and maps
them to (i) a NAIVE name-matched schema (what a practitioner gets by matching column
names) and (ii) the CANONICAL schema in which every quantity has the same physical
meaning (IP-layer bytes, packets, seconds).

Environments (networks / exporters):
    UNSW   : UNSW-NB15 official train+test sets      (Argus)        benign vs DoS
    BOTIOT : Bot-IoT (stratified subset)              (Argus)        benign vs DDoS/DoS
    TONIOT : ToN-IoT network Train_Test (subset)      (Zeek)         benign vs DoS+DDoS
    CIC17  : CIC-IDS2017 Friday afternoon             (CICFlowMeter) benign vs DDoS

Byte semantics (verified empirically on the data, see paper Sec. 3.2):
    Argus  sbytes/dbytes            : IP-layer bytes (min mean size 28 = IP+UDP/ICMP hdr)
    Zeek   src_ip_bytes/dst_ip_bytes: IP-layer bytes;  src_bytes/dst_bytes = payload only
    CICFlowMeter Total Length       : payload only;    Fwd Header Length = L4 headers only
            -> IP bytes = payload + L4 header bytes + 20 * packets
Duration: Argus/Zeek seconds; CICFlowMeter microseconds.
"""
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ENVS = ["UNSW", "BOTIOT", "TONIOT", "CIC17"]
EXPORTER = {"UNSW": "Argus", "BOTIOT": "Argus", "TONIOT": "Zeek", "CIC17": "CICFlowMeter"}

CANON_COLS = ["dur", "fpk", "bpk", "fby", "bby", "proto"]


def _proto(s):
    s = s.astype(str).str.strip().str.lower()
    s = s.replace({"6": "tcp", "17": "udp", "1": "icmp", "0": "other"})
    return np.where(s.isin(["tcp", "udp", "icmp"]), s, "other")


def load_env(name, path=DATA):
    """Return DataFrame with canonical + naive raw columns and binary label y."""
    if name == "UNSW":
        d = pd.concat([pd.read_csv(f"{path}/unsw_train.csv", low_memory=False),
                       pd.read_csv(f"{path}/unsw_test.csv", low_memory=False)], ignore_index=True)
        d = d[d.attack_name.isin(["Normal", "DoS"])]
        out = pd.DataFrame({
            "dur": d.dur, "fpk": d.spkts, "bpk": d.dpkts,
            "fby": d.sbytes, "bby": d.dbytes,            # IP-layer bytes
            "proto": _proto(d.protocol),
            # naive (name-matched) view == canonical for Argus
            "n_dur": d.dur, "n_fby": d.sbytes, "n_bby": d.dbytes,
            "y": (d.attack_name == "DoS").astype(int).values})
    elif name == "BOTIOT":
        # benign: all 9,543 Normal flows of the full Bot-IoT release (redistributed subset file);
        # attacks: every DoS/DDoS flow (TCP, UDP, HTTP) of the official 5% release
        b = pd.read_csv(f"{path}/botiot.csv", low_memory=False)
        b.columns = [c.strip() for c in b.columns]
        b = b[b.category == "Normal"]
        a = pd.read_csv(f"{path}/botiot_full5pc.csv", low_memory=False)
        a.columns = [c.strip() for c in a.columns]
        a = a[a.category.isin(["DoS", "DDoS"])]
        cols = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "proto"]
        d = pd.concat([b[cols].assign(y=0), a[cols].assign(y=1)], ignore_index=True)
        out = pd.DataFrame({
            "dur": d.dur, "fpk": d.spkts, "bpk": d.dpkts,
            "fby": d.sbytes, "bby": d.dbytes,
            "proto": _proto(d.proto),
            "n_dur": d.dur, "n_fby": d.sbytes, "n_bby": d.dbytes,
            "y": d.y.values})
    elif name == "TONIOT":
        # official Train_Test_Network split (all DoS/DDoS records; 50,000 normal records)
        d = pd.read_csv(f"{path}/toniot_train_test_network.csv", low_memory=False)
        d.columns = [c.strip().lstrip("﻿") for c in d.columns]
        d = d[d["type"].isin(["normal", "dos", "ddos"])]
        num = lambda c: pd.to_numeric(d[c], errors="coerce").fillna(0).clip(lower=0)
        out = pd.DataFrame({
            "dur": num("duration"), "fpk": num("src_pkts"), "bpk": num("dst_pkts"),
            "fby": num("src_ip_bytes"), "bby": num("dst_ip_bytes"),   # IP-layer bytes
            "proto": _proto(d.proto),
            # naive: Zeek columns literally named *_bytes are payload bytes
            "n_dur": num("duration"), "n_fby": num("src_bytes"), "n_bby": num("dst_bytes"),
            "y": (d["type"] != "normal").astype(int).values})
    elif name == "CIC17":
        d = pd.read_csv(f"{path}/cicids2017_fri_ddos.csv", low_memory=False, encoding="latin1")
        d.columns = [c.strip() for c in d.columns]
        num = lambda c: pd.to_numeric(d[c], errors="coerce").fillna(0).clip(lower=0)
        fpk, bpk = num("Total Fwd Packets"), num("Total Backward Packets")
        out = pd.DataFrame({
            "dur": num("Flow Duration") / 1e6, "fpk": fpk, "bpk": bpk,
            "fby": num("Total Length of Fwd Packets") + num("Fwd Header Length") + 20 * fpk,
            "bby": num("Total Length of Bwd Packets") + num("Bwd Header Length") + 20 * bpk,
            "proto": _proto(d.Protocol),
            # naive: duration as exported (microseconds), payload-only "Total Length"
            "n_dur": num("Flow Duration"), "n_fby": num("Total Length of Fwd Packets"),
            "n_bby": num("Total Length of Bwd Packets"),
            "y": (d.Label.astype(str).str.strip() != "BENIGN").astype(int).values})
    else:
        raise ValueError(name)
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    for c in ["dur", "fpk", "bpk", "fby", "bby", "n_dur", "n_fby", "n_bby"]:
        out[c] = out[c].astype(float).clip(lower=0)
    out["env"] = name
    return out


def featurize(df, view="canon"):
    """Dimensionless volumetric descriptors.

    view='canon' uses harmonized semantics, view='naive' uses name-matched raw fields.
    Packet counts are identical in both views (all exporters count packets the same way).
    """
    if view == "canon":
        dur, fby, bby = df.dur.values, df.fby.values, df.bby.values
    else:
        dur, fby, bby = df.n_dur.values, df.n_fby.values, df.n_bby.values
    fpk, bpk = df.fpk.values, df.bpk.values
    eps = 1e-6
    pk, by = fpk + bpk, fby + bby
    X = pd.DataFrame({
        "log_dur": np.log1p(dur),
        "log_fpk": np.log1p(fpk), "log_bpk": np.log1p(bpk),
        "log_fby": np.log1p(fby), "log_bby": np.log1p(bby),
        "log_fsize": np.log1p(fby / np.maximum(fpk, 1)),
        "log_bsize": np.log1p(bby / np.maximum(bpk, 1)),
        "log_pps": np.log1p(pk / np.maximum(dur, eps)),
        "log_bps": np.log1p(by / np.maximum(dur, eps)),
        "pk_asym": (fpk - bpk) / np.maximum(pk, 1),
        "by_asym": (fby - bby) / np.maximum(by, 1),
        "no_resp": (bpk == 0).astype(float),
        "is_tcp": (df.proto.values == "tcp").astype(float),
        "is_udp": (df.proto.values == "udp").astype(float),
        "is_icmp": (df.proto.values == "icmp").astype(float),
    })
    return X


CONT = ["log_dur", "log_fpk", "log_bpk", "log_fby", "log_bby", "log_fsize", "log_bsize",
        "log_pps", "log_bps"]


class BaselineRelative:
    """Baseline-Relative Encoding (BRE).

    Re-expresses each continuous descriptor of a network relative to that network's own
    benign reference window R:  z = (x - median_R) / (IQR_R + s0).
    Fitted per environment; needs NO labels from the target network, only a short window
    of traffic believed to be attack-free (contamination is studied explicitly).
    """
    def __init__(self, cols=CONT, s0=0.1):
        self.cols, self.s0 = cols, s0

    def fit(self, Xref):
        q = Xref[self.cols].quantile([0.25, 0.5, 0.75])
        self.med = q.loc[0.5].values
        self.iqr = (q.loc[0.75] - q.loc[0.25]).values
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols] = (X[self.cols].values - self.med) / (self.iqr + self.s0)
        return X


class GlobalStandardize:
    """Common practice baseline: per-dataset z-score using ALL (unlabeled) data of the network."""
    def __init__(self, cols=CONT):
        self.cols = cols

    def fit(self, X):
        self.mu = X[self.cols].mean().values
        self.sd = X[self.cols].std().values + 1e-6
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols] = (X[self.cols].values - self.mu) / self.sd
        return X


if __name__ == "__main__":
    for e in ENVS:
        d = load_env(e)
        dup = d.duplicated(subset=CANON_COLS + ["y"]).mean()
        print(f"{e:7s} {EXPORTER[e]:12s} n={len(d):7d} benign={int((d.y==0).sum()):7d} "
              f"attack={int(d.y.sum()):7d} dup={dup:.3f} "
              f"proto={dict(pd.Series(d.proto).value_counts().round(0))}")
