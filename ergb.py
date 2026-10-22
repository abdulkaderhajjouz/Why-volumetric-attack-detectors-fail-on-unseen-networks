"""
ergb.py -- Environment-aware histogram gradient boosting for binary classification.

Split criteria (selected with `criterion`):
  'pooled'   : standard second-order gain (XGBoost/LightGBM style) on pooled data.
  'era_mean' : Era splitting (DeLise) with alpha = 0 -> mean of per-environment gains,
               each computed with environment-specific optimal leaf values.
  'era_min'  : Era splitting limit alpha -> -inf -> worst-environment gain, again with
               environment-specific optimal leaves (always >= 0).
  'slweg'    : Shared-Leaf Worst-Environment Gain (proposed). The candidate split is scored
               by the loss reduction it brings to EACH environment *under the leaf values the
               model will actually share* (pooled Newton leaves). Unlike env-specific gains,
               this quantity becomes negative when a split that helps the pooled data harms
               one environment, so harmful (shortcut) splits are rejected.
               robust = H_node * min_e  dL_e / H_e   ;  score = (1-rho)*pooled + rho*robust

All per-environment quantities are normalised by the environment's Hessian mass in the
node, so environments of different size are compared on a per-sample scale.
"""
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class EnvBoost:
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=5, reg_lambda=1.0,
                 min_child_weight=1.0, n_bins=64, criterion="slweg", rho=1.0,
                 min_env_frac=0.05, env_balance=True, class_balance=True, subsample=1.0,
                 colsample=1.0, random_state=0):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    # ---------------- binning ----------------
    def _fit_bins(self, X):
        self.edges_ = []
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        for j in range(X.shape[1]):
            e = np.unique(np.quantile(X[:, j], qs))
            self.edges_.append(e)

    def _bin(self, X):
        B = np.empty(X.shape, dtype=np.int32)
        for j, e in enumerate(self.edges_):
            B[:, j] = np.searchsorted(e, X[:, j], side="right")
        return B

    # ---------------- fitting ----------------
    def fit(self, X, y, env):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        env_codes, env = np.unique(np.asarray(env), return_inverse=True)
        self.E_ = E = len(env_codes)
        n, F = X.shape
        self._fit_bins(X)
        B = self._bin(X)
        nb = self.n_bins
        # sample weights: every environment (and class within it) gets equal total mass
        w = np.ones(n)
        if self.env_balance or self.class_balance:
            for e in range(E):
                for c in (0, 1):
                    m = (env == e) & (y == c)
                    if m.sum() == 0:
                        continue
                    share = (1.0 / E if self.env_balance else m.sum() / n) * \
                            (0.5 if self.class_balance else 1.0)
                    w[m] = share * n / m.sum()
        p0 = np.clip(np.sum(w * y) / np.sum(w), 1e-6, 1 - 1e-6)
        self.base_ = np.log(p0 / (1 - p0))
        Fx = np.full(n, self.base_)
        self.trees_ = []
        offs = (np.arange(F) * E * nb)[None, :]
        for t in range(self.n_estimators):
            p = _sigmoid(Fx)
            g = w * (p - y)
            h = w * p * (1 - p)
            rows = np.arange(n) if self.subsample >= 1 else \
                np.sort(rng.choice(n, int(self.subsample * n), replace=False))
            feats = np.arange(F) if self.colsample >= 1 else \
                np.sort(rng.choice(F, max(1, int(self.colsample * F)), replace=False))
            tree = self._grow(B, g, h, env, rows, feats, offs, E, F, nb)
            self.trees_.append(tree)
            Fx += self.learning_rate * self._predict_tree(tree, B)
        return self

    def _best_split(self, Bn, gn, hn, en, offs, E, F, nb, feats):
        m = len(gn)
        flat = (offs + (en * nb)[:, None] + Bn).ravel()
        size = F * E * nb
        G = np.bincount(flat, weights=np.repeat(gn, F), minlength=size).reshape(F, E, nb)
        H = np.bincount(flat, weights=np.repeat(hn, F), minlength=size).reshape(F, E, nb)
        GL = np.cumsum(G, axis=2)[:, :, :-1]          # (F,E,nb-1): left = bins <= k
        HL = np.cumsum(H, axis=2)[:, :, :-1]
        Ge = G.sum(2)[:, :, None]                     # (F,E,1) node totals per env
        He = H.sum(2)[:, :, None]
        GR, HR = Ge - GL, He - HL
        lam = self.reg_lambda
        GLp, HLp, GRp, HRp = GL.sum(1), HL.sum(1), GR.sum(1), HR.sum(1)   # (F,nb-1)
        Gp, Hp = Ge.sum(1), He.sum(1)                                       # (F,1)
        pooled = 0.5 * (GLp ** 2 / (HLp + lam) + GRp ** 2 / (HRp + lam) - Gp ** 2 / (Hp + lam))
        valid = (HLp >= self.min_child_weight) & (HRp >= self.min_child_weight)
        crit = self.criterion
        He0 = He[0, :, 0]                                  # env hessian mass in node
        active = He0 >= self.min_env_frac * He0.sum()
        if crit == "pooled" or active.sum() < 2:
            score = pooled
        else:
            a = active
            if crit in ("era_mean", "era_min"):
                ge = 0.5 * (GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - Ge ** 2 / (He + lam))
                ge = ge[:, a, :] / He[:, a, :]             # per-unit-hessian gain
                agg = ge.mean(1) if crit == "era_mean" else ge.min(1)
                score = Hp * agg
            elif crit == "era_dir":
                # directional era splitting: pooled gain x agreement of per-env split direction
                vL = -GL[:, a, :] / (HL[:, a, :] + lam)
                vR = -GR[:, a, :] / (HR[:, a, :] + lam)
                D = np.abs(np.sign(vL - vR).mean(1))
                score = pooled * D
            elif crit == "slweg":
                wL = -GLp / (HLp + lam)
                wR = -GRp / (HRp + lam)
                wP = -Gp / (Hp + lam)
                lossP = Ge * wP[:, None, :] + 0.5 * He * wP[:, None, :] ** 2
                lossC = (GL * wL[:, None, :] + 0.5 * HL * wL[:, None, :] ** 2 +
                         GR * wR[:, None, :] + 0.5 * HR * wR[:, None, :] ** 2)
                dL = (lossP - lossC)[:, a, :] / He[:, a, :]
                robust = Hp * dL.min(1)
                score = (1 - self.rho) * pooled + self.rho * robust
            else:
                raise ValueError(crit)
        score = np.where(valid, score, -np.inf)
        mask = np.full(F, False)
        mask[feats] = True
        score[~mask, :] = -np.inf
        j, k = np.unravel_index(np.argmax(score), score.shape)
        return j, k, score[j, k]

    def _grow(self, B, g, h, env, rows, feats, offs, E, F, nb):
        feat, thr, left, right, value = [], [], [], [], []

        def new_node():
            feat.append(-1); thr.append(0); left.append(-1); right.append(-1); value.append(0.0)
            return len(feat) - 1

        root = new_node()
        frontier = [(root, rows)]
        lam = self.reg_lambda
        for depth in range(self.max_depth + 1):
            nxt = []
            for nid, idx in frontier:
                gs, hs = g[idx].sum(), h[idx].sum()
                value[nid] = -gs / (hs + lam)
                if depth == self.max_depth or len(idx) < 2 or hs < 2 * self.min_child_weight:
                    continue
                j, k, s = self._best_split(B[idx], g[idx], h[idx], env[idx], offs, E, F, nb, feats)
                if not np.isfinite(s) or s <= 1e-12:
                    continue
                go_left = B[idx, j] <= k
                l, r = new_node(), new_node()
                feat[nid], thr[nid], left[nid], right[nid] = j, k, l, r
                nxt.append((l, idx[go_left]))
                nxt.append((r, idx[~go_left]))
            frontier = nxt
            if not frontier:
                break
        return (np.array(feat), np.array(thr), np.array(left), np.array(right), np.array(value))

    @staticmethod
    def _predict_tree(tree, B):
        feat, thr, left, right, value = tree
        node = np.zeros(B.shape[0], dtype=np.int64)
        for _ in range(64):
            f = feat[node]
            internal = f >= 0
            if not internal.any():
                break
            ii = np.where(internal)[0]
            go_left = B[ii, f[ii]] <= thr[node[ii]]
            node[ii] = np.where(go_left, left[node[ii]], right[node[ii]])
        return value[node]

    def decision_function(self, X):
        B = self._bin(np.asarray(X, dtype=float))
        out = np.full(B.shape[0], self.base_)
        for tree in self.trees_:
            out += self.learning_rate * self._predict_tree(tree, B)
        return out

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def feature_gain(self):
        """Number of splits per feature (for analysis)."""
        cnt = {}
        for feat, *_ in self.trees_:
            for f in feat[feat >= 0]:
                cnt[int(f)] = cnt.get(int(f), 0) + 1
        return cnt
