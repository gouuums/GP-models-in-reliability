import secrets
import numpy as np
from smt.sampling_methods import LHS
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve

from models import MultiFidelityModel
from learning import lf_eff


class MFEGRA:
    def __init__(
        self,
        functions,
        costs,
        xlimits,
        S_candidate=None,
        n_init=None,
        n_mc=100000,
        max_iter=200,
        eff_stop=1e-3,
        Nc=1000,
        failure_if_positive=False,
        random_state=None,   # kept for compatibility, unused
        dist="uniform",
        dist_mean=None,
        dist_std=None,
        use_normal_lhs=False,
        cov_stop=0.05,
        delta_S=10_000,
        max_mc=200_000,
        nugget=1e-10,
    ):
        self.functions = functions
        self.costs = np.asarray(costs, dtype=float)
        self.xlimits = np.asarray(xlimits, dtype=float)

        self.n_mc = int(n_mc)
        self.max_iter = int(max_iter)
        self.eff_stop = float(eff_stop)
        self.Nc = int(Nc)
        self.failure_if_positive = bool(failure_if_positive)

        self.n_init = int(n_init or 10)

        self.rng = np.random.default_rng()

        self.dist = dist
        self.use_normal_lhs = bool(use_normal_lhs)

        d = self.xlimits.shape[0]
        self.dist_mean = np.array(
            dist_mean if dist_mean is not None else [0.0] * d,
            dtype=float,
        )
        self.dist_std = np.array(
            dist_std if dist_std is not None else [1.0] * d,
            dtype=float,
        )

        self.cov_stop = float(cov_stop)
        self.delta_S = int(delta_S)
        self.max_mc = int(max_mc)
        self.nugget = float(nugget)

        if S_candidate is not None:
            self.S = np.asarray(S_candidate, dtype=float)
            self.n_mc = self.S.shape[0]
        else:
            self.S = self._build_candidate_set(self.n_mc)

        self.xt = [None] * len(functions)
        self.yt = [None] * len(functions)

        self.total_cost = 0.0
        self.eval_counts = np.zeros(len(functions), dtype=int)

        self.model = MultiFidelityModel()
        self.initialize_doe()

        self._cache_valid = False
        self._K_chol = None
        self._train_levels = None
        self._train_X = None
        self._rebuild_cov_cache()

    def _rand_seed(self) -> int:
        return secrets.randbits(31)

    def _build_candidate_set(self, n: int) -> np.ndarray:
        d = self.xlimits.shape[0]

        if self.dist == "normal":
            mean = self.dist_mean
            std = self.dist_std

            if self.use_normal_lhs:
                sampler = LHS(
                    xlimits=np.array([[0.0, 1.0]] * d),
                    criterion="maximin",
                    seed=self._rand_seed(),
                )
                U = np.clip(sampler(n), 1e-12, 1.0 - 1e-12)
                return mean + std * norm.ppf(U)

            return self.rng.normal(loc=mean, scale=std, size=(n, d))

        sampler = LHS(
            xlimits=self.xlimits,
            criterion="maximin",
            seed=self._rand_seed(),
        )
        return sampler(n)

    def _extend_S(self):
        if self.n_mc >= self.max_mc:
            return False

        add_n = min(self.delta_S, self.max_mc - self.n_mc)
        extra = self._build_candidate_set(add_n)
        self.S = np.vstack([self.S, extra])
        self.n_mc = self.S.shape[0]
        return True

    def initialize_doe(self):
        sampler = LHS(xlimits=self.xlimits, seed=self._rand_seed())
        x_init = sampler(self.n_init)

        for fid in range(len(self.functions)):
            self.xt[fid] = x_init.copy()
            yi = self.functions[fid](x_init)
            self.yt[fid] = np.asarray(yi).reshape(-1, 1)

            n = x_init.shape[0]
            self.eval_counts[fid] += n
            self.total_cost += float(self.costs[fid]) * n

        self.model.train(self.xt, self.yt)

    # ------------------------------------------------------------
    # Prior / posterior covariance reconstruction
    # ------------------------------------------------------------

    def _already_evaluated(self, x, fid, tol=1e-8):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        if self.xt[fid] is None or self.xt[fid].shape[0] == 0:
            return False
        dists = np.linalg.norm(self.xt[fid] - x, axis=1)
        return np.min(dists) < tol


    def _select_next_pair(self, eff_S, verbose=False):
        # candidats classés par EFF décroissante
        order = np.argsort(eff_S)[::-1]

        Nc = min(self.Nc, self.S.shape[0])
        idx_Z = np.argsort(eff_S)[-Nc:]

        for idx in order:
            x_next = self.S[idx]

            best_fid = None
            best_score = -np.inf

            for fid in range(len(self.functions)):
                # on saute les couples déjà évalués
                if self._already_evaluated(x_next, fid):
                    continue

                raw_score = self._weighted_info_gain_objective_exact_raw(
                    x_next=x_next,
                    fid=fid,
                    eff_S=eff_S,
                    idx_Z=idx_Z,
                )
                score = raw_score / float(self.costs[fid])

                if verbose:
                    print(
                        f"     x_idx={idx} | fid={fid} | raw={raw_score:.6e} | "
                        f"cost={self.costs[fid]:.3g} | norm={score:.6e}"
                    )

                if score > best_score:
                    best_score = score
                    best_fid = fid

            # si au moins une fidélité est disponible pour ce point, on le prend
            if best_fid is not None:
                return x_next, best_fid, best_score

        # aucun couple nouveau disponible
        return None, None, None

    def _rbf_kernel(self, X, Y, theta, sigma2):
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        theta = np.asarray(theta).reshape(1, -1)
        diff2 = (X[:, None, :] - Y[None, :, :]) ** 2
        r = np.exp(-np.sum(theta * diff2, axis=2))
        return float(sigma2) * r

    def _k_pr(self, l, X, lp, Y):
        theta0 = self.model.get_theta_for_fid(0)
        sigma20 = self.model.get_sigma2_for_fid(0)
        K0 = self._rbf_kernel(X, Y, theta0, sigma20)

        if l == lp and l != 0:
            thetal = self.model.get_theta_for_fid(l)
            sigma2l = self.model.get_sigma2_for_fid(l)
            Kl = self._rbf_kernel(X, Y, thetal, sigma2l)
            return K0 + Kl

        return K0

    def _rebuild_cov_cache(self):
        train = self.model.get_training_data()

        levels = []
        Xs = []
        for fid, X in train:
            levels.extend([fid] * X.shape[0])
            Xs.append(X)

        levels = np.array(levels, dtype=int)
        X_all = np.vstack(Xs)
        n = X_all.shape[0]

        K = np.empty((n, n), dtype=float)
        for i in range(n):
            li = int(levels[i])
            xi = X_all[i:i + 1]
            for j in range(i, n):
                lj = int(levels[j])
                xj = X_all[j:j + 1]
                kij = self._k_pr(li, xi, lj, xj)[0, 0]
                K[i, j] = kij
                K[j, i] = kij

        K = K + np.eye(n) * (self.nugget + 1e-10) 
        self._K_chol = cho_factor(K, lower=True, check_finite=False)
        self._train_levels = levels
        self._train_X = X_all
        self._cache_valid = True

    def _k_vec_to_train(self, l, X):
        X = np.atleast_2d(X)
        n_eval = X.shape[0]
        n_train = self._train_X.shape[0]
        out = np.empty((n_eval, n_train), dtype=float)

        for j in range(n_train):
            lj = int(self._train_levels[j])
            xj = self._train_X[j:j + 1]
            out[:, j] = self._k_pr(l, X, lj, xj).reshape(-1)

        return out

    def _post_cov(self, l, X, lp, Y):
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        kXY = self._k_pr(l, X, lp, Y)
        kX = self._k_vec_to_train(l, X)
        kY = self._k_vec_to_train(lp, Y)

        alpha = cho_solve(self._K_chol, kY.T, check_finite=False)
        return kXY - kX @ alpha

    def _D_kl_local(self, sigmaP0, sigmaF0, sigma_bar2):
        sigmaP0 = np.asarray(sigmaP0, dtype=float)
        sigmaF0 = np.asarray(sigmaF0, dtype=float)
        sigma_bar2 = np.asarray(sigma_bar2, dtype=float)

        ratio = sigmaF0 / sigmaP0
        return np.log(ratio) + (sigmaP0**2 + sigma_bar2) / (2.0 * sigmaF0**2) - 0.5

    def _weighted_info_gain_objective_exact_raw(self, x_next, fid, eff_S, idx_Z):
        Z = self.S[idx_Z]
        w = eff_S[idx_Z]

        x_next = np.asarray(x_next, dtype=float).reshape(1, -1)

        cov_0z_f = self._post_cov(0, Z, fid, x_next).reshape(-1)
        var_f = self._post_cov(fid, x_next, fid, x_next)[0, 0]
        if fid == 0:
    # On simule que même h0 a une toute petite incertitude résiduelle 
    # pour éviter l'explosion du score à l'infini
            var_f += 3e-1
        var_0z = np.diag(self._post_cov(0, Z, 0, Z))

        var_f = float(max(var_f, 1e-12))
        var_0z = np.maximum(var_0z, 1e-12)

        sigmaP0 = np.sqrt(var_0z)
        sigma_bar2 = (cov_0z_f**2) / var_f
        sigma_bar2 = np.minimum(sigma_bar2, var_0z - 1e-15)

        sigmaF2 = np.maximum(var_0z - sigma_bar2, 1e-12)
        sigmaF0 = np.sqrt(sigmaF2)

        D = self._D_kl_local(sigmaP0, sigmaF0, sigma_bar2)
        return float(np.sum(w * D))

    def _weighted_info_gain_objective_exact(self, x_next, fid, eff_S, idx_Z):
        raw_score = self._weighted_info_gain_objective_exact_raw(
            x_next=x_next,
            fid=fid,
            eff_S=eff_S,
            idx_Z=idx_Z,
        )
        return float(raw_score / float(self.costs[fid]))

    def run(self, verbose=True):
        it = -1

        for it in range(self.max_iter):
            mu_S, sigma_S = self.model.predict_hf(self.S)
            eff_S = lf_eff(mu_S, sigma_S)
            max_eff = float(np.max(eff_S))

            if self.failure_if_positive:
                pf = float(np.mean(mu_S > 0.0))
            else:
                pf = float(np.mean(mu_S <= 0.0))

            cov_pf = np.sqrt((1.0 - pf) / (self.n_mc * pf)) if pf > 0 else np.inf

            if verbose:
                print(
                    f"Iter {it:03d} | Pf≈{pf:.6f} | COV≈{cov_pf:.3f} | "
                    f"max(EFF)={max_eff:.2e} | n_mc={self.n_mc} | "
                    f"cost={self.total_cost:.2f} | evals={self.eval_counts.tolist()}"
                )

            if max_eff < self.eff_stop:
                if cov_pf > self.cov_stop:
                    old = self.n_mc
                    if self._extend_S():
                        if verbose:
                            print(f"  -> Err OK but COV too high: extend S {old} -> {self.n_mc}")
                        continue
                if verbose:
                    print("  -> STOP")
                break

            x_next, best_fid, best_score = self._select_next_pair(eff_S, verbose=verbose)

            if x_next is None:
                if verbose:
                    print("  -> no new admissible (x, fid) pair found, STOP")
                break

            if verbose:
                print(f"  -> choose x_next={x_next}, best fid={best_fid}, score={best_score:.3e}")

            self.enrich(x_next, best_fid)

        mu_S, _ = self.model.predict_hf(self.S)
        if self.failure_if_positive:
            pf_final = float(np.mean(mu_S > 0.0))
        else:
            pf_final = float(np.mean(mu_S <= 0.0))

        cov_pf_final = (
            np.sqrt((1.0 - pf_final) / (self.n_mc * pf_final))
            if pf_final > 0
            else np.inf
        )

        return {
            "pf": pf_final,
            "cov_pf": float(cov_pf_final),
            "total_cost": float(self.total_cost),
            "eval_counts": self.eval_counts.copy(),
            "n_iter": int(it + 1),
            "n_mc": int(self.n_mc),
        }

    def enrich(self, x, fid):
        x = np.asarray(x, dtype=float).reshape(1, -1)
    
    # Vérifier si le point existe déjà pour cette fidélité
        if self.xt[fid] is not None:
            dists = np.linalg.norm(self.xt[fid] - x, axis=1)
            if np.min(dists) < 1e-8:
                print(f"  !! Point déjà évalué pour fid={fid}, on ignore l'enrichissement.")
                return
        # AMGPRA-like nested enrichment:
        # when HF is selected, evaluate all fidelities at the same point
        y_new = np.asarray(self.functions[fid](x)).reshape(-1, 1)
        self.xt[fid] = np.vstack([self.xt[fid], x])
        self.yt[fid] = np.vstack([self.yt[fid], y_new])
        self.eval_counts[fid] += 1
        self.total_cost += float(self.costs[fid])

        self.model.train(self.xt, self.yt)
        self._rebuild_cov_cache()