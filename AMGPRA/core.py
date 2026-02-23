import numpy as np
from smt.sampling_methods import LHS
from .models import MultiFidelityModel
from .learning import lf_eff


class AMGPRA:
    def __init__(self, functions, costs, xlimits, S_candidate=None, n_init=None, n_mc=100000):
        self.functions = functions
        self.costs = costs
        self.xlimits = xlimits
        self.n_mc = n_mc
        self.n_init = n_init or min(12, (xlimits.shape[0] + 1) * (xlimits.shape[0] + 2) // 2)

        if S_candidate is not None:
            self.S = S_candidate
            self.n_mc = S_candidate.shape[0]
        else:
            sampler = LHS(xlimits=self.xlimits, criterion="maximin")
            self.S = sampler(self.n_mc)

        self.xt = [None] * len(functions)
        self.yt = [None] * len(functions)
        self.model = MultiFidelityModel()
        self.initialize_doe()

    def initialize_doe(self):
        """ Draw initial points for all fidelities using LHS"""
        sampler = LHS(xlimits=self.xlimits)
        x_init = sampler(self.n_init)
        for i in range(len(self.functions)):
            self.xt[i] = x_init
            self.yt[i] = self.functions[i](x_init).reshape(-1, 1)
        self.model.train(self.xt, self.yt)

    def compute_clf(self, x_cand, fid_idx, Nc_indices, mu_S, sigma_S):
        """Implementation  of the collective learning function"""
        hf_level = len(self.functions) - 1
        thetas = self.model.model.optimal_theta[hf_level]

        mu_Nc = mu_S[Nc_indices]
        sigma_Nc = sigma_S[Nc_indices]
        coords_Nc = self.S[Nc_indices]

        diff = (coords_Nc - x_cand.reshape(1, -1)) ** 2
        correlation = np.exp(-np.sum(thetas * diff, axis=1))

        sigma_bar2 = (sigma_Nc ** 2) * (correlation ** 2)
        sigma_F = np.sqrt(np.maximum(sigma_Nc ** 2 - sigma_bar2, 1e-12))

        lf_prev = lf_eff(mu_Nc, sigma_Nc)
        lf_future = lf_eff(mu_Nc, sigma_F)

        improvement = np.mean(lf_prev - lf_future)
        return improvement / self.costs[fid_idx]

    def run(self):
        """Adaptative loop for the AMGPRA procedure"""
        for i in range(200):
            mu_S, sigma_S = self.model.predict(self.S)
            eff_S = lf_eff(mu_S, sigma_S)
            pf = np.mean(mu_S <= 0)
            cov_pf = np.sqrt((1 - pf) / (self.n_mc * pf)) if pf > 0 else 1.0

            print(f"Iteration {i:02d} | Pf: {pf:.4e} | max(EFF): {np.max(eff_S):.2e}")

            if np.max(eff_S) < 0.001 and (pf == 0 or cov_pf <= 0.05):
                print("--- Convergence reached ---")
                break

            Nc_indices = np.argsort(eff_S)[-min(1000, len(eff_S)):]
            best_clf_score = -1.0
            best_x, best_fid = None, 0

            sample_indices = np.random.choice(Nc_indices, min(50, len(Nc_indices)), replace=False)
            for idx in sample_indices:
                x_cand = self.S[idx]
                for f_idx in range(len(self.functions)):
                    score = self.compute_clf(x_cand, f_idx, Nc_indices, mu_S, sigma_S)
                    if score > best_clf_score:
                        best_clf_score = score
                        best_x, best_fid = x_cand, f_idx

            self.enrich(best_x, best_fid)
        return pf

    def enrich(self, x, fid):
        """Update the DoE and train again"""
        x = x.reshape(1, -1)
        new_y = self.functions[fid](x)
        self.xt[fid] = np.vstack([self.xt[fid], x])
        self.yt[fid] = np.vstack([self.yt[fid], new_y.reshape(-1, 1)])

        if fid == 0:
            for low_fid in range(1, len(self.functions)):
                y_low = self.functions[low_fid](x)
                self.xt[low_fid] = np.vstack([self.xt[low_fid], x])
                self.yt[low_fid] = np.vstack([self.yt[low_fid], y_low.reshape(-1, 1)])

        self.model.train(self.xt, self.yt)
