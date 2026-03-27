import numpy as np
from smt.applications import MFK


class MultiFidelityModel:
    """
    Wrapper around SMT MFK.

    Convention used in this project:
      - fid = 0 : high fidelity (HF), stored in SMT with name=None
      - fid > 0 : lower fidelities, stored with integer names

    For 3 fidelities:
      fid=0 -> HF -> SMT name=None
      fid=1 -> MF -> SMT name=1
      fid=2 -> LF -> SMT name=0
    """

    def __init__(self):
        self.model = None
        self.num_levels = None

    def train(self, xt_list, yt_list):
        self.num_levels = len(xt_list)

        self.model = MFK(
            print_global=False,
            hyper_opt="Cobyla",
            rho_regr="constant",
        )

        for fid in range(self.num_levels):
            smt_name = None if fid == 0 else (self.num_levels - 1 - fid)
            xt = np.atleast_2d(xt_list[fid])
            yt = np.atleast_2d(yt_list[fid]).reshape(-1, 1)
            self.model.set_training_values(xt, yt, name=smt_name)

        self.model.train()

    def predict_hf(self, x):
        x = np.atleast_2d(x)
        mu = self.model.predict_values(x).reshape(-1)
        var = self.model.predict_variances(x).reshape(-1)
        sigma = np.sqrt(np.maximum(var, 1e-12))
        return mu, sigma

    def theta_index_for_fid(self, fid: int) -> int:
        if self.num_levels is None:
            raise RuntimeError("Model not trained yet.")
        if fid == 0:
            return self.num_levels - 1
        return self.num_levels - 1 - fid

    def get_theta_for_fid(self, fid: int):
        idx = self.theta_index_for_fid(fid)
        return np.asarray(self.model.optimal_theta[idx]).reshape(-1)

    def get_sigma2_for_fid(self, fid: int):
        idx = self.theta_index_for_fid(fid)
        return float(self.model.optimal_par[idx]["sigma2"])

    def get_training_data(self):
        """
        Returns a list of (fid, X) for all fidelities.
        """
        pts = []
        tp = self.model.training_points

        for smt_name in range(self.num_levels - 1):
            X = np.atleast_2d(tp[smt_name][0][0])
            fid = self.num_levels - 1 - smt_name
            pts.append((fid, X))

        Xh = np.atleast_2d(tp[None][0][0])
        pts.append((0, Xh))

        return pts