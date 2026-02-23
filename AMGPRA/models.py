import numpy as np
from smt.applications import MFK

class MultiFidelityModel:
    """Implement the Gaussian Process regression for multi-fidelity problems,
    using the MFK method from SMT"""
    def __init__(self):
        self.model = None

    def train(self, xt_list, yt_list):
        self.model = MFK(print_global=False, hyper_opt='Cobyla')
        num_levels = len(xt_list)

        for i in range(num_levels):
            smt_name = None if i == 0 else (num_levels - 1 - i)

            xt = np.atleast_2d(xt_list[i])
            yt = np.atleast_2d(yt_list[i])
            self.model.set_training_values(xt, yt, name=smt_name)

        self.model.train()

    def predict(self, x):
        mu = self.model.predict_values(x)
        sigma2 = self.model.predict_variances(x)
        return mu.flatten(), np.sqrt(np.maximum(sigma2.flatten(), 1e-12))