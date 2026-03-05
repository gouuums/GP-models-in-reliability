import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from smt.surrogate_models import KRG  

class AKMCS:
    def __init__(self, performance_function, population, n_init=6, eff_stop=0.001):
        self.g_func = performance_function
        self.S = population.copy()
        self.n_mc = self.S.shape[0]
        self.dim = self.S.shape[1]
        self.n_init = n_init
        self.eff_stop = eff_stop
        
        self.doe_x = None
        self.doe_y = None
        self.pf_history = [] 
        
    def init_doe(self):
        idx = np.random.choice(self.S.shape[0], self.n_init, replace=False)
        self.doe_x = self.S[idx]
        self.doe_y = self.g_func(self.doe_x)
        self.S = np.delete(self.S, idx, axis=0)
        
    def run(self):
        self.init_doe()
        iteration = 0
        
        while True:
            model = KRG(print_global=False)
            model.set_training_values(self.doe_x, self.doe_y)
            model.train()
            self.model = model
            
            y_pred = model.predict_values(self.S)
            y_var = model.predict_variances(self.S)
            y_std = np.sqrt(np.maximum(y_var, 1e-10))
            
            a = 0.0
            epsilon = 2.0 * y_std
            a_minus = a - epsilon
            a_plus = a + epsilon
            
            z1 = (a - y_pred) / y_std
            z2 = (a_minus - y_pred) / y_std
            z3 = (a_plus - y_pred) / y_std
            
            term1 = (y_pred - a) * (2 * norm.cdf(z1) - norm.cdf(z2) - norm.cdf(z3))
            term2 = y_std * (2 * norm.pdf(z1) - norm.pdf(z2) - norm.pdf(z3))
            term3 = epsilon * (norm.cdf(z3) - norm.cdf(z2))
            
            eff_val = term1 - term2 + term3
            max_eff_idx = np.argmax(eff_val)
            max_eff = eff_val[max_eff_idx][0]
            
            all_preds = np.vstack((self.doe_y, y_pred))
            pf = np.sum(all_preds <= 0) / self.n_mc
            self.pf_history.append(pf) 
            
            print(f"Iter {iteration:02d} | Calls to G: {len(self.doe_y)} | Max EFF: {max_eff:.6e} | Pf: {pf:.6e}")
            
            if max_eff < self.eff_stop:
                print("\n Stopping criterion reached (max(EFF) < 0.001)")
                return pf
                
            x_new = self.S[max_eff_idx:max_eff_idx+1]
            y_new = self.g_func(x_new)
            
            self.doe_x = np.vstack((self.doe_x, x_new))
            self.doe_y = np.vstack((self.doe_y, y_new))
            self.S = np.delete(self.S, max_eff_idx, axis=0)
            iteration += 1

