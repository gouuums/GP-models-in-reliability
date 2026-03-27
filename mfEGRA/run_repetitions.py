import numpy as np
from tqdm import trange

from core import MFEGRA


def g0(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return 2.0 - ((x1**2 + 4.0) * (x2 - 1.0)) / 20.0 + np.sin(5.0 * x1 / 2.0)


def g1(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return g0(X) - np.sin(5.0 * x1 / 22.0 + 5.0 * x2 / 44.0 + 5.0 / 4.0)


def g2(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return g0(X) - np.sin(5.0 * x1 / 11.0 + 5.0 * x2 / 22.0 + 35.0 / 11.0)


functions = [g0, g1, g2]
costs = [1.0, 0.1, 0.01]

xlimits = np.array([
    [-4.0, 7.0],
    [-3.0, 8.0],
])

# Référence pour l'ARE
PF_REF = 0.0313

# Nombre de répétitions
N_REP = 20


def build_algo():
    return MFEGRA(
        functions=functions,
        costs=costs,
        xlimits=xlimits,
        n_init=6,
        n_mc=10_000,
        Nc=1000,
        max_iter=100,
        eff_stop=1e-3,
        cov_stop=0.05,
        delta_S=10_000,
        max_mc=200_000,
        failure_if_positive=False,
        random_state=None,
        dist="normal",
        dist_mean=[1.5, 2.5],
        dist_std=[1.0, 1.0],
        use_normal_lhs=False,
    )


outs = []
pfs = []
costs_total = []
covs = []
n_iters = []
n_mc_final = []
eval_counts = []

for _ in trange(N_REP, desc="Répétitions"):
    algo = build_algo()
    out = algo.run(verbose=False)

    outs.append(out)
    pfs.append(out["pf"])
    costs_total.append(out["total_cost"])
    covs.append(out["cov_pf"])
    n_iters.append(out["n_iter"])
    n_mc_final.append(out["n_mc"])
    eval_counts.append(out["eval_counts"])

pfs = np.array(pfs, dtype=float)
costs_total = np.array(costs_total, dtype=float)
covs = np.array(covs, dtype=float)
n_iters = np.array(n_iters, dtype=float)
n_mc_final = np.array(n_mc_final, dtype=float)
eval_counts = np.array(eval_counts, dtype=float)

avg_pf = float(np.mean(pfs))
avg_cost = float(np.mean(costs_total))
avg_cov = float(np.mean(covs))
avg_n_iter = float(np.mean(n_iters))
avg_n_mc = float(np.mean(n_mc_final))
avg_eval_counts = np.mean(eval_counts, axis=0)

are = float(np.mean(np.abs((pfs - PF_REF) / PF_REF)) * 100.0)

print("\n=== Résumé sur 20 répétitions ===")
print(f"Average cost             : {avg_cost:.4f}")
print(f"Average Pf estimate      : {avg_pf:.6f}")
print(f"Average relative error % : {are:.4f}")
print(f"Average COV(Pf)          : {avg_cov:.4f}")
print(f"Average n_iter           : {avg_n_iter:.2f}")
print(f"Average n_mc final       : {avg_n_mc:.1f}")
print(f"Average eval counts/fid  : {avg_eval_counts}")

print("\n=== Détail des Pf ===")
print(pfs)