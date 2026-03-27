import numpy as np

from core import MFEGRA
from learning import lf_eff

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def plot_limit_states(algo, hf_function, xlimits, iteration=None, n_grid=300):
    x1 = np.linspace(xlimits[0, 0], xlimits[0, 1], n_grid)
    x2 = np.linspace(xlimits[1, 0], xlimits[1, 1], n_grid)
    X1, X2 = np.meshgrid(x1, x2)

    Xgrid = np.column_stack([X1.ravel(), X2.ravel()])

    # vraie fonction HF
    G_true = hf_function(Xgrid).reshape(X1.shape)

    # métamodèle HF
    mu_hf, _ = algo.model.predict_hf(Xgrid)
    G_pred = mu_hf.reshape(X1.shape)

    fig, ax = plt.subplots(figsize=(10, 4))

    # contour réel g0(x)=0
    ax.contour(
        X1, X2, G_true,
        levels=[0.0],
        colors='red',
        linewidths=2.0
    )

    # contour métamodèle mu_HF(x)=0
    ax.contour(
        X1, X2, G_pred,
        levels=[0.0],
        colors='blue',
        linewidths=2.0,
        linestyles='--'
    )

    # points DOE par fidélité
    markers = ['*', 's', 'o']
    labels = [r'$g_0$ training points', r'$g_1$ training points', r'$g_2$ training points']
    facecolors = ['none', 'none', 'none']

    for fid in range(len(algo.xt)):
        Xfid = algo.xt[fid]
        if Xfid is None or len(Xfid) == 0:
            continue

        if markers[fid] == '*':
            ax.scatter(
                Xfid[:, 0], Xfid[:, 1],
                marker=markers[fid],
                s=80,
                c='tab:blue',
                linewidths=1.2
            )
        else:
            ax.scatter(
                Xfid[:, 0], Xfid[:, 1],
                marker=markers[fid],
                s=70,
                facecolors='none',
                edgecolors=('tab:orange' if fid == 1 else 'teal'),
                linewidths=1.5
            )

    # légende propre
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Actual limit state'),
        Line2D([0], [0], color='blue', lw=2, ls='--', label='Metamodel limit state'),
        Line2D([0], [0], marker='*', color='tab:blue', lw=0, markersize=10,
               label=r'$g_0$ training points'),
        Line2D([0], [0], marker='s', color='tab:orange', markerfacecolor='none',
               lw=0, markersize=8, markeredgewidth=1.5,
               label=r'$g_1$ training points'),
        Line2D([0], [0], marker='o', color='teal', markerfacecolor='none',
               lw=0, markersize=9, markeredgewidth=1.8,
               label=r'$g_2$ training points'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=False,
              edgecolor='black', fontsize=11, handlelength=3.5)

    if iteration is not None:
        ax.set_title(f"Iteration {iteration}", fontsize=16)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim(xlimits[0, 0], xlimits[0, 1])
    ax.set_ylim(xlimits[1, 0], xlimits[1, 1])

    plt.tight_layout()
    plt.show()


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

algo = MFEGRA(
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

mu_S, sigma_S = algo.model.predict_hf(algo.S)
eff_S = lf_eff(mu_S, sigma_S)

idx_next = int(np.argmax(eff_S))
x_next = algo.S[idx_next]

Nc = min(algo.Nc, algo.S.shape[0])
idx_Z = np.argsort(eff_S)[-Nc:]

print("\n=== Diagnostic mfEGRA (Eq.13) ===")
print("x_next (argmax EFF) =", x_next)

for fid in range(len(functions)):
    raw_score = algo._weighted_info_gain_objective_exact_raw(
        x_next=x_next,
        fid=fid,
        eff_S=eff_S,
        idx_Z=idx_Z,
    )
    score = raw_score / costs[fid]
    print(
        f"fid={fid} | raw={raw_score:.6e} | "
        f"cost={costs[fid]:.3g} | norm={score:.6e}"
    )

out = algo.run(verbose=True)

print("\n=== Résultat ===")
print(f"Pf estimée        : {out['pf']:.6f}")
print(f"COV(Pf)           : {out['cov_pf']:.4f}")
print(f"Coût total        : {out['total_cost']:.4f}")
print(f"Nb evals / fid    : {out['eval_counts']}")
print(f"Nb itérations     : {out['n_iter']}")
print(f"n_mc final        : {out['n_mc']}")

plot_limit_states(
    algo=algo,
    hf_function=g0,
    xlimits=xlimits,
    iteration=out["n_iter"]
)