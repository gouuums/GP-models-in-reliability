import os
import re
import time
import tempfile
import subprocess
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from core import MFEGRA


# ============================================================
# Configuration générale
# ============================================================

# Exécutable Solar dans le même dossier que ce script
SOLAR_EXE = r".\SOLAR_WINDOWS.exe"

# Problème solar10.1
PROBLEM_ID = 10

# Deux niveaux de fidélité
FID_HF = 1.0
FID_LF = 0.5

# Seuil de fiabilité : tau = (1 + DELTA) * F*
DELTA = 0.10

# Valeur de référence F* (à remplacer si tu as la vraie valeur exacte)
F_STAR_OVERRIDE = 42.0

# Bornes de solar10.1
XLIMITS = np.array([
    [793.0, 995.0],   # x1
    [2.0,   50.0],    # x2
    [2.0,   30.0],    # x3
    [0.01,   5.0],    # x4
    [0.01,   5.0],    # x5
], dtype=float)

# Paramètres mfEGRA
N_INIT = 8
N_MC = 10000
NC = 1000
MAX_ITER = 80
EFF_STOP = 1e-3
COV_STOP = 0.05
DELTA_S = 10000
MAX_MC = 200000

# Convention : g(x)=tau-F(x), défaillance si g(x)<0
FAILURE_IF_POSITIVE = False

# Distribution d'entrée
DIST = "uniform"

# Format du fichier x.txt :
#   "lines" -> une valeur par ligne
#   "space" -> une seule ligne avec valeurs séparées par espaces
X_FILE_FORMAT = "lines"

# Modes d'exécution
RUN_BENCHMARK = False
RUN_MFEGRA = True


# ============================================================
# Outils Solar
# ============================================================

def write_x_file(x: np.ndarray) -> str:
    """
    Ecrit x dans un fichier temporaire pour Solar.
    """
    x = np.asarray(x, dtype=float).ravel()

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    try:
        if X_FILE_FORMAT == "lines":
            for xi in x:
                tmp.write(f"{xi:.16g}\n")
        elif X_FILE_FORMAT == "space":
            tmp.write(" ".join(f"{xi:.16g}" for xi in x) + "\n")
        else:
            raise ValueError("X_FILE_FORMAT must be 'lines' or 'space'.")

        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


def build_solar_command(x: np.ndarray, fid: float) -> tuple[list[str], str]:
    """
    Construit la commande Solar :
        SOLAR_WINDOWS.exe 10 x.txt -fid=0.5
    """
    x_file = write_x_file(x)
    cmd = [
        SOLAR_EXE,
        str(PROBLEM_ID),
        x_file,
        f"-fid={fid}",
    ]
    return cmd, x_file


def parse_objective_from_output(stdout: str, stderr: str = "") -> float:
    """
    Extrait la valeur objectif de la sortie de Solar.
    Si besoin, adapter les regex après un premier test.
    """
    text = stdout + "\n" + stderr

    patterns = [
        r"f1\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        r"objective\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        r"obj\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        r"value\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))

    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if nums:
        return float(nums[-1])

    raise RuntimeError(
        "Impossible de parser la sortie de SOLAR.\n"
        f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
    )


def solar_objective(x: np.ndarray, fid: float, verbose_cmd: bool = False) -> tuple[float, float]:
    """
    Retourne (valeur objectif, temps d'évaluation).
    """
    cmd, x_file = build_solar_command(x, fid)

    try:
        if verbose_cmd:
            print("CMD =", cmd)

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            raise RuntimeError(
                f"SOLAR a échoué avec code {proc.returncode}\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        value = parse_objective_from_output(proc.stdout, proc.stderr)
        return value, elapsed

    finally:
        if os.path.exists(x_file):
            os.remove(x_file)


def solar_objective_batch(X: np.ndarray, fid: float) -> np.ndarray:
    """
    Evalue Solar sur une matrice X de taille (n,d).
    """
    X = np.atleast_2d(X)
    vals = np.empty(X.shape[0], dtype=float)
    for i, x in enumerate(X):
        vals[i], _ = solar_objective(x, fid=fid, verbose_cmd=False)
    return vals


# ============================================================
# Transformation optimisation -> fiabilité
# ============================================================

def get_reference_objective() -> float:
    """
    Valeur de référence F* utilisée pour définir tau.
    """
    return float(F_STAR_OVERRIDE)


def make_limit_state_function(tau: float, fid: float):
    """
    Construit g(x) = tau - F(x; fid)
    """
    def g(X: np.ndarray) -> np.ndarray:
        Fvals = solar_objective_batch(X, fid=fid)
        return tau - Fvals
    return g


# ============================================================
# Echantillonnage / coûts
# ============================================================

def lhs_uniform(n: int, xlimits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Petit LHS uniforme.
    """
    d = xlimits.shape[0]
    X = np.empty((n, d), dtype=float)

    for j in range(d):
        perm = rng.permutation(n)
        u = (perm + rng.random(n)) / n
        lo, hi = xlimits[j]
        X[:, j] = lo + (hi - lo) * u

    return X


def estimate_relative_costs(n_probe: int = 5, rng_seed: int = 1234):
    """
    Estime les coûts relatifs HF/LF à partir des temps de calcul moyens.
    On normalise le coût HF à 1.
    """
    rng = np.random.default_rng(rng_seed)
    Xprobe = lhs_uniform(n_probe, XLIMITS, rng)

    times_hf = []
    times_lf = []

    for x in Xprobe:
        _, th = solar_objective(x, fid=FID_HF, verbose_cmd=False)
        _, tl = solar_objective(x, fid=FID_LF, verbose_cmd=False)
        times_hf.append(th)
        times_lf.append(tl)

    mean_hf = float(np.mean(times_hf))
    mean_lf = float(np.mean(times_lf))

    if mean_hf <= 0.0:
        raise RuntimeError("Temps HF moyen invalide.")

    costs = [1.0, mean_lf / mean_hf]
    stats = {
        "mean_time_hf": mean_hf,
        "mean_time_lf": mean_lf,
        "relative_cost_lf": costs[1],
    }
    return costs, stats


# ============================================================
# Benchmark fidélité / temps / erreur
# ============================================================

@dataclass
class FidelityBenchmarkResult:
    fid: float
    pct_time: float
    pct_error: float


def benchmark_fidelity_curve(
    fidelities=(0.25, 0.50, 0.75, 1.00),
    n_points: int = 10,
    rng_seed: int = 2025,
):
    """
    Benchmark dans l'esprit du graphique du papier :
      - % temps de calcul par rapport à fid=1
      - % erreur par rapport à fid=1

    Le point fid=1 est imposé à (0, 100) par construction.
    """
    rng = np.random.default_rng(rng_seed)
    Xtest = lhs_uniform(n_points, XLIMITS, rng)

    ref_vals = []
    ref_times = []
    for x in Xtest:
        v, t = solar_objective(x, fid=1.0, verbose_cmd=False)
        ref_vals.append(v)
        ref_times.append(t)

    ref_vals = np.array(ref_vals, dtype=float)
    ref_times = np.array(ref_times, dtype=float)
    mean_ref_time = float(np.mean(ref_times))

    out = []

    for fid in fidelities:
        if abs(fid - 1.0) < 1e-12:
            out.append(FidelityBenchmarkResult(
                fid=1.0,
                pct_time=100.0,
                pct_error=0.0,
            ))
            continue

        vals = []
        times_ = []
        for x in Xtest:
            v, t = solar_objective(x, fid=float(fid), verbose_cmd=False)
            vals.append(v)
            times_.append(t)

        vals = np.array(vals, dtype=float)
        times_ = np.array(times_, dtype=float)

        pct_time = 100.0 * float(np.mean(times_)) / mean_ref_time
        pct_err = 100.0 * np.mean(
            np.abs(vals - ref_vals) / np.maximum(np.abs(ref_vals), 1e-12)
        )

        out.append(FidelityBenchmarkResult(
            fid=float(fid),
            pct_time=float(pct_time),
            pct_error=float(pct_err),
        ))

    return out


def plot_fidelity_time_error(results: list[FidelityBenchmarkResult]) -> None:
    xs = np.array([r.pct_error for r in results], dtype=float)
    ys = np.array([r.pct_time for r in results], dtype=float)
    cs = np.array([r.fid for r in results], dtype=float)

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(xs, ys, c=cs, s=90)

    for r in results:
        plt.annotate(
            f"{r.fid:.2f}",
            (r.pct_error, r.pct_time),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    plt.xlabel("Percentage error (vs fidelity 1)")
    plt.ylabel("Percentage computing time (vs fidelity 1)")
    plt.title("solar10.1: fidelity / runtime / error benchmark")
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc)
    cbar.set_label("Fidelity")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    print("=== solar10.1 bi-fidelity setup ===")

    # Test initial de l'exécutable
    x_test = np.array([900.0, 20.0, 10.0, 1.0, 1.0], dtype=float)
    try:
        val_test, t_test = solar_objective(x_test, fid=FID_HF, verbose_cmd=True)
        print(f"Test objective at fid={FID_HF}: {val_test:.6f} (time={t_test:.4f}s)")
    except Exception:
        print("\nERREUR lors du test initial de SOLAR.")
        print("Si ça plante encore, il faudra sans doute ajuster le format de x.txt ou le parsing de sortie.")
        raise

    # 1) Benchmark fidélité / temps / erreur
    if RUN_BENCHMARK:
        print("\n=== Benchmark fidelity / runtime / error ===")
        bench = benchmark_fidelity_curve(
            fidelities=(0.25, 0.50, 0.75, 1.00),
            n_points=10,
            rng_seed=2025,
        )

        for row in bench:
            print(
                f"fid={row.fid:.2f} | "
                f"pct_time={row.pct_time:.2f} | "
                f"pct_error={row.pct_error:.2f}"
            )

        plot_fidelity_time_error(bench)

    # 2) Partie mfEGRA / fiabilité (désactivée par défaut)
    if RUN_MFEGRA:
        f_star = get_reference_objective()
        tau = (1.0 + DELTA) * f_star

        print(f"\nBest/reference objective F* : {f_star:.6f}")
        print(f"Threshold tau              : {tau:.6f}  (delta={DELTA:.0%})")

        g_hf = make_limit_state_function(tau=tau, fid=FID_HF)
        g_lf = make_limit_state_function(tau=tau, fid=FID_LF)

        print("\n=== Estimation des coûts relatifs ===")
        costs, cost_stats = estimate_relative_costs(n_probe=5, rng_seed=123)
        print(f"Mean HF time (s): {cost_stats['mean_time_hf']:.6f}")
        print(f"Mean LF time (s): {cost_stats['mean_time_lf']:.6f}")
        print(f"Relative costs   : HF={costs[0]:.4f}, LF={costs[1]:.4f}")

        algo = MFEGRA(
            functions=[g_hf, g_lf],
            costs=costs,
            xlimits=XLIMITS,
            n_init=N_INIT,
            n_mc=N_MC,
            Nc=NC,
            max_iter=MAX_ITER,
            eff_stop=EFF_STOP,
            cov_stop=COV_STOP,
            delta_S=DELTA_S,
            max_mc=MAX_MC,
            failure_if_positive=FAILURE_IF_POSITIVE,
            random_state=None,
            dist=DIST,
            use_normal_lhs=False,
        )

        out = algo.run(verbose=True)

        print("\n=== Résultat solar10.1 / bi-fidélité ===")
        print(f"Pf estimée        : {out['pf']:.6f}")
        print(f"COV(Pf)           : {out['cov_pf']:.4f}")
        print(f"Coût total        : {out['total_cost']:.4f}")
        print(f"Nb evals / fid    : {out['eval_counts']}")
        print(f"Nb itérations     : {out['n_iter']}")
        print(f"n_mc final        : {out['n_mc']}")


if __name__ == "__main__":
    main()