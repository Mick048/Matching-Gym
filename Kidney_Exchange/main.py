import os
import sys
from datetime import datetime

import numpy as np

from utils.Data_Structure import Graph
from utils.Data_Bank import DATA_BANK
from utils.Emprical_Sampler import EmpiricalSampler
from utils.Sample_Arrivals import sample_arrivals
from utils.Build_Edges import build_edges
from utils.Step_Pool import step_pool


def setup_logging(method_name, log_dir="experimental_results/logs", *, also_console=True):
    """Redirect all print() output to a log file (and optionally still show on console)."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{method_name}_{ts}.log")
    log_f = open(log_path, "w", encoding="utf-8")

    orig_out, orig_err = sys.stdout, sys.stderr

    if also_console:
        class Tee:
            def __init__(self, *files):
                self.files = files

            def write(self, data):
                for f in self.files:
                    f.write(data)
                    f.flush()

            def flush(self):
                for f in self.files:
                    f.flush()

        sys.stdout = Tee(orig_out, log_f)
        sys.stderr = Tee(orig_err, log_f)
    else:
        sys.stdout = log_f
        sys.stderr = log_f

    print(f"[LOG] {log_path}")
    return log_path, log_f, orig_out, orig_err


if __name__ == "__main__":
    # ---- knobs ----
    SOLVER = "picef"     # one of: "scipy", "gurobi", "picef", "pief"
    SEED = 0

    T = 40
    lam_p = 4.77
    lam_a = 0.15
    K = 3
    L = 2

    # Log for this run
    log_path, log_f, orig_out, orig_err = setup_logging(f"{SOLVER}_K{K}L{L}_seed{SEED}")

    rng = np.random.default_rng(SEED)
    g = Graph()

    # Load data once
    pair_bank, altruist_bank = DATA_BANK()
    f_p = EmpiricalSampler(pair_bank, rng)
    f_a = EmpiricalSampler(altruist_bank, rng)

    # init pool
    sample_arrivals(0, g, lam_p=3.0, lam_a=1.0, f_p=f_p, f_a=f_a, rng=rng)
    build_edges(g, rng)

    print("SOLVER =", SOLVER, "| seed =", SEED)
    print("T =", T, "| lam_p =", lam_p, "| lam_a =", lam_a, "| K =", K, "| L =", L)
    print("Initial: |V(0)| =", len(g.V), ", |E(0)| =", len(g.E))

    for t in range(T):
        g, D = step_pool(
            graph=g,
            t=t,
            lam_p=lam_p,
            lam_a=lam_a,
            f_p=f_p,
            f_a=f_a,
            rng=rng,
            max_cycle_len=K,
            max_chain_len=L,
            solveip_method=SOLVER,
            profile_solveip=True
        )
        print("t =", t, "|departures| =", len(D), ", |V(t)| =", len(g.V), ", |E(t)| =", len(g.E))

    print("DONE | final |V| =", len(g.V), "| final |E| =", len(g.E))

    # Restore + close log
    sys.stdout = orig_out
    sys.stderr = orig_err
    log_f.close()
    print(f"[LOG CLOSED] {log_path}")
