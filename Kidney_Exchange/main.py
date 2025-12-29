import numpy as np
from utils.Data_Structure import Graph
from utils.Data_Bank import DATA_BANK
from utils.Emprical_Sampler import EmpiricalSampler
from utils.Sample_Arrivals import sample_arrivals
from utils.Build_Edges import build_edges
from utils.Step_Pool import step_pool

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    g = Graph()

    # Pair records must include: candidate_abo, paired_donor_abo, cpra, wait_days
    # We additionally include HLA antigens to enable true 0-ABDR mismatch computation in w_optn.
    pair_bank, altruist_bank = DATA_BANK() # Load Data from csv file (./data/)
    # pair_bank, altruist_bank = DATA_BANK(path= "") # Default Settings

    f_p = EmpiricalSampler(pair_bank, rng)
    f_a = EmpiricalSampler(altruist_bank, rng)

    sample_arrivals(0, g, lam_p=3.0, lam_a=1.0, f_p=f_p, f_a=f_a, rng=rng)
    build_edges(g, rng)

    print("Initial: |V(0)| =", len(g.V), ", |E(0)| =", len(g.E))

    for t in range(20):
        g, D = step_pool(
            graph=g,
            t=t,
            lam_p=4.77,
            lam_a=0.15,
            f_p=f_p,
            f_a=f_a,
            rng=rng,
        )
        print(
            "t =", t,
            "|departures| =", len(D),
            ", |V(t)| =", len(g.V),
            ", |E(t)| =", len(g.E),
        )
