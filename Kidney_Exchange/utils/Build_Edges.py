from collections import Counter
from utils.Data_Structure import Vertex, Graph
import time

def abo_compatible(donor_abo: str, cand_abo: str) -> bool:
    '''
    Blood Type Compatiable Funciton:
        O compatible to A, B, AB, O
        A compatible to A, AB
        B compatible to B, AB
        AB compatible to AB
    '''
    d = donor_abo.upper()
    c = cand_abo.upper()
    if d == "O":
        return True
    if d == "A":
        return c in ("A", "AB")
    if d == "B":
        return c in ("B", "AB")
    if d == "AB":
        return c == "AB"
    raise ValueError(f"Unknown ABO: donor={donor_abo}, candidate={cand_abo}")


CAND_ABO_POINTS = {"O": 100, "B": 50, "A": 25, "AB": 0} # O only can recieve O type kidney -> higher priority
PAIRED_DONOR_ABO_POINTS = {"O": 0, "B": 100, "A": 250, "AB": 500} # AB only can donor their kidney to AB -> higher priority


def cpra_points(cpra: int) -> int:
    cpra = int(cpra)
    if not (0 <= cpra <= 100):
        raise ValueError("CPRA must be in [0, 100].")

    if 0 <= cpra <= 19:  return 0
    if 20 <= cpra <= 29: return 5
    if 30 <= cpra <= 39: return 10
    if 40 <= cpra <= 49: return 15
    if 50 <= cpra <= 59: return 20
    if 60 <= cpra <= 69: return 25
    if 70 <= cpra <= 74: return 50
    if 75 <= cpra <= 79: return 75
    if 80 <= cpra <= 84: return 125
    if 85 <= cpra <= 89: return 200
    if 90 <= cpra <= 94: return 300
    if cpra == 95: return 500
    if cpra == 96: return 700
    if cpra == 97: return 900
    if cpra == 98: return 1250
    if cpra == 99: return 1500
    if cpra == 100: return 2000
    raise RuntimeError("unreachable")


def _paired_donor_abo_points(paired_donor_abo):
    """If multiple ABO candidates exist, take the fewest points (conservative)."""
    if paired_donor_abo is None:
        return 0
    if isinstance(paired_donor_abo, str):
        return PAIRED_DONOR_ABO_POINTS[paired_donor_abo.upper()]
    pts = [PAIRED_DONOR_ABO_POINTS[a.upper()] for a in paired_donor_abo]
    return min(pts)


def donor_abo_of(vertex: Vertex) -> str:
    f = vertex.features
    if vertex.is_altruist:
        return f["donor_abo"]
    return f["paired_donor_abo"]


def candidate_abo_of(vertex: Vertex) -> str:
    return vertex.features["candidate_abo"]


def _extract_donor_hla(donor_vertex: Vertex) -> dict:
    """
    Return donor HLA antigens as dict: {"A":[..,..], "B":[..,..], "DR":[..,..]}
    - altruist: donor_hla_A/B/DR
    - pair: paired_donor_hla_A/B/DR
    """
    f = donor_vertex.features or {}
    if donor_vertex.is_altruist:
        A = f.get("donor_hla_A", None)
        B = f.get("donor_hla_B", None)
        DR = f.get("donor_hla_DR", None)
    else:
        A = f.get("paired_donor_hla_A", None)
        B = f.get("paired_donor_hla_B", None)
        DR = f.get("paired_donor_hla_DR", None)

    if A is None or B is None or DR is None:
        raise KeyError(f"Missing donor HLA features for vertex {donor_vertex.id}")

    return {"A": list(A), "B": list(B), "DR": list(DR)}


def _extract_candidate_hla(cand_vertex: Vertex) -> dict:
    """
    Return candidate (recipient) HLA antigens as dict: {"A":[..,..], "B":[..,..], "DR":[..,..]}
    """
    f = cand_vertex.features or {}
    A = f.get("candidate_hla_A", None)
    B = f.get("candidate_hla_B", None)
    DR = f.get("candidate_hla_DR", None)

    if A is None or B is None or DR is None:
        raise KeyError(f"Missing candidate HLA features for vertex {cand_vertex.id}")

    return {"A": list(A), "B": list(B), "DR": list(DR)}


def _zero_abdr_mismatch(donor_hla: dict, cand_hla: dict) -> bool:
    """
    True iff donor and candidate match exactly at A, B, DR (order-insensitive, count-sensitive).
    """
    for locus in ("A", "B", "DR"):
        if Counter(donor_hla[locus]) != Counter(cand_hla[locus]):
            return False
    return True


def w_optn(donor_vertex: Vertex, cand_vertex: Vertex, graph: Graph) -> float:
    """
    Updated w_optn based on your new databank features:
      - 0-ABDR mismatch computed from donor/candidate HLA antigens (edge-level)
      - Previous crossmatch from graph.crossmatch_hist[(u,v)] (edge-level)
    """
    f = cand_vertex.features or {}
    df = donor_vertex.features or {}

    cand_abo = str(f["candidate_abo"]).upper()
    cpra = int(f.get("cpra", 0))
    wait_days = int(f.get("wait_days", 0) or 0)

    w = 100.0 + 0.07 * max(0, wait_days)

    # ---- 0-ABDR mismatch (computed from HLA antigens) ----
    donor_hla = _extract_donor_hla(donor_vertex)
    cand_hla = _extract_candidate_hla(cand_vertex)
    if _zero_abdr_mismatch(donor_hla, cand_hla):
        w += 10.0

    # ---- Same hospital/center ----
    c1 = df.get("center", None)
    c2 = f.get("center", None)
    if c1 is not None and c2 is not None and c1 == c2:
        w += 75.0

    # ---- Previous crossmatch record (edge-level) ----
    prev_ok = graph.crossmatch_hist.get((donor_vertex.id, cand_vertex.id), None)
    if prev_ok is True:
        w += 75.0

    # ---- Candidate age ----
    age = f.get("candidate_age", f.get("age", None))
    if age is not None and int(age) < 18:
        w += 100.0

    # ---- Prior living donor ----
    if bool(f.get("prior_living_donor", False)):
        w += 150.0

    # ---- Your original points ----
    w += float(CAND_ABO_POINTS[cand_abo])
    w += float(_paired_donor_abo_points(f.get("paired_donor_abo", None)))
    w += float(cpra_points(cpra))

    if bool(f.get("orphan", False)):
        w += 1_000_000.0
    return float(w)


def build_edges(graph: Graph,
               profile: bool = False, return_timing: bool = False):
    """
    Build feasible directed edges (u -> v) and set weights graph.E[(u,v)].

    If profile=True: print timing.
    If return_timing=True: return (timing_dict), else return None.

    timing_dict:
      {
        "seconds": float,
        "n_vertices": int,
        "n_edges": int,
        "n_pairs_checked": int
      }
    """
    t0 = time.perf_counter()

    graph.E.clear()
    vids = list(graph.V.keys())

    n_pairs_checked = 0  # number of (i,j) considered with i!=j

    for i in vids:
        u = graph.V[i]
        d_abo = donor_abo_of(u)

        for j in vids:
            if i == j:
                continue
            n_pairs_checked += 1

            v = graph.V[j]
            if v.is_altruist:
                continue

            c_abo = candidate_abo_of(v)
            if not abo_compatible(d_abo, c_abo):
                continue

            graph.E[(i, j)] = float(w_optn(u, v, graph))

    dt = time.perf_counter() - t0

    timing = {
        "seconds": dt,
        "n_vertices": len(vids),
        "n_edges": len(graph.E),
        "n_pairs_checked": n_pairs_checked,
    }

    if profile:
        print(
            f"[build_edges] time={dt:.6f}s | "
            f"|V|={timing['n_vertices']} | |E|={timing['n_edges']} | checked={timing['n_pairs_checked']}"
        )

    if return_timing:
        return timing
    return None

            
