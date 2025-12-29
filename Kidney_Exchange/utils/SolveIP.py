import collections
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
import time


def enumerate_structures(graph, max_cycle_len=3, max_chain_len=4):
    """
    Enumerate feasible cycles (pairs only) and chains (altruist-start; every prefix allowed).

    Returns:
      structures: list of dict, each:
        {
          "type": "cycle" or "chain",
          "nodes": tuple(...),
          "edges": tuple((u,v), ...),
          "w": float
        }
    """
    # Build outgoing adjacency from current edges
    out = collections.defaultdict(list)
    for (u, v), w in graph.E.items():
        if u in graph.V and v in graph.V:
            out[u].append(v)

    structures = []  # each: {"type","nodes","edges","w"}

    # ---- Enumerate cycles (patients/pairs only) ----
    patients = [vid for vid, vv in graph.V.items() if not vv.is_altruist]
    for start in patients:
        stack = [(start, [start])]
        while stack:
            cur, path = stack.pop()

            if len(path) > max_cycle_len:
                continue

            for nxt in out.get(cur, []):
                if nxt == start:
                    # found a directed cycle
                    if len(path) >= 2 and start == min(path):  # de-duplicate by smallest vid
                        cyc = path[:]
                        edges = [(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))]
                        if all(e in graph.E for e in edges):
                            w = float(sum(graph.E[e] for e in edges))
                            structures.append(
                                {"type": "cycle", "nodes": tuple(cyc), "edges": tuple(edges), "w": w}
                            )
                else:
                    # extend DFS path
                    if nxt in path:
                        continue
                    if nxt not in graph.V:
                        continue
                    if graph.V[nxt].is_altruist:
                        continue  # cycles cannot include altruists
                    stack.append((nxt, path + [nxt]))

    # ---- Enumerate chains (altruist-start only); every prefix is valid ----
    altruists = [vid for vid, vv in graph.V.items() if vv.is_altruist]
    for a in altruists:
        stack = [(a, [a], [], 0.0)]  # (cur, nodes_path, edges_path, w)
        while stack:
            cur, nodes_path, edges_path, w = stack.pop()

            if len(edges_path) >= max_chain_len:
                continue

            for nxt in out.get(cur, []):
                if nxt not in graph.V:
                    continue
                if graph.V[nxt].is_altruist:
                    continue  # altruists cannot receive
                if nxt in nodes_path:
                    continue  # simple path only

                e = (cur, nxt)
                if e not in graph.E:
                    continue

                new_nodes = nodes_path + [nxt]
                new_edges = edges_path + [e]
                new_w = w + float(graph.E[e])

                # every prefix is a valid chain
                structures.append(
                    {"type": "chain", "nodes": tuple(new_nodes), "edges": tuple(new_edges), "w": float(new_w)}
                )

                # extend further
                stack.append((nxt, new_nodes, new_edges, new_w))

    return structures

def SolveIP_scipy(graph, max_cycle_len=3, max_chain_len=4, *, profile=False):
    t0 = time.perf_counter()
    structures = enumerate_structures(graph, max_cycle_len=max_cycle_len, max_chain_len=max_chain_len)
    t_enum = time.perf_counter() - t0

    if not structures:
        if profile:
            print(f"[SolveIP_scipy] enumerate_structures={t_enum:.6f}s | structures=0 | solve=0.000000s")
        return []

    # ---- MILP build + solve ----
    t0 = time.perf_counter()

    vids = list(graph.V.keys())
    vid_to_row = {vid: i for i, vid in enumerate(vids)}
    n_rows = len(vids)
    n_vars = len(structures)

    rows, cols, data = [], [], []
    for j, s in enumerate(structures):
        for vid in set(s["nodes"]):
            rows.append(vid_to_row[vid])
            cols.append(j)
            data.append(1.0)

    A = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_vars))
    lc = LinearConstraint(A, -np.inf * np.ones(n_rows), np.ones(n_rows))
    bounds = Bounds(np.zeros(n_vars), np.ones(n_vars))
    integrality = np.ones(n_vars, dtype=int)
    c = -np.array([s["w"] for s in structures], dtype=float)

    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=[lc])

    t_solve = time.perf_counter() - t0

    if res.x is None or res.status != 0:
        msg = getattr(res, "message", "MILP failed")
        raise RuntimeError(f"SolveIP_scipy MILP failed (status={res.status}). {msg}")

    chosen = [structures[j] for j in range(n_vars) if res.x[j] > 0.5]

    if profile:
        print(
            f"[SolveIP_scipy] enumerate_structures={t_enum:.6f}s | "
            f"structures={n_vars} | solve={t_solve:.6f}s"
        )

    return chosen

def SolveIP_gurobi(graph, max_cycle_len=3, max_chain_len=4, *,
                   verbose=False, time_limit=None, mip_gap=None,
                   profile=False, return_timing=False):
    """
    Solve the standard kidney exchange IP via Gurobi:

        max   sum_s w_s x_s
        s.t.  for each vertex v: sum_{s: v in nodes(s)} x_s <= 1
              x_s in {0,1}

    If profile=True: print timing.
    If return_timing=True: return (chosen, timing_dict); else return chosen.

    timing_dict:
      {
        "enumerate_seconds": float,
        "solve_seconds": float,
        "total_seconds": float,
        "n_structures": int,
        "n_vertices": int
      }
    """
    import gurobipy as gp
    from gurobipy import GRB

    t_all = time.perf_counter()

    # 1) enumerate structures timing
    t0 = time.perf_counter()
    structures = enumerate_structures(graph, max_cycle_len=max_cycle_len, max_chain_len=max_chain_len)
    t_enum = time.perf_counter() - t0

    if not structures:
        timing = {
            "enumerate_seconds": t_enum,
            "solve_seconds": 0.0,
            "total_seconds": time.perf_counter() - t_all,
            "n_structures": 0,
            "n_vertices": len(graph.V),
        }
        if profile:
            print(f"[SolveIP_gurobi] enum={t_enum:.6f}s | structures=0 | solve=0.000000s | total={timing['total_seconds']:.6f}s")
        return ([], timing) if return_timing else []

    # 2) build+solve timing (we measure from model construction through optimize)
    t0 = time.perf_counter()

    m = gp.Model("kidney_exchange_ip")
    m.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

    n_vars = len(structures)

    # Binary variable per structure
    x = m.addVars(n_vars, vtype=GRB.BINARY, name="x")

    # Objective
    m.setObjective(gp.quicksum(structures[j]["w"] * x[j] for j in range(n_vars)), GRB.MAXIMIZE)

    # Vertex-disjoint constraints: each vertex used at most once
    v_to_structs = collections.defaultdict(list)
    for j, s in enumerate(structures):
        for vid in set(s["nodes"]):
            v_to_structs[vid].append(j)

    for vid, idxs in v_to_structs.items():
        m.addConstr(gp.quicksum(x[j] for j in idxs) <= 1, name=f"v_{vid}_used")

    m.optimize()

    t_solve = time.perf_counter() - t0
    t_total = time.perf_counter() - t_all

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"SolveIP_gurobi failed: status={m.Status}")

    chosen = [structures[j] for j in range(n_vars) if x[j].X > 0.5]

    timing = {
        "enumerate_seconds": t_enum,
        "solve_seconds": t_solve,
        "total_seconds": t_total,
        "n_structures": n_vars,
        "n_vertices": len(graph.V),
    }

    if profile:
        print(
            f"[SolveIP_gurobi] enum={t_enum:.6f}s | structures={n_vars} | "
            f"solve={t_solve:.6f}s | total={t_total:.6f}s"
        )

    return chosen
