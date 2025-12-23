import collections
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

def SolveIP_structures(graph, max_cycle_len=3, max_chain_len=4):
    # Build outgoing adjacency
    out = collections.defaultdict(list)
    for (u, v), w in graph.E.items():
        if u in graph.V and v in graph.V:
            out[u].append(v)

    structures = []  # each: {"type","nodes","edges","w"}

    # Enumerate cycles (patients only)
    patients = [vid for vid, vv in graph.V.items() if not vv.is_altruist]
    for start in patients:
        stack = [(start, [start])]
        while stack:
            cur, path = stack.pop()
            if len(path) > max_cycle_len:
                continue
            for nxt in out.get(cur, []):
                if nxt == start:
                    if len(path) >= 2 and start == min(path):  # de-duplicate by smallest vid
                        cyc = path[:]
                        edges = [(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))]
                        if all(e in graph.E for e in edges):
                            w = float(sum(graph.E[e] for e in edges))
                            structures.append({"type": "cycle", "nodes": tuple(cyc), "edges": tuple(edges), "w": w})
                else:
                    if nxt in path:
                        continue
                    if nxt not in graph.V:
                        continue
                    if graph.V[nxt].is_altruist:
                        continue
                    stack.append((nxt, path + [nxt]))

    # Enumerate chains (altruist-start only); every prefix is valid
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
                    continue
                if nxt in nodes_path:
                    continue
                e = (cur, nxt)
                if e not in graph.E:
                    continue

                new_nodes = nodes_path + [nxt]
                new_edges = edges_path + [e]
                new_w = w + float(graph.E[e])

                structures.append({"type": "chain", "nodes": tuple(new_nodes), "edges": tuple(new_edges), "w": float(new_w)})
                stack.append((nxt, new_nodes, new_edges, new_w))

    if not structures:
        return []

    # Build A matrix for Ax <= 1 (vertex-disjoint)
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
    lc = LinearConstraint(A, -np.inf * np.ones(n_rows), np.ones(n_rows))  # Ax <= 1
    bounds = Bounds(np.zeros(n_vars), np.ones(n_vars))
    integrality = np.ones(n_vars, dtype=int)

    # maximize sum(w_j x_j) == minimize -w
    c = -np.array([s["w"] for s in structures], dtype=float)
    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=[lc])

    if res.x is None or res.status != 0:
        msg = getattr(res, "message", "MILP failed")
        raise RuntimeError(f"SolveIP MILP failed (status={res.status}). {msg}")

    chosen = [structures[j] for j in range(n_vars) if res.x[j] > 0.5]
    return chosen