import collections, time, math
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds



def _build_out_adjacency(graph):
    """Outgoing adjacency list from graph.E restricted to vertices in graph.V."""
    out = collections.defaultdict(list)
    for (u, v), w in graph.E.items():
        if u in graph.V and v in graph.V:
            out[u].append(v)
    return out


def enumerate_cycles(graph, max_cycle_len=3, *, out=None):
    """
    Enumerate directed cycles among patient/pair vertices only (no altruists).

    Returns:
      cycles: list of dict, each:
        {"type":"cycle","nodes": tuple(...), "edges": tuple((u,v),...), "w": float}
    """
    if out is None:
        out = _build_out_adjacency(graph)

    cycles = []
    patients = [vid for vid, vv in graph.V.items() if not vv.is_altruist]

    for start in patients:
        stack = [(start, [start])]
        while stack:
            cur, path = stack.pop()

            if len(path) > max_cycle_len:
                continue

            for nxt in out.get(cur, []):
                if nxt == start:
                    # found a directed cycle (path closes back to start)
                    if len(path) >= 2 and start == min(path):  # de-duplicate by smallest vid
                        cyc = path[:]
                        edges = [(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))]

                        # Guard (out is built from graph.E, but keep it safe)
                        if all(e in graph.E for e in edges):
                            w = float(sum(graph.E[e] for e in edges))
                            cycles.append(
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

    return cycles


def enumerate_chains(graph, max_chain_len=4, *, out=None):
    """
    Enumerate altruist-start chains; every prefix is a valid chain.

    Returns:
      chains: list of dict, each:
        {"type":"chain","nodes": tuple(...), "edges": tuple((u,v),...), "w": float}
    """
    if out is None:
        out = _build_out_adjacency(graph)

    chains = []
    altruists = [vid for vid, vv in graph.V.items() if vv.is_altruist]

    for a in altruists:
        stack = [(a, [a], [], 0.0)]  # (cur, nodes_path, edges_path, w_so_far)
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
                chains.append(
                    {"type": "chain", "nodes": tuple(new_nodes), "edges": tuple(new_edges), "w": float(new_w)}
                )

                # extend further
                stack.append((nxt, new_nodes, new_edges, new_w))

    return chains


def SolveIP_scipy(graph, max_cycle_len=3, max_chain_len=4, *, profile=False):
    # ---- enumerate (split timing) ----
    t_enum_all0 = time.perf_counter()

    out = _build_out_adjacency(graph)

    t0 = time.perf_counter()
    cycles = enumerate_cycles(graph, max_cycle_len=max_cycle_len, out=out)
    t_cycles = time.perf_counter() - t0

    t0 = time.perf_counter()
    chains = enumerate_chains(graph, max_chain_len=max_chain_len, out=out)
    t_chains = time.perf_counter() - t0

    structures = cycles + chains
    t_enum_total = time.perf_counter() - t_enum_all0

    if not structures:
        if profile:
            print(
                f"[SolveIP_scipy] enum_total={t_enum_total:.6f}s "
                f"(cycles={t_cycles:.6f}s, chains={t_chains:.6f}s) | "
                f"cycles={len(cycles)} chains={len(chains)} | "
                f"structures=0 | solve=0.000000s"
            )
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
            f"[SolveIP_scipy] enum_total={t_enum_total:.6f}s "
            f"(cycles={t_cycles:.6f}s, chains={t_chains:.6f}s) | "
            f"cycles={len(cycles)} chains={len(chains)} | "
            f"structures={n_vars} | solve={t_solve:.6f}s"
        )

    return chosen


def SolveIP_gurobi(
    graph, max_cycle_len=3, max_chain_len=4, *,
    verbose=False, time_limit=None, mip_gap=None,
    profile=False, return_timing=False
):
    import gurobipy as gp
    from gurobipy import GRB

    t_all = time.perf_counter()

    # ---- enumerate (split timing) ----
    t_enum_all0 = time.perf_counter()

    out = _build_out_adjacency(graph)

    t0 = time.perf_counter()
    cycles = enumerate_cycles(graph, max_cycle_len=max_cycle_len, out=out)
    t_cycles = time.perf_counter() - t0

    t0 = time.perf_counter()
    chains = enumerate_chains(graph, max_chain_len=max_chain_len, out=out)
    t_chains = time.perf_counter() - t0

    structures = cycles + chains
    t_enum_total = time.perf_counter() - t_enum_all0

    if not structures:
        timing = {
            "enumerate_seconds": t_enum_total,
            "enumerate_cycles_seconds": t_cycles,
            "enumerate_chains_seconds": t_chains,
            "solve_seconds": 0.0,
            "total_seconds": time.perf_counter() - t_all,
            "n_structures": 0,
            "n_vertices": len(graph.V),
            "n_cycles": len(cycles),
            "n_chains": len(chains),
        }
        if profile:
            print(
                f"[SolveIP_gurobi] enum_total={t_enum_total:.6f}s "
                f"(cycles={t_cycles:.6f}s, chains={t_chains:.6f}s) | "
                f"cycles=0 chains=0 | structures=0 | solve=0.000000s | "
                f"total={timing['total_seconds']:.6f}s"
            )
        return ([], timing) if return_timing else []

    # ---- build + solve ----
    t0 = time.perf_counter()

    m = gp.Model("kidney_exchange_ip")
    m.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

    n_vars = len(structures)
    x = m.addVars(n_vars, vtype=GRB.BINARY, name="x")

    m.setObjective(gp.quicksum(structures[j]["w"] * x[j] for j in range(n_vars)), GRB.MAXIMIZE)

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
        "enumerate_seconds": t_enum_total,
        "enumerate_cycles_seconds": t_cycles,
        "enumerate_chains_seconds": t_chains,
        "solve_seconds": t_solve,
        "total_seconds": t_total,
        "n_structures": n_vars,
        "n_vertices": len(graph.V),
        "n_cycles": len(cycles),
        "n_chains": len(chains),
    }

    if profile:
        print(
            f"[SolveIP_gurobi] enum_total={t_enum_total:.6f}s "
            f"(cycles={t_cycles:.6f}s, chains={t_chains:.6f}s) | "
            f"cycles={len(cycles)} chains={len(chains)} | "
            f"structures={n_vars} | solve={t_solve:.6f}s | total={t_total:.6f}s"
        )

    return (chosen, timing) if return_timing else chosen



def _compute_shortest_dist_from_ndds(graph):
    """
    For Reduced PICEF:
    d(i) = shortest number of arcs from any NDD to i in the directed graph.
    If unreachable, d(i)=+inf.
    """
    out = collections.defaultdict(list)
    for (u, v), w in graph.E.items():
        if u in graph.V and v in graph.V:
            out[u].append(v)

    INF = float("inf")
    d = {vid: INF for vid in graph.V.keys()}

    # multi-source BFS from all altruists (NDDs)
    q = collections.deque()
    for vid, vv in graph.V.items():
        if vv.is_altruist:
            d[vid] = 0
            q.append(vid)

    while q:
        u = q.popleft()
        for v in out.get(u, []):
            if v not in graph.V:
                continue
            if d[v] == INF:
                d[v] = d[u] + 1
                q.append(v)

    return d


def SolveIP_PICEF_gurobi(
    graph,
    max_cycle_len=3,
    max_chain_len=4,
    *,
    reduced=False,
    verbose=False,
    time_limit=None,
    mip_gap=None,
    profile=False,
    return_timing=False
):
    """
    PICEF: Position-Indexed Chain-Edge Formulation.
    - cycles: enumerate up to max_cycle_len (binary z_c)
    - chains: arc-position binaries y_{i,j,k} (k=1..L), no chain enumeration

    Output:
      chosen: list of structures dicts, in your old format:
        cycle: {"type":"cycle","nodes":..., "edges":..., "w":...}
        chain: {"type":"chain","nodes":..., "edges":..., "w":...}
      If return_timing=True: return (chosen, timing_dict)
    """
    import gurobipy as gp
    from gurobipy import GRB

    t_all = time.perf_counter()

    # -------------------------
    # 0) precompute distances for Reduced PICEF (optional)
    # -------------------------
    t0 = time.perf_counter()
    d = _compute_shortest_dist_from_ndds(graph) if reduced else None
    t_dist = time.perf_counter() - t0

    # -------------------------
    # 1) enumerate cycles (still needed in PICEF)
    # -------------------------
    t0 = time.perf_counter()
    # reuse your enumerate_cycles(...) which returns dicts with type="cycle"
    cycles = enumerate_cycles(graph, max_cycle_len=max_cycle_len)
    t_cycles = time.perf_counter() - t0

    # -------------------------
    # 2) build arc-position variable keys for chains
    #    y[(i,j,k)] = 1 if arc (i->j) used at position k
    # -------------------------
    t0 = time.perf_counter()

    L = int(max_chain_len)

    # Collect vertices
    altruists = [vid for vid, vv in graph.V.items() if vv.is_altruist]
    patients  = [vid for vid, vv in graph.V.items() if not vv.is_altruist]

    # Build edge lists restricted to current pool
    # For PICEF chains, recipient j must be a patient (altruists cannot receive).
    outgoing_pat_edges = collections.defaultdict(list)  # tail -> list(head)
    incoming_pat_edges = collections.defaultdict(list)  # head -> list(tail)
    for (u, v), w in graph.E.items():
        if u not in graph.V or v not in graph.V:
            continue
        if graph.V[v].is_altruist:
            continue  # recipients in chains must be patients
        outgoing_pat_edges[u].append(v)
        incoming_pat_edges[v].append(u)

    # Allowed positions K(i,j):
    # - If tail i is altruist: k = 1 only
    # - If tail i is patient: k = 2..L (or reduced: max(2, d(i)+1)..L)
    y_keys = []
    for i, heads in outgoing_pat_edges.items():
        is_ndd = graph.V[i].is_altruist
        if is_ndd:
            k_min, k_max = 1, 1
        else:
            # patient tail
            k_min = 2
            if reduced:
                di = d.get(i, float("inf"))
                if math.isinf(di):
                    continue  # unreachable from any NDD => cannot appear in a chain
                k_min = max(k_min, int(di) + 1)
            k_max = L
            if k_min > k_max:
                continue

        for j in heads:
            for k in range(k_min, k_max + 1):
                y_keys.append((i, j, k))

    t_ykeys = time.perf_counter() - t0

    # -------------------------
    # 3) build & solve MILP
    # -------------------------
    t0 = time.perf_counter()

    m = gp.Model("kidney_exchange_picef")
    m.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

    # Variables
    y = m.addVars(y_keys, vtype=GRB.BINARY, name="y")  # y[i,j,k]
    z = m.addVars(len(cycles), vtype=GRB.BINARY, name="z")  # z[c]

    # Objective: sum w_ij y_ijk + sum w_c z_c
    obj_chain = gp.quicksum(graph.E[(i, j)] * y[i, j, k] for (i, j, k) in y_keys)
    obj_cycle = gp.quicksum(cycles[c]["w"] * z[c] for c in range(len(cycles)))
    m.setObjective(obj_chain + obj_cycle, GRB.MAXIMIZE)

    # ---- Constraints ----

    # (A) Patient capacity: each patient i used at most once
    # incoming chain arcs to i (any allowed k) + cycles containing i <= 1
    # Build quick index: incoming terms for each patient
    incoming_terms = {i: [] for i in patients}
    for (tail, head, k) in y_keys:
        if head in incoming_terms:
            incoming_terms[head].append((tail, head, k))

    # Build cycle membership index
    patient_to_cycles = collections.defaultdict(list)
    for c, cyc in enumerate(cycles):
        for vid in set(cyc["nodes"]):
            if vid in patient_to_cycles:
                patient_to_cycles[vid].append(c)
            else:
                # only patients should appear in cycles, but keep safe
                patient_to_cycles[vid].append(c)

    for i in patients:
        lhs_chain = gp.quicksum(y[tail, i, k] for (tail, _, k) in incoming_terms.get(i, []))
        lhs_cycle = gp.quicksum(z[c] for c in patient_to_cycles.get(i, []))
        m.addConstr(lhs_chain + lhs_cycle <= 1, name=f"cap_{i}")

    # (B) NDD capacity: each altruist can start at most one chain
    for a in altruists:
        # only k=1 variables exist for NDD tails
        lhs = gp.quicksum(y[a, j, 1] for j in outgoing_pat_edges.get(a, []) if (a, j, 1) in y)
        m.addConstr(lhs <= 1, name=f"ndd_{a}")

    # (C) Chain flow (position-indexed):
    # For each patient i and k=1..L-1:
    #   sum_{j} y[j,i,k] >= sum_{j} y[i,j,k+1]
    #
    # Build outgoing-at-position lists for efficiency
    outgoing_terms_by_pos = {i: {k: [] for k in range(2, L + 1)} for i in patients}
    incoming_terms_by_pos = {i: {k: [] for k in range(1, L + 1)} for i in patients}

    for (tail, head, k) in y_keys:
        if head in incoming_terms_by_pos:
            incoming_terms_by_pos[head][k].append((tail, head, k))
        if tail in outgoing_terms_by_pos and k >= 2:
            outgoing_terms_by_pos[tail][k].append((tail, head, k))

    for i in patients:
        for k in range(1, L):
            lhs = gp.quicksum(
                y[tail, i, k] for (tail, _, _) in incoming_terms_by_pos[i].get(k, [])
            )
            rhs = gp.quicksum(
                y[i, head, k + 1] for (_, head, _) in outgoing_terms_by_pos[i].get(k + 1, [])
            )
            m.addConstr(lhs >= rhs, name=f"flow_{i}_{k}")

    m.optimize()

    t_solve = time.perf_counter() - t0
    t_total = time.perf_counter() - t_all

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"SolveIP_PICEF_gurobi failed: status={m.Status}")

    # -------------------------
    # 4) Extract solution: cycles + reconstruct chains from y
    # -------------------------
    chosen = []

    # chosen cycles
    for c in range(len(cycles)):
        if z[c].X > 0.5:
            chosen.append(cycles[c])

    # reconstruct one chain per NDD from selected y variables
    # We assume integrality => for each (tail,k) at most one outgoing selected.
    # Start from k=1 at each altruist, then follow k=2..L by tail continuity.
    # Build lookup: (tail, k) -> head if selected
    succ = {}  # key=(tail,k) -> head
    for (i, j, k) in y_keys:
        if y[i, j, k].X > 0.5:
            succ[(i, k)] = j

    for a in altruists:
        if (a, 1) not in succ:
            continue
        nodes = [a]
        edges = []
        wsum = 0.0

        # position 1
        cur = a
        head = succ[(cur, 1)]
        edges.append((cur, head))
        wsum += float(graph.E[(cur, head)])
        nodes.append(head)

        cur = head  # now at first patient
        # positions 2..L
        for k in range(2, L + 1):
            nxt = succ.get((cur, k), None)
            if nxt is None:
                break
            edges.append((cur, nxt))
            wsum += float(graph.E[(cur, nxt)])
            nodes.append(nxt)
            cur = nxt

        chosen.append({"type": "chain", "nodes": tuple(nodes), "edges": tuple(edges), "w": float(wsum)})

    timing = {
        "dist_seconds": t_dist,
        "enumerate_cycles_seconds": t_cycles,
        "build_ykeys_seconds": t_ykeys,
        "solve_seconds": t_solve,
        "total_seconds": t_total,
        "n_vertices": len(graph.V),
        "n_cycles": len(cycles),
        "n_y_vars": len(y_keys),
        "reduced": bool(reduced),
    }

    if profile:
        print(
            f"[SolveIP_PICEF_gurobi] reduced={bool(reduced)} | "
            f"dist={t_dist:.6f}s | cycles_enum={t_cycles:.6f}s | "
            f"ykeys={t_ykeys:.6f}s (n_y={len(y_keys)}) | "
            f"solve={t_solve:.6f}s | total={t_total:.6f}s | "
            f"chosen={len(chosen)}"
        )

    return (chosen, timing) if return_timing else chosen

def SolveIP_PIEF_gurobi(
    graph,
    max_cycle_len=3,
    *,
    verbose=False,
    time_limit=None,
    mip_gap=None,
    profile=False,
    return_timing=False
):
    """
    PIEF (cycles only): Position-Indexed Edge Formulation for cycles up to length K.

    Variables:
      y[i,j,k] in {0,1} indicates edge (i->j) is chosen at position k in some cycle.

    Key constraints (cycles only):
      - Vertex-disjoint: each patient has at most one incoming and at most one outgoing across all positions.
      - Flow across positions: for each patient v and k=1..K-1:
            sum_u y[u,v,k] == sum_w y[v,w,k+1]
        This enforces that if v is entered at position k, it must be left at position k+1 (and vice versa),
        thus chaining edges into cycles/paths of length up to K.
      - Closure: connect position K back to position 1:
            sum_u y[u,v,K] == sum_w y[v,w,1]
        This enforces wrap-around so chains close into cycles (no open paths).

    Output:
      chosen: list of cycle structures in your old format:
        {"type":"cycle","nodes": tuple(...), "edges": tuple((u,v),...), "w": float}
      If return_timing=True: return (chosen, timing_dict)
    """
    import gurobipy as gp
    from gurobipy import GRB

    t_all = time.perf_counter()

    # -------------------------
    # 1) Build patient-only edge set (cycles cannot include altruists)
    # -------------------------
    t0 = time.perf_counter()

    K = int(max_cycle_len)
    patients = [vid for vid, vv in graph.V.items() if not vv.is_altruist]
    patient_set = set(patients)

    # directed edges among patients only
    outP = collections.defaultdict(list)
    inP  = collections.defaultdict(list)
    edgesP = []  # list of (i,j)
    for (i, j), w in graph.E.items():
        if i in patient_set and j in patient_set:
            edgesP.append((i, j))
            outP[i].append(j)
            inP[j].append(i)

    t_prep = time.perf_counter() - t0

    if not edgesP or K < 2:
        timing = {
            "prep_seconds": t_prep,
            "solve_seconds": 0.0,
            "total_seconds": time.perf_counter() - t_all,
            "n_vertices": len(graph.V),
            "n_patients": len(patients),
            "n_edges_patients": len(edgesP),
            "K": K,
        }
        if profile:
            print(
                f"[SolveIP_PIEF_gurobi] K={K} | prep={t_prep:.6f}s | "
                f"edgesP={len(edgesP)} | solve=0.000000s | total={timing['total_seconds']:.6f}s | chosen=0"
            )
        return ([], timing) if return_timing else []

    # -------------------------
    # 2) Build & solve MILP
    # -------------------------
    t0 = time.perf_counter()

    m = gp.Model("kidney_exchange_pief_cycles")
    m.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

    # variables y[i,j,k] for k=1..K
    y_keys = [(i, j, k) for (i, j) in edgesP for k in range(1, K + 1)]
    y = m.addVars(y_keys, vtype=GRB.BINARY, name="y")

    # objective: sum_{i,j,k} w_ij y_ijk
    m.setObjective(gp.quicksum(graph.E[(i, j)] * y[i, j, k] for (i, j, k) in y_keys), GRB.MAXIMIZE)

    # -------------------------
    # Constraints
    # -------------------------

    # (A) Vertex-disjoint in/out degree <= 1 (across all positions)
    # outgoing <=1 for each v
    for v in patients:
        lhs_out = gp.quicksum(
            y[v, j, k]
            for k in range(1, K + 1)
            for j in outP.get(v, [])
            if (v, j, k) in y
        )
        m.addConstr(lhs_out <= 1, name=f"out_{v}")

    # incoming <=1 for each v
    for v in patients:
        lhs_in = gp.quicksum(
            y[i, v, k]
            for k in range(1, K + 1)
            for i in inP.get(v, [])
            if (i, v, k) in y
        )
        m.addConstr(lhs_in <= 1, name=f"in_{v}")

    # (B) Position-flow equalities:
    # for each v and k=1..K-1:
    #   incoming at k == outgoing at k+1
    for v in patients:
        for k in range(1, K):
            inc_k = gp.quicksum(
                y[i, v, k] for i in inP.get(v, []) if (i, v, k) in y
            )
            out_k1 = gp.quicksum(
                y[v, j, k + 1] for j in outP.get(v, []) if (v, j, k + 1) in y
            )
            m.addConstr(inc_k == out_k1, name=f"flow_{v}_{k}")

    # (C) Closure wrap-around:
    # incoming at K == outgoing at 1
    for v in patients:
        inc_K = gp.quicksum(
            y[i, v, K] for i in inP.get(v, []) if (i, v, K) in y
        )
        out_1 = gp.quicksum(
            y[v, j, 1] for j in outP.get(v, []) if (v, j, 1) in y
        )
        m.addConstr(inc_K == out_1, name=f"wrap_{v}")

    m.optimize()

    t_solve = time.perf_counter() - t0
    t_total = time.perf_counter() - t_all

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"SolveIP_PIEF_gurobi failed: status={m.Status}")

    # -------------------------
    # 3) Extract selected edges and reconstruct cycles
    # -------------------------
    # collect chosen directed edges (ignore positions; each (i,j) should appear at most once overall due to constraints)
    chosen_edges = set()
    for (i, j, k) in y_keys:
        if y[i, j, k].X > 0.5:
            chosen_edges.add((i, j))

    # reconstruct cycles from chosen_edges
    # because in/out <= 1, this is a disjoint union of directed cycles (or empty)
    succ = {i: j for (i, j) in chosen_edges}  # outgoing map
    visited = set()
    chosen = []

    for start in list(succ.keys()):
        if start in visited:
            continue
        # walk until repeat or dead-end (shouldn't dead-end if constraints correct)
        path = []
        cur = start
        while cur not in visited and cur in succ:
            visited.add(cur)
            path.append(cur)
            cur = succ[cur]

        # if we returned to a node in this path, we found a cycle
        if cur in path:
            idx = path.index(cur)
            cyc_nodes = path[idx:]  # nodes in the cycle
            # ensure length within K (should be by construction)
            if 2 <= len(cyc_nodes) <= K:
                cyc_edges = [(cyc_nodes[t], cyc_nodes[(t + 1) % len(cyc_nodes)]) for t in range(len(cyc_nodes))]
                w = float(sum(graph.E[e] for e in cyc_edges))
                chosen.append({"type": "cycle", "nodes": tuple(cyc_nodes), "edges": tuple(cyc_edges), "w": w})

    timing = {
        "prep_seconds": t_prep,
        "solve_seconds": t_solve,
        "total_seconds": t_total,
        "n_vertices": len(graph.V),
        "n_patients": len(patients),
        "n_edges_patients": len(edgesP),
        "n_y_vars": len(y_keys),
        "K": K,
    }

    if profile:
        print(
            f"[SolveIP_PIEF_gurobi] K={K} | prep={t_prep:.6f}s | "
            f"edgesP={len(edgesP)} | y_vars={len(y_keys)} | "
            f"solve={t_solve:.6f}s | total={t_total:.6f}s | chosen_cycles={len(chosen)}"
        )

    return (chosen, timing) if return_timing else chosen
