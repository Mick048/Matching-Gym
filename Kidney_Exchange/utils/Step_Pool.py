from utils.SolveIP import SolveIP_structures
from utils.Events import expire, renege, negative_crossmatch
from utils.Sample_Arrivals import sample_arrivals
from utils.Build_Edges import build_edges

def step_pool(graph, t, lam_p, lam_a, f_p, f_a, rng,
              expire_prob=0.01, renege_prob=0.02,
              max_cycle_len=3, max_chain_len=4):
    """
    Returns:
      graph (updated in-place),
      departures: set[int]
    """

    # 1) SolveIP -> return chosen STRUCTURES (cycle/chain grouped with order)
    chosen = SolveIP_structures(graph, max_cycle_len=max_cycle_len, max_chain_len=max_chain_len)

    # 2) Expire on V(t)
    expired = set()
    for vid, v in list(graph.V.items()):
        if expire(v, rng, prob=expire_prob):
            expired.add(vid)

    departures = set(expired)

    # Cut a chain before an expired recipient
    def _cut_chain_on_expire(nodes, edges, expired_set):
        # nodes: (a0, v1, v2, ...), edges: ((a0,v1),(v1,v2),...)
        if nodes[0] in expired_set:
            return None  # altruist expired => drop whole chain

        for i in range(1, len(nodes)):
            if nodes[i] in expired_set:
                new_nodes = nodes[:i]
                new_edges = edges[:i-1]
                if len(new_edges) >= 1:
                    return (new_nodes, new_edges)
                return None

        return (nodes, edges)

    # Remove/trim structures that touch expired vertices
    filtered = []
    for s in chosen:
        if s["type"] == "cycle":
            if any(x in expired for x in s["nodes"]):
                continue
            filtered.append(s)
        else:  # chain
            cut = _cut_chain_on_expire(s["nodes"], s["edges"], expired)
            if cut is None:
                continue
            new_nodes, new_edges = cut
            filtered.append({"type": "chain", "nodes": new_nodes, "edges": new_edges})

    chosen = filtered

    # 3) Cycles: all-or-nothing crossmatch
    # MIN CHANGE: evaluate crossmatch per executed edge (u->v) and record graph.crossmatch_hist[(u,v)]
    kept = []
    for s in chosen:
        if s["type"] != "cycle":
            kept.append(s)
            continue

        ok = True
        for (u, v) in s["edges"]:
            fail = negative_crossmatch(graph.V[v], rng)   # True means fail
            graph.crossmatch_hist[(u, v)] = (not fail)    # True means OK (negative crossmatch)
            if fail:
                ok = False
                break

        if ok:
            kept.append(s)

    chosen = kept

    # 4) Chains: sequential with tail cut (crossmatch/renege)
    # MIN CHANGE: record crossmatch outcome for each attempted edge (u->v)
    kept = []
    for s in chosen:
        if s["type"] != "chain":
            kept.append(s)
            continue

        nodes = s["nodes"]  # (a0, v1, v2, ...)
        edges = s["edges"]  # ((a0,v1),(v1,v2),...)

        if len(edges) == 0:
            continue

        cut_nodes = nodes
        cut_edges = edges

        for i, (u, v) in enumerate(edges):
            fail = negative_crossmatch(graph.V[v], rng)
            graph.crossmatch_hist[(u, v)] = (not fail)

            if fail:
                # stop BEFORE this edge executes => keep prefix
                cut_edges = edges[:i]
                cut_nodes = nodes[:i+1]
                break

            if renege(graph.V[v], rng, default_prob=renege_prob):
                # edge executes into v; stop AFTER v
                cut_edges = edges[:i+1]
                cut_nodes = nodes[:i+2]
                break

        if len(cut_edges) >= 1:
            kept.append({"type": "chain", "nodes": cut_nodes, "edges": cut_edges})

    chosen = kept
    print(len(chosen))

    # 5) Collect executed edges, then departures
    executed_edges = []
    for s in chosen:
        executed_edges.extend(list(s["edges"]))

    # recipients who actually receive depart; altruists depart iff they donated
    for (u, v) in executed_edges:
        departures.add(v)
        if u in graph.V and graph.V[u].is_altruist:
            departures.add(u)

    # 6) Remove departures
    for vid in departures:
        graph.V.pop(vid, None)

    # 7) New arrivals + rebuild edges/weights
    sample_arrivals(t, graph, lam_p, lam_a, f_p, f_a, rng)
    build_edges(graph, rng)

    return graph, departures
