from utils.Data_Structure import Vertex

def sample_arrivals(t, graph, lam_p, lam_a, f_p, f_a, rng):
    next_id = max(graph.V.keys()) + 1 if graph.V else 0
    num_pairs = rng.poisson(lam_p)
    num_altruists = rng.poisson(lam_a)

    for _ in range(num_pairs):
        vid = next_id
        next_id += 1
        graph.V[vid] = Vertex(vid, False, f_p(), arrival_time=t + 1)

    for _ in range(num_altruists):
        vid = next_id
        next_id += 1
        graph.V[vid] = Vertex(vid, True, f_a(), arrival_time=t + 1)