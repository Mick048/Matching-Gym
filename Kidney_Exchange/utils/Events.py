def expire(vertex, rng, prob=0.01):
    """Paper: expire with (calibrated) constant probability. Here keep as parameter."""
    return rng.random() < prob


def negative_crossmatch(patient_vertex, rng):
    """
    Paper: failure probability depends on patient's CPRA.
    P(fail) = CPRA/100
    CPRA=100 -> fail prob = 1
    """
    cpra = int(patient_vertex.features.get("cpra", 0))
    cpra = max(0, min(100, cpra))
    return rng.random() < (cpra / 100.0)


def renege(pair_vertex, rng, default_prob=0.02):
    """
    Paper: only relevant for CHAINS (non-simultaneous) — the paired donor may renege
    on continuing the chain.
    We model it as a Bernoulli event whose probability can be a constant, or stored
    per-vertex in features['renege_prob'].
    """
    p = float(pair_vertex.features.get("renege_prob", default_prob))
    p = max(0.0, min(1.0, p))
    return rng.random() < p