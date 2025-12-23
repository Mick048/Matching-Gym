class Vertex:
    def __init__(self, vid, is_altruist, features, arrival_time=0):
        self.id = vid
        self.is_altruist = is_altruist
        self.features = features
        self.arrival_time = arrival_time


class Graph:
    def __init__(self):
        self.V = {}  # {int: Vertex}
        self.E = {}  # {(int, int): float}
        self.crossmatch_hist = {}  # {(u,v): True/False}