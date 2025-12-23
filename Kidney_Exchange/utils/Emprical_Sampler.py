import copy
class EmpiricalSampler:
    def __init__(self, bank, rng):
        if not bank:
            raise ValueError("bank is empty. Provide at least 1 record.")
        self.bank = bank
        self.rng = rng

    def __call__(self):
        idx = int(self.rng.integers(0, len(self.bank)))
        return copy.deepcopy(self.bank[idx])