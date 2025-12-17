import numpy as np
import pulp
from pulp import value
import os, sys, shutil


class StaticSolver:
    def __init__(self, matches: np.ndarray, arrival_rates: np.ndarray, rewards: np.ndarray):
        """Initialize static matching model.

        Args:
            matches: Match matrix (d x n) where d is num match types, n is num agent types
            arrival_rates: Arrival rates vector of length n
            rewards: Rewards vector of length d
        """
        self.matches = matches
        self.d = matches.shape[0]  # number of match types
        self.n = matches.shape[1]  # number of agent types

        # Create optimization model
        self.model = pulp.LpProblem("StaticMatching", pulp.LpMaximize)

        # Decision variables
        self.x = [pulp.LpVariable(f"x_{i}", lowBound=0) for i in range(self.d)]

        # Objective
        self.model += pulp.lpSum(rewards[i] * self.x[i] for i in range(self.d))

        # Constraints
        for j in range(self.n):
            self.model += (
                pulp.lpSum(matches[i, j] * self.x[i] for i in range(self.d)) == arrival_rates[j],
                f"flow_conservation_{j}",
            )

    def _cbc_path(self) -> str:
        """Return an absolute path to CBC binary."""
        return (
            os.environ.get("PULP_CBC_PATH")
            or shutil.which("cbc")
            or os.path.join(sys.prefix, "bin", "cbc")
        )

    def solve(self, use_gurobi: bool = False) -> None:
        """Solve the optimization problem.

        Default: use CBC via COIN_CMD (robust on conda).
        If use_gurobi=True, try Gurobi first (requires gurobi license + setup).
        """
        # Optional: try Gurobi first
        if use_gurobi:
            try:
                gurobi = pulp.GUROBI(msg=0)
                if gurobi.available():
                    self.model.solve(gurobi)
                    return
                else:
                    print("Warning: Gurobi not available, falling back to CBC solver")
            except Exception as e:
                print(f"Warning: Gurobi solver failed: {e}, falling back to CBC solver")

        # Use external CBC binary through COIN_CMD (NOT PULP_CBC_CMD)
        cbc_path = self._cbc_path()
        solver = pulp.COIN_CMD(path=cbc_path, msg=0)
        self.model.solve(solver)

    def get_primal_solution(self) -> np.ndarray:
        """Get primal solution (matching rates)."""
        if self.model.status != pulp.LpStatusOptimal:
            raise ValueError(f"Model not solved optimally. Status: {pulp.LpStatus[self.model.status]}")
        return np.array([v.varValue for v in self.x])

    def get_dual_solution(self) -> np.ndarray:
        """Get dual solution (shadow prices)."""
        if self.model.status != pulp.LpStatusOptimal:
            raise ValueError(f"Model not solved optimally. Status: {pulp.LpStatus[self.model.status]}")
        return np.array([self.model.constraints[f"flow_conservation_{j}"].pi for j in range(self.n)])

    def update_arrival_rates(self, new_rates: np.ndarray) -> None:
        """Update arrival rates in the model."""
        for j in range(self.n):
            self.model.constraints[f"flow_conservation_{j}"].changeRHS(new_rates[j])
        # Reset solution status
        self.model.status = pulp.LpStatusNotSolved

    def opt_obj(self):
        return value(self.model.objective)
