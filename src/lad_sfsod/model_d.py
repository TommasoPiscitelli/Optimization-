from dataclasses import dataclass
import numpy as np

import gurobipy as gp
from gurobipy import GRB


@dataclass
class LADSFSSODSolution:
    status_code: int
    status_name: str
    objective_value: float | None
    best_bound: float | None
    runtime: float | None
    mip_gap: float | None
    node_count: float | None

    w: np.ndarray | None
    z: float | None
    p: np.ndarray | None
    f: np.ndarray | None
    s: np.ndarray | None

    selected_features: np.ndarray | None
    detected_outliers: np.ndarray | None
    residuals: np.ndarray | None


def _status_name(status_code: int) -> str:
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return mapping.get(status_code, f"STATUS_{status_code}")


def solve_formulation_D(
    A: np.ndarray,
    r: np.ndarray,
    d0: int,
    k0: int,
    WL: np.ndarray,
    WU: np.ndarray,
    RU: np.ndarray,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    output_flag: bool = True,
    normalize_objective: bool = True,
) -> LADSFSSODSolution:
    """
    Solves the disjunctive-based LAD-SFSOD formulation (D).

    Mathematical meaning:
    - A[i, j] = value of feature j for observation i
    - r[i] = response of observation i
    - w[j] = regression coefficient
    - z = intercept
    - f[j] = 1 if feature j is selected
    - s[i] = 1 if observation i is authentic, 0 if it is an outlier
    - p[i] >= |r_i - A_i w - z|

    Formulation D:
        min (1/n) sum_i [p_i - (1 - s_i) R^U_i]

    subject to:
        p_i >= r_i - A_i w - z
        p_i >= z + A_i w - r_i
        w_j >= W^L_j f_j
        w_j <= W^U_j f_j
        sum_j f_j <= d0
        sum_i s_i >= n - k0
        p_i >= R^U_i (1 - s_i)
        p_i <= R^U_i
        f_j, s_i binary
    """
    A = np.asarray(A, dtype=float)
    r = np.asarray(r, dtype=float)
    WL = np.asarray(WL, dtype=float)
    WU = np.asarray(WU, dtype=float)
    RU = np.asarray(RU, dtype=float)

    if A.ndim != 2:
        raise ValueError("A must be a 2D array.")

    if r.ndim != 1:
        raise ValueError("r must be a 1D array.")

    n_samples, n_features = A.shape

    if len(r) != n_samples:
        raise ValueError("r length must match the number of rows of A.")

    if len(WL) != n_features or len(WU) != n_features:
        raise ValueError("WL and WU must have length equal to n_features.")

    if len(RU) != n_samples:
        raise ValueError("RU must have length equal to n_samples.")

    if not (0 <= d0 <= n_features):
        raise ValueError("d0 must be between 0 and n_features.")

    if not (0 <= k0 < n_samples):
        raise ValueError("k0 must be between 0 and n_samples - 1.")

    if np.any(RU <= 0):
        raise ValueError("All RU values must be strictly positive.")

    model = gp.Model("LAD_SFSOD_D")

    model.Params.OutputFlag = 1 if output_flag else 0

    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)

    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)

    I = range(n_samples)
    J = range(n_features)

    # Regression coefficients and intercept
    w = model.addVars(
        J,
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="w",
    )

    z = model.addVar(
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="z",
    )

    # Feature selection variables
    f = model.addVars(
        J,
        vtype=GRB.BINARY,
        name="f",
    )

    # Switch variables: s_i = 1 authentic, s_i = 0 outlier
    s = model.addVars(
        I,
        vtype=GRB.BINARY,
        name="s",
    )

    # Residual upper-bound variables p_i
    p = {
        i: model.addVar(
            lb=0.0,
            ub=float(RU[i]),
            vtype=GRB.CONTINUOUS,
            name=f"p[{i}]",
        )
        for i in I
    }

    model.update()

    # Residual absolute value linearization:
    # p_i >= r_i - A_i w - z
    # p_i >= z + A_i w - r_i
    for i in I:
        prediction_i = gp.quicksum(A[i, j] * w[j] for j in J) + z

        model.addConstr(
            p[i] >= r[i] - prediction_i,
            name=f"abs_pos[{i}]",
        )

        model.addConstr(
            p[i] >= prediction_i - r[i],
            name=f"abs_neg[{i}]",
        )

    # Feature selection constraints:
    # if f_j = 0, then w_j = 0
    # if f_j = 1, then WL_j <= w_j <= WU_j
    for j in J:
        model.addConstr(
            w[j] >= WL[j] * f[j],
            name=f"coef_lb[{j}]",
        )

        model.addConstr(
            w[j] <= WU[j] * f[j],
            name=f"coef_ub[{j}]",
        )

    # At most d0 selected features
    model.addConstr(
        gp.quicksum(f[j] for j in J) <= d0,
        name="feature_budget",
    )

    # At least n_samples - k0 authentic points
    # Equivalently, at most k0 outliers.
    model.addConstr(
        gp.quicksum(s[i] for i in I) >= n_samples - k0,
        name="outlier_budget",
    )

    # Disjunctive formulation D constraints:
    # p_i >= R_U_i (1 - s_i)
    # p_i <= R_U_i is already enforced through the upper bound of p_i,
    # but we keep the explicit bound in the variable definition.
    for i in I:
        model.addConstr(
            p[i] >= RU[i] * (1 - s[i]),
            name=f"disjunctive_link[{i}]",
        )

    objective = gp.quicksum(
        p[i] - (1 - s[i]) * RU[i]
        for i in I
    )

    if normalize_objective:
        objective = objective / n_samples

    model.setObjective(objective, GRB.MINIMIZE)

    model.optimize()

    status_code = model.Status
    status_name = _status_name(status_code)

    if model.SolCount == 0:
        return LADSFSSODSolution(
            status_code=status_code,
            status_name=status_name,
            objective_value=None,
            best_bound=None,
            runtime=getattr(model, "Runtime", None),
            mip_gap=None,
            node_count=getattr(model, "NodeCount", None),
            w=None,
            z=None,
            p=None,
            f=None,
            s=None,
            selected_features=None,
            detected_outliers=None,
            residuals=None,
        )

    w_val = np.array([w[j].X for j in J])
    z_val = float(z.X)
    p_val = np.array([p[i].X for i in I])
    f_val = np.array([f[j].X for j in J])
    s_val = np.array([s[i].X for i in I])

    selected_features = np.where(f_val > 0.5)[0]
    detected_outliers = np.where(s_val < 0.5)[0]

    residuals = r - (A @ w_val + z_val)

    gap = None
    try:
        gap = float(model.MIPGap)
    except gp.GurobiError:
        pass

    return LADSFSSODSolution(
        status_code=status_code,
        status_name=status_name,
        objective_value=float(model.ObjVal),
        best_bound=float(model.ObjBound),
        runtime=float(model.Runtime),
        mip_gap=gap,
        node_count=float(model.NodeCount),
        w=w_val,
        z=z_val,
        p=p_val,
        f=f_val,
        s=s_val,
        selected_features=selected_features,
        detected_outliers=detected_outliers,
        residuals=residuals,
    )