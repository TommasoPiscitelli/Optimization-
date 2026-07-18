from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
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

    # For compatibility with formulation D:
    # in formulation L, p stores the q values.
    p: np.ndarray | None

    f: np.ndarray | None
    s: np.ndarray | None

    selected_features: np.ndarray | None
    detected_outliers: np.ndarray | None
    residuals: np.ndarray | None

    # Specific variables of formulation L.
    q: np.ndarray | None = None
    t: np.ndarray | None = None


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


def solve_formulation_L(
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
    Solves formulation (L) of the LAD-SFSOD problem.

    Mathematical meaning:
    - A[i, j] = value of feature j for observation i
    - r[i] = response of observation i
    - w[j] = regression coefficient
    - z = intercept
    - f[j] = 1 if feature j is selected
    - s[i] = 1 if observation i is authentic, 0 if it is an outlier
    - q[i] >= |r_i - A_i w - z - t_i|
    - t[i] is a residual correction allowed only for outliers

    Formulation L:

        min (1/n) sum_i q_i

    subject to:

        q_i + r_i - A_i w - z - t_i >= 0
        r_i - A_i w - z - t_i - q_i <= 0

        t_i >= -(1 - s_i) R^U_i
        t_i <=  (1 - s_i) R^U_i

        w_j >= W^L_j f_j
        w_j <= W^U_j f_j

        sum_j f_j <= d0
        sum_i s_i >= n - k0

        f_j, s_i binary
        q_i >= 0
        w_j, z, t_i continuous

    Interpretation:
    - If s_i = 1, then t_i = 0, so q_i measures the true absolute residual.
    - If s_i = 0, then t_i can move inside [-R^U_i, R^U_i], allowing the
      model to cancel the residual of an outlier.
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

    if np.any(WL > WU):
        raise ValueError("Each lower coefficient bound must be <= upper bound.")

    model = gp.Model("LAD_SFSOD_L")

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

    # Switch variables:
    # s_i = 1 authentic point
    # s_i = 0 outlier
    s = model.addVars(
        I,
        vtype=GRB.BINARY,
        name="s",
    )

    # q_i >= |r_i - A_i w - z - t_i|
    q = model.addVars(
        I,
        lb=0.0,
        ub=GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="q",
    )

    # t_i is the correction variable for possible outliers
    t = model.addVars(
        I,
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="t",
    )

    model.update()

    # Absolute value linearization for:
    #
    # q_i >= |r_i - A_i w - z - t_i|
    #
    # Equivalent to:
    #
    # q_i + r_i - A_i w - z - t_i >= 0
    # r_i - A_i w - z - t_i - q_i <= 0
    for i in I:
        prediction_i = gp.quicksum(A[i, j] * w[j] for j in J) + z

        model.addConstr(
            q[i] + r[i] - prediction_i - t[i] >= 0,
            name=f"abs_pos_L[{i}]",
        )

        model.addConstr(
            r[i] - prediction_i - t[i] - q[i] <= 0,
            name=f"abs_neg_L[{i}]",
        )

    # t_i is forced to zero when s_i = 1.
    # If s_i = 0, then t_i can vary in [-RU_i, RU_i].
    for i in I:
        model.addConstr(
            t[i] >= -RU[i] * (1 - s[i]),
            name=f"t_lb[{i}]",
        )

        model.addConstr(
            t[i] <= RU[i] * (1 - s[i]),
            name=f"t_ub[{i}]",
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

    objective = gp.quicksum(q[i] for i in I)

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
            q=None,
            t=None,
        )

    w_val = np.array([w[j].X for j in J])
    z_val = float(z.X)
    q_val = np.array([q[i].X for i in I])
    t_val = np.array([t[i].X for i in I])
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
        p=q_val,
        f=f_val,
        s=s_val,
        selected_features=selected_features,
        detected_outliers=detected_outliers,
        residuals=residuals,
        q=q_val,
        t=t_val,
    )