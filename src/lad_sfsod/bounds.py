from dataclasses import dataclass

import numpy as np
import csv
from pathlib import Path


@dataclass
class FeasibleSparseSolution:
    w: np.ndarray
    z: float
    selected_features: np.ndarray
    detected_outliers: np.ndarray
    residuals: np.ndarray
    rss: float
    tau: np.ndarray | None = None
    rss_all: float | None = None

@dataclass
class EllipsoidBounds:
    WL: np.ndarray
    WU: np.ndarray
    RU: np.ndarray
    beta_ls: np.ndarray
    v_star: float
    v_min: float
    delta: float
    rank: int
    condition_number: float




def build_true_hyperplane_solution(
    A: np.ndarray,
    r: np.ndarray,
    true_z: float,
    n_true_features: int = 5,
    k0: int = 0,
) -> FeasibleSparseSolution:
    """
    Builds the true synthetic hyperplane solution used in the paper.

    For the synthetic datasets, the true coefficient vector is:

        w = (1, 1, 1, 1, 1, 0, ..., 0)

    and true_z is the true intercept used to generate the instance.
    """
    A = np.asarray(A, dtype=float)
    r = np.asarray(r, dtype=float)

    n_samples, n_features = A.shape

    if n_true_features > n_features:
        raise ValueError(
            f"n_true_features={n_true_features} is larger than "
            f"n_features={n_features}."
        )

    if not (0 <= k0 <= n_samples):
        raise ValueError("k0 must be between 0 and n_samples.")

    w = np.zeros(n_features)
    w[:n_true_features] = 1.0

    z = float(true_z)

    residuals = r - (A @ w + z)
    residual_abs = np.abs(residuals)

    if k0 > 0:
        detected_outliers = np.argsort(residual_abs)[-k0:]
        detected_outliers = np.sort(detected_outliers.astype(int))
    else:
        detected_outliers = np.array([], dtype=int)

    rss = float(np.sum(residuals**2))

    return FeasibleSparseSolution(
        w=w,
        z=z,
        selected_features=np.arange(n_true_features, dtype=int),
        detected_outliers=detected_outliers,
        residuals=residuals,
        rss=rss,
    )


def compute_RU_from_true_hyperplane(
    A: np.ndarray,
    r: np.ndarray,
    true_z: float,
    phi: float = 1.0,
    n_true_features: int = 5,
    min_width: float = 1e-6,
) -> tuple[np.ndarray, FeasibleSparseSolution]:
    """
    Computes R^U as done in the synthetic experiments of the paper.

    The paper defines:

        E_i = |r_i - a_i omega - zeta|

    where (omega, zeta) is the authentic hyperplane used to generate
    the synthetic data. Then:

        R^U = phi * E

    with phi >= 1 controlling the tightness of the big-M values.
    """
    if phi < 1.0:
        raise ValueError("phi must be greater than or equal to 1.")

    true_solution = build_true_hyperplane_solution(
        A=A,
        r=r,
        true_z=true_z,
        n_true_features=n_true_features,
        k0=0,
    )

    E = np.abs(true_solution.residuals)
    RU = phi * E

    # Numerical safety: avoid exactly zero upper bounds.
    RU = np.maximum(RU, min_width)

    return RU, true_solution


def refine_RU_by_bisection(
    A: np.ndarray,
    r: np.ndarray,
    RU: np.ndarray,
    incumbent: FeasibleSparseSolution,
    k0: int,
    n_iter: int = 10,
    mu: float = 20.0,
    min_width: float = 1e-6,
    feasibility_tol: float = 1e-9,
) -> np.ndarray:
    """
    Refines the residual big-M vector R^U using Algorithm 1 of the paper.

    Starting from a vector R^U, the algorithm repeatedly tries to reduce
    the whole vector while preserving feasibility of the incumbent solution.

    After the bisection phase, the k0 largest entries of R^U are multiplied
    by mu.

    Paper defaults:
        n_iter = 10
        mu = 20
    """
    A = np.asarray(A, dtype=float)
    r = np.asarray(r, dtype=float)
    RU = np.asarray(RU, dtype=float).copy()

    n_samples = A.shape[0]

    if RU.shape != (n_samples,):
        raise ValueError(
            f"RU must have shape ({n_samples},), got {RU.shape}."
        )

    if not (0 <= k0 <= n_samples):
        raise ValueError("k0 must be between 0 and n_samples.")

    if mu < 1.0:
        raise ValueError("mu must be greater than or equal to 1.")

    w_star = np.asarray(incumbent.w, dtype=float)
    z_star = float(incumbent.z)

    residual_abs = np.abs(r - (A @ w_star + z_star))

    # Ensure that the starting vector makes the incumbent feasible.
    RU = np.maximum(RU, residual_abs + feasibility_tol)
    RU = np.maximum(RU, min_width)

    new_RU = RU / 2.0

    for _ in range(n_iter):
        incumbent_still_feasible = np.all(
            residual_abs <= new_RU + feasibility_tol
        )

        if incumbent_still_feasible:
            RU = new_RU
            new_RU = RU / 2.0
        else:
            new_RU = new_RU + (RU - new_RU) / 2.0

    if k0 > 0:
        largest_indices = np.argsort(-RU)[:k0]
        RU[largest_indices] = mu * RU[largest_indices]

    RU = np.maximum(RU, min_width)

    return RU


def fit_sparse_trimmed_least_squares(
    A: np.ndarray,
    r: np.ndarray,
    d0: int,
    k0: int,
    n_iter: int = 10,
) -> FeasibleSparseSolution:
    """
    Builds a simple feasible sparse solution.

    Heuristic:
    1. Fit least squares.
    2. Select the d0 largest coefficients in absolute value.
    3. Refit only on those features.
    4. Remove the k0 observations with largest absolute residual.
    5. Repeat.

    The returned solution has at most d0 nonzero coefficients.
    """
    A = np.asarray(A, dtype=float)
    r = np.asarray(r, dtype=float)

    n_samples, n_features = A.shape

    if not (0 < d0 <= n_features):
        raise ValueError("d0 must be between 1 and n_features.")

    if not (0 <= k0 < n_samples):
        raise ValueError("k0 must be between 0 and n_samples - 1.")

    authentic_mask = np.ones(n_samples, dtype=bool)
    selected = np.arange(min(d0, n_features))

    for _ in range(n_iter):
        A_auth = A[authentic_mask]
        r_auth = r[authentic_mask]

        # Full LS on currently authentic rows, used only to rank features.
        X_full = np.column_stack([A_auth, np.ones(A_auth.shape[0])])
        beta_full, *_ = np.linalg.lstsq(X_full, r_auth, rcond=None)

        w_full = beta_full[:-1]

        selected = np.argsort(np.abs(w_full))[-d0:]
        selected = np.sort(selected)

        # Refit only selected features.
        X_sel = np.column_stack([A_auth[:, selected], np.ones(A_auth.shape[0])])
        beta_sel, *_ = np.linalg.lstsq(X_sel, r_auth, rcond=None)

        w = np.zeros(n_features)
        w[selected] = beta_sel[:-1]
        z = float(beta_sel[-1])

        residuals = r - (A @ w + z)

        if k0 > 0:
            outliers = np.argsort(np.abs(residuals))[-k0:]
            authentic_mask = np.ones(n_samples, dtype=bool)
            authentic_mask[outliers] = False
        else:
            outliers = np.array([], dtype=int)
            authentic_mask = np.ones(n_samples, dtype=bool)

    residuals = r - (A @ w + z)
    rss = float(np.sum(residuals**2))

    return FeasibleSparseSolution(
        w=w,
        z=z,
        selected_features=selected,
        detected_outliers=np.sort(outliers.astype(int)),
        residuals=residuals,
        rss=rss,
    )


def read_true_intercept_from_synthetic_file(
    dat_path: str | Path,
    intercept_csv_path: str | Path | None = None,
) -> float:
    """
    Reads the true intercept of a synthetic instance.

    Expected .dat filename format:

        test150322_100_50_01_2_0_5_-10_1.dat

    where:
        - 100 is the original number of samples k;
        - 1 is the replication id.

    Expected intercept CSV format:

        k,replication,intercept
        100,1,-0.830890005467626
        ...
    """
    dat_path = Path(dat_path)
    stem = dat_path.stem
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected synthetic filename format: {dat_path.name}")

    k = int(parts[1])
    replication = int(parts[-1])

    if intercept_csv_path is None:
        date_prefix = parts[0].replace("test", "")
        intercept_csv_path = dat_path.parent / f"test{date_prefix}_intercept_values.csv"
    else:
        intercept_csv_path = Path(intercept_csv_path)

    if not intercept_csv_path.exists():
        raise FileNotFoundError(f"Intercept CSV not found: {intercept_csv_path}")

    with intercept_csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            row_k = int(row["k"])
            row_replication = int(row["replication"])

            if row_k == k and row_replication == replication:
                return float(row["intercept"])

    raise ValueError(
        f"Could not find intercept for k={k}, replication={replication} "
        f"in {intercept_csv_path}"
    )


def hard_threshold_top_k(x: np.ndarray, k: int) -> np.ndarray:
    """
    Projection onto the set of vectors with at most k nonzero entries.

    Keeps the k largest entries in absolute value and sets the others to zero.
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)

    if k <= 0:
        return result

    if k >= x.size:
        return x.copy()

    keep_indices = np.argsort(-np.abs(x))[:k]
    result[keep_indices] = x[keep_indices]

    return result


def project_beta_with_free_intercept(
    beta: np.ndarray,
    n_features: int,
    d0: int,
) -> np.ndarray:
    """
    Projects beta = (w, z) onto the set ||w||_0 <= d0.

    The intercept z is never thresholded.
    """
    beta = np.asarray(beta, dtype=float).copy()

    w = beta[:n_features]
    z = beta[n_features]

    w_projected = hard_threshold_top_k(w, d0)

    beta[:n_features] = w_projected
    beta[n_features] = z

    return beta


def fit_palm_ls_sfsod(
    A: np.ndarray,
    r: np.ndarray,
    d0: int,
    k0: int,
    max_iter: int = 500,
    tol: float = 1e-6,
    rho_beta: float | None = None,
    rho_tau: float = 1.0,
    coef_tol: float = 1e-8,
) -> FeasibleSparseSolution:
    """
    PALM algorithm for the LS-SFSOD feasible solution.

    It approximately solves:

        min_{beta, tau} 1/2 ||X beta + tau - r||_2^2

    subject to:

        ||w||_0 <= d0
        ||tau||_0 <= k0

    where X = [A, 1] and beta = (w, z).

    tau_i != 0 means that observation i is treated as an outlier.
    """
    A = np.asarray(A, dtype=float)
    r = np.asarray(r, dtype=float)

    n_samples, n_features = A.shape

    if not (0 <= d0 <= n_features):
        raise ValueError("d0 must be between 0 and n_features.")

    if not (0 <= k0 <= n_samples):
        raise ValueError("k0 must be between 0 and n_samples.")

    X = np.column_stack([A, np.ones(n_samples)])

    # Step size for beta block.
    # The gradient with respect to beta has Lipschitz constant ||X||_2^2.
    if rho_beta is None:
        spectral_norm = np.linalg.norm(X, ord=2)
        lipschitz_beta = spectral_norm**2
        rho_beta = 0.99 / (lipschitz_beta + 1e-12)

    # Initialization: ordinary least-squares, then projection on d0 features.
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    beta = project_beta_with_free_intercept(
        beta=beta,
        n_features=n_features,
        d0=d0,
    )

    tau = np.zeros(n_samples)

    previous_objective = np.inf

    for _ in range(max_iter):
        beta_old = beta.copy()
        tau_old = tau.copy()

        # beta update:
        # beta^{l+1} = P_d0(beta^l - rho_beta X^T(X beta^l + tau^l - r))
        residual_beta = X @ beta + tau - r
        beta_candidate = beta - rho_beta * (X.T @ residual_beta)

        beta = project_beta_with_free_intercept(
            beta=beta_candidate,
            n_features=n_features,
            d0=d0,
        )

        # tau update:
        # tau^{l+1} = P_k0(tau^l - rho_tau(X beta^{l+1} + tau^l - r))
        residual_tau = X @ beta + tau - r
        tau_candidate = tau - rho_tau * residual_tau
        tau = hard_threshold_top_k(tau_candidate, k0)

        corrected_residual = X @ beta + tau - r
        objective = 0.5 * float(np.sum(corrected_residual**2))

        beta_step = np.linalg.norm(beta - beta_old) / (np.linalg.norm(beta_old) + 1e-12)
        tau_step = np.linalg.norm(tau - tau_old) / (np.linalg.norm(tau_old) + 1e-12)

        if np.isfinite(previous_objective):
            objective_change = abs(previous_objective - objective) / (
                abs(previous_objective) + 1e-12
            )

            if objective_change < tol and beta_step < tol and tau_step < tol:
                break

        previous_objective = objective

    w = beta[:n_features]
    z = float(beta[n_features])

    raw_residuals = r - (A @ w + z)

    # In the PALM model, the corrected residual is:
    # X beta + tau - r.
    corrected_residuals = X @ beta + tau - r

    detected_outliers = np.flatnonzero(np.abs(tau) > coef_tol).astype(int)
    selected_features = np.flatnonzero(np.abs(w) > coef_tol).astype(int)

    rss_corrected = float(np.sum(corrected_residuals**2))
    rss_all = float(np.sum(raw_residuals**2))

    return FeasibleSparseSolution(
        w=w,
        z=z,
        selected_features=selected_features,
        detected_outliers=detected_outliers,
        residuals=raw_residuals,
        rss=rss_corrected,
        tau=tau,
        rss_all=rss_all,
    )



def compute_ellipsoid_big_m_bounds(
    A: np.ndarray,
    r: np.ndarray,
    d0: int,
    k0: int,
    n_iter_heuristic: int = 10,
    expansion_factor: float = 1.05,
    min_width: float = 1e-6,
    ru_method: str ="ellipsoid",
    true_z: float | None = None,
    dat_path: str | Path | None = None,
    intercept_csv_path: str | Path | None = None,
    ru_phi: float = 1.0,
    n_true_features: int = 5,
    refine_RU: bool = True,
    ru_bisection_iter: int = 10,
    ru_bisection_mu: float = 20.0,
    feasible_solution_method: str = "palm",
    palm_max_iter: int = 500,
    palm_tol: float = 1e-6,
    use_palm_adjusted_response: bool = True,
) -> EllipsoidBounds:
    """
    Computes W^L, W^U and R^U using the ellipsoid idea described
    in the paper.

    Let X = [A, 1]. We first compute a feasible sparse solution and
    define:

        v_star = ||r - X beta_star||_2^2

    Then we compute the least squares solution beta_ls and v_min.

    The bounds are obtained from the ellipsoid:

        ||r - X beta||_2^2 <= v_star

    In full-rank regimes, this gives:

        beta_j in beta_ls_j +- sqrt((v_star - v_min) * H_inv[j,j])

    where H_inv = (X^T X)^(-1).

    For numerical robustness we use the pseudo-inverse.
    """
    A = np.asarray(A, dtype=float)
    r = np.asarray(r, dtype=float)

    n_samples, n_features = A.shape

    if feasible_solution_method == "palm":
        sparse_solution = fit_palm_ls_sfsod(
            A=A,
            r=r,
            d0=d0,
            k0=k0,
            max_iter=palm_max_iter,
            tol=palm_tol,
        )

    elif feasible_solution_method == "trimmed_ls":
        sparse_solution = fit_sparse_trimmed_least_squares(
            A=A,
            r=r,
            d0=d0,
            k0=k0,
            n_iter=n_iter_heuristic,
        )

    else:
        raise ValueError(
            "Unknown feasible_solution_method. "
            "Expected 'palm' or 'trimmed_ls'."
        )

    X = np.column_stack([A, np.ones(n_samples)])

    beta_star = np.concatenate(
        [sparse_solution.w, np.array([sparse_solution.z])]
    )

    # If PALM is used, we can build the ellipsoid on the adjusted response:
    #
    #     r_adjusted = r - tau
    #
    # because PALM solves:
    #
    #     min ||X beta + tau - r||^2
    #
    # which is equivalent, for fixed tau, to:
    #
    #     min ||X beta - (r - tau)||^2.
    if (
        feasible_solution_method == "palm"
        and use_palm_adjusted_response
        and sparse_solution.tau is not None
    ):
        r_for_ellipsoid = r - sparse_solution.tau
    else:
        r_for_ellipsoid = r

    v_star = float(np.sum((r_for_ellipsoid - X @ beta_star) ** 2))

    beta_ls, *_ = np.linalg.lstsq(X, r_for_ellipsoid, rcond=None)
    residuals_ls = r_for_ellipsoid - X @ beta_ls
    v_min = float(np.sum(residuals_ls**2))

    delta = max(v_star - v_min, 0.0)

    H = X.T @ X

    rank = int(np.linalg.matrix_rank(H))
    condition_number = float(np.linalg.cond(H))

    H_inv = np.linalg.pinv(H)

    beta_radius = np.sqrt(
        np.maximum(delta * np.diag(H_inv), 0.0)
    )

    beta_radius = expansion_factor * beta_radius

    beta_lower = beta_ls - beta_radius
    beta_upper = beta_ls + beta_radius

    # Avoid zero-width intervals due to numerical degeneracy.
    beta_lower = np.minimum(beta_lower, beta_ls - min_width)
    beta_upper = np.maximum(beta_upper, beta_ls + min_width)

    WL = beta_lower[:n_features]
    WU = beta_upper[:n_features]

    if ru_method == "ellipsoid":
        # Prediction bounds for each row.
        pred_center = X @ beta_ls

        pred_radius = np.empty(n_samples)

        for i in range(n_samples):
            x_i = X[i, :]
            value = float(x_i @ H_inv @ x_i)
            pred_radius[i] = np.sqrt(max(delta * value, 0.0))

        pred_radius = expansion_factor * pred_radius

        SL = pred_center - pred_radius
        SU = pred_center + pred_radius

        RU = np.abs(r) + np.maximum(np.abs(SL), np.abs(SU))

        # Numerical safety.
        RU = np.maximum(RU, min_width)

        if refine_RU:
            RU = refine_RU_by_bisection(
                A=A,
                r=r,
                RU=RU,
                incumbent=sparse_solution,
                k0=k0,
                n_iter=ru_bisection_iter,
                mu=ru_bisection_mu,
                min_width=min_width,
            )

    elif ru_method == "synthetic_true":
        if true_z is None:
            if dat_path is None:
                raise ValueError(
                    "Either true_z or dat_path must be provided when ru_method='synthetic_true'."
                )
            true_z = read_true_intercept_from_synthetic_file(
                dat_path=dat_path,
                intercept_csv_path=intercept_csv_path
            )

        RU, true_solution = compute_RU_from_true_hyperplane(
            A=A,
            r=r,
            true_z=true_z,
            phi=ru_phi,
            n_true_features=n_true_features,
            min_width=min_width,
        )

        if refine_RU:
            RU = refine_RU_by_bisection(
                A=A,
                r=r,
                RU=RU,
                incumbent=true_solution,
                k0=k0,
                n_iter=ru_bisection_iter,
                mu=ru_bisection_mu,
                min_width=min_width,
            )

    else:
        raise ValueError(
            "Unknown ru_method. Expected 'ellipsoid' or 'synthetic_true'."
        )

    return EllipsoidBounds(
        WL=WL,
        WU=WU,
        RU=RU,
        beta_ls=beta_ls,
        v_star=v_star,
        v_min=v_min,
        delta=delta,
        rank=rank,
        condition_number=condition_number,
    )