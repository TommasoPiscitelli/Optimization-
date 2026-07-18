"""
Scalability experiment formulations on the synthetic datasets.

One CSV row corresponds to one MILP run:
    (training .dat file, d0, k0)

Default experiment:
    - corruption scheme: vertical_outliers
    - n_samples: 100, 150, 200
    - max 3 replicates per sample size
    - d0 values: 5, 7, 9
    - k0 ratios: 0.10, 0.15, 0.20
    - bounds: ellipsoid bounds

Example:
    python scalability.py \
        --data-dir data/raw/synthetic \
        --output-csv results/scalability_formulation_D_vertical.csv \
        --time-limit 300 \
        --mip-gap 1e-4
"""

import argparse
import json
import re
from pathlib import Path
import numpy as np

from lad_sfsod.bounds import compute_ellipsoid_big_m_bounds
from lad_sfsod.data import read_dat_file
from lad_sfsod.utils import append_csv_row, parse_float_list, parse_int_list, round_half_up
from lad_sfsod.model_d import solve_formulation_D
from lad_sfsod.model_l import solve_formulation_L


CSV_COLUMNS = [
    "run_id",
    "formulation",
    "train_file",
    "corruption_scheme",
    "n_samples",
    "n_features",
    "d0",
    "k0",
    "k0_ratio",
    "time_limit",
    "mip_gap_target",
    "status_name",
    "solved_to_optimality",
    "objective_value",
    "best_bound",
    "mip_gap",
    "runtime_sec",
    "node_count",
    "selected_features",
    "detected_outliers",
    "estimated_z",
    "selected_coefficients",
    "train_lad_authentic",
    "train_mae_all",
    "train_mse_all",
    "true_features",
     "true_outliers",
    "error_message",
]





def parse_dat_filename(path: Path) -> dict[str, Any]:
    """
    Parses file names like:
        test150322_100_50_01_2_0_5_-10_1.dat

    Returns:
        date_prefix: '150322'
        n_samples_from_name: 100
    """
    match = re.match(r"^test(?P<date_prefix>\d+)_(?P<n_samples>\d+)_", path.name)
    if not match:
        return {
            "date_prefix": None,
            "n_samples_from_name": None,
        }

    return {
        "date_prefix": match.group("date_prefix"),
        "n_samples_from_name": int(match.group("n_samples")),
    }


def infer_corruption_scheme(path: Path, date_prefix: str | None = None) -> str:
    """Infers corruption scheme from folder name or date prefix."""
    path_parts = set(path.parts)

    if "vertical_outliers" in path_parts:
        return "vertical_outliers"
    if "bad_leverage_points" in path_parts:
        return "bad_leverage_points"

    if date_prefix == "150322":
        return "vertical_outliers"
    if date_prefix == "210322":
        return "bad_leverage_points"

    return "unknown"


def find_training_files(
    data_dir: Path,
    corruption_scheme: str,
    n_samples_values: list[int],
    max_replicates_per_size: int,
) -> list[Path]:
    """
    Finds training .dat files and keeps up to max_replicates_per_size
    files for each n_samples value.
    """
    search_root = data_dir / corruption_scheme

    grouped: dict[int, list[Path]] = {n: [] for n in n_samples_values}

    for path in sorted(search_root.rglob("*.dat")):
        if path.name.endswith("Val.dat"):
            continue

        if ":Zone.Identifier" in path.name:
            continue

        info = parse_dat_filename(path)
        n_samples = info["n_samples_from_name"]

        if n_samples not in grouped:
            continue

        grouped[n_samples].append(path)

    selected_files: list[Path] = []

    for n_samples in n_samples_values:
        selected_files.extend(
            sorted(grouped[n_samples])[:max_replicates_per_size]
        )

    return selected_files


def run_single_experiment(
    *,
    run_id: int,
    train_path: Path,
    corruption_scheme: str,
    d0: int,
    k0_ratio: float,
    time_limit: float,
    mip_gap_target: float,
    output_flag: bool,
    formulation: str,
    coef_tol: float = 1e-6,
) -> dict:
    """
    Runs one formulation on one training file with one (d0, k0_ratio) setting.
    """
    A, r = read_dat_file(train_path)

    n_samples, n_features = A.shape
    k0 = round_half_up(k0_ratio * n_samples)

    if k0 < 0 or k0 >= n_samples:
        raise ValueError(f"Invalid k0={k0} for n_samples={n_samples}.")

    bounds = compute_ellipsoid_big_m_bounds(
        A=A,
        r=r,
        d0=d0,
        k0=k0,
        n_iter_heuristic=10,
        expansion_factor=1.05,
        ru_method="synthetic_true",
        dat_path=train_path,
        ru_phi=3.0,
        refine_RU=True,
        ru_bisection_iter=10,
        ru_bisection_mu=20.0,
    )

    if formulation == "D":
        solution = solve_formulation_D(
            A=A,
            r=r,
            d0=d0,
            k0=k0,
            WL=bounds.WL,
            WU=bounds.WU,
            RU=bounds.RU,
            time_limit=time_limit,
            mip_gap=mip_gap_target,
            output_flag=output_flag,
            normalize_objective=True,
        )

    elif formulation == "L":
        solution = solve_formulation_L(
            A=A,
            r=r,
            d0=d0,
            k0=k0,
            WL=bounds.WL,
            WU=bounds.WU,
            RU=bounds.RU,
            time_limit=time_limit,
            mip_gap=mip_gap_target,
            output_flag=output_flag,
            normalize_objective=True,
        )

    else:
        raise ValueError(f"Unknown formulation: {formulation}")

    objective_value = (
        float(solution.objective_value)
        if solution.objective_value is not None
        else None
    )

    best_bound = (
        float(solution.best_bound)
        if solution.best_bound is not None
        else None
    )

    mip_gap = (
        float(solution.mip_gap)
        if solution.mip_gap is not None
        else None
    )

    runtime_sec = (
        float(solution.runtime)
        if solution.runtime is not None
        else None
    )

    node_count = (
        float(solution.node_count)
        if solution.node_count is not None
        else None
    )

    solved_to_optimality = (
        solution.status_name == "OPTIMAL"
        or (mip_gap is not None and mip_gap <= mip_gap_target)
    )

    row = {
        "run_id": run_id,
        "formulation": formulation,
        "train_file": str(train_path),
        "corruption_scheme": corruption_scheme,
        "n_samples": n_samples,
        "n_features": n_features,
        "d0": d0,
        "k0": k0,
        "k0_ratio": f"{k0_ratio:.2f}",
        "time_limit": time_limit,
        "mip_gap_target": mip_gap_target,
        "status_name": solution.status_name,
        "solved_to_optimality": solved_to_optimality,
        "objective_value": objective_value,
        "best_bound": best_bound,
        "mip_gap": mip_gap,
        "runtime_sec": runtime_sec,
        "node_count": node_count,
        "true_features": json.dumps([0, 1, 2, 3, 4], separators=(",", ":")),
        "error_message": "",
    }

    if solution.w is None or solution.z is None:
        return row

    selected_features = np.flatnonzero(
        np.abs(solution.w) > coef_tol
    ).astype(int)

    detected_outliers = (
        np.asarray(solution.detected_outliers, dtype=int)
        if solution.detected_outliers is not None
        else np.array([], dtype=int)
    )

    selected_coefficients = {
        int(j): float(solution.w[int(j)])
        for j in selected_features
    }

    residuals = np.asarray(solution.residuals, dtype=float)
    s_values = np.asarray(solution.s, dtype=float)

    authentic_mask = s_values > 0.5

    train_lad_authentic = (
        float(np.sum(np.abs(residuals[authentic_mask])))
        if authentic_mask.any()
        else float("nan")
    )

    train_mae_all = float(np.mean(np.abs(residuals)))
    train_mse_all = float(np.mean(residuals**2))

    row.update(
        {
            "selected_features": json.dumps(
                selected_features.tolist(),
                separators=(",", ":"),
            ),
            "detected_outliers": json.dumps(
                detected_outliers.tolist(),
                separators=(",", ":"),
            ),
            "estimated_z": float(solution.z),
            "selected_coefficients": json.dumps(
                selected_coefficients,
                separators=(",", ":"),
            ),
            "train_lad_authentic": train_lad_authentic,
            "train_mae_all": train_mae_all,
            "train_mse_all": train_mse_all,
        }
    )

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scalability experiments for formulation D."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/synthetic"),
        help="Root directory containing synthetic data folders.",
    )

    parser.add_argument(
        "--formulation",
        type=str,
        default="D",
        choices=["D", "L"],
        help="MILP formulation to solve: D or L.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/scalability_formulation_D_vertical.csv"),
        help="CSV file where results will be appended.",
    )
    
    parser.add_argument(
        "--corruption-scheme",
        type=str,
        default="vertical_outliers",
        choices=["vertical_outliers", "bad_leverage_points", "all"],
        help="Corruption scheme to test.",
    )
    
    parser.add_argument(
        "--n-samples-values",
        type=str,
        default="100,150,200",
        help="Comma-separated n_samples values to include.",
    )
    
    parser.add_argument(
        "--max-replicates-per-size",
        type=int,
        default=3,
        help="Maximum number of training files per n_samples value.",
    )
    
    parser.add_argument(
        "--d0-values",
        type=str,
        default="5,7,9",
        help="Comma-separated d0 values.",
    )
    
    parser.add_argument(
        "--k0-ratios",
        type=str,
        default="0.10,0.15,0.20",
        help="Comma-separated k0 ratios.",
    )
    
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="Gurobi time limit in seconds for each run.",
    )
    
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=1e-4,
        help="Target relative MIP gap.",
    )
    
    parser.add_argument(
        "--output-flag",
        action="store_true",
        help="If set, print Gurobi logs.",
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="If set, do not skip runs already present in the output CSV.",
    )
    
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional debug limit on the number of runs to execute.",
    )

    args = parser.parse_args()

    n_samples_values = parse_int_list(args.n_samples_values)
    d0_values = parse_int_list(args.d0_values)
    k0_ratios = parse_float_list(args.k0_ratios)

    training_files = find_training_files(
        data_dir=args.data_dir,
        corruption_scheme=args.corruption_scheme,
        n_samples_values=n_samples_values,
        max_replicates_per_size=args.max_replicates_per_size,
    )

    if not training_files:
        raise FileNotFoundError(
            f"No training .dat files found under {args.data_dir} "
            f"for corruption_scheme={args.corruption_scheme}."
        )

    run_id = 0
    executed = 0

    for train_path in training_files:
        info = parse_dat_filename(train_path)
        inferred_scheme = infer_corruption_scheme(train_path, info["date_prefix"])


        for d0 in d0_values:
            for k0_ratio in k0_ratios:
                run_id += 1

                if args.max_runs is not None and executed >= args.max_runs:
                    print(f"Reached --max-runs={args.max_runs}. Stopping.")
                    return

                row = run_single_experiment(
                        run_id=run_id,
                        formulation=args.formulation,
                        train_path=train_path,
                        corruption_scheme=inferred_scheme,
                        d0=d0,
                        k0_ratio=k0_ratio,
                        time_limit=args.time_limit,
                        mip_gap_target=args.mip_gap,
                        output_flag=args.output_flag,
                    )

                append_csv_row(args.output_csv, row)

                executed += 1

    print("\nExperiment completed.")
    print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()