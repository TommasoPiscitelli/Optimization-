"""
Small test script for the project.

It executes the proposed pipeline on a small instance:

    - load one synthetic training .dat file from the paper data;
    - keep only the first k observations;
    - keep only the first d features;
    - compute ellipsoid big-M bounds;
    - solve formulation for a grid of d0 and k0 values;
    - compare selected features and detected outliers with the known truth;

Default small instance:
    k = 30
    d = 20
    d0 values = 5, 7, 9
    k0 ratios = 0.10, 0.15, 0.20

Example:
    python test.py --data-dir data/raw/synthetic --time-limit 60
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lad_sfsod.bounds import compute_ellipsoid_big_m_bounds
from lad_sfsod.data import (
    find_first_training_dat_file,
    infer_outlier_path,
    read_dat_file,
    read_outlier_file,
)
from lad_sfsod.utils import append_csv_row, parse_float_list, parse_int_list, round_half_up

from lad_sfsod.model_d import solve_formulation_D
from lad_sfsod.model_l import solve_formulation_L




def truncate_instance(
    A: np.ndarray,
    r: np.ndarray,
    true_outliers_full: np.ndarray,
    n_small: int,
    p_small: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_small > A.shape[0]:
        raise ValueError(
            f"n_small={n_small} is larger than available rows={A.shape[0]}."
        )

    if p_small > A.shape[1]:
        raise ValueError(
            f"p_small={p_small} is larger than available features={A.shape[1]}."
        )

    A_small = A[:n_small, :p_small]
    r_small = r[:n_small]

    # Keep only the true outliers that are still present after row truncation.
    true_outliers_small = true_outliers_full[true_outliers_full < n_small]

    return A_small, r_small, np.sort(true_outliers_small.astype(int))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run formulation D on a small synthetic instance."
    )

    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/synthetic"))
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument("--date-prefix", type=str, default="150322", choices=["150322", "210322"])
    parser.add_argument("--n-samples", type=int, default=100, choices=[100, 150, 200])
    parser.add_argument("--n-small", type=int, default=30)
    parser.add_argument("--p-small", type=int, default=20)

    parser.add_argument("--d0-values", type=str, default="5")
    parser.add_argument("--k0-ratios", type=str, default="0.10")
    parser.add_argument("--formulation", type=str, default="L", choices=["D", "L"])

    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--output-flag", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=None)

    args = parser.parse_args()

    d0_values = parse_int_list(args.d0_values)
    k0_ratios = parse_float_list(args.k0_ratios)

    if args.file is not None:
        dat_path = args.file
    else:
        dat_path = find_first_training_dat_file(
            data_dir=args.data_dir,
            date_prefix=args.date_prefix,
            n_samples=args.n_samples,
        )

    outlier_path = infer_outlier_path(dat_path)

    A_full, r_full = read_dat_file(dat_path)
    true_outliers_full = read_outlier_file(outlier_path, n_samples=A_full.shape[0])

    A, r, true_outliers = truncate_instance(
        A=A_full,
        r=r_full,
        true_outliers_full=true_outliers_full,
        n_small=args.n_small,
        p_small=args.p_small,
    )

    n_samples, n_features = A.shape
    true_features = np.array([0, 1, 2, 3, 4], dtype=int)

    
    csv_columns = [
        "run_id",
        "train_file",
        "n_samples",
        "n_features",
        "d0",
        "k0",
        "k0_ratio",
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
    ]

    run_id = 0

    for d0 in d0_values:
        for k0_ratio in k0_ratios:
            run_id += 1
            k0 = round_half_up(k0_ratio * n_samples)

            print(f"---------- RUN {run_id} ----------")
            print(f"d0={d0}, k0_ratio={k0_ratio:.2f}, k0={k0}")

            bounds = compute_ellipsoid_big_m_bounds(
                A=A,
                r=r,
                d0=d0,
                k0=k0,
                n_iter_heuristic=10,
                expansion_factor=1.05,
                ru_method="synthetic_true",
                dat_path=dat_path,
                ru_phi=1.0,
                refine_RU=True,
                ru_bisection_iter=10,
                ru_bisection_mu=20.0,
            )

            print(f"Solving formulation {args.formulation}...")

            if args.formulation == "D":
                solution = solve_formulation_D(
                    A=A,
                    r=r,
                    d0=d0,
                    k0=k0,
                    WL=bounds.WL,
                    WU=bounds.WU,
                    RU=bounds.RU,
                    time_limit=args.time_limit,
                    mip_gap=args.mip_gap,
                    output_flag=args.output_flag,
                    normalize_objective=True,
                )

            elif args.formulation == "L":
                solution = solve_formulation_L(
                    A=A,
                    r=r,
                    d0=d0,
                    k0=k0,
                    WL=bounds.WL,
                    WU=bounds.WU,
                    RU=bounds.RU,
                    time_limit=args.time_limit,
                    mip_gap=args.mip_gap,
                    output_flag=args.output_flag,
                    normalize_objective=True,
                )

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
                or (mip_gap is not None and mip_gap <= args.mip_gap)
            )

            row = {
                "run_id": run_id,
                "train_file": str(dat_path),
                "n_samples": n_samples,
                "n_features": n_features,
                "d0": d0,
                "k0": k0,
                "k0_ratio": f"{k0_ratio:.2f}",
                "status_name": solution.status_name,
                "solved_to_optimality": solved_to_optimality,
                "objective_value": objective_value,
                "best_bound": best_bound,
                "mip_gap": mip_gap,
                "runtime_sec": runtime_sec,
                "node_count": node_count,
                "true_features": json.dumps(true_features.tolist(), separators=(",", ":")),
                "true_outliers": json.dumps(true_outliers.tolist(), separators=(",", ":")),
            }

        
            if solution.w is not None and solution.z is not None:
                coef_tol = 1e-5

                selected_features = np.flatnonzero(
                    np.abs(solution.w) > coef_tol
                ).astype(int)

                detected_outliers = np.asarray(solution.detected_outliers, dtype=int)

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

                coeffs = {
                    int(j): float(solution.w[int(j)])
                    for j in selected_features
                }

                row.update(
                    {
                        "selected_features": json.dumps(selected_features.tolist(), separators=(",", ":")),
                        "detected_outliers": json.dumps(detected_outliers.tolist(), separators=(",", ":")),
                        "selected_coefficients": json.dumps(coeffs, separators=(",", ":")),
                        "estimated_z": float(solution.z),
                        "train_lad_authentic": train_lad_authentic,
                        "train_mae_all": train_mae_all,
                        "train_mse_all": train_mse_all,
                    }
                )


            if args.output_csv is not None:
                append_csv_row(args.output_csv, row, csv_columns)
                print(f"Saved row to: {args.output_csv}")

            print()

    print("Small test completed.")


if __name__ == "__main__":
    main()