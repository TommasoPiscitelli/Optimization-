from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_scalability_csv(csv_path: Path, formulation: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "formulation" in df.columns:
        df = df[df["formulation"].astype(str).str.upper() == formulation.upper()].copy()

    if "error_message" in df.columns:
        valid_error = df["error_message"].isna() | (
            df["error_message"].astype(str).str.strip() == ""
        )
        df = df[valid_error].copy()

    required_columns = [
        "train_file",
        "d0",
        "k0",
        "k0_ratio",
        "status_name",
        "runtime_sec",
        "mip_gap",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    df["formulation"] = formulation.upper()
    df["d0"] = pd.to_numeric(df["d0"], errors="coerce")
    df["k0"] = pd.to_numeric(df["k0"], errors="coerce")
    df["k0_ratio"] = pd.to_numeric(df["k0_ratio"], errors="coerce")
    df["runtime_sec"] = pd.to_numeric(df["runtime_sec"], errors="coerce")
    df["mip_gap"] = pd.to_numeric(df["mip_gap"], errors="coerce")
    df["status_name"] = df["status_name"].astype(str).str.upper()

    df = df.dropna(subset=["d0", "k0", "k0_ratio"])
    df["d0"] = df["d0"].astype(int)
    df["k0"] = df["k0"].astype(int)

    return df


def build_summary(df_d: pd.DataFrame, df_l: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for formulation, df in [("D", df_d), ("L", df_l)]:
        optimal_df = df[df["status_name"] == "OPTIMAL"].copy()

        rows.append(
            {
                "formulation": formulation,
                "n_tests_total": len(df),
                "n_optimal": len(optimal_df),
                "optimal_rate": len(optimal_df) / len(df) if len(df) > 0 else np.nan,
                "mean_runtime_optimal": optimal_df["runtime_sec"].mean(),
                "std_runtime_optimal": optimal_df["runtime_sec"].std(),
                "mean_mip_gap_all": df["mip_gap"].mean(),
                "mean_mip_gap_nonoptimal": df.loc[
                    df["status_name"] != "OPTIMAL", "mip_gap"
                ].mean(),
            }
        )

    return pd.DataFrame(rows)


def save_summary_barplot(summary_df: pd.DataFrame, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    formulations = summary_df["formulation"].tolist()
    x = np.arange(len(formulations))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].bar(x, summary_df["n_optimal"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(formulations)
    axes[0].set_title("Optimal tests")
    axes[0].set_ylabel("Count")

    axes[1].bar(x, summary_df["mean_runtime_optimal"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(formulations)
    axes[1].set_title("Mean CPU time on OPTIMAL")
    axes[1].set_ylabel("Seconds")

    axes[2].bar(x, summary_df["mean_mip_gap_all"])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(formulations)
    axes[2].set_title("Mean MIP gap")
    axes[2].set_ylabel("MIP gap")

    fig.suptitle("Scalability summary: formulation D vs L")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def common_optimal_runs(df_d: pd.DataFrame, df_l: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only tests solved to OPTIMAL by both formulations.

    Matching key:
        train_file, d0, k0, k0_ratio
    """
    key_cols = ["train_file", "d0", "k0", "k0_ratio"]

    d_opt = df_d[df_d["status_name"] == "OPTIMAL"].copy()
    l_opt = df_l[df_l["status_name"] == "OPTIMAL"].copy()

    d_opt = d_opt[key_cols + ["runtime_sec"]].rename(
        columns={"runtime_sec": "runtime_D"}
    )
    l_opt = l_opt[key_cols + ["runtime_sec"]].rename(
        columns={"runtime_sec": "runtime_L"}
    )

    merged = pd.merge(d_opt, l_opt, on=key_cols, how="inner")

    return merged.sort_values(["d0", "k0", "train_file"]).reset_index(drop=True)


def save_runtime_boxplot_by_d0(common_df: pd.DataFrame, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    d0_values = sorted(common_df["d0"].unique())

    if not d0_values:
        raise ValueError("No common OPTIMAL runs found between D and L.")

    data = []
    positions = []
    labels = []

    base_positions = np.arange(len(d0_values)) * 3.0

    for base_pos, d0 in zip(base_positions, d0_values):
        group = common_df[common_df["d0"] == d0]

        data.append(group["runtime_D"].dropna().to_numpy())
        positions.append(base_pos)
        labels.append("D")

        data.append(group["runtime_L"].dropna().to_numpy())
        positions.append(base_pos + 1.0)
        labels.append("L")

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(d0_values)), 5))

    ax.boxplot(
        data,
        positions=positions,
        widths=0.7,
        showfliers=True,
    )

    # Overlay points, useful when there are few replications.
    for values, pos in zip(data, positions):
        if len(values) == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(np.full(len(values), pos) + jitter, values, alpha=0.7, s=20)

    ax.set_ylabel("CPU Time (seconds)")
    ax.set_title(
        "CPU time distribution on instances solved by both D and L\n"
        "Grouped by d0"
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    for base_pos, d0 in zip(base_positions, d0_values):
        ax.text(
            base_pos + 0.5,
            -0.12,
            rf"$d_0={d0}$",
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
        )

    ax.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare scalability results of formulations D and L."
    )

    parser.add_argument(
        "--csv-d",
        type=Path,
        required=True,
        help="Scalability CSV for formulation D.",
    )

    parser.add_argument(
        "--csv-l",
        type=Path,
        required=True,
        help="Scalability CSV for formulation L.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/comparison"),
        help="Directory where output files will be saved.",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="comparison",
        help="Tag used in output filenames, for example phi_2.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_d = load_scalability_csv(args.csv_d, formulation="D")
    df_l = load_scalability_csv(args.csv_l, formulation="L")

    summary_df = build_summary(df_d, df_l)
    common_df = common_optimal_runs(df_d, df_l)

    summary_csv = args.output_dir / f"{args.tag}_summary.csv"
    common_csv = args.output_dir / f"{args.tag}_common_optimal_runs.csv"
    summary_png = args.output_dir / f"{args.tag}_summary_barplot.png"
    boxplot_png = args.output_dir / f"{args.tag}_runtime_boxplot_by_d0.png"

    summary_df.to_csv(summary_csv, index=False)
    common_df.to_csv(common_csv, index=False)

    save_summary_barplot(summary_df, summary_png)
    save_runtime_boxplot_by_d0(common_df, boxplot_png)

    print("Generated files:")
    print(f"  Summary CSV:              {summary_csv}")
    print(f"  Common OPTIMAL runs CSV:  {common_csv}")
    print(f"  Summary barplot:          {summary_png}")
    print(f"  Runtime boxplot by d0:    {boxplot_png}")

    print("\nSummary:")
    print(summary_df)


if __name__ == "__main__":
    main()