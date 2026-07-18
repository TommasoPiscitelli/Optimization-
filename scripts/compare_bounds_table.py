import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clean_valid_rows(
    df: pd.DataFrame,
    metric: str,
    require_optimal: bool,
) -> pd.DataFrame:
    df = df.copy()

    if "error_message" in df.columns:
        error_is_empty = (
            df["error_message"].isna()
            | (df["error_message"].astype(str).str.strip() == "")
        )
        df = df[error_is_empty]

    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not found in CSV.")

    required_columns = [
        "train_file",
        "n_samples",
        "d0",
        "k0",
        "k0_ratio",
        metric,
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if require_optimal:
        if "status_name" not in df.columns:
            raise ValueError("require_optimal=True but 'status_name' column is missing.")
        df = df[df["status_name"].astype(str).str.upper() == "OPTIMAL"]

    numeric_columns = [
        "n_samples",
        "d0",
        "k0",
        "k0_ratio",
        metric,
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_columns)

    df["n_samples"] = df["n_samples"].astype(int)
    df["d0"] = df["d0"].astype(int)
    df["k0"] = df["k0"].astype(int)
    df["k0_ratio"] = df["k0_ratio"].astype(float)
    df[metric] = df[metric].astype(float)

    if "run_id" in df.columns:
        df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
        df = df.sort_values("run_id")

    # Avoid duplicated runs due to resume/relaunch.
    dedup_keys = [
        "train_file",
        "n_samples",
        "d0",
        "k0",
        "k0_ratio",
    ]

    df = df.drop_duplicates(subset=dedup_keys, keep="last")

    return df


def prepare_formulation_df(
    csv_path: str | Path,
    formulation: str,
    metric: str,
    denormalize_by_n: bool,
    require_optimal: bool,
) -> pd.DataFrame:
    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)

    # If the CSV contains both formulations, keep only the requested one.
    if "formulation" in df.columns:
        mask = df["formulation"].astype(str).str.upper() == formulation.upper()
        if mask.any():
            df = df[mask].copy()

    df = clean_valid_rows(
        df=df,
        metric=metric,
        require_optimal=require_optimal,
    )

    df["formulation"] = formulation.upper()

    # The models in your code usually solve the normalized objective:
    # objective = original objective / n_samples.
    # To reproduce the paper-style table, we multiply by n_samples.
    df["v_value"] = df[metric].astype(float)

    if denormalize_by_n:
        df["v_value"] = df["v_value"] * df["n_samples"]

    df["k0_percent"] = 100.0 * df["k0_ratio"]

    return df


def build_paper_like_table(
    df_d: pd.DataFrame,
    df_l: pd.DataFrame,
    d0_for_table: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds the table:

        k | k0 (%) | v(L) | v(D) | Incr. (%)

    Values are averaged over the three replications.

    Since the paper states that d0 does not affect these values, by default
    we select one d0 value, usually the smallest common d0, and average only
    over the replications.
    """
    common_d0 = sorted(set(df_d["d0"].unique()) & set(df_l["d0"].unique()))

    if not common_d0:
        raise ValueError("No common d0 values found between D and L CSVs.")

    if d0_for_table is None:
        d0_for_table = common_d0[0]

    if d0_for_table not in common_d0:
        raise ValueError(
            f"d0_for_table={d0_for_table} is not present in both CSVs. "
            f"Common d0 values are: {common_d0}"
        )

    df_d_sel = df_d[df_d["d0"] == d0_for_table].copy()
    df_l_sel = df_l[df_l["d0"] == d0_for_table].copy()

    d_summary = (
        df_d_sel.groupby(["n_samples", "k0_ratio", "k0_percent"], as_index=False)
        .agg(
            n_replicates_D=("train_file", "nunique"),
            v_D=("v_value", "mean"),
            v_D_std=("v_value", "std"),
        )
    )

    l_summary = (
        df_l_sel.groupby(["n_samples", "k0_ratio", "k0_percent"], as_index=False)
        .agg(
            n_replicates_L=("train_file", "nunique"),
            v_L=("v_value", "mean"),
            v_L_std=("v_value", "std"),
        )
    )

    merged = pd.merge(
        l_summary,
        d_summary,
        on=["n_samples", "k0_ratio", "k0_percent"],
        how="inner",
    )

    merged["incr_percent"] = np.where(
        np.abs(merged["v_L"]) > 1e-12,
        100.0 * (merged["v_D"] - merged["v_L"]) / merged["v_L"],
        np.nan,
    )

    merged["d0_used_for_table"] = d0_for_table

    merged = merged.sort_values(["n_samples", "k0_ratio"]).reset_index(drop=True)

    # Add average rows for each n_samples, as in the paper.
    rows = []

    for n_samples, group in merged.groupby("n_samples", sort=True):
        for _, row in group.iterrows():
            rows.append(
                {
                    "n_samples": int(row["n_samples"]),
                    "k0_percent": row["k0_percent"],
                    "v_L": row["v_L"],
                    "v_D": row["v_D"],
                    "incr_percent": row["incr_percent"],
                    "n_replicates_L": int(row["n_replicates_L"]),
                    "n_replicates_D": int(row["n_replicates_D"]),
                    "d0_used_for_table": d0_for_table,
                    "row_type": "value",
                }
            )

        rows.append(
            {
                "n_samples": int(n_samples),
                "k0_percent": "Avg.",
                "v_L": np.nan,
                "v_D": np.nan,
                "incr_percent": group["incr_percent"].mean(),
                "n_replicates_L": np.nan,
                "n_replicates_D": np.nan,
                "d0_used_for_table": d0_for_table,
                "row_type": "average",
            }
        )

    paper_table = pd.DataFrame(rows)

    return merged, paper_table


def save_table_figure(
    paper_table: pd.DataFrame,
    phi: str,
    output_png: str | Path,
) -> None:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    display_rows = []

    previous_n = None

    for _, row in paper_table.iterrows():
        is_avg = row["row_type"] == "average"

        if is_avg:
            k_display = ""
            k0_display = "Avg."
            v_l_display = ""
            v_d_display = ""
            incr_display = f"{row['incr_percent']:.2f}"
        else:
            n = int(row["n_samples"])
            k_display = str(n) if previous_n != n else ""
            previous_n = n

            k0_display = f"{float(row['k0_percent']):.0f}"
            v_l_display = f"{row['v_L']:.2f}"
            v_d_display = f"{row['v_D']:.2f}"
            incr_display = f"{row['incr_percent']:.2f}"

        display_rows.append(
            [
                k_display,
                k0_display,
                v_l_display,
                v_d_display,
                incr_display,
            ]
        )

    columns = [
        "k",
        r"$k_0$ (%)",
        rf"$\phi={phi}$" + "\n" + r"$v(\mathcal{L})$",
        rf"$\phi={phi}$" + "\n" + r"$v(\mathcal{D})$",
        "Incr. (%)",
    ]

    n_rows = len(display_rows)

    fig_width = 8.0
    fig_height = max(3.0, 0.42 * n_rows + 1.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)

    # Header style.
    for col_idx in range(len(columns)):
        cell = table[(0, col_idx)]
        cell.set_text_props(weight="bold")
        cell.set_linewidth(1.2)

    # Add slightly thicker lines for average rows.
    for row_idx, row in enumerate(paper_table.itertuples(index=False), start=1):
        if row.row_type == "average":
            for col_idx in range(len(columns)):
                cell = table[(row_idx, col_idx)]
                cell.set_text_props(weight="bold")
                cell.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare formulation D and L bound values for a fixed phi."
    )

    parser.add_argument(
        "--csv-d",
        type=Path,
        required=True,
        help="CSV generated by scalability.py for formulation D.",
    )

    parser.add_argument(
        "--csv-l",
        type=Path,
        required=True,
        help="CSV generated by scalability.py for formulation L.",
    )

    parser.add_argument(
        "--phi",
        type=str,
        required=True,
        help="Phi value used to generate the two CSV files, e.g. 1, 1.2, 1.5, 2.0.",
    )

    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Prefix for generated output image.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="best_bound",
        choices=["best_bound", "objective_value"],
        help="Column to use for v(L), v(D). For paper-style bound comparison use best_bound.",
    )

    parser.add_argument(
        "--d0-for-table",
        type=int,
        default=None,
        help=(
            "d0 value used for the table. "
            "If omitted, the smallest common d0 is used."
        ),
    )

    parser.add_argument(
        "--no-denormalize-by-n",
        action="store_true",
        help=(
            "If set, do not multiply the metric by n_samples. "
            "Use this only if the CSV already contains unnormalized objectives/bounds."
        ),
    )

    parser.add_argument(
        "--require-optimal",
        action="store_true",
        help="If set, use only rows with status_name == OPTIMAL.",
    )

    args = parser.parse_args()

    denormalize_by_n = not args.no_denormalize_by_n

    df_d = prepare_formulation_df(
        csv_path=args.csv_d,
        formulation="D",
        metric=args.metric,
        denormalize_by_n=denormalize_by_n,
        require_optimal=args.require_optimal,
    )

    df_l = prepare_formulation_df(
        csv_path=args.csv_l,
        formulation="L",
        metric=args.metric,
        denormalize_by_n=denormalize_by_n,
        require_optimal=args.require_optimal,
    )

    _, paper_table = build_paper_like_table(
        df_d=df_d,
        df_l=df_l,
        d0_for_table=args.d0_for_table,
    )

    output_prefix = args.output_prefix
    table_png_path = output_prefix.with_name(
        output_prefix.name + "_paper_table.png"
    )

    save_table_figure(
        paper_table=paper_table,
        phi=args.phi,
        output_png=table_png_path,
    )

    print(f"Saved paper table figure to: {table_png_path}")


if __name__ == "__main__":
    main()