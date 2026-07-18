from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def prepare_validation_table(csv_path: Path, formulation: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    columns_to_keep = [
        "n_samples",
        "mean_rmspe",
        "mean_mae",
        "mean_feature_fpr",
        "mean_feature_fnr",
        "mean_feature_f1",
    ]

    missing = [col for col in columns_to_keep if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    out = df[columns_to_keep].copy()
    out.insert(0, "formulation", formulation)

    return out


def save_table_image(table_df: pd.DataFrame, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    display_df = table_df.copy()

    numeric_columns = [
        "mean_rmspe",
        "mean_mae",
        "mean_feature_fpr",
        "mean_feature_fnr",
        "mean_feature_f1",
    ]

    for col in numeric_columns:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")

    display_df["n_samples"] = display_df["n_samples"].astype(int).astype(str)

    display_df = display_df.rename(
        columns={
            "formulation": "Formulation",
            "n_samples": "n",
            "mean_rmspe": "RMSPE",
            "mean_mae": "MAE",
            "mean_feature_fpr": "FPR",
            "mean_feature_fnr": "FNR",
            "mean_feature_f1": "F1 score",
        }
    )

    fig_height = max(2.5, 0.45 * len(display_df) + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.3)

    for col_idx in range(len(display_df.columns)):
        header_cell = table[(0, col_idx)]
        header_cell.set_text_props(weight="bold")
        header_cell.set_linewidth(1.2)

    fig.suptitle("Validation metrics by instance size", fontsize=14, y=0.98)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a validation metrics table comparing formulations D and L."
    )

    parser.add_argument(
        "--csv-d",
        type=Path,
        default=Path("results/D/validation/validation_metrics_by_size_D_phi_2.csv"),
        help="Validation metrics CSV for formulation D.",
    )

    parser.add_argument(
        "--csv-l",
        type=Path,
        default=Path("results/L/validation/validation_metrics_by_size_L_phi_2.csv"),
        help="Validation metrics CSV for formulation L.",
    )

    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("results/comparison/validation_metrics_table_phi_2.png"),
        help="Output image path.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/comparison/validation_metrics_table_phi_2.csv"),
        help="Optional combined CSV output path.",
    )

    args = parser.parse_args()

    df_d = prepare_validation_table(args.csv_d, formulation="D")
    df_l = prepare_validation_table(args.csv_l, formulation="L")

    table_df = pd.concat([df_d, df_l], ignore_index=True)
    table_df = table_df.sort_values(["n_samples", "formulation"]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(args.output_csv, index=False)

    save_table_image(table_df, args.output_png)

    print(f"Saved combined CSV to: {args.output_csv}")
    print(f"Saved table image to:  {args.output_png}")


if __name__ == "__main__":
    main()