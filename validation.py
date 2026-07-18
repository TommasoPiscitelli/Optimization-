import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lad_sfsod.data import read_dat_file


def parse_json_cell(value: Any, default: Any):
    """
    Parses CSV cells containing JSON-like lists/dictionaries.

    Examples:
        "[0,1,2,3,4]"
        '{"0": 1.0, "1": 0.9}'
    """
    if value is None:
        return default

    if isinstance(value, float) and np.isnan(value):
        return default

    if isinstance(value, (list, dict)):
        return value

    text = str(value).strip()

    if text == "" or text.lower() in {"nan", "none"}:
        return default

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def compute_aic_for_csv_row(
    row: pd.Series,
    base_dir: str | Path = ".",
    coef_tol: float = 1e-6,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """
    Computes the AIC-like criterion used in the paper for one CSV row.

    The CSV row contains:
        - train_file
        - detected_outliers
        - estimated_z
        - selected_coefficients

    The function reopens the training .dat file, reconstructs w and z,
    removes the detected outliers, and computes:

        L = ||r' - A'w - z1||_2^2 / kappa

        AIC = kappa * delta + kappa * log(L)

    where:
        kappa = number of non-outlier points
        delta = number of nonzero coefficients + intercept
    """
    base_dir = Path(base_dir)

    train_path = Path(row["train_file"])
    if not train_path.is_absolute():
        train_path = base_dir / train_path

    A, r = read_dat_file(train_path)

    n_samples, n_features = A.shape

    coeffs = parse_json_cell(row["selected_coefficients"], default={})
    detected_outliers = parse_json_cell(row["detected_outliers"], default=[])

    w = np.zeros(n_features)

    for key, value in coeffs.items():
        j = int(key)
        if 0 <= j < n_features:
            w[j] = float(value)

    z = float(row["estimated_z"])

    authentic_mask = np.ones(n_samples, dtype=bool)

    for idx in detected_outliers:
        i = int(idx)
        if 0 <= i < n_samples:
            authentic_mask[i] = False

    kappa = int(np.sum(authentic_mask))

    if kappa <= 0:
        return {
            "kappa_non_outliers": kappa,
            "delta_nonzero_with_intercept": int(np.sum(np.abs(w) > coef_tol) + 1),
            "ls_avg_error_non_outliers": float("nan"),
            "aic_score": float("nan"),
        }

    residuals_authentic = r[authentic_mask] - (
        A[authentic_mask] @ w + z
    )

    rss_authentic = float(np.sum(residuals_authentic**2))

    L = rss_authentic / kappa

    delta = int(np.sum(np.abs(w) > coef_tol) + 1)

    L_safe = max(L, eps)

    aic_score = kappa * delta + kappa * math.log(L_safe)

    return {
        "kappa_non_outliers": kappa,
        "delta_nonzero_with_intercept": delta,
        "ls_avg_error_non_outliers": L,
        "aic_score": aic_score,
    }


def select_best_solution_by_aic(
    csv_path: str | Path,
    base_dir: str | Path = ".",
    output_scored_csv: str | Path | None = None,
    output_best_csv: str | Path | None = None,
    require_optimal: bool = False,
    coef_tol: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads a scalability CSV, computes the AIC-like criterion for each row,
    and selects the best solution among the 9 combinations for each train file.

    Returns:
        scored_df:
            original CSV plus AIC columns.

        best_df:
            one best row for each train_file, selected by minimum AIC.
    """
    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)

    valid_mask = df["error_message"].isna() | (df["error_message"].astype(str).str.strip() == "")

    valid_mask &= df["selected_coefficients"].notna()
    valid_mask &= df["detected_outliers"].notna()
    valid_mask &= df["estimated_z"].notna()

    if require_optimal:
        valid_mask &= df["solved_to_optimality"].astype(str).str.lower().eq("true")

    df["kappa_non_outliers"] = np.nan
    df["delta_nonzero_with_intercept"] = np.nan
    df["ls_avg_error_non_outliers"] = np.nan
    df["aic_score"] = np.nan

    for idx, row in df[valid_mask].iterrows():
        aic_data = compute_aic_for_csv_row(
            row=row,
            base_dir=base_dir,
            coef_tol=coef_tol,
        )

        df.loc[idx, "kappa_non_outliers"] = aic_data["kappa_non_outliers"]
        df.loc[idx, "delta_nonzero_with_intercept"] = aic_data["delta_nonzero_with_intercept"]
        df.loc[idx, "ls_avg_error_non_outliers"] = aic_data["ls_avg_error_non_outliers"]
        df.loc[idx, "aic_score"] = aic_data["aic_score"]

    scored_valid_df = df[valid_mask & df["aic_score"].notna()].copy()

    best_indices = (
        scored_valid_df
        .groupby("train_file")["aic_score"]
        .idxmin()
    )

    best_df = (
        scored_valid_df
        .loc[best_indices]
        .sort_values(["n_samples", "train_file"])
        .reset_index(drop=True)
    )

    if output_scored_csv is not None:
        output_scored_csv = Path(output_scored_csv)
        output_scored_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_scored_csv, index=False)

    if output_best_csv is not None:
        output_best_csv = Path(output_best_csv)
        output_best_csv.parent.mkdir(parents=True, exist_ok=True)
        best_df.to_csv(output_best_csv, index=False)

    return df, best_df


def infer_validation_file(train_file: str | Path, base_dir: str | Path = ".") -> Path:
    """
    Given a training file:

        test150322_100_50_01_2_0_5_-10_1.dat

    returns the corresponding validation/test file:

        test150322_100_50_01_2_0_5_-10_1Val.dat
    """
    base_dir = Path(base_dir)

    train_path = Path(train_file)
    if not train_path.is_absolute():
        train_path = base_dir / train_path

    val_path = train_path.with_name(train_path.stem + "Val.dat")

    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    return val_path


def feature_classification_metrics(
    true_features: np.ndarray,
    predicted_features: np.ndarray,
    n_features: int,
) -> dict[str, float | int]:
    """
    Computes FPR, FNR and F1 score for selected features.
    """
    true_set = set(int(j) for j in true_features)
    pred_set = set(int(j) for j in predicted_features)

    all_features = set(range(n_features))

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    tn = len(all_features - true_set - pred_set)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "feature_tp": tp,
        "feature_fp": fp,
        "feature_fn": fn,
        "feature_tn": tn,
        "feature_fpr": fpr,
        "feature_fnr": fnr,
        "feature_precision": precision,
        "feature_recall": recall,
        "feature_f1": f1,
    }


def evaluate_best_solution_on_validation(
    row: pd.Series,
    base_dir: str | Path = ".",
    coef_tol: float = 1e-6,
) -> dict[str, Any]:
    """
    Evaluates one selected best model on the corresponding Val.dat file.

    Computes:
        - RMSPE
        - MAE
        - feature FPR
        - feature FNR
        - feature F1
    """
    base_dir = Path(base_dir)

    train_file = Path(row["train_file"])
    val_path = infer_validation_file(train_file, base_dir=base_dir)

    A_val, r_val = read_dat_file(val_path)

    n_test, n_features = A_val.shape

    coeffs = parse_json_cell(row["selected_coefficients"], default={})

    w = np.zeros(n_features)

    for key, value in coeffs.items():
        j = int(key)
        if 0 <= j < n_features:
            w[j] = float(value)

    z = float(row["estimated_z"])

    predictions = A_val @ w + z
    errors = r_val - predictions

    mae = float(np.mean(np.abs(errors)))
    rmspe = float(np.sqrt(np.mean(errors**2)))

    true_features = parse_json_cell(row.get("true_features", None), default=[0, 1, 2, 3, 4])
    true_features = np.asarray(true_features, dtype=int)

    predicted_features = np.flatnonzero(np.abs(w) > coef_tol).astype(int)

    feature_metrics = feature_classification_metrics(
        true_features=true_features,
        predicted_features=predicted_features,
        n_features=n_features,
    )

    result = {
        "train_file": str(row["train_file"]),
        "validation_file": str(val_path),
        "corruption_scheme": row["corruption_scheme"],
        "n_samples": int(row["n_samples"]),
        "n_features": int(row["n_features"]),
        "d0": int(row["d0"]),
        "k0": int(row["k0"]),
        "k0_ratio": float(row["k0_ratio"]),
        "aic_score": float(row["aic_score"]),
        "rmspe": rmspe,
        "mae": mae,
        "true_features": json.dumps(true_features.tolist(), separators=(",", ":")),
        "predicted_features": json.dumps(predicted_features.tolist(), separators=(",", ":")),
    }

    result.update(feature_metrics)

    return result


def evaluate_best_models_and_average_by_size(
    best_df: pd.DataFrame,
    base_dir: str | Path = ".",
    output_evaluation_csv: str | Path | None = None,
    output_average_csv: str | Path | None = None,
    coef_tol: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates the best models selected by AIC.

    For each best solution:
        - opens the corresponding Val.dat file
        - computes RMSPE and MAE
        - computes feature FPR, FNR and F1

    Then averages the metrics over the three best solutions for each n_samples.
    """
    evaluation_rows: list[dict[str, Any]] = []

    for _, row in best_df.iterrows():
        evaluation_row = evaluate_best_solution_on_validation(
            row=row,
            base_dir=base_dir,
            coef_tol=coef_tol,
        )
        evaluation_rows.append(evaluation_row)

    evaluation_df = pd.DataFrame(evaluation_rows)

    average_df = (
        evaluation_df
        .groupby("n_samples", as_index=False)
        .agg(
            n_models=("train_file", "count"),
            mean_rmspe=("rmspe", "mean"),
            std_rmspe=("rmspe", "std"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_feature_fpr=("feature_fpr", "mean"),
            std_feature_fpr=("feature_fpr", "std"),
            mean_feature_fnr=("feature_fnr", "mean"),
            std_feature_fnr=("feature_fnr", "std"),
            mean_feature_precision=("feature_precision", "mean"),
            std_feature_precision=("feature_precision", "std"),
            mean_feature_recall=("feature_recall", "mean"),
            std_feature_recall=("feature_recall", "std"),
            mean_feature_f1=("feature_f1", "mean"),
            std_feature_f1=("feature_f1", "std"),
        )
        .sort_values("n_samples")
        .reset_index(drop=True)
    )

    if output_evaluation_csv is not None:
        output_evaluation_csv = Path(output_evaluation_csv)
        output_evaluation_csv.parent.mkdir(parents=True, exist_ok=True)
        evaluation_df.to_csv(output_evaluation_csv, index=False)

    if output_average_csv is not None:
        output_average_csv = Path(output_average_csv)
        output_average_csv.parent.mkdir(parents=True, exist_ok=True)
        average_df.to_csv(output_average_csv, index=False)

    return evaluation_df, average_df



if __name__ == "__main__":
        input_csv = "results/L/scalability_formulation_L_palm_vertical_ru_phi_2.csv"

        scored_df, best_df = select_best_solution_by_aic(
            csv_path=input_csv,
            base_dir=".",
            output_scored_csv="results/L/scalability_formulation_L_palm_vertical_ru_phi_2_scored.csv",
            output_best_csv="results/L/best_models_L_phi_2_by_aic.csv",
            require_optimal=False,
            coef_tol=1e-6,
        )

        evaluation_df, average_df = evaluate_best_models_and_average_by_size(
            best_df=best_df,
            base_dir=".",
            output_evaluation_csv="results/L/validation_best_models_L_phi_2.csv",
            output_average_csv="results/L/validation_metrics_by_size_L_phi_2.csv",
            coef_tol=1e-6,
        )

        print("\nBest models selected by AIC:")
        print(
            best_df[
                [
                    "train_file",
                    "n_samples",
                    "d0",
                    "k0",
                    "k0_ratio",
                    "aic_score",
                ]
            ]
        )

        print("\nValidation metrics for each selected model:")
        print(
            evaluation_df[
                [
                    "train_file",
                    "n_samples",
                    "d0",
                    "k0",
                    "k0_ratio",
                    "aic_score",
                    "rmspe",
                    "mae",
                    "feature_fpr",
                    "feature_fnr",
                    "feature_f1",
                ]
            ]
        )

        print("\nAverage validation metrics by instance size:")
        print(average_df)