from dataclasses import dataclass
import numpy as np


@dataclass
class SFSODInstance:
    A: np.ndarray
    r: np.ndarray
    true_w: np.ndarray
    true_z: float
    true_relevant_features: np.ndarray
    true_outliers: np.ndarray



from pathlib import Path
import ast
import pandas as pd


def _extract_top_level_bracket_blocks(text: str) -> list[str]:
    """
    Extracts top-level bracket expressions from a .dat file.

    Example:
        [[1, 2], [3, 4]]
        [10, 20]

    returns two strings:
        '[[1, 2], [3, 4]]'
        '[10, 20]'
    """
    blocks = []
    level = 0
    start = None

    for pos, ch in enumerate(text):
        if ch == "[":
            if level == 0:
                start = pos
            level += 1
        elif ch == "]":
            level -= 1
            if level == 0 and start is not None:
                blocks.append(text[start : pos + 1])
                start = None

    return blocks


def read_dat_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads a .dat instance from the paper repository.

    The file contains:
    - first bracket block: design matrix A
    - second bracket block: response vector r

    Returns:
        A: shape (n_samples, n_features)
        r: shape (n_samples,)
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        text = f.read()

    text = text.replace(";", "")
    blocks = _extract_top_level_bracket_blocks(text)

    if len(blocks) < 2:
        raise ValueError(
            f"Could not find both design matrix and response vector in {path}."
        )

    A = np.array(ast.literal_eval(blocks[0]), dtype=float)
    r = np.array(ast.literal_eval(blocks[1]), dtype=float)

    if A.ndim != 2:
        raise ValueError(f"Design matrix in {path} is not 2-dimensional.")

    if r.ndim != 1:
        raise ValueError(f"Response vector in {path} is not 1-dimensional.")

    if A.shape[0] != r.shape[0]:
        raise ValueError(
            f"Inconsistent shapes in {path}: A has {A.shape[0]} rows, "
            f"but r has length {r.shape[0]}."
        )

    return A, r


def read_outlier_file(path: str | Path, n_samples: int | None = None) -> np.ndarray:
    """
    Reads an Outlier.csv file from the paper repository.

    The file has two columns:
    - row index
    - 1 if outlier, 0 otherwise

    It may contain a header row such as:
        index,outlier

    Returns zero-based indices of true outlier rows.
    """
    path = Path(path)

    df = pd.read_csv(path, header=None, sep=None, engine="python")

    if df.shape[1] < 2:
        raise ValueError(f"Outlier file {path} should have at least 2 columns.")

    # Convert columns to numeric without assigning back into the original df.
    row_indices_numeric = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    labels_numeric = pd.to_numeric(df.iloc[:, 1], errors="coerce")

    # Keep only rows where both columns are numeric.
    valid_mask = row_indices_numeric.notna() & labels_numeric.notna()

    row_indices = row_indices_numeric[valid_mask].to_numpy(dtype=int)
    labels = labels_numeric[valid_mask].to_numpy(dtype=int)

    outlier_rows = row_indices[labels == 1]

    # The repository may use either 0-based or 1-based row indices.
    # We detect the convention and always return 0-based indices.
    if n_samples is not None and len(row_indices) > 0:
        min_idx = int(row_indices.min())
        max_idx = int(row_indices.max())

        if min_idx == 1 and max_idx == n_samples:
            outlier_rows = outlier_rows - 1
        elif min_idx == 0 and max_idx == n_samples - 1:
            pass
        else:
            print(
                f"Warning: could not infer row-index convention in {path}. "
                "Assuming indices are already zero-based."
            )

    return np.sort(outlier_rows.astype(int))


def infer_outlier_path(dat_path: str | Path) -> Path:
    """
    Given a training .dat path, returns the expected Outlier.csv path.

    Example:
        test150322_100_..._1.dat
    becomes:
        test150322_100_..._1Outlier.csv
    """
    dat_path = Path(dat_path)

    if dat_path.name.endswith("Val.dat"):
        raise ValueError(
            "Validation files do not have corresponding Outlier.csv files. "
            "Use a training .dat file instead."
        )

    return dat_path.with_name(dat_path.stem + "Outlier.csv")


def find_first_training_dat_file(
    data_dir: str | Path,
    date_prefix: str | None = None,
    n_samples: int | None = None,
) -> Path:
    """
    Finds the first training .dat file in data_dir.

    Excludes validation files ending with Val.dat.

    Optional filters:
    - date_prefix='150322' for vertical outliers
    - date_prefix='210322' for bad leverage points
    - n_samples=100, 150, or 200
    """
    data_dir = Path(data_dir)

    candidates = sorted(data_dir.rglob("*.dat"))
    candidates = [p for p in candidates if not p.name.endswith("Val.dat")]

    if date_prefix is not None:
        candidates = [
            p for p in candidates
            if p.name.startswith(f"test{date_prefix}_")
        ]

    if n_samples is not None:
        candidates = [
            p for p in candidates
            if f"_{n_samples}_" in p.name
        ]

    if not candidates:
        raise FileNotFoundError(
            f"No training .dat file found in {data_dir} "
            f"with date_prefix={date_prefix}, n_samples={n_samples}."
        )

    return candidates[0]