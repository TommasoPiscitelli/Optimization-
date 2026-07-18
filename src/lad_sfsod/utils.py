import csv
from pathlib import Path
import math



def append_csv_row(
    output_csv: Path,
    row: dict,
    fieldnames: list[str],
) -> None:
    """
    Appends one row to a CSV file.

    If the file does not exist yet, or is empty, the header is written first.
    Missing fields are written as empty strings.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    write_header = not output_csv.exists() or output_csv.stat().st_size == 0

    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                key: row.get(key, "")
                for key in fieldnames
            }
        )


def parse_int_list(text: str) -> list[int]:
    """Parses comma-separated integers, e.g. '5,7,9'."""
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_float_list(text: str) -> list[float]:
    """Parses comma-separated floats, e.g. '0.10,0.15,0.20'."""
    values = []

    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(float(token))

    if not values:
        raise ValueError("Expected at least one float value.")

    return values


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))

