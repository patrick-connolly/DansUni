import csv
import sys
from pathlib import Path

# THIS IS AN AI GENERATED HELPER PYTHON SCRIPT TO FIX CSV FILES TO THEY WORK FOR OUR CSV READER!!!!

def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def clean_header_cell(cell: str) -> str:
    cell = cell.strip()
    if len(cell) >= 2 and cell[0] == '"' and cell[-1] == '"':
        cell = cell[1:-1]
    return cell.strip()


def main():
    if len(sys.argv) != 3:
        print("Usage: python fix_csv.py input.csv output.csv")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    with input_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError("Input CSV is empty.")

    header = [clean_header_cell(c) for c in rows[0]]
    data_rows = rows[1:]

    # Drop empty first header cell like: ,crim,zn,...
    if header and header[0] == "":
        header = header[1:]

    cleaned_rows = []
    expected_cols = len(header)

    for row_idx, row in enumerate(data_rows, start=2):
        row = [c.strip() for c in row]

        if not any(row):
            continue

        # Drop leading index/id column if present
        if len(row) == expected_cols + 1:
            row = row[1:]

        if len(row) != expected_cols:
            raise ValueError(
                f"Row {row_idx} has {len(row)} columns, expected {expected_cols}."
            )

        for col_idx, value in enumerate(row, start=1):
            if not is_number(value):
                raise ValueError(
                    f"Row {row_idx}, column {col_idx} is not numeric: {value!r}"
                )

        cleaned_rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(cleaned_rows)

    print(f"Cleaned CSV written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())