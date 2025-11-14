#!/usr/bin/env python3
# ^ Use the system's Python 3 interpreter when run as an executable.

import argparse, csv, json, sys
from pathlib import Path
from datetime import datetime


def log(msg, verbose, log_file=None):
    """
    Write a log message to stdout (if verbose) and to an optional log file.

    Args:
        msg (str): The message to log.
        verbose (bool): If True, also print to the console.
        log_file (io.TextIOBase | None): If provided, write the message to this file.
    """
    # Ensure each message ends with a newline (avoid double newlines).
    line = msg if msg.endswith("\n") else msg + "\n"
    if verbose:
        # Print immediately; flush so progress is visible during long runs.
        print(line, end="", flush=True)
    if log_file:
        log_file.write(line)


def extract_csv_info(csv_path: Path):
    """
    Read a CSV and return (description_cell_below_header, headers_without_Description).

    Behavior:
      - Tries multiple encodings to robustly open diverse CSV files.
      - Finds the header row (first row).
      - Removes the 'Description' header (case-insensitive) from the header list.
      - If a 'Description' column exists, reads the first data row's value from that column.

    Returns:
        tuple[str, list[str]]: (description_text, headers_wo_desc)

    Raises:
        Exception: Re-raises the last encountered error if all encodings fail.
    """
    # Common encodings to try (BOM-friendly first).
    try_encodings = ("utf-8-sig", "utf-8", "cp1252")
    last_err = None

    for enc in try_encodings:
        try:
            # newline="" is recommended for csv module to handle newlines correctly.
            with csv_path.open("r", encoding=enc, newline="") as f:
                reader = csv.reader(f)

                # First row is assumed to be the header. If missing, treat as empty.
                header = next(reader, None)
                if not header:
                    return ("", [])

                # Build a case-insensitive index for header names.
                lower_index = {}
                for i, h in enumerate(header):
                    hn = (h or "").strip()  # normalize None -> "" and trim whitespace
                    hl = hn.lower()
                    if hl not in lower_index:
                        lower_index[hl] = i  # keep first occurrence

                # Locate 'Description' header (if present).
                desc_idx = lower_index.get("description")

                # Keep all headers except 'Description' (case-insensitive compare).
                headers_wo_desc = [
                    (h or "").strip()
                    for h in header
                    if (h or "").strip().lower() != "description"
                ]

                # Peek at the first data row to extract the description cell.
                first_data_row = next(reader, None)
                description = ""
                if (
                    desc_idx is not None
                    and first_data_row is not None
                    and desc_idx < len(first_data_row)
                ):
                    description = (first_data_row[desc_idx] or "").strip()

                return (description, headers_wo_desc)

        except Exception as e:
            # Save the last error and try the next encoding.
            last_err = e
            continue

    # If all encodings failed, bubble up the last error for diagnostics.
    raise last_err if last_err else RuntimeError("Unknown CSV read error")


def find_csvs(root: Path, recursive: bool):
    """
    Find .csv files under a directory, case-insensitively.

    Args:
        root (Path): Directory to search.
        recursive (bool): If True, search all subdirectories; else only the top level.

    Returns:
        list[Path]: Sorted list of matching .csv file paths.
    """
    # Use a glob pattern based on recursion preference.
    pattern = "**/*" if recursive else "*"

    # Filter to regular files with a .csv extension (case-insensitive).
    return sorted(
        p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() == ".csv"
    )


def main():
    """
    CLI entry point:
      - Parses args
      - Scans for CSVs
      - Extracts Description (first data row under Description header) + headers (excluding Description)
      - Writes a JSON inventory and a companion .log file
    """
    ap = argparse.ArgumentParser(
        description="Build JSON of CSV headers + first Description cell."
    )
    ap.add_argument("--input-dir", required=True, help="Folder to scan for .csv files")
    ap.add_argument("--output-dir", required=True, help="Folder to write JSON/log")
    ap.add_argument(
        "--out-name", required=True, help="Output JSON filename (with or without .json)"
    )
    ap.add_argument(
        "--recursive", action="store_true", help="Recurse into subdirectories"
    )
    ap.add_argument("--verbose", action="store_true", help="Print progress to stdout")
    args = ap.parse_args()

    # Normalize paths (expand ~, resolve symlinks).
    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    # Ensure JSON filename ends with .json.
    out_name = (
        args.out_name if args.out_name.lower().endswith(".json") else f"{args.out_name}.json"
    )
    out_path = out_dir / out_name

    # Log file shares the same stem as the JSON.
    log_path = out_dir / (out_path.stem + ".log")

    # Make sure output directory exists.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open the log file once and keep appending throughout the run.
    with log_path.open("w", encoding="utf-8") as lf:
        log(f"[start] {datetime.now().isoformat()}", True, lf)
        log(f"[args] input-dir={in_dir}", True, lf)
        log(f"[args] output-dir={out_dir}", True, lf)
        log(f"[args] out-name={out_name}", True, lf)
        log(f"[args] recursive={args.recursive}", True, lf)

        # Validate input directory early to avoid confusing errors later.
        if not in_dir.exists() or not in_dir.is_dir():
            log(f"[ERROR] Input directory not found or not a directory: {in_dir}", True, lf)
            sys.exit(1)

        # Discover candidate CSV files.
        csv_files = find_csvs(in_dir, args.recursive)
        log(f"[scan] Found {len(csv_files)} .csv file(s).", True, lf)
        for p in csv_files:
            log(f"  - {p}", True, lf)

        # inventory maps filename -> {"Description": str, "Columns": [str, ...]}
        inventory = {}
        # errors maps filename -> error string (if any file fails to parse)
        errors = {}

        # Process each CSV independently so one bad file doesn't stop the run.
        for csv_file in csv_files:
            try:
                desc, cols = extract_csv_info(csv_file)
                inventory[csv_file.name] = {"Description": desc, "Columns": cols}
                # Log a short preview of the description (or "(blank)").
                log(
                    f"[ok] {csv_file.name} -> {len(cols)} column(s), desc={'(blank)' if not desc else desc[:60]}",
                    True,
                    lf,
                )
            except Exception as e:
                # Collect failures and continue.
                errors[csv_file.name] = str(e)
                log(f"[fail] {csv_file.name}: {e}", True, lf)

        # Always write JSON (even if empty), so the user can confirm the output path.
        with out_path.open("w", encoding="utf-8") as jf:
            json.dump(inventory, jf, indent=2, ensure_ascii=False)

        log(f"[write] JSON -> {out_path}", True, lf)

        # Summarize any errors at the end (does not fail the run).
        if errors:
            log("[warn] Some files had errors:", True, lf)
            for k, v in errors.items():
                log(f"  - {k}: {v}", True, lf)

        log("[done] Finished.", True, lf)


if __name__ == "__main__":
    # Ensure stdout is line-buffered so logs appear promptly even if piped.
    sys.stdout.reconfigure(line_buffering=True)
    main()
