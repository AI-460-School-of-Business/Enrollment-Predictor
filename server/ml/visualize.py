import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import textwrap
from pathlib import Path

# visualize.py
# ----------------
# Simple helper script to query the project's PostgreSQL database and produce
# a PNG showing enrollment totals over time. This file is intentionally
# lightweight: it reads SQL from the built-in `DEFAULT_QUERY` or an inline
# query provided with `--data-query`, fetches data using psycopg2/pandas,
# and writes a single PNG to `server/ml/plots/enrollment_over_time.png`.
#
# Design notes:
# - We only support the project's single data source (configured via
#   environment variables) and produce a single plot file. No interactive
#   prompts or additional I/O are included to keep automation simple.
# - Keeping the default SQL in-code makes the script portable in CI /
#   container environments where relative file paths may be inconvenient.


# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "database": os.getenv("POSTGRES_DB", "postgres"),
}

DEFAULT_QUERY = textwrap.dedent(
    """
    -- Default enrollment over time query
    -- Adjust the WHERE clause or add additional columns as needed.

    SELECT
        term,
        term_desc,
        SUM(COALESCE(act, 0)) AS total_enrollment
    FROM section_detail_report_sbussection_detail_report_sbus
    WHERE (term / 100) >= 2023
    GROUP BY term, term_desc
    ORDER BY term;
    """
).strip()

def fetch_enrollment_data(query: str) -> pd.DataFrame:
    """Execute the provided SQL query against the configured PostgreSQL
    database and return the results as a pandas DataFrame.

    Inputs:
      - query: SQL string expected to be a SELECT that returns the columns
        needed by the plotting function (e.g. 'term' and one or more numeric
        enrollment columns).

    Output:
      - pandas.DataFrame with the query results. This function will raise
        any underlying exceptions from psycopg2/pandas if the query fails.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df

def plot_enrollment_over_time(df: pd.DataFrame, output_path: Path = None):
    """Create and save a simple line plot of enrollment totals over time.

    Behavior:
      - Accepts a DataFrame that must include either a `term` column or a
        `year` column. Rows are grouped by that column and numeric columns
        are summed to produce the plotted series.
      - Writes a PNG to `output_path` if provided. The function ensures the
        parent directory exists before saving.

    Inputs:
      - df: pandas.DataFrame containing the query results.
      - output_path: optional Path or string where the PNG will be saved.
    """
    # Determine x-axis column and prepare grouped numeric series
    if "term" in df.columns:
        df_grouped = df.groupby("term").sum(numeric_only=True).sort_index()
        x_vals = df_grouped.index
        x_label = "Term"
    elif "year" in df.columns:
        df_grouped = df.groupby("year").sum(numeric_only=True).sort_index()
        x_vals = df_grouped.index
        x_label = "Year"
    else:
        # Fail fast with a clear message for callers/automation
        print("Error: query result must include a 'term' or 'year' column.")
        sys.exit(1)

    # Select numeric columns to plot; avoid non-numeric metadata
    numeric_cols = [col for col in df_grouped.columns if df_grouped[col].dtype.kind in {"i", "u", "f"}]
    if not numeric_cols:
        print("Error: no numeric enrollment columns found to plot.")
        sys.exit(1)

    # Build the matplotlib figure
    plt.figure(figsize=(10, 6))
    for col in numeric_cols:
        plt.plot(x_vals, df_grouped[col], marker="o", label=col)

    plt.xlabel(x_label)
    plt.ylabel("Enrollment Count")
    plt.title("Enrollment Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save to disk if a path was provided (ensures folder exists)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")

    # Close the figure to free memory in automated runs
    plt.close()

def main():
    """Parse CLI args, select a query, fetch data, and write the plot.

    The CLI is intentionally minimal: only an inline SQL string is accepted.
    If the inline query is empty or omitted the built-in DEFAULT_QUERY is
    used. The generated PNG is written to the project's `server/ml/plots`
    folder to keep behavior simple and deterministic for automation.
    """
    parser = argparse.ArgumentParser(description="Visualize enrollment data from the database")
    parser.add_argument("--data-query", type=str,
                        help="Inline SQL query to retrieve data")
    args = parser.parse_args()

    inline_query = (args.data_query or "").strip()
    if inline_query:
        custom_query = inline_query
        print("Using custom query provided via --data-query")
    else:
        custom_query = DEFAULT_QUERY
        if args.data_query is not None:
            print("Received empty inline query. Using built-in default query.")
        else:
            print("Using built-in default query.")

    # Print a short preview of the SQL that will be executed so users can
    # quickly verify the query without opening the code. We limit the
    # preview to a few lines for brevity.
    preview_lines = custom_query.strip().splitlines()
    preview_display = preview_lines[:5]
    if len(preview_lines) > 5:
        preview_display.append("...")
    print("Query preview:")
    print(textwrap.indent("\n".join(preview_display), "    "))
    print()

    # Execute the query and produce the PNG. Any database or SQL errors
    # will surface here (intentionally not swallowed) so CI or calling
    # automation can detect failures.
    df = fetch_enrollment_data(custom_query)
    print(f"Fetched {df.shape[0]} rows.")
    output_path = Path(__file__).parent / "plots" / "enrollment_over_time.png"
    plot_enrollment_over_time(df, output_path=output_path)

if __name__ == "__main__":
    main()