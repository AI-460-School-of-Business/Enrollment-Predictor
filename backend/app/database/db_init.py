"""
Database initialization script for the Docker container.

FLOW:
1. Wait for PostgreSQL to be ready.
2. Check if the database already has user tables.
3. If no tables exist, import CSV files.
4. Generate SQL dump files for future use.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parent
BACKEND_ROOT = APP_DIR.parent

# Add the backend/app directory to Python path so we can import our modules
sys.path.insert(0, str(APP_DIR))

DATA_DIR_CANDIDATES = [
    Path("/app/data"),
    BACKEND_ROOT / "data",
]

DATABASE_DIR = APP_DIR / "database"
READ_CSV_SCRIPT = DATABASE_DIR / "read_csv.py"
GENERATE_FEATURES_SCRIPT = DATABASE_DIR / "generate_auto_features.py"


def _find_data_subdir(subdir_name: str):
    """Return the first existing data subdir, checking both /app/data and backend/data."""
    for base in DATA_DIR_CANDIDATES:
        candidate = base / subdir_name
        if candidate.exists():
            return candidate
    return None


def _ensure_data_subdir(subdir_name: str) -> Path:
    """Return a writable data subdir path, creating the /app/data structure if needed."""
    existing = _find_data_subdir(subdir_name)
    if existing is not None:
        return existing

    base = DATA_DIR_CANDIDATES[0]
    target = base / subdir_name
    target.mkdir(parents=True, exist_ok=True)
    return target

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("psycopg2 not available, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
    import psycopg2
    import psycopg2.extras

# Database configuration from environment
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "db"),  # Use 'db' service name in Docker
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "database": os.getenv("POSTGRES_DB", "postgres"),
}

def wait_for_postgres(max_attempts=30):
    """Wait for PostgreSQL to be ready."""
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.close()
            print("PostgreSQL is ready!")
            return True
        except psycopg2.Error:
            print(f"Waiting for PostgreSQL... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(2)
    
    print("Failed to connect to PostgreSQL after maximum attempts")
    return False

def check_existing_data():
    """Check if database already has user tables."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        count = cur.fetchone()[0]
        conn.close()
        return count > 0
    except Exception as e:
        print(f"Error checking existing data: {e}")
        return False

def check_sql_files():
    """Check if SQL files exist in data/sql directory."""
    sql_dir = _find_data_subdir("sql")
    if not sql_dir:
        return []

    return list(sql_dir.glob("*.sql"))

def check_csv_files():
    """Check if CSV files exist in data/csv directory."""
    csv_dir = _find_data_subdir("csv")
    if not csv_dir:
        return []

    return list(csv_dir.glob("*.csv"))

def run_sql_files(sql_files):
    """Execute SQL files in the database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        for sql_file in sorted(sql_files):
            print(f"Executing SQL file: {sql_file}")
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
                
            # Split by semicolon and execute each statement
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            for statement in statements:
                cur.execute(statement)
            
            conn.commit()
            print(f"Successfully executed: {sql_file}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error executing SQL files: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

def run_csv_import_with_sql_export():
    """Run CSV import and also generate SQL dumps for future use."""
    try:
        if not READ_CSV_SCRIPT.exists():
            print(f"read_csv.py not found at {READ_CSV_SCRIPT}")
            return False
        
        # First, generate SQL dumps from CSV files
        print("Generating SQL dumps from CSV files...")
        export_result = subprocess.run([
            sys.executable, 
            str(READ_CSV_SCRIPT), 
            "--export-sql"
        ], capture_output=True, text=True)
        
        if export_result.returncode == 0:
            print("SQL dumps generated successfully!")
            print(export_result.stdout)
            
            # Now check if we have generated SQL files and use them instead
            sql_files = check_sql_files()
            if sql_files:
                print("Using generated SQL files for faster import...")
                return run_sql_files(sql_files)
        else:
            print("SQL export failed, falling back to direct CSV import...")
            print("Export STDERR:", export_result.stderr)
        
        # Fallback: Run the direct CSV importer
        print("Running direct CSV import...")
        result = subprocess.run([
            sys.executable, 
            str(READ_CSV_SCRIPT), 
            "--import-all"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("CSV import completed successfully!")
            print(result.stdout)
            
            # After successful import, generate feature schema
            generate_feature_schema()
            
            return True
        else:
            print("CSV import failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"Error running CSV import: {e}")
        return False

def generate_feature_schema():
    """Generate auto feature schema from imported data."""
    try:
        if not GENERATE_FEATURES_SCRIPT.exists():
            print(f"generate_auto_features.py not found at {GENERATE_FEATURES_SCRIPT}")
            return

        print("Generating feature schema from database...")
        result = subprocess.run([
            sys.executable, 
            str(GENERATE_FEATURES_SCRIPT)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Feature schema generated successfully!")
            print(result.stdout)
        else:
            print("Feature schema generation failed!")
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error generating feature schema: {e}")

def export_data_on_shutdown():
    """Export SQL dumps and feature schema on container shutdown."""
    try:
        print("Exporting data before shutdown...")
        
        # Export SQL dumps
        export_result = subprocess.run([
            sys.executable, 
            str(READ_CSV_SCRIPT), 
            "--export-sql"
        ], capture_output=True, text=True)
        
        if export_result.returncode == 0:
            print("SQL export completed!")
        else:
            print("SQL export failed:", export_result.stderr)
        
        # Generate/update feature schema
        generate_feature_schema()
        
        print("Data export completed!")
        
    except Exception as e:
        print(f"Error during shutdown export: {e}")

def main():
    print("Starting database initialization...")
    
    # Wait for PostgreSQL to be ready
    if not wait_for_postgres():
        sys.exit(1)
    
    # Check if we already have data
    if check_existing_data():
        print("Database already contains tables. Skipping import.")
        return
    
    # Check for SQL files first (preferred method)
    sql_files = check_sql_files()
    if sql_files:
        sql_dir = _find_data_subdir("sql") or Path("/app/data/sql")
        print(f"Found {len(sql_files)} SQL files in {sql_dir}:")
        for sql_file in sql_files:
            print(f"  - {sql_file}")
        
        print("Executing SQL files...")
        if run_sql_files(sql_files):
            print("Database initialization completed successfully from SQL files!")
            return
        else:
            print("SQL file execution failed, falling back to CSV import...")
    
    # Fallback to CSV import
    csv_files = check_csv_files()
    if not csv_files:
        sql_dir = _find_data_subdir("sql") or Path("/app/data/sql")
        csv_dir = _find_data_subdir("csv") or Path("/app/data/csv")
        print(f"No SQL files in {sql_dir} and no CSV files found in {csv_dir}. Skipping import.")
        return
    
    csv_dir = _find_data_subdir("csv") or Path("/app/data/csv")
    print(f"Found {len(csv_files)} CSV files in {csv_dir}:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    # Run CSV import with SQL export
    print("No SQL files found. Importing from CSV data and generating SQL dumps...")
    if run_csv_import_with_sql_export():
        print("Database initialization completed successfully from CSV files!")
    else:
        print("Database initialization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
