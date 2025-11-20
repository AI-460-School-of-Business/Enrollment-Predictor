"""
CSV Reader and Database Importer

This script reads CSV files and imports them into PostgreSQL database.
"""

import os
import sys
import pandas as pd
import psycopg2
import psycopg2.extras
from pathlib import Path
import argparse

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "db"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "database": os.getenv("POSTGRES_DB", "postgres"),
}

def get_db_connection():
    """Create database connection."""
    return psycopg2.connect(**DB_CONFIG)

def clean_column_name(col_name):
    """Clean column names for PostgreSQL compatibility."""
    # Remove special characters and spaces, convert to lowercase
    cleaned = str(col_name).lower()
    cleaned = cleaned.replace(' ', '_')
    cleaned = cleaned.replace('(', '').replace(')', '')
    cleaned = cleaned.replace('-', '_').replace('.', '_')
    cleaned = cleaned.replace('/', '_').replace('\\', '_')
    # Remove multiple underscores
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    cleaned = cleaned.strip('_')
    return cleaned

def infer_sql_type(series):
    """Infer SQL data type from pandas series."""
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(series):
        return "NUMERIC"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    else:
        # For text, determine max length
        max_len = series.astype(str).str.len().max()
        if max_len <= 255:
            return "VARCHAR(255)"
        else:
            return "TEXT"

def import_csv_file(csv_path, table_name=None):
    """Import a single CSV file into the database."""
    print(f"Importing CSV file: {csv_path}")
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Read {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False
    
    if len(df) == 0:
        print("CSV file is empty")
        return False
    
    # Generate table name if not provided
    if not table_name:
        table_name = clean_column_name(Path(csv_path).stem)
    
    print(f"Creating table: {table_name}")
    
    # Clean column names
    original_columns = df.columns.tolist()
    cleaned_columns = [clean_column_name(col) for col in original_columns]
    df.columns = cleaned_columns
    
    print(f"Cleaned columns: {cleaned_columns[:5]}...")
    
    # Connect to database
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Drop table if exists
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create table schema
        column_definitions = []
        for col in df.columns:
            sql_type = infer_sql_type(df[col])
            column_definitions.append(f"{col} {sql_type}")
        
        create_table_sql = f"""
            CREATE TABLE {table_name} (
                {', '.join(column_definitions)}
            )
        """
        
        print("Creating table with schema:")
        print(create_table_sql)
        cur.execute(create_table_sql)
        
        # Insert data
        print("Inserting data...")
        
        # Prepare data for insertion
        data_tuples = []
        for _, row in df.iterrows():
            # Convert NaN to None for SQL NULL
            row_data = []
            for val in row:
                if pd.isna(val):
                    row_data.append(None)
                else:
                    row_data.append(val)
            data_tuples.append(tuple(row_data))
        
        # Build insert statement
        placeholders = ', '.join(['%s'] * len(df.columns))
        insert_sql = f"""
            INSERT INTO {table_name} ({', '.join(df.columns)})
            VALUES ({placeholders})
        """
        
        # Execute batch insert
        cur.executemany(insert_sql, data_tuples)
        
        conn.commit()
        print(f"Successfully imported {len(df)} rows into table '{table_name}'")
        
        # Show sample data
        cur.execute(f"SELECT * FROM {table_name} LIMIT 3")
        sample_rows = cur.fetchall()
        print(f"Sample data from {table_name}:")
        for i, row in enumerate(sample_rows):
            print(f"  Row {i+1}: {row[:3]}...")  # Show first 3 columns
        
        return True
        
    except Exception as e:
        print(f"Error importing CSV: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()

def import_all_csv_files():
    """Import all CSV files from the data/csv directory."""
    csv_dir = Path("/app/data/csv")
    if not csv_dir.exists():
        print("CSV directory not found")
        return False
    
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found")
        return False
    
    print(f"Found {len(csv_files)} CSV files")
    
    success_count = 0
    for csv_file in csv_files:
        if import_csv_file(str(csv_file)):
            success_count += 1
    
    print(f"Successfully imported {success_count}/{len(csv_files)} files")
    return success_count > 0

def export_sql_dumps():
    """Export all database tables to SQL files."""
    print("Exporting database tables to SQL files...")
    
    conn = get_db_connection()
    try:
        # Get all tables
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        
        tables = [row[0] for row in cur.fetchall()]
        if not tables:
            print("No tables found to export")
            return False
        
        # Create SQL export directory
        sql_dir = Path("/app/data/sql")
        sql_dir.mkdir(exist_ok=True)
        
        for table_name in tables:
            print(f"Exporting table: {table_name}")
            
            # Get table structure
            cur.execute(f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = '{table_name}'
                ORDER BY ordinal_position;
            """)
            
            columns = cur.fetchall()
            
            # Generate CREATE TABLE statement
            create_sql = f"-- Table: {table_name}\n"
            create_sql += f"DROP TABLE IF EXISTS {table_name};\n"
            create_sql += f"CREATE TABLE {table_name} (\n"
            
            column_defs = []
            for col_name, data_type, nullable, default in columns:
                col_def = f"    {col_name} {data_type.upper()}"
                if nullable == 'NO':
                    col_def += " NOT NULL"
                if default:
                    col_def += f" DEFAULT {default}"
                column_defs.append(col_def)
            
            create_sql += ",\n".join(column_defs)
            create_sql += "\n);\n\n"
            
            # Get table data
            cur.execute(f"SELECT * FROM {table_name}")
            rows = cur.fetchall()
            
            if rows:
                # Generate INSERT statements
                column_names = [col[0] for col in columns]
                create_sql += f"-- Data for table: {table_name}\n"
                
                for row in rows:
                    values = []
                    for val in row:
                        if val is None:
                            values.append("NULL")
                        elif isinstance(val, str):
                            # Escape single quotes
                            escaped_val = val.replace("'", "''")
                            values.append(f"'{escaped_val}'")
                        else:
                            values.append(str(val))
                    
                    insert_sql = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({', '.join(values)});\n"
                    create_sql += insert_sql
            
            # Write to file
            sql_file = sql_dir / f"{table_name}.sql"
            with open(sql_file, 'w', encoding='utf-8') as f:
                f.write(create_sql)
            
            print(f"Exported {len(rows)} rows to: {sql_file}")
        
        print(f"Successfully exported {len(tables)} tables to SQL files")
        return True
        
    except Exception as e:
        print(f"Error exporting SQL: {e}")
        return False
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Import CSV files to PostgreSQL")
    parser.add_argument("--import-all", action="store_true", help="Import all CSV files")
    parser.add_argument("--export-sql", action="store_true", help="Export SQL dumps")
    parser.add_argument("--file", help="Import specific CSV file")
    
    args = parser.parse_args()
    
    if args.export_sql:
        export_sql_dumps()
        return
    
    if args.import_all:
        import_all_csv_files()
    elif args.file:
        import_csv_file(args.file)
    else:
        print("Please specify --import-all or --file <filename>")

if __name__ == "__main__":
    main()