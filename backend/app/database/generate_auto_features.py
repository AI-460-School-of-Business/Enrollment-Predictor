"""
Database Schema Generator for Enrollment Prediction

This script connects to the PostgreSQL database and generates a simple schema
showing all tables and their columns from the imported CSV data.
Creates the enrollment_features_auto.json file from database analysis.

Usage:
    python server/database/generate_auto_features.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("Error: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "database": os.getenv("POSTGRES_DB", "postgres"),
}


def get_db_connection():
    """Create and return a database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Database connection failed: {e}")
        raise


def get_all_tables():
    """Get all user-created tables (excluding system tables)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        tables = [row['table_name'] for row in cur.fetchall()]
        return tables
    finally:
        conn.close()


def get_table_columns(table_name: str) -> List[str]:
    """Get column names for a specific table."""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = %s
            ORDER BY ordinal_position;
        """, (table_name,))
        
        columns = [row['column_name'] for row in cur.fetchall()]
        return columns
    finally:
        conn.close()


def generate_database_schema():
    """Generate a simple schema from database tables."""
    print("Connecting to database and reading table structure...")
    
    # Get all tables
    tables = get_all_tables()
    if not tables:
        print("No tables found in database. Make sure CSV data has been imported.")
        return None
    
    print(f"Found {len(tables)} tables")
    
    # Initialize schema in the format requested
    schema = {}
    
    total_columns = 0
    # Get columns for each table
    for table_name in tables:
        print(f"Reading table: {table_name}")
        columns = get_table_columns(table_name)
        
        # Create the table structure with empty column values
        table_schema = {}
        for column in columns:
            table_schema[column] = ""
        
        schema[table_name] = table_schema
        total_columns += len(columns)
        
        print(f"  {table_name}: {len(columns)} columns")
    
    print(f"\nTotal: {len(tables)} tables, {total_columns} columns")
    return schema


def save_schema(schema: Dict[str, Any], output_path: str = None):
    """Save the generated schema to a JSON file."""
    if output_path is None:
        # Save to the same directory as this script
        script_dir = Path(__file__).parent
        output_path = script_dir / "enrollment_features_auto.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    
    print(f"Schema saved to: {output_path}")


def main():
    """Main execution function."""
    print("=== Database Schema Generator ===")
    
    try:
        # Generate schema from database tables
        schema = generate_database_schema()
        
        if schema is None:
            print("Failed to generate schema - no tables found.")
            sys.exit(1)
        
        # Save schema
        save_schema(schema)
        
        # Print summary
        print(f"\n=== Generation Complete ===")
        
        print(f"\nTables and columns:")
        for table_name, columns in schema.items():
            print(f"  {table_name}: {list(columns.keys())}")
        
        print(f"\nSchema file: enrollment_features_auto.json")
        print("This schema shows all tables and columns from your database.")
        
    except Exception as e:
        print(f"Error generating schema: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()