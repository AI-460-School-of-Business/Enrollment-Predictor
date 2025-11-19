#!/usr/bin/env python3
"""
Data Export Script

Exports SQL dumps and feature schemas when called.
This can be run manually or on container shutdown.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the server directory to Python path
sys.path.insert(0, '/app/server')

def export_sql_dumps():
    """Export database to SQL files."""
    try:
        print("Exporting SQL dumps...")
        result = subprocess.run([
            sys.executable, 
            "server/database/read_csv.py", 
            "--export-sql"
        ], capture_output=True, text=True, cwd="/app")
        
        if result.returncode == 0:
            print("SQL dumps exported successfully!")
            return True
        else:
            print("SQL export failed:", result.stderr)
            return False
    
    except Exception as e:
        print(f"Error exporting SQL: {e}")
        return False

def export_feature_schema():
    """Export auto-generated feature schema."""
    try:
        print("Generating and exporting feature schema...")
        result = subprocess.run([
            sys.executable, 
            "server/database/generate_auto_features.py"
        ], capture_output=True, text=True, cwd="/app")
        
        if result.returncode == 0:
            print("Feature schema exported successfully!")
            
            # Copy the auto-generated schema to the feature_schema directory
            src_path = Path("/app/server/database/enrollment_features_auto.json")
            dest_path = Path("/app/server/ml/feature_schema/enrollment_features_auto.json")
            
            if src_path.exists():
                import shutil
                shutil.copy2(str(src_path), str(dest_path))
                print(f"Feature schema copied to: {dest_path}")
            
            return True
        else:
            print("Feature schema generation failed:", result.stderr)
            return False
    
    except Exception as e:
        print(f"Error generating feature schema: {e}")
        return False

def main():
    """Main export function."""
    print("=== Data Export Script ===")
    
    # Change to app directory
    os.chdir("/app")
    
    success = True
    
    # Export SQL dumps
    if not export_sql_dumps():
        success = False
    
    # Export feature schema
    if not export_feature_schema():
        success = False
    
    if success:
        print("\n=== Export Complete ===")
        print("All data exported successfully!")
    else:
        print("\n=== Export Failed ===")
        print("Some exports failed. Check logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()