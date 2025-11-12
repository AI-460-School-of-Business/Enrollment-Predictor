"""
Data Validation for Enrollment Predictor

Validates data integrity in the enrollment database, including:
- Checking for subj+crse combinations that have multiple different titles
- Ensuring data consistency for model training

Usage: python server/database/data_validation.py
"""

import os
import sys
import psycopg2
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "database": os.getenv("POSTGRES_DB", "postgres"),
}


def connect_to_db():
    """Establish a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)


def check_subj_crse_title_consistency(conn) -> Tuple[bool, List[Dict]]:
    """
    Check if any subj+crse combination has multiple different titles.
    
    This validation ensures that a course identifier (e.g., BUS 250) always
    has the same title throughout the dataset. If a subj+crse combination has
    multiple titles, it may indicate:
    - Data entry errors
    - Course title changes over time (requires manual review)
    - Duplicate course codes used for different courses (data quality issue)
    
    Returns:
        Tuple of (is_valid, inconsistencies_list)
        - is_valid: True if all subj+crse combinations have consistent titles
        - inconsistencies_list: List of dictionaries containing inconsistent records
    """
    query = """
    SELECT 
        subj,
        crse,
        COUNT(DISTINCT title) as title_count,
        ARRAY_AGG(DISTINCT title) as titles
    FROM section_detail_report_sbussection_detail_report_sbus
    WHERE subj IS NOT NULL 
      AND crse IS NOT NULL 
      AND title IS NOT NULL
    GROUP BY subj, crse
    HAVING COUNT(DISTINCT title) > 1
    ORDER BY subj, crse;
    """
    
    try:
        df = pd.read_sql(query, conn)
        
        if df.empty:
            print("✓ PASS: All subj+crse combinations have consistent titles.")
            return True, []
        else:
            print(f"✗ FAIL: Found {len(df)} subj+crse combinations with multiple titles:\n")
            inconsistencies = []
            for _, row in df.iterrows():
                inconsistency = {
                    'subj': row['subj'],
                    'crse': row['crse'],
                    'title_count': row['title_count'],
                    'titles': row['titles']
                }
                inconsistencies.append(inconsistency)
                print(f"  {row['subj']} {row['crse']}: {row['title_count']} different titles")
                for title in row['titles']:
                    print(f"    - {title}")
                print()
            
            return False, inconsistencies
    except Exception as e:
        print(f"Error running validation query: {e}")
        sys.exit(1)


def check_crn_uniqueness_per_term(conn) -> Tuple[bool, List[Dict]]:
    """
    Check if CRNs are unique within each term.
    
    CRNs should be unique identifiers for course sections within a given term.
    This validation ensures no duplicate CRNs exist in the same term.
    
    Returns:
        Tuple of (is_valid, duplicates_list)
    """
    query = """
    SELECT 
        term,
        crn,
        COUNT(*) as occurrence_count
    FROM section_detail_report_sbussection_detail_report_sbus
    WHERE crn IS NOT NULL AND term IS NOT NULL
    GROUP BY term, crn
    HAVING COUNT(*) > 1
    ORDER BY term, crn;
    """
    
    try:
        df = pd.read_sql(query, conn)
        
        if df.empty:
            print("✓ PASS: All CRNs are unique within their terms.")
            return True, []
        else:
            print(f"✗ FAIL: Found {len(df)} CRN duplicates within terms:\n")
            duplicates = []
            for _, row in df.iterrows():
                duplicate = {
                    'term': row['term'],
                    'crn': row['crn'],
                    'count': row['occurrence_count']
                }
                duplicates.append(duplicate)
                print(f"  Term {row['term']}, CRN {row['crn']}: appears {row['occurrence_count']} times")
            print()
            
            return False, duplicates
    except Exception as e:
        print(f"Error running CRN uniqueness query: {e}")
        sys.exit(1)


def main():
    """Run all data validation checks."""
    print("=" * 70)
    print("ENROLLMENT DATA VALIDATION")
    print("=" * 70)
    print()
    
    conn = connect_to_db()
    
    all_valid = True
    
    # Check 1: Subject + Course consistency
    print("Check 1: Subject + Course Title Consistency")
    print("-" * 70)
    title_valid, title_issues = check_subj_crse_title_consistency(conn)
    all_valid = all_valid and title_valid
    print()
    
    # Check 2: CRN uniqueness per term
    print("Check 2: CRN Uniqueness Per Term")
    print("-" * 70)
    crn_valid, crn_issues = check_crn_uniqueness_per_term(conn)
    all_valid = all_valid and crn_valid
    print()
    
    # Summary
    print("=" * 70)
    if all_valid:
        print("✓ ALL VALIDATION CHECKS PASSED")
        print("=" * 70)
        sys.exit(0)
    else:
        print("✗ VALIDATION FAILED - Please review the issues above")
        print("=" * 70)
        sys.exit(1)
    
    conn.close()


if __name__ == "__main__":
    main()
