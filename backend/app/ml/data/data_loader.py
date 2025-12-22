"""Database extraction and table joining logic."""
from typing import Dict, List, Any, Optional
import pandas as pd
import psycopg2
import psycopg2.extras
from ml.utils.db_config import DB_CONFIG


class DataLoader:
    """Extract training data from database."""
    
    def __init__(self, custom_query: Optional[str] = None):
        self.custom_query = custom_query
    
    def get_db_connection(self):
        """Create database connection."""
        try:
            return psycopg2.connect(**DB_CONFIG)
        except psycopg2.Error as e:
            print(f"Database connection failed: {e}")
            raise
    
    def extract_training_data(self) -> pd.DataFrame:
        """Extract training data from database and join related tables."""
        print("Extracting training data...")
        
        conn = self.get_db_connection()
        try:
            if self.custom_query:
                print("Custom SQL query provided. Skipping automatic table discovery.")
                print("Executing custom query...\n")
                custom_data = pd.read_sql(self.custom_query, conn)
                print(f"Custom query returned {custom_data.shape[0]} rows and {custom_data.shape[1]} columns")
                return custom_data
            
            # Automatic table discovery
            table_metadata = self._analyze_table_structures(conn)
            primary_table = self._identify_primary_table(table_metadata)
            
            if not primary_table:
                raise ValueError("Could not find primary table with Subject, Course, Semester, Year, and Enrollment")
            
            print(f"Primary table: {primary_table['name']}")
            
            joinable_tables = self._identify_joinable_tables(table_metadata, primary_table)
            print(f"Found {len(joinable_tables)} joinable tables: {[t['name'] for t in joinable_tables]}")
            combined_data = self._extract_and_join_data(conn, primary_table, joinable_tables)
            
            return combined_data
            
        finally:
            conn.close()
    
    def _analyze_table_structures(self, conn) -> List[Dict[str, Any]]:
        """Analyze all tables and their column structures."""
        print("Analyzing table structures...")
        
        tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """
        
        tables_df = pd.read_sql(tables_query, conn)
        available_tables = tables_df['table_name'].tolist()
        print(f"Available tables: {available_tables}")
        
        table_metadata = []
        
        for table_name in available_tables:
            # Get detailed column info
            columns_query = f"""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = '{table_name}'
                ORDER BY ordinal_position;
            """
            
            columns_df = pd.read_sql(columns_query, conn)
            column_info = columns_df.to_dict('records')
            
            column_names = [col['column_name'] for col in column_info]
            column_names_lower = [col.lower() for col in column_names]
            
            has_subj = any('subj' in col for col in column_names_lower)
            has_crse = any('crse' in col or 'course' in col for col in column_names_lower)
            has_term = any(col == 'term' for col in column_names_lower)
            has_semester = any('semester' in col for col in column_names_lower) or has_term
            has_year = any('year' in col for col in column_names_lower) or has_term
            has_enrollment = any('enrollment' in col or 'headcount' in col or col == 'act' for col in column_names_lower)
            
            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            row_count = pd.read_sql(count_query, conn).iloc[0]['row_count']
            
            metadata = {
                'name': table_name,
                'columns': column_info,
                'column_names': column_names,
                'column_names_lower': column_names_lower,
                'has_subj': has_subj,
                'has_crse': has_crse,
                'has_term': has_term,
                'has_semester': has_semester,
                'has_year': has_year,
                'has_enrollment': has_enrollment,
                'row_count': row_count
            }
            
            table_metadata.append(metadata)
            
            print(f"Table {table_name}: {row_count} rows, Subj={has_subj}, Crse={has_crse}, Term={has_term}, Semester={has_semester}, Year={has_year}, Enrollment={has_enrollment}")
            print(f"  Columns: {column_names[:10]}...")
        
        return table_metadata
    
    def _identify_primary_table(self, table_metadata: List[Dict]) -> Optional[Dict]:
        """Identify the primary table that has Subject, Course, Semester, Year, and Enrollment."""
        candidates = []
        
        for table in table_metadata:
            # Must have Subject, Course, Semester, and Enrollment at minimum
            if table['has_subj'] and table['has_crse'] and table['has_semester'] and table['has_enrollment']:
                score = 0
                score += 10 if table['has_subj'] else 0
                score += 10 if table['has_crse'] else 0
                score += 10 if table['has_semester'] else 0
                score += 10 if table['has_year'] else 0
                score += 10 if table['has_enrollment'] else 0
                score += min(table['row_count'] / 1000, 10)
                
                candidates.append((score, table))
        
        if not candidates:
            return None
        
        # Return the highest scoring table
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    def _identify_joinable_tables(self, table_metadata: List[Dict], primary_table: Dict) -> List[Dict]:
        """Identify tables that can be joined to the primary table via Year/Semester."""
        joinable = []
        
        for table in table_metadata:
            if table['name'] == primary_table['name']:
                continue
            
            # Must have at least Term, Year or Semester to be joinable
            if table.get('has_term') or table['has_year'] or table['has_semester']:
                # Determine join strategy
                join_strategy = []
                if table.get('has_term'):
                    join_strategy.append('term')
                if table['has_year']:
                    join_strategy.append('year')
                if table['has_semester']:
                    join_strategy.append('semester')
                
                table['join_strategy'] = join_strategy
                joinable.append(table)
                
                print(f"  {table['name']}: joinable via {join_strategy}")
        
        return joinable
    
    def _extract_and_join_data(self, conn, primary_table: Dict, joinable_tables: List[Dict]) -> pd.DataFrame:
        """Extract data from primary table and join with related tables."""
        print("Extracting and joining data...")
        
        print(f"Loading primary table: {primary_table['name']} (filtered to 2023 onward)")
        
        year_col = None
        if 'year' in [col.lower() for col in primary_table['column_names']]:
            for col in primary_table['column_names']:
                if col.lower() == 'year':
                    year_col = col
                    break
        
        # Build the query with year filter
        if year_col:
            print(f"Filtering by year column: {year_col}")
            primary_query = f"SELECT * FROM {primary_table['name']} WHERE {year_col} >= 2023"
        elif any('term' == col.lower() for col in primary_table['column_names']):
            term_col = next(col for col in primary_table['column_names'] if col.lower() == 'term')
            print(f"Filtering by term column: {term_col}")
            primary_query = f"SELECT * FROM {primary_table['name']} WHERE ({term_col} / 100) >= 2023"
        else:
            print("WARNING: Could not find year or term column for filtering. Using all data.")
            primary_query = f"SELECT * FROM {primary_table['name']}"
        
        print(f"Query: {primary_query}")
        base_data = pd.read_sql(primary_query, conn)
        
        print(f"Primary data shape after 2023 filtering: {base_data.shape}")
        
        # Identify key columns in primary table
        primary_cols = self._identify_key_columns(base_data.columns)
        print(f"Primary table key columns: {primary_cols}")
        
        # Join with each related table
        for related_table in joinable_tables:
            try:
                print(f"\nJoining with: {related_table['name']}")
                
                # Load related table data from 2023 onward
                # Determine which column to use for year filtering in related table
                year_col = None
                term_col = None
                
                for col in related_table['column_names']:
                    if col.lower() == 'year':
                        year_col = col
                        break
                    elif col.lower() == 'term':
                        term_col = col
                
                # Build the query with year filter
                if year_col:
                    print(f"Filtering related table by year column: {year_col}")
                    related_query = f"SELECT * FROM {related_table['name']} WHERE {year_col} >= 2023"
                elif term_col:
                    print(f"Filtering related table by term column: {term_col}")
                    related_query = f"SELECT * FROM {related_table['name']} WHERE ({term_col} / 100) >= 2023"
                else:
                    print(f"WARNING: Could not find year or term column for filtering in {related_table['name']}. Using all data.")
                    related_query = f"SELECT * FROM {related_table['name']}"
                
                print(f"Related query: {related_query}")
                related_data = pd.read_sql(related_query, conn)
                
                print(f"Related data shape (2023+ only): {related_data.shape}")
                
                # Identify key columns in related table
                related_cols = self._identify_key_columns(related_data.columns)
                print(f"Related table key columns: {related_cols}")
                
                # Determine join keys: prefer 'term' if present in both; else fall back to year/semester
                join_keys = []
                if primary_cols.get('term') and related_cols.get('term'):
                    join_keys = [('term', 'term')]
                else:
                    if primary_cols['year'] and related_cols['year']:
                        join_keys.append(('year', 'year'))
                    if primary_cols['semester'] and related_cols['semester']:
                        join_keys.append(('semester', 'semester'))
                
                if not join_keys:
                    print(f"  Cannot join {related_table['name']} - no common keys")
                    continue
                
                # Perform the join
                left_keys = [primary_cols[k[0]] for k in join_keys if primary_cols[k[0]]]
                right_keys = [related_cols[k[1]] for k in join_keys if related_cols[k[1]]]
                
                if left_keys and right_keys:
                    print(f"  Joining on: {list(zip(left_keys, right_keys))}")
                    
                    # Add table prefix to avoid column name conflicts
                    related_data_prefixed = related_data.copy()
                    for col in related_data_prefixed.columns:
                        if col not in right_keys:  # Don't prefix join keys
                            related_data_prefixed = related_data_prefixed.rename(columns={col: f"{related_table['name']}_{col}"})
                    
                    # Perform left join
                    base_data = pd.merge(
                        base_data, 
                        related_data_prefixed, 
                        left_on=left_keys, 
                        right_on=right_keys, 
                        how='left',
                        suffixes=('', f'_{related_table["name"]}')
                    )
                    
                    print(f"  After join shape: {base_data.shape}")
                else:
                    print(f"  Missing join keys for {related_table['name']}")
                    
            except Exception as e:
                print(f"  Error joining {related_table['name']}: {e}")
                continue
        
        print(f"\nFinal combined data shape: {base_data.shape}")
        print(f"Final columns: {len(base_data.columns)} total")
        
        # Show data distribution by year to confirm our filtering worked
        print("\n=== Confirming data is from 2023 onward ===")
        
        # Check if we have a year column
        year_col = None
        term_col = None
        
        for col in base_data.columns:
            if col.lower() == 'year':
                year_col = col
                break
            elif col.lower() == 'term':
                term_col = col
        
        # Display year distribution
        if year_col:
            year_counts = base_data[year_col].value_counts().sort_index()
            print(f"\nData distribution by year column '{year_col}':")
            for year, count in year_counts.items():
                print(f"  Year {year}: {count} records")
                
            # Validate no records before 2023
            if any(year < 2023 for year in year_counts.index if isinstance(year, (int, float))):
                print("WARNING: Some records are from before 2023!")
                
        elif term_col:
            # Extract year from term and count
            base_data['extracted_year'] = (base_data[term_col] / 100).astype(int)
                
            year_counts = base_data['extracted_year'].value_counts().sort_index()
            print(f"\nData distribution by year (extracted from '{term_col}'):")
            for year, count in year_counts.items():
                if isinstance(year, (int, float)):
                    print(f"  Year {year}: {count} records")
                    
            # Validate no records before 2023
            if any(year < 2023 for year in year_counts.index if isinstance(year, (int, float))):
                print("WARNING: Some records are from before 2023!")
        
        return base_data
    
    @staticmethod
    def _identify_key_columns(columns: List[str]) -> Dict[str, Optional[str]]:
        """Identify key columns (Subject, Course, Term/Semester/Year, Enrollment) in a list of column names."""
        key_cols = {
            'subj': None,
            'crse': None,
            'term': None,
            'semester': None, 
            'year': None,
            'enrollment': None
        }
        
        for col in columns:
            col_lower = col.lower()
            
            if 'subj' in col_lower and key_cols['subj'] is None:
                key_cols['subj'] = col
            elif ('crse' in col_lower or (col_lower == 'course' and 'crse' not in [c.lower() for c in columns])) and key_cols['crse'] is None:
                key_cols['crse'] = col
            elif (col_lower == 'term') and key_cols['term'] is None:
                key_cols['term'] = col
            elif ('semester' in col_lower) and key_cols['semester'] is None:
                key_cols['semester'] = col
            elif ('year' in col_lower) and key_cols['year'] is None:
                key_cols['year'] = col
            elif ('enrollment' in col_lower or 'headcount' in col_lower or col_lower == 'act') and key_cols['enrollment'] is None:
                key_cols['enrollment'] = col
        
        return key_cols
