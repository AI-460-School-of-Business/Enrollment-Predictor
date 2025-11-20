# Database

## Why Database?

The original implementation strategy relied on file based storage to store data used in our prediction models. However, this approach presented several scalability and maintainability challenges that led us to adopt a PostgreSQL database solution.

## Challenges with CSV Approach

### Scalability Issues
- **File count**: We anticipate managing 50+ CSV files as the system grows
- **Data fragmentation**: Related data spread across multiple files made quick analysis difficult
- **Performance degradation**: CSV parsing and processing became increasingly slow with the inclusion of more data

### Maintainability Problems
- **Holistic modifications**: Making system-wide data changes required updating multiple CSV files
- **Data consistency**: Ensuring data integrity across multiple files was error-prone
- **Version control**: Tracking changes across numerous CSV files was cumbersome
- **Backup complexity**: Managing backups of multiple CSV files was inefficient

## Database Solution Benefits

### Performance & Scalability
- **Query optimization**: SQL queries are significantly faster than CSV parsing
- **Indexing**: Database indexes provide rapid data access for large datasets
- **Memory efficiency**: Only load required data into memory
- **Concurrent access**: Multiple processes can safely access data simultaneously

### Data Management
- **Single source of truth**: All data centralized in one location
- **Easy backup/restore**: Single SQL dump file contains entire dataset

### Development Workflow
- **Simplified deployment**: Import/export single SQL file instead of managing 50+ CSVs
- **Better tooling**: SQL provides powerful data manipulation capabilities
- **Schema evolution**: Database migrations allow structured schema changes
- **Data validation**: Built-in constraints prevent invalid data entry

## Feature Schema Strategies

We implement two distinct approaches to enrollment prediction, each serving different use cases:

### KISS Method (Keep It Simple Stupid)
**File**: `enrollment_features_min.json`

**Philosophy**: Dr. G's original minimal approach focusing on core identifiers.

**Features**:
- **CRN** (Course Reference Number)
- **Semester** 
- **Year**
- **Headcount** (enrollment count)

**Advantages**:
- Fast model training and inference
- Easy to understand and debug
- Minimal data requirements
- Low computational overhead

### Feature-Rich Strategy
**File**: `enrollment_features_rich.json`

**Philosophy**: Comprehensive approach incorporating external factors and contextual information.

**Feature Categories**:
- **Course Attributes**: Prerequisites, level, department, credits
- **Student Demographics**: Age distribution, academic standing, major
- **Historical Context**: Previous enrollment patterns, trends
- **Institutional Factors**: University events, policy changes
- **External Factors**: Economic indicators, seasonal patterns
- **Geographic Context**: Student distribution, regional factors

**Use Cases**:
- Production-ready models requiring high accuracy
- Complex enrollment pattern analysis
- Multi-factor scenario planning
- Advanced predictive analytics

**Advantages**:
- Higher prediction accuracy
- Better handling of edge cases
- Comprehensive trend analysis
- Robust to external changes

## Implementation Notes

The database architecture supports both approaches seamlessly:
- **Flexible querying**: Extract minimal or comprehensive feature sets as needed
- **Schema validation**: JSON schemas ensure consistent feature extraction
- **Performance optimization**: Database indexes support both simple and complex queries
- **Data lineage**: Track how features are derived from raw enrollment data

This dual-schema approach allows us to maintain the simplicity of the KISS method while enabling sophisticated analysis through the feature-rich strategy, all backed by a scalable database infrastructure.