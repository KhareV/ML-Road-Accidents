import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('air_quality_data.csv')

# Print column names to debug
print("Original column names:", df.columns.tolist())

# Check which column name format is used and standardize
if 'State/UT' in df.columns:
    df = df.rename(columns={'State/UT': 'State_Union_Territory'})
elif 'State / Union Territory' in df.columns:
    df = df.rename(columns={'State / Union Territory': 'State_Union_Territory'})
else:
    # Try to find the column containing state names
    first_column = df.columns[0]
    if 'State' in first_column or 'UT' in first_column:
        df = df.rename(columns={first_column: 'State_Union_Territory'})
    else:
        print(f"Could not identify state column. First column is: '{first_column}'")
        # If your data is in quotes with state as first value, try this:
        if df[first_column].str.contains('Andhra Pradesh').any():
            print("Found states in the first column data, treating as state names")
            df = df.rename(columns={first_column: 'State_Union_Territory'})

# Standardize state/UT names
name_mapping = {
    'Chattisgarh': 'Chhattisgarh',
    'Lakshwadeep': 'Lakshadweep'
}

# Now use the standardized column name
df['State_Union_Territory'] = df['State_Union_Territory'].replace(name_mapping)

# Check if Year column exists, otherwise try to extract it
if 'Year' not in df.columns:
    # Try to find year in the data - this depends on your data format
    print("Year column not found, checking for year values in the data...")
    # Example: If year might be in another column or in the state name column
    # You might need to customize this based on your actual data structure

# Standardize column names for pollutant measurements
pollutant_mappings = {
    'SO2 (µg/m³)': 'SO2 (Annual Average)',
    'NO2 (µg/m³)': 'NO2 (Annual Average)',
    'PM10 (µg/m³)': 'PM10 (Annual Average)',
    'PM2.5 (µg/m³)': 'PM2.5 (Annual Average)'
}

# Rename columns if they exist
for old_name, new_name in pollutant_mappings.items():
    if old_name in df.columns:
        df = df.rename(columns={old_name: new_name})

# Print updated column names
print("Updated column names:", df.columns.tolist())

# Replace missing values with NaN
df = df.replace(['-', ''], np.nan)

# Convert all measurement values to numeric
for col in df.columns:
    if any(pollutant in col for pollutant in ['SO2', 'NO2', 'PM10', 'PM2.5']):
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Add Region column if it doesn't exist
if 'Region' not in df.columns:
    print("Region column not found, will need to be added manually")
    # If you have region data for later years, you could implement the region mapping here

# Save normalized data
df.to_csv('normalized_air_quality_data.csv', index=False)
print("Data normalized and saved to normalized_air_quality_data.csv")