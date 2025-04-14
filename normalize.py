import pandas as pd
import numpy as np

# Load your datasets
df_national = pd.read_csv('new_aqi/1970-2021 data.csv')
df_state_accidents = pd.read_csv('new_aqi/2018-2022 data.csv')
df_locations = pd.read_csv('new_aqi/statewise_loactions.csv')

# Create a standardization mapping dictionary
state_mapping = {
    # For & vs and
    'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands',
    'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Jammu & Kashmir': 'Jammu and Kashmir',
    
    # For hyphenated vs space
    'Tamil-Nadu': 'Tamil Nadu',
    'Andhra-Pradesh': 'Andhra Pradesh',
    
    # For old vs new names
    'Orissa': 'Odisha',
    'Pondicherry': 'Puducherry',
    
    # For variations in spacing or capitalization
    'TAMIL NADU': 'Tamil Nadu',
    'Tamilnadu': 'Tamil Nadu',
    'DELHI': 'Delhi',
    'delhi': 'Delhi'
}

def standardize_state_names(df, column_name):
    """
    Standardize state/UT names in a dataframe
    
    Parameters:
    df (pandas.DataFrame): The dataframe containing state names
    column_name (str): Name of the column containing state names
    
    Returns:
    pandas.DataFrame: Dataframe with standardized state names
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Apply the mapping
    df_copy[column_name] = df_copy[column_name].map(lambda x: state_mapping.get(x, x))
    
    # Strip any leading/trailing whitespace
    df_copy[column_name] = df_copy[column_name].str.strip()
    
    return df_copy

# Apply standardization to the datasets
df_state_accidents = standardize_state_names(df_state_accidents, 'State/UT')
df_locations = standardize_state_names(df_locations, 'State/UT')

# Handle the duplicate entry for Dadra and Nagar Haveli and Daman and Diu
# Keep only the first occurrence and sum the values
if len(df_state_accidents[df_state_accidents['State/UT'] == 'Dadra and Nagar Haveli and Daman and Diu']) > 1:
    # Get indices of duplicate rows
    duplicate_indices = df_state_accidents[df_state_accidents['State/UT'] == 
                                        'Dadra and Nagar Haveli and Daman and Diu'].index.tolist()
    
    if len(duplicate_indices) > 1:
        # Keep the first instance and aggregate numeric values
        first_idx = duplicate_indices[0]
        for col in df_state_accidents.columns:
            if col != 'S.No' and col != 'State/UT' and pd.api.types.is_numeric_dtype(df_state_accidents[col]):
                # Sum the values from duplicates
                df_state_accidents.at[first_idx, col] = df_state_accidents.loc[duplicate_indices, col].sum()
        
        # Drop the subsequent duplicates
        df_state_accidents = df_state_accidents.drop(duplicate_indices[1:])
        # Reset the index
        df_state_accidents = df_state_accidents.reset_index(drop=True)

# Fix the missing Mizoram in locations dataset
if 'Mizoram' not in df_locations['State/UT'].values:
    # Adding Mizoram with approximate coordinates
    mizoram_row = pd.DataFrame({
        'State/UT': ['Mizoram'],
        'latitude': [23.1645],
        'longitude': [92.9376]
    })
    df_locations = pd.concat([df_locations, mizoram_row], ignore_index=True)

# Fix for Ladakh if missing in locations
if 'Ladakh' not in df_locations['State/UT'].values and 'Ladakh' in df_state_accidents['State/UT'].values:
    ladakh_row = pd.DataFrame({
        'State/UT': ['Ladakh'],
        'latitude': [34.1526],
        'longitude': [77.5770]
    })
    df_locations = pd.concat([df_locations, ladakh_row], ignore_index=True)

# Save the standardized datasets
df_state_accidents.to_csv('new_aqi/standardized_accident_data.csv', index=False)
df_locations.to_csv('new_aqi/standardized_locations.csv', index=False)

# Verify standardization by checking for mismatches between datasets
accident_states = set(df_state_accidents['State/UT'].unique())
location_states = set(df_locations['State/UT'].unique())

# Find states in accident data but not in location data
missing_in_locations = accident_states - location_states
if missing_in_locations:
    print("States in accident data but missing in location data:", missing_in_locations)

# Find states in location data but not in accident data
missing_in_accidents = location_states - accident_states
if missing_in_accidents:
    print("States in location data but missing in accident data:", missing_in_accidents)

print("Standardization complete!")