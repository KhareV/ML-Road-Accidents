import pandas as pd
import re

def standardize_state_names(df, state_column):
    """
    Standardizes state names in a DataFrame.
    
    Parameters:
    - df: DataFrame containing state names
    - state_column: Name of column containing state names
    
    Returns:
    - DataFrame with standardized state names
    """
    if state_column not in df.columns:
        print(f"Column '{state_column}' not found in DataFrame.")
        return df
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Common state name variations and their standardized forms
    state_mapping = {
        # Ampersand vs. 'and'
        r'(\w+)\s*&\s*(\w+)': r'\1 and \2',
        
        # Handle hyphenated names
        r'Tamil-Nadu': 'Tamil Nadu',
        r'Andhra-Pradesh': 'Andhra Pradesh',
        r'Arunachal-Pradesh': 'Arunachal Pradesh',
        r'Himachal-Pradesh': 'Himachal Pradesh',
        r'Madhya-Pradesh': 'Madhya Pradesh',
        r'Uttar-Pradesh': 'Uttar Pradesh',
        r'West-Bengal': 'West Bengal',
        
        # Various spellings and formats
        r'Orissa': 'Odisha',
        r'Pondicherry': 'Puducherry',
        r'Delhi( UT)?': 'Delhi',
        r'Jammu & Kashmir': 'Jammu and Kashmir',
        r'Uttaranchal': 'Uttarakhand',
        r'Chattisgarh': 'Chhattisgarh',
        r'Jharkhand': 'Jharkhand',
        r'D & N Haveli': 'Dadra and Nagar Haveli',
        r'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli',
        r'Daman & Diu': 'Daman and Diu',

        # Combined UT after 2020
        r'Dadra and Nagar Haveli and Daman and Diu': 'Dadra and Nagar Haveli and Daman and Diu',
        
        # Additional mappings for other potential variations
        r'A & N Islands': 'Andaman and Nicobar Islands',
        r'Andaman & Nicobar': 'Andaman and Nicobar Islands',
        r'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands'
    }
    
    # Apply all replacements
    for pattern, replacement in state_mapping.items():
        df_copy[state_column] = df_copy[state_column].str.replace(pattern, replacement, regex=True)
    
    # Strip whitespace
    df_copy[state_column] = df_copy[state_column].str.strip()
    
    return df_copy

def main():
    # Load datasets
    try:
        # This dataset doesn't have state-level data but included for completeness
        national_data = pd.read_csv('new_aqi/1970-2021 data.csv')
        
        # State-level accident data
        accident_data = pd.read_csv('new_aqi/2018-2022 data.csv')
        
        # Assuming you have an air quality dataset with state names
        # air_quality = pd.read_csv('path_to_air_quality.csv')
        
        # Standardize state names in accident data
        standardized_accident = standardize_state_names(accident_data, 'State/UT')
        
        # Standardize state names in air quality data (uncomment when available)
        # standardized_air_quality = standardize_state_names(air_quality, 'State / Union Territory')
        
        # Save standardized datasets
        standardized_accident.to_csv('new_aqi/standardized_accident_data.csv', index=False)
        # standardized_air_quality.to_csv('new_aqi/standardized_air_quality.csv', index=False)
        
        print("Standardization complete. Files saved with prefix 'standardized_'")
        
        # Print sample of standardized states for verification
        print("\nSample of standardized state names:")
        print(standardized_accident['State/UT'].head(10))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()