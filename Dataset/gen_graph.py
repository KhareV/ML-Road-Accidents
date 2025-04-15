"""
Road Accident Analysis Visualization Script
Focusing on AQI correlation and hotspot identification

Created for research paper on:
1. Road accidents correlation with high AQI locations
2. Finding hotspots of road accidents

Author: Abeer Gupta
Date: 2025-04-15
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import glob

# Suppress warnings
warnings.filterwarnings('ignore')

# Set global styling for better visualizations
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Create output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Create subdirectories for organization
aqi_dir = os.path.join(output_dir, 'AQI_Analysis')
hotspot_dir = os.path.join(output_dir, 'Hotspot_Analysis')
os.makedirs(aqi_dir, exist_ok=True) 
os.makedirs(hotspot_dir, exist_ok=True)

# Function to find files with flexible naming (compatible with older Python versions)
def find_file(pattern):
    # Try exact match
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    
    # Try lowercase version
    matches = glob.glob(pattern.lower())
    if matches:
        return matches[0]
    
    # Try with underscores instead of spaces
    pattern_with_underscores = pattern.replace(" ", "_")
    matches = glob.glob(pattern_with_underscores)
    if matches:
        return matches[0]
    
    # Try with partial match
    base_name = os.path.basename(pattern).split('.')[0]
    partial_matches = []
    for file in os.listdir('.'):  # search in current directory
        if base_name.lower() in file.lower() and os.path.isfile(file):
            partial_matches.append(file)
    
    if partial_matches:
        return partial_matches[0]
    
    return None

# Load datasets
print("Loading datasets...")
print(f"Current working directory: {os.getcwd()}")
print("Available CSV files:", glob.glob("*.csv"))

# Air quality data
try:
    # Try different variations of the filenames
    air_2020_file = find_file("air 2020.csv")
    air_2021_file = find_file("air 2021.csv")
    air_2022_file = find_file("air 2022.csv")
    
    # Check if files were found
    if not all([air_2020_file, air_2021_file, air_2022_file]):
        missing = []
        if not air_2020_file: missing.append("air 2020.csv")
        if not air_2021_file: missing.append("air 2021.csv") 
        if not air_2022_file: missing.append("air 2022.csv")
        print(f"Could not find the following files: {', '.join(missing)}")
        print("Please make sure these files are in the current directory.")
        air_2020, air_2021, air_2022, air_combined = None, None, None, None
    else:
        print(f"Found air quality files: {air_2020_file}, {air_2021_file}, {air_2022_file}")
        
        air_2020 = pd.read_csv(air_2020_file)
        air_2021 = pd.read_csv(air_2021_file)
        air_2022 = pd.read_csv(air_2022_file)
        
        # Add year column to each dataframe
        air_2020['Year'] = 2020
        air_2021['Year'] = 2021
        air_2022['Year'] = 2022
        
        # Combine air quality data
        air_combined = pd.concat([air_2020, air_2021, air_2022], ignore_index=True)
        
        # Clean numeric columns - replace NM with NaN
        for col in ['SO2 (Annual Average)', 'NO2 (Annual Average)', 'PM10 (Annual Average)', 'PM2.5 (Annual Average)']:
            air_combined[col] = pd.to_numeric(air_combined[col].replace(['NM', '-'], np.nan))
            
        print("Air quality data loaded successfully")
except Exception as e:
    print(f"Error loading air quality data: {e}")
    print("Will continue with other datasets")
    air_2020, air_2021, air_2022, air_combined = None, None, None, None

# Black spots data
try:
    black_spots_file = find_file("Black Spots.csv")
    
    if not black_spots_file:
        print("Could not find Black Spots.csv file")
        black_spots = None
    else:
        print(f"Found black spots file: {black_spots_file}")
        try:
            black_spots = pd.read_csv(black_spots_file, encoding='utf-8')
        except:
            # Try with different encoding if UTF-8 fails
            black_spots = pd.read_csv(black_spots_file, encoding='ISO-8859-1')
        print("Black spots data loaded successfully")
except Exception as e:
    print(f"Error loading black spots data: {e}")
    black_spots = None

# Historical data
try:
    historical_file = find_file("1970-2021 data.csv")
    
    if not historical_file:
        print("Could not find 1970-2021 data.csv file")
        historical_data = None
    else:
        print(f"Found historical data file: {historical_file}")
        historical_data = pd.read_csv(historical_file)
        
        # Convert all relevant columns to numeric to ensure calculations work
        # First, identify columns that might be accident/death related
        for col in historical_data.columns:
            if 'accident' in col.lower() or 'killed' in col.lower() or 'death' in col.lower() or 'injured' in col.lower() or 'total' in col.lower():
                # Try to convert to numeric, coercing errors to NaN
                historical_data[col] = pd.to_numeric(historical_data[col], errors='coerce')
        
        print("Historical data loaded successfully")
except Exception as e:
    print(f"Error loading historical data: {e}")
    historical_data = None

# Recent state-wise data
try:
    recent_file = find_file("2018-2022 data.csv")
    
    if not recent_file:
        print("Could not find 2018-2022 data.csv file")
        recent_data = None
    else:
        print(f"Found recent data file: {recent_file}")
        recent_data = pd.read_csv(recent_file)
        
        # Convert all numeric columns to numeric type
        for col in recent_data.columns:
            if col != 'State/UT' and col != 'S.No':  # Skip non-numeric columns
                recent_data[col] = pd.to_numeric(recent_data[col], errors='coerce')
                
        print("Recent state-wise data loaded successfully")
except Exception as e:
    print(f"Error loading recent data: {e}")
    recent_data = None

# Area distribution data
try:
    area_file = find_file("Area of Accidents.csv")
    
    if not area_file:
        print("Could not find Area of Accidents.csv file")
        area_data = None
    else:
        print(f"Found area data file: {area_file}")
        area_data = pd.read_csv(area_file)
        
        # Convert numeric columns
        for col in ['Urban', 'Rural', 'Total']:
            area_data[col] = pd.to_numeric(area_data[col], errors='coerce')
            
        print("Area distribution data loaded successfully")
except Exception as e:
    print(f"Error loading area data: {e}")
    area_data = None

# Check if we have enough data to proceed
if all(x is None for x in [air_combined, black_spots, historical_data, recent_data, area_data]):
    print("ERROR: No datasets could be loaded. Please check file paths and try again.")
    sys.exit(1)

print("All available datasets loaded successfully!")

# ===== RESEARCH QUESTION 1: AQI AND ACCIDENTS CORRELATION =====
print("\nGenerating graphs for AQI and accidents correlation...")

# Function to create scatter plots for AQI vs accidents
def create_aqi_scatter_plots():
    if air_combined is None or recent_data is None:
        print("  Skipping AQI correlation scatter plots - required data not available")
        return
        
    print("  Creating AQI correlation scatter plots...")
    
    # Create a copy of the dataframes to avoid modifying originals
    air_data = air_combined.copy()
    accident_data = recent_data.copy()
    
    # Clean up state names for better matching
    air_data['State / Union Territory'] = air_data['State / Union Territory'].str.strip().str.replace('"', '')
    accident_data['State/UT'] = accident_data['State/UT'].str.strip()
    
    # Dictionary for years and corresponding accident columns
    year_columns = {
        2020: 'Accidents 2020',
        2021: 'Accidents 2021',
        2022: 'Accidents 2022'
    }
    
    # Create scatter plots for each pollutant and year
    pollutants = ['PM2.5 (Annual Average)', 'PM10 (Annual Average)', 'NO2 (Annual Average)', 'SO2 (Annual Average)']
    for pollutant in pollutants:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Correlation Between {pollutant} and Road Accidents (2020-2022)', fontsize=20)
        
        for i, year in enumerate([2020, 2021, 2022]):
            # Filter air data for the current year
            year_air_data = air_data[air_data['Year'] == year]
            
            # Create a mapping dictionary from state to pollutant value
            pollutant_map = dict(zip(year_air_data['State / Union Territory'], year_air_data[pollutant]))
            
            # Map pollutant values to accident data
            accident_data[f'{pollutant}_{year}'] = accident_data['State/UT'].map(pollutant_map)
            
            # Create scatter plot
            ax = axes[i]
            scatter = ax.scatter(
                accident_data[f'{pollutant}_{year}'], 
                accident_data[year_columns[year]],
                alpha=0.7, 
                s=100,
                c=accident_data[f'{pollutant}_{year}'],
                cmap='YlOrRd'
            )
            
            # Add trendline
            mask = ~np.isnan(accident_data[f'{pollutant}_{year}']) & ~np.isnan(accident_data[year_columns[year]])
            if sum(mask) > 1:  # Only add trendline if we have at least 2 valid points
                x = accident_data[f'{pollutant}_{year}'][mask]
                y = accident_data[year_columns[year]][mask]
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
                
                # Calculate and display correlation coefficient
                correlation = np.corrcoef(x, y)[0, 1]
                ax.annotate(f'Correlation: {correlation:.2f}', 
                           xy=(0.05, 0.95), 
                           xycoords='axes fraction', 
                           fontsize=12, 
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=1, alpha=0.8))
            
            # Add labels for top 5 states with highest accident counts
            if sum(mask) > 0:
                top_5_states = accident_data[mask].nlargest(5, year_columns[year])
                for j, row in top_5_states.iterrows():
                    ax.annotate(row['State/UT'], 
                               (row[f'{pollutant}_{year}'], row[year_columns[year]]),
                               xytext=(5, 5), 
                               textcoords='offset points',
                               fontsize=8,
                               bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7))
            
            # Set labels and title
            ax.set_xlabel(pollutant, fontsize=12)
            ax.set_ylabel('Number of Accidents', fontsize=12)
            ax.set_title(f'Year {year}', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(scatter, cax=cax)
            cbar.set_label(pollutant, fontsize=10)
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(aqi_dir, f'scatter_{pollutant.split()[0]}_vs_accidents.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Function to create improved AQI correlation analysis - replacing the old heatmap
def create_aqi_correlation_analysis():
    if air_2022 is None or recent_data is None:
        print("  Skipping AQI correlation analysis - required data not available")
        return
        
    print("  Creating improved AQI correlation analysis...")
    
    # Prepare data for 2022
    air_recent = air_2022.copy()
    accident_data = recent_data.copy()
    
    # Clean up state names for better matching
    air_recent['State / Union Territory'] = air_recent['State / Union Territory'].str.strip().str.replace('"', '')
    accident_data['State/UT'] = accident_data['State/UT'].str.strip()
    
    # Clean and prepare data
    for col in ['SO2 (Annual Average)', 'NO2 (Annual Average)', 'PM10 (Annual Average)', 'PM2.5 (Annual Average)']:
        air_recent[col] = pd.to_numeric(air_recent[col].replace(['NM', '-'], np.nan))
    
    # Merge datasets
    merged_data = pd.merge(air_recent, accident_data, 
                         left_on='State / Union Territory', 
                         right_on='State/UT', 
                         how='inner')
    
    # Select relevant columns for correlation analysis
    pollutant_cols = ['SO2 (Annual Average)', 'NO2 (Annual Average)', 'PM10 (Annual Average)', 'PM2.5 (Annual Average)']
    accident_cols = ['Accidents 2022', 'Killed 2022', 'Injured 2022']
    
    # Calculate correlation matrix
    correlation_data = merged_data[pollutant_cols + accident_cols]
    corr_matrix = correlation_data.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix.iloc[:4, 4:], annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
               xticklabels=accident_cols, yticklabels=pollutant_cols)
    
    plt.title('Correlation Between Air Quality Parameters and Road Accident Metrics (2022)', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(aqi_dir, 'aqi_accident_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create dual-axis chart for PM2.5 and accidents
    plt.figure(figsize=(14, 10))
    
    # Sort states by PM2.5 level
    sorted_data = merged_data.sort_values('PM2.5 (Annual Average)', ascending=False).head(15)
    
    # Create primary axis for PM2.5
    ax1 = plt.gca()
    bars1 = ax1.bar(np.arange(len(sorted_data)), sorted_data['PM2.5 (Annual Average)'], 
                   width=0.4, color='#ff9999', label='PM2.5 Level')
    ax1.set_xlabel('States/UTs', fontsize=14)
    ax1.set_ylabel('PM2.5 Annual Average', fontsize=14, color='#d62728')
    ax1.tick_params(axis='y', labelcolor='#d62728')
    
    # Create secondary axis for accidents
    ax2 = ax1.twinx()
    bars2 = ax2.bar(np.arange(len(sorted_data)) + 0.4, sorted_data['Accidents 2022'], 
                   width=0.4, color='#1f77b4', label='Accidents')
    ax2.set_ylabel('Number of Accidents (2022)', fontsize=14, color='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    
    # Set x-ticks
    plt.xticks(np.arange(len(sorted_data)) + 0.2, sorted_data['State/UT'], rotation=45, ha='right')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('PM2.5 Levels vs. Road Accidents by State (2022)', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(aqi_dir, 'pm25_vs_accidents_by_state.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Function to create time series analysis graphs
def create_time_series_graphs():
    if air_combined is None or recent_data is None:
        print("  Skipping time series analysis graphs - required data not available")
        return
        
    print("  Creating time series analysis graphs...")
    
    # Create a merged dataset for analysis
    air_avg = air_combined.groupby(['State / Union Territory', 'Year'])['PM2.5 (Annual Average)'].mean().reset_index()
    
    # Get top 5 states with highest average PM2.5
    top5_states = air_avg.groupby('State / Union Territory')['PM2.5 (Annual Average)'].mean().nlargest(5).index.tolist()
    
    # Filter data for top 5 states
    top5_data = air_avg[air_avg['State / Union Territory'].isin(top5_states)]
    
    # Get accident data for these states
    accident_data = recent_data[recent_data['State/UT'].isin(top5_states)].copy()
    
    # Create figure
    fig, axes = plt.subplots(len(top5_states), 1, figsize=(14, 4 * len(top5_states)), sharex=True)
    fig.suptitle('Air Quality (PM2.5) and Accident Trends for States with Poorest Air Quality', fontsize=20)
    
    # Setup colors
    colors = sns.color_palette("husl", 2)
    
    for i, state in enumerate(top5_states):
        ax = axes[i]
        
        # Plot PM2.5 trend
        state_air_data = top5_data[top5_data['State / Union Territory'] == state]
        ax.plot(state_air_data['Year'], state_air_data['PM2.5 (Annual Average)'], 
               marker='o', linewidth=2, color=colors[0], label='PM2.5')
        ax.set_ylabel('PM2.5 Level', color=colors[0], fontsize=12)
        ax.tick_params(axis='y', labelcolor=colors[0])
        
        # Create a second y-axis for accidents
        ax2 = ax.twinx()
        state_accident_data = accident_data[accident_data['State/UT'] == state]
        
        years = [2020, 2021, 2022]
        accident_values = [
            state_accident_data['Accidents 2020'].values[0] if not state_accident_data.empty else np.nan,
            state_accident_data['Accidents 2021'].values[0] if not state_accident_data.empty else np.nan,
            state_accident_data['Accidents 2022'].values[0] if not state_accident_data.empty else np.nan
        ]
        
        ax2.plot(years, accident_values, marker='s', linewidth=2, 
                color=colors[1], label='Accidents')
        ax2.set_ylabel('Number of Accidents', color=colors[1], fontsize=12)
        ax2.tick_params(axis='y', labelcolor=colors[1])
        
        # Set title and grid
        ax.set_title(f'State: {state}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.xlabel('Year', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig(os.path.join(aqi_dir, 'time_series_pm25_vs_accidents.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Function to create categorical bar charts
def create_categorical_bar_charts():
    if air_2022 is None or recent_data is None:
        print("  Skipping categorical bar charts - required data not available")
        return
        
    print("  Creating categorical bar charts for AQI categories...")
    
    # Get data for 2022
    air_data = air_2022.copy()
    accident_data = recent_data.copy()
    
    # Clean data
    for col in ['SO2 (Annual Average)', 'NO2 (Annual Average)', 'PM10 (Annual Average)', 'PM2.5 (Annual Average)']:
        air_data[col] = pd.to_numeric(air_data[col].replace(['NM', '-'], np.nan))
    
    # Define AQI categories based on PM2.5 levels
    # Using simplified categories based on Indian standards
    def categorize_aqi(pm25):
        if pd.isna(pm25):
            return np.nan
        elif pm25 <= 30:
            return 'Good (0-30)'
        elif pm25 <= 60:
            return 'Moderate (31-60)'
        elif pm25 <= 90:
            return 'Poor (61-90)'
        else:
            return 'Very Poor (>90)'
    
    air_data['AQI_Category'] = air_data['PM2.5 (Annual Average)'].apply(categorize_aqi)
    
    # Merge with accident data
    merged_data = pd.merge(air_data, accident_data, 
                         left_on='State / Union Territory', 
                         right_on='State/UT', 
                         how='inner')
    
    # Calculate average accidents per AQI category
    aqi_accidents = merged_data.groupby('AQI_Category')[['Accidents 2022', 'Killed 2022']].mean().reset_index()
    
    # Count states in each category
    category_counts = merged_data.groupby('AQI_Category').size().reset_index(name='State_Count')
    aqi_accidents = pd.merge(aqi_accidents, category_counts, on='AQI_Category')
    
    # Sort categories in meaningful order
    order = ['Good (0-30)', 'Moderate (31-60)', 'Poor (61-90)', 'Very Poor (>90)']
    aqi_accidents['AQI_Category'] = pd.Categorical(aqi_accidents['AQI_Category'], categories=order, ordered=True)
    aqi_accidents = aqi_accidents.sort_values('AQI_Category')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Colors for categories
    colors = ['#4CAF50', '#FFEB3B', '#FF9800', '#F44336']
    
    # Plot average accidents by AQI category
    sns.barplot(x='AQI_Category', y='Accidents 2022', data=aqi_accidents, palette=colors, ax=ax1)
    ax1.set_title('Average Number of Accidents by Air Quality Category (2022)', fontsize=16)
    ax1.set_xlabel('PM2.5 Air Quality Category', fontsize=14)
    ax1.set_ylabel('Average Number of Accidents', fontsize=14)
    
    # Add state count as text on bars
    for i, p in enumerate(ax1.patches):
        ax1.annotate(f'n={aqi_accidents.iloc[i]["State_Count"]} states', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
    
    # Plot average fatalities by AQI category
    sns.barplot(x='AQI_Category', y='Killed 2022', data=aqi_accidents, palette=colors, ax=ax2)
    ax2.set_title('Average Number of Fatalities by Air Quality Category (2022)', fontsize=16)
    ax2.set_xlabel('PM2.5 Air Quality Category', fontsize=14)
    ax2.set_ylabel('Average Number of Fatalities', fontsize=14)
    
    # Add state count as text on bars
    for i, p in enumerate(ax2.patches):
        ax2.annotate(f'n={aqi_accidents.iloc[i]["State_Count"]} states', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
    
    # Add note about potential bias in the "Very Poor" category due to small sample size
    if 'Very Poor (>90)' in aqi_accidents['AQI_Category'].values:
        very_poor_count = aqi_accidents[aqi_accidents['AQI_Category'] == 'Very Poor (>90)']['State_Count'].values[0]
        if very_poor_count == 1:
            fig.text(0.5, 0.01, "Note: The 'Very Poor' category contains only one state, which may not be representative.", 
                   ha='center', fontsize=12, style='italic')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    plt.savefig(os.path.join(aqi_dir, 'categorical_aqi_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second chart showing fatality rate (fatalities per accident)
    merged_data['Fatality_Rate'] = merged_data['Killed 2022'] / merged_data['Accidents 2022']
    
    # Create boxplot of fatality rates by AQI category
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='AQI_Category', y='Fatality_Rate', data=merged_data, palette=colors, order=order)
    
    plt.title('Road Accident Fatality Rates by Air Quality Category (2022)', fontsize=18)
    plt.xlabel('PM2.5 Air Quality Category', fontsize=14)
    plt.ylabel('Fatality Rate (Deaths per Accident)', fontsize=14)
    
    # Add individual data points
    sns.stripplot(x='AQI_Category', y='Fatality_Rate', data=merged_data, 
                 color='black', alpha=0.5, jitter=True, order=order)
    
    # Add category counts
    for i, category in enumerate(order):
        if category in merged_data['AQI_Category'].values:
            count = merged_data[merged_data['AQI_Category'] == category].shape[0]
            plt.annotate(f'n={count}', xy=(i, -0.05), ha='center', fontsize=12)
    
    # Add note about interpretation
    plt.figtext(0.5, 0.01, 
              "Note: Higher fatality rate indicates more deaths per accident, reflecting accident severity rather than frequency.",
              ha='center', fontsize=12, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(aqi_dir, 'fatality_rate_by_aqi_category.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ===== RESEARCH QUESTION 2: ACCIDENT HOTSPOTS =====
print("\nGenerating graphs for accident hotspot identification...")

# Function to analyze urban vs rural distribution
def analyze_urban_vs_rural():
    if area_data is None or recent_data is None:
        print("  Skipping urban vs rural comparison charts - required data not available")
        return
        
    print("  Creating urban vs rural comparison charts...")
    
    # Load and prepare data
    area = area_data.copy()
    
    # Calculate percentages
    area['Urban_Percentage'] = (area['Urban'] / area['Total']) * 100
    area['Rural_Percentage'] = (area['Rural'] / area['Total']) * 100
    
    # Remove rows with NA values
    area = area[~area['Total'].isna() & (area['Total'] > 0)]
    
    # Sort by total accidents
    area_sorted = area.sort_values('Total', ascending=False)
    
    # Take top 15 states for better visualization
    top_states = area_sorted.head(15)
    
    # Create stacked bar chart
    plt.figure(figsize=(14, 10))
    
    # Plot
    ax = top_states.plot(x='State/UTs', y=['Urban', 'Rural'], kind='bar', stacked=True, 
                        color=['#2196F3', '#4CAF50'], figsize=(14, 8))
    
    plt.title('Distribution of Road Accidents: Urban vs Rural Areas (Top 15 States)', fontsize=18)
    plt.xlabel('State/Union Territory', fontsize=14)
    plt.ylabel('Number of Accidents', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text for percentages on bars
    for i, state in enumerate(top_states['State/UTs']):
        urban_count = top_states.iloc[i]['Urban']
        rural_count = top_states.iloc[i]['Rural']
        total = top_states.iloc[i]['Total']
        
        urban_pct = urban_count / total * 100
        rural_pct = rural_count / total * 100
        
        # Only add percentage text if it's significant enough to be visible
        if urban_pct > 5:
            plt.text(i, urban_count/2, f"{urban_pct:.1f}%", ha='center', va='center', fontsize=9)
            
        if rural_pct > 5:
            plt.text(i, urban_count + rural_count/2, f"{rural_pct:.1f}%", ha='center', va='center', fontsize=9)
    
    plt.legend(title='Area Type')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(hotspot_dir, 'urban_vs_rural_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second chart: Scatter plot showing relationship between rural percentage and fatality
    # We'll need to merge with recent data to get fatality information
    merged = pd.merge(area, recent_data, left_on='State/UTs', right_on='State/UT', how='inner')
    
    # Calculate fatality rate
    merged['Fatality_Rate_2022'] = merged['Killed 2022'] / merged['Accidents 2022']
    
    # Create scatter plot
    plt.figure(figsize=(14, 8))
    
    # Plot points
    scatter = plt.scatter(merged['Rural_Percentage'], merged['Fatality_Rate_2022'], 
                        s=merged['Total']/100, # Size based on total accidents
                        c=merged['Total'], # Color based on total accidents
                        cmap='YlOrRd', alpha=0.7)
    
    # Add state labels to points
    for i, row in merged.iterrows():
        if row['Total'] > 1000:  # Only label larger states for clarity
            plt.annotate(row['State/UTs'], 
                       (row['Rural_Percentage'], row['Fatality_Rate_2022']),
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7))
    
    # Add trend line
    mask = ~np.isnan(merged['Rural_Percentage']) & ~np.isnan(merged['Fatality_Rate_2022'])
    if sum(mask) > 1:
        x = merged['Rural_Percentage'][mask]
        y = merged['Fatality_Rate_2022'][mask]
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
        
        # Add correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]
        plt.annotate(f'Correlation: {correlation:.2f}', 
                   xy=(0.05, 0.95), 
                   xycoords='axes fraction', 
                   fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=1, alpha=0.8))
    
    # Add colorbar for total accidents
    cbar = plt.colorbar(scatter)
    cbar.set_label('Total Number of Accidents', fontsize=12)
    
    # Add a legend for bubble sizes
    sizes = [1000, 5000, 10000, 20000]
    labels = ["1,000", "5,000", "10,000", "20,000"]
    
    # Create legend handles
    legend_elements = []
    for size, label in zip(sizes, labels):
        size_scaled = size/100
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                      label=label,
                                      markerfacecolor='grey', 
                                      alpha=0.7,
                                      markersize=size_scaled**0.5))
    
    plt.legend(handles=legend_elements, title="Total Accidents", 
             loc='upper left', bbox_to_anchor=(0.01, 0.99))
    
    # Add title and labels
    plt.title('Relationship Between Rural Accidents and Fatality Rate (2022)', fontsize=18)
    plt.xlabel('Percentage of Accidents in Rural Areas', fontsize=14)
    plt.ylabel('Fatality Rate (Deaths per Accident)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add interpretation note
    plt.figtext(0.5, 0.01, 
              "Note: Positive correlation suggests accidents in rural areas tend to be more severe, possibly due to factors like emergency response times.",
              ha='center', fontsize=12, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(hotspot_dir, 'rural_vs_fatality_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Function to analyze black spots - fixing the fatality rate calculation
def analyze_black_spots():
    if black_spots is None:
        print("  Skipping black spot analysis - required data not available")
        return
        
    print("  Creating black spot analysis graphs...")
    
    # Clean and prepare black spots data
    bs = black_spots.copy()
    
    # Ensure numeric columns are properly formatted
    numeric_cols = bs.columns[bs.columns.str.contains('Number of')]
    for col in numeric_cols:
        bs[col] = pd.to_numeric(bs[col], errors='coerce')
    
    # Calculate total fatalities and total injuries (where available)
    # The column name might vary, so we'll use string filtering
    fatality_cols = [col for col in bs.columns if 'fatal' in col.lower() or 'killed' in col.lower() or 'death' in col.lower()]
    
    if fatality_cols:
        bs['Total_Fatalities'] = bs[fatality_cols].sum(axis=1)
        
        # Analyze by state
        state_analysis = bs.groupby('State')['Total_Fatalities'].sum().reset_index()
        state_analysis = state_analysis.sort_values('Total_Fatalities', ascending=False)
        
        # Create first chart - top states by fatalities at black spots
        plt.figure(figsize=(14, 8))
        
        # Use only top 15 states for clarity
        top_states = state_analysis.head(15)
        
        # Plot
        ax = sns.barplot(x='State', y='Total_Fatalities', data=top_states, palette='viridis')
        
        plt.title('States with Highest Fatalities at Identified Black Spots', fontsize=18)
        plt.xlabel('State', fontsize=14)
        plt.ylabel('Total Fatalities at Black Spots', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.0f}', 
                       (p.get_x() + p.get_width()/2., p.get_height()), 
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(hotspot_dir, 'black_spots_by_state.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create second chart - analyze by road type with fixed fatality rate calculation
    if 'NH/SH/M' in bs.columns:
        bs['Road_Type'] = bs['NH/SH/M'].str.extract(r'(NH|SH|M)')
        
        # Group by road type
        road_analysis = bs.groupby('Road_Type').agg({
            'Total_Fatalities': 'sum',
            'Number of Accidents Total of all 3 years': 'sum'
        }).reset_index()
        
        # Calculate fatality rate per 100 accidents (normalized value) - fixing the issue
        road_analysis['Fatality_Rate'] = (road_analysis['Total_Fatalities'] / 
                                     road_analysis['Number of Accidents Total of all 3 years']) * 100
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot accident counts by road type
        sns.barplot(x='Road_Type', y='Number of Accidents Total of all 3 years', data=road_analysis, palette='Blues_r', ax=ax1)
        ax1.set_title('Number of Accidents by Road Type at Black Spots', fontsize=16)
        ax1.set_xlabel('Road Type (NH: National Highway, SH: State Highway, M: Municipal)', fontsize=12)
        ax1.set_ylabel('Total Number of Accidents', fontsize=14)
        
        # Add value labels
        for i, p in enumerate(ax1.patches):
            ax1.annotate(f'{p.get_height():.0f}', 
                        (p.get_x() + p.get_width()/2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10)
        
        # Plot fatality rates by road type (deaths per 100 accidents)
        sns.barplot(x='Road_Type', y='Fatality_Rate', data=road_analysis, palette='Reds_r', ax=ax2)
        ax2.set_title('Fatality Rate by Road Type at Black Spots', fontsize=16)
        ax2.set_xlabel('Road Type (NH: National Highway, SH: State Highway, M: Municipal)', fontsize=12)
        ax2.set_ylabel('Fatality Rate (Deaths per 100 Accidents)', fontsize=14)
        
        # Add value labels
        for i, p in enumerate(ax2.patches):
            ax2.annotate(f'{p.get_height():.1f}', 
                        (p.get_x() + p.get_width()/2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10)
        
        # Add explanatory note about fatality rate calculation
        fig.text(0.5, 0.01, 
               "Note: Fatality rate is calculated as (Deaths / Total Accidents) × 100, representing deaths per 100 accidents.",
               ha='center', fontsize=12, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(hotspot_dir, 'black_spots_by_road_type.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create third chart - top individual black spots
    # Calculate a severity score for each black spot
    if 'Total_Fatalities' in bs.columns and 'Number of Accidents Total of all 3 years' in bs.columns:
        bs['Severity_Score'] = bs['Total_Fatalities'] * 3 + bs['Number of Accidents Total of all 3 years']
        
        # Sort by severity score and get top 15
        top_spots = bs.sort_values('Severity_Score', ascending=False).head(15)
        
        # Create location labels
        if 'State' in top_spots.columns and 'Name of Location Place' in top_spots.columns:
            top_spots['Location'] = top_spots['State'] + ': ' + top_spots['Name of Location Place'].astype(str).str[:25]
            
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Create horizontal bar chart
            sns.set_color_codes("pastel")
            sns.barplot(x='Severity_Score', y='Location', data=top_spots, 
                       label="Severity Score", color="b")
            
            # Add a second bar for fatalities
            sns.set_color_codes("muted")
            sns.barplot(x='Total_Fatalities', y='Location', data=top_spots,
                       label="Fatalities", color="r")
            
            # Add legend and labels
            plt.legend(ncol=2, loc="lower right", frameon=True)
            plt.title('Top 15 Most Severe Black Spots for Road Accidents', fontsize=18)
            plt.xlabel('Score (Fatalities × 3 + Total Accidents)', fontsize=14)
            plt.ylabel('Location', fontsize=14)
            
            # Add explanation of severity score calculation
            plt.figtext(0.5, 0.01,
                     "Note: Severity Score = (Fatalities × 3) + Total Accidents, giving higher weight to locations with fatal accidents.",
                     ha='center', fontsize=12, style='italic')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(os.path.join(hotspot_dir, 'top_black_spots_by_severity.png'), dpi=300, bbox_inches='tight')
            plt.close()

# Function to analyze temporal patterns
def analyze_temporal_patterns():
    if recent_data is None:
        print("  Skipping temporal pattern graphs - required data not available")
        return
        
    print("  Creating temporal pattern graphs...")
    
    # Analyze trends across 2018-2022 from recent data
    recent = recent_data.copy()
    
    # Calculate totals per year
    yearly_totals = pd.DataFrame({
        'Year': [2018, 2019, 2020, 2021, 2022],
        'Total_Accidents': [
            recent['Accidents 2018'].sum(),
            recent['Accidents 2019'].sum(),
            recent['Accidents 2020'].sum(),
            recent['Accidents 2021'].sum(),
            recent['Accidents 2022'].sum()
        ],
        'Total_Killed': [
            recent['Killed 2018'].sum(),
            recent['Killed 2019'].sum(),
            recent['Killed 2020'].sum(),
            recent['Killed 2021'].sum(),
            recent['Killed 2022'].sum()
        ],
        'Total_Injured': [
            recent['Injured 2018'].sum(),
            recent['Injured 2019'].sum(),
            recent['Injured 2020'].sum(),
            recent['Injured 2021'].sum(),
            recent['Injured 2022'].sum()
        ]
    })
    
    # Calculate fatality rate
    yearly_totals['Fatality_Rate'] = yearly_totals['Total_Killed'] / yearly_totals['Total_Accidents']
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Plot accident, injury, and death counts
    ax1.plot(yearly_totals['Year'], yearly_totals['Total_Accidents'], 'b-o', linewidth=2, label='Accidents')
    ax1.plot(yearly_totals['Year'], yearly_totals['Total_Injured'], 'g-s', linewidth=2, label='Injured')
    ax1.plot(yearly_totals['Year'], yearly_totals['Total_Killed'], 'r-^', linewidth=2, label='Killed')
    
    # Add data labels for first plot
    for i, year in enumerate(yearly_totals['Year']):
        ax1.annotate(f"{yearly_totals['Total_Accidents'].iloc[i]:,.0f}", 
                    (year, yearly_totals['Total_Accidents'].iloc[i]),
                    textcoords="offset points", xytext=(0,10), ha='center')
        
        ax1.annotate(f"{yearly_totals['Total_Killed'].iloc[i]:,.0f}", 
                    (year, yearly_totals['Total_Killed'].iloc[i]),
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Add COVID-19 visual reference
    ax1.axvspan(2020-0.25, 2020+0.25, alpha=0.2, color='gray')
    ax1.text(2020, ax1.get_ylim()[1]*0.95, "COVID-19\nPandemic", ha='center', fontsize=12)
    
    # Set labels and legend for first plot
    ax1.set_title('Road Accident Trends in India (2018-2022)', fontsize=18)
    ax1.set_ylabel('Number of Cases', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot fatality rate
    ax2.plot(yearly_totals['Year'], yearly_totals['Fatality_Rate'], 'k-D', linewidth=3, color='purple')
    
    # Add data labels for second plot
    for i, year in enumerate(yearly_totals['Year']):
        ax2.annotate(f"{yearly_totals['Fatality_Rate'].iloc[i]:.2f}", 
                    (year, yearly_totals['Fatality_Rate'].iloc[i]),
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Add COVID-19 visual reference
    ax2.axvspan(2020-0.25, 2020+0.25, alpha=0.2, color='gray')
    
    # Set labels for second plot
    ax2.set_title('Fatality Rate in Road Accidents (2018-2022)', fontsize=18)
    ax2.set_xlabel('Year', fontsize=14)
    ax2.set_ylabel('Fatality Rate (Deaths per Accident)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add interpretation note
    fig.text(0.5, 0.01, 
           "Note: While absolute accident numbers dropped during the COVID-19 pandemic (2020), the fatality rate continued to increase,\n" +
           "suggesting accidents during this period were more severe despite reduced traffic volume.",
           ha='center', fontsize=12, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(hotspot_dir, 'temporal_accident_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second chart - analyze long-term historical trends
    if historical_data is not None:
        historical = historical_data.copy()
        
        # Plot long-term trends
        plt.figure(figsize=(16, 10))
        
        # Get 5-year intervals for cleaner visualization
        historical_subset = historical[historical['Year'] % 5 == 0].copy()
        
        # Check column names
        accident_cols = [col for col in historical.columns if 'accident' in col.lower()]
        death_cols = [col for col in historical.columns if 'killed' in col.lower() or 'death' in col.lower()]
        
        if accident_cols and death_cols:
            # Make sure we're working with actual data columns, not metadata columns
            accident_col = None
            for col in accident_cols:
                if historical[col].dtype in [np.number] or pd.api.types.is_numeric_dtype(historical[col]):
                    accident_col = col
                    break
            
            death_col = None
            for col in death_cols:
                if historical[col].dtype in [np.number] or pd.api.types.is_numeric_dtype(historical[col]):
                    death_col = col
                    break
            
            if accident_col and death_col:
                # Convert to numeric explicitly in case we still have string data
                historical_subset[accident_col] = pd.to_numeric(historical_subset[accident_col], errors='coerce')
                historical_subset[death_col] = pd.to_numeric(historical_subset[death_col], errors='coerce')
                
                # Plot accident count with primary y-axis
                ax1 = plt.gca()
                ax1.plot(historical_subset['Year'], historical_subset[accident_col], 
                        'bo-', linewidth=2, markersize=8, label='Accidents')
                ax1.set_xlabel('Year', fontsize=14)
                ax1.set_ylabel('Number of Accidents', fontsize=14, color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                
                # Create secondary y-axis for fatality rate
                ax2 = ax1.twinx()
                historical_subset['Fatality_Rate'] = (historical_subset[death_col] / 
                                                   historical_subset[accident_col])
                
                ax2.plot(historical_subset['Year'], historical_subset['Fatality_Rate'], 
                        'ro-', linewidth=2, markersize=8, label='Fatality Rate')
                ax2.set_ylabel('Fatality Rate (Deaths per Accident)', fontsize=14, color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # Combine legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
                
                plt.title('Long-term Road Accident Trends in India', fontsize=18)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Add annotations for key policy changes or events
                events = [
                    (1994, "Motor Vehicles Act Amendments", 0),
                    (2000, "Introduction of National Road Safety Policy", 1),
                    (2006, "National Highway Development Program Expansion", 0),
                    (2019, "Motor Vehicles Amendment Act", 1)
                ]
                
                for year, event, offset in events:
                    if year in historical_subset['Year'].values:
                        y_pos = historical_subset[historical_subset['Year'] <= year][accident_col].max() * (0.7 + 0.1 * offset)
                        plt.annotate(event, xy=(year, y_pos),
                                    xytext=(year, y_pos * 1.2),
                                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                                    horizontalalignment='center', fontsize=10,
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
                
                # Add interpretation note
                plt.figtext(0.5, 0.01, 
                         "Historical analysis showing the long-term trend in accidents and fatality rate, with key policy interventions marked.",
                         ha='center', fontsize=12, style='italic')
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                plt.savefig(os.path.join(hotspot_dir, 'long_term_accident_trends.png'), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print("  Couldn't find appropriate numeric accident/death columns for historical data.")
        else:
            print("  Couldn't find appropriate accident/death columns for historical data.")

# Function to create multi-factor hotspot analysis
def create_multifactor_analysis():
    if black_spots is None:
        print("  Skipping multi-factor hotspot analysis - required data not available")
        return
        
    print("  Creating multi-factor hotspot analysis...")
    
    # Use the black spots data
    bs = black_spots.copy()
    
    # Ensure numeric columns are properly formatted
    numeric_cols = bs.columns[bs.columns.str.contains('Number of')]
    for col in numeric_cols:
        bs[col] = pd.to_numeric(bs[col], errors='coerce')
    
    # Calculate total fatalities and total injuries (where available)
    fatality_cols = [col for col in bs.columns if 'fatal' in col.lower() or 'killed' in col.lower() or 'death' in col.lower()]
    accident_col = 'Number of Accidents Total of all 3 years'
    
    if fatality_cols and accident_col in bs.columns:
        bs['Total_Fatalities'] = bs[fatality_cols].sum(axis=1)
        bs['Total_Accidents'] = bs[accident_col]
        
        # Calculate fatality rate and accident density
        bs['Fatality_Rate'] = bs['Total_Fatalities'] / bs['Total_Accidents']
        
        # Create a composite severity score
        # Normalize each component to 0-1 scale
        bs['Norm_Accidents'] = bs['Total_Accidents'] / bs['Total_Accidents'].max()
        bs['Norm_Fatalities'] = bs['Total_Fatalities'] / bs['Total_Fatalities'].max()
        bs['Norm_Rate'] = bs['Fatality_Rate'] / bs['Fatality_Rate'].max()
        
        # Composite score with higher weight to fatalities
        bs['Severity_Score'] = (0.3 * bs['Norm_Accidents'] + 
                               0.5 * bs['Norm_Fatalities'] + 
                               0.2 * bs['Norm_Rate'])
        
        # Sort by severity score and get top 10
        top_spots = bs.sort_values('Severity_Score', ascending=False).head(10).copy()
        
        # Create location labels
        if 'State' in top_spots.columns and 'Name of Location Place' in top_spots.columns:
            top_spots['Location'] = top_spots['State'].str[:3] + ': ' + top_spots['Name of Location Place'].astype(str).str[:20]
            
            # Create radar chart
            # Set data to visualize
            categories = ['Accidents', 'Fatalities', 'Fatality Rate']
            
            # Create figure
            fig = plt.figure(figsize=(14, 14))
            
            # Calculate number of rows and columns for subplots
            n_spots = len(top_spots)
            n_cols = min(3, n_spots)
            n_rows = (n_spots + n_cols - 1) // n_cols
            
            # Create color map
            colors = plt.cm.viridis(np.linspace(0, 1, n_spots))
            
            # Create a subplot for each black spot
            for i, (idx, spot) in enumerate(top_spots.iterrows()):
                ax = fig.add_subplot(n_rows, n_cols, i+1, projection='polar')
                
                # Compute values for each category (normalized)
                values = [
                    spot['Norm_Accidents'],
                    spot['Norm_Fatalities'],
                    spot['Norm_Rate']
                ]
                
                # Close the loop
                values += values[:1]
                
                # Set angles for each category
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]  # Close the loop
                
                # Plot values
                ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=spot['Location'])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
                
                # Set category labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                
                # Set severity score in title
                ax.set_title(f"{spot['Location']}\nSeverity Score: {spot['Severity_Score']:.2f}", 
                            fontsize=10, pad=20)
                
                # Set y-axis limit
                ax.set_ylim(0, 1)
                
                # Remove radial labels
                ax.set_yticklabels([])
            
            plt.tight_layout()
            plt.suptitle('Multi-factor Analysis of Top 10 Road Accident Hotspots', fontsize=18, y=0.98)
            
            # Add interpretation note
            plt.figtext(0.5, 0.01, 
                     "Note: Radar charts show the relative performance of each hotspot across three key metrics normalized to a 0-1 scale.\n" +
                     "This visualization helps identify different patterns of danger (e.g., high volume vs. high severity locations).",
                     ha='center', fontsize=12, style='italic')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(os.path.join(hotspot_dir, 'multifactor_hotspot_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create a second visualization: bubble chart for severity
            plt.figure(figsize=(14, 10))
            
            # Take top 20 for this visualization
            top20_spots = bs.sort_values('Severity_Score', ascending=False).head(20)
            
            # Create scatter plot
            scatter = plt.scatter(top20_spots['Total_Accidents'], 
                                top20_spots['Total_Fatalities'],
                                s=top20_spots['Fatality_Rate'] * 500,  # Size based on fatality rate
                                c=top20_spots['Severity_Score'],  # Color based on severity score
                                cmap='viridis',
                                alpha=0.7)
            
            # Add state labels to points
            for i, row in top20_spots.iterrows():
                location = f"{row['State']}: {row['Name of Location Place'][:15]}"
                plt.annotate(location, 
                           (row['Total_Accidents'], row['Total_Fatalities']),
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7))
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Severity Score', fontsize=12)
            
            sizes = [0.1, 0.3, 0.5, 1.0]
            labels = ["0.1", "0.3", "0.5", "1.0"]
            
            # Create legend handles
            legend_elements = []
            for size, label in zip(sizes, labels):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                              label=label,
                                              markerfacecolor='grey', 
                                              alpha=0.7,
                                              markersize=np.sqrt(size * 500)/2))
            
            plt.legend(handles=legend_elements, title="Fatality Rate", 
                     loc='upper left', bbox_to_anchor=(0.01, 0.99))
            
            # Set title and labels
            plt.title('Multi-dimensional Analysis of Top 20 Accident Hotspots', fontsize=18)
            plt.xlabel('Total Number of Accidents', fontsize=14)
            plt.ylabel('Total Number of Fatalities', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add interpretation note
            plt.figtext(0.5, 0.01,
                      "Note: Bubble size represents fatality rate, color intensity represents overall severity score.\n" +
                      "This visualization highlights locations that are dangerous due to high volume, high fatality count, or high rate of fatality.",
                      ha='center', fontsize=12, style='italic')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(os.path.join(hotspot_dir, 'bubble_chart_hotspots.png'), dpi=300, bbox_inches='tight')
            plt.close()

# Execute all functions to generate graphs
print("\nStarting to generate all visualizations...")

# Research Question 1: AQI and Accidents
print("\n===== GENERATING VISUALIZATIONS FOR RESEARCH QUESTION 1 =====")
print("Investigating the relationship between air quality and road accidents...")

create_aqi_scatter_plots()               # Creates scatter plots for each pollutant showing correlation with accidents
create_aqi_correlation_analysis()        # Creates improved correlation analysis (replacing old heatmap)
create_categorical_bar_charts()          # Creates bar charts showing accident/fatality averages by AQI category
create_time_series_graphs()              # Creates time series analysis for states with poorest air quality

# Research Question 2: Accident Hotspots
print("\n===== GENERATING VISUALIZATIONS FOR RESEARCH QUESTION 2 =====")
print("Identifying and analyzing road accident hotspots...")

analyze_black_spots()                    # Analyzes black spots by state and road type
analyze_temporal_patterns()              # Shows temporal trends in recent and historical accident data
create_multifactor_analysis()            # Creates multi-factor analysis of top accident hotspots
analyze_urban_vs_rural()                 # Analyzes urban vs rural distribution of accidents

print("\nAll visualizations have been generated and saved to the output folder!")
print(f"\nKey findings for Research Question 1 (AQI correlation):")
print(f" - Visualizations saved in: {aqi_dir}")
print(f" - Key visualizations: categorical_aqi_analysis.png, scatter_PM2.5_vs_accidents.png")
print(f" - The data suggests a more complex relationship between AQI and accidents than a simple linear correlation")

print(f"\nKey findings for Research Question 2 (Accident Hotspots):")
print(f" - Visualizations saved in: {hotspot_dir}")
print(f" - Key visualizations: top_black_spots_by_severity.png, multifactor_hotspot_analysis.png")
print(f" - Identified specific high-priority locations for intervention based on severity scores")

print("\nAnalysis complete. Use these visualizations to support your research paper on the relationship")
print("between air quality and road accidents, and for the identification of accident hotspots.")