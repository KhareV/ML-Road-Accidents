import os

# Define the mapping of current filenames to new filenames
file_mapping = {
  "pm25_vs_accidents_by_state.png": "fig1.png",
  "top_black_spots_by_severity.png": "fig2.png",
  "visibility_safety_impact.png": "fig3.png",
  "black_spots_by_state.png": "fig4.png",
  "urban_vs_rural_distribution.png": "fig5.png",
  "long_term_accident_trends.png": "fig6.png",
  "pollution_accident_hotspots.png": "fig7.png",
  "pollution_accident_parallel_trends.png": "fig8.png",
  "temporal_accident_trends.png": "fig9.png",
  "scatter_SO2_vs_accidents.png": "fig10.png",
  "scatter_NO2_vs_accidents.png": "fig11.png",
  "pollution_accident_correlation_2021.png": "fig12.png",
  "multifactor_hotspot_analysis.png": "fig13.png",
  "time_series_pm25_vs_accidents.png": "fig14.png",
}

# Directory containing the files
directory = "/home/abeer/Downloads/git/ML-Road-Accidents/output_Final copy"

# Rename files based on the mapping
for current_name, new_name in file_mapping.items():
  current_path = os.path.join(directory, current_name)
  new_path = os.path.join(directory, new_name)
  try:
    os.rename(current_path, new_path)
    print(f"Renamed: {current_name} -> {new_name}")
  except FileNotFoundError:
    print(f"File not found: {current_name}")
  except Exception as e:
    print(f"Error renaming {current_name} to {new_name}: {e}")