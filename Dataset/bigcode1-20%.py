import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

class RoadSafetyAnalyzer:
    """
    A comprehensive system for analyzing and predicting road safety metrics in India.
    This class handles data preprocessing, statistical analysis, prediction modeling,
    hotspot identification, and visualization of results.
    """
    
    def __init__(self, data_folder='./'):
        """
        Initialize the analyzer by loading all datasets.
        
        Parameters:
        -----------
        data_folder : str
            Path to the folder containing all data files
        """
        print("Loading datasets...")
        # Load all datasets
        try:
            self.historical_data = pd.read_csv(f'{data_folder}1970-2021 data.csv')
            self.state_data = pd.read_csv(f'{data_folder}2018-2022 data.csv')
            self.age_sex_data = pd.read_csv(f'{data_folder}age_sex_corrected.csv')
            self.air_quality = pd.read_csv(f'{data_folder}airquality.csv')
            self.collision_type = pd.read_csv(f'{data_folder}collision_type_corrected.csv')
            self.junction_types = pd.read_csv(f'{data_folder}junction_types_corrected.csv')
            self.license_data = pd.read_csv(f'{data_folder}license_corrected.csv')
            self.road_features = pd.read_csv(f'{data_folder}road_features_corrected.csv')
            self.locations = pd.read_csv(f'{data_folder}statewise_loactions.csv')
            self.traffic_control = pd.read_csv(f'{data_folder}traffic_control_corrected.csv')
            self.traffic_violations = pd.read_csv(f'{data_folder}traffic_violation_corrected.csv')
            self.weather_data = pd.read_csv(f'{data_folder}Weather conditions.csv')
            self.black_spots = pd.read_csv(f'{data_folder}Black Spots.csv')
            
            print("All datasets loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error: {e.filename} not found. Please ensure all data files are in the specified directory.")
            raise
        except Exception as e:
            print(f"Error loading datasets: {e}")
            raise
        
        # Preprocess the datasets
        self._preprocess_data()
        
    def _preprocess_data(self):
        """
        Preprocess all datasets for analysis:
        - Clean missing values
        - Merge geographic information
        - Calculate derived metrics
        """
        print("Preprocessing datasets...")
        # Clean historical data
        self.historical_data.dropna(inplace=True)
        
        # Merge state data with locations for geospatial analysis
        self.state_data = pd.merge(
            self.state_data, 
            self.locations,
            on='State/UT', 
            how='left'
        )
        
        # Clean black spots data
        self.black_spots['Starting'] = self.black_spots['Starting'].astype(str)
        self.black_spots['Ending to'] = self.black_spots['Ending to'].astype(str)
        
        # Calculate accident rates and safety metrics
        self._calculate_safety_metrics()
        
        print("Data preprocessing complete.")
    
    def _calculate_safety_metrics(self):
        """
        Calculate various safety metrics for analysis:
        - Accident rates per population
        - Fatality rates
        - Injury severity
        """
        # Using the most recent year data (2022)
        latest_data = self.state_data[self.state_data['S.No'] <= 36].copy()
        
        # Calculate accidents per lakh population (using approximate population data)
        latest_data['Accident_Rate_Population'] = latest_data['Accidents_2022'] / 100000
        
        # Calculate fatality rate (deaths per 100 accidents)
        latest_data['Fatality_Rate'] = (latest_data['Killed_2022'] / latest_data['Accidents_2022']) * 100
        
        # Calculate injury severity (injuries per accident)
        latest_data['Injury_Severity'] = latest_data['Injured_2022'] / latest_data['Accidents_2022']
        
        # Calculate risk index combining fatality rate and accident frequency
        max_accidents = latest_data['Accidents_2022'].max()
        latest_data['Risk_Index'] = (
            (0.7 * latest_data['Fatality_Rate'] / latest_data['Fatality_Rate'].max()) + 
            (0.3 * latest_data['Accidents_2022'] / max_accidents)
        ) * 100
        
        self.latest_data = latest_data

    def predict_national_trends(self, years_to_predict=10, display_efficiency=True):
        """
        Predict national road safety trends for specified number of years.
        
        Parameters:
        -----------
        years_to_predict : int
            Number of years to forecast
        display_efficiency : bool
            Whether to print prediction accuracy metrics
            
        Returns:
        --------
        dict
            Dictionary containing prediction results for each metric
        """
        if display_efficiency:
            print("\n===== NATIONAL TRENDS PREDICTION =====")
            
        results = {}
        efficiency_results = {}
        prediction_columns = [
            'Total Number of Road Accidents (in numbers)',
            'Total Number of Persons Killed (in numbers)',
            'Total Number of Persons Injured (in numbers)'
        ]
        
        for column in prediction_columns:
            # Prepare the data
            df = self.historical_data[['Year', column]].copy()
            df = df.rename(columns={column: 't'})
            df.set_index('Year', inplace=True)
            
            # Create lagged features manually
            data = np.log(df + 1)  # Adding 1 to handle zeros
            data['t-1'] = data['t'].shift(1)
            data = data.dropna()
            
            # Split into X and y for training
            X_train = data['t-1'].values[:-1].reshape(-1, 1)
            y_train = data['t'].values[:-1]
            
            # Hold out the last year for validation
            X_test = data['t-1'].values[-1:].reshape(-1, 1)
            y_test = data['t'].values[-1:]
            
            # Train the model
            model = RandomForestRegressor(n_estimators=1000, random_state=42)
            model.fit(X_train, y_train)
            
            # Make validation prediction
            y_pred_log = model.predict(X_test)[0]
            y_pred = np.exp(y_pred_log) - 1
            y_actual = np.exp(y_test[0]) - 1
            
            # Calculate error
            mae = abs(y_actual - y_pred)
            
            # Retrain with all data including validation year
            X_full = data['t-1'].values.reshape(-1, 1)
            y_full = data['t'].values
            model.fit(X_full, y_full)
            
            # Display validation metrics
            if display_efficiency:
                metric_name = column.split('(')[0].strip()
                print(f"\nPrediction for {metric_name}:")
                print(f">Expected={y_actual:.1f}, Predicted={y_pred:.1f}")
                print(f"Mean Absolute Error: {mae:.3f}")
                percent_error = (mae/y_actual)*100
                print(f"Percent Error: {percent_error:.3f}%")
                efficiency_results[column] = {
                    'mae': mae,
                    'percent_error': percent_error
                }
            
            # Make predictions for future years
            future_predictions = []
            last_year = int(df.index.max())
            last_value = data['t'].iloc[-1]
            
            for i in range(years_to_predict):
                # Use the previous prediction as input
                next_input = np.array([last_value]).reshape(1, -1)
                next_log_pred = model.predict(next_input)[0]
                next_pred = np.exp(next_log_pred) - 1
                
                future_predictions.append(next_pred)
                last_value = next_log_pred  # Use log prediction for next iteration
                
                if display_efficiency:
                    print(f"Year {last_year + i + 1} prediction: {next_pred:.1f}")
            
            results[column] = {
                'years': list(range(last_year + 1, last_year + years_to_predict + 1)),
                'predictions': future_predictions,
                'efficiency': efficiency_results.get(column, {})
            }
        
        # Generate inferences from the predictions
        if display_efficiency:
            self._generate_national_inferences(results)
            
        return results
    
    def _generate_national_inferences(self, results):
        """
        Generate insights and inferences from national predictions.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing prediction results
        """
        print("\n===== NATIONAL TRENDS INFERENCES =====")
        
        try:
            # Extract key metrics
            accident_preds = results['Total Number of Road Accidents (in numbers)']['predictions']
            fatality_preds = results['Total Number of Persons Killed (in numbers)']['predictions']
            injury_preds = results['Total Number of Persons Injured (in numbers)']['predictions']
            years = results['Total Number of Road Accidents (in numbers)']['years']
            
            # Calculate compound annual growth rates (CAGR)
            accident_growth = ((accident_preds[-1] / accident_preds[0]) ** (1/(len(accident_preds)-1)) - 1) * 100
            fatality_growth = ((fatality_preds[-1] / fatality_preds[0]) ** (1/(len(fatality_preds)-1)) - 1) * 100
            injury_growth = ((injury_preds[-1] / injury_preds[0]) ** (1/(len(injury_preds)-1)) - 1) * 100
            
            # Calculate fatality rate trend
            fatality_rates = [fatality_preds[i] / accident_preds[i] * 100 for i in range(len(accident_preds))]
            fatality_rate_change = fatality_rates[-1] - fatality_rates[0]
            
            # Present inferences
            print(f"1. Annual Accident Growth Rate: {accident_growth:.2f}% per year")
            print(f"2. Annual Fatality Growth Rate: {fatality_growth:.2f}% per year")
            print(f"3. Annual Injury Growth Rate: {injury_growth:.2f}% per year")
            
            if accident_growth > 0:
                print(f"4. Road accidents are projected to increase by approximately {int(accident_preds[-1] - accident_preds[0])} incidents over the next {len(years)} years.")
            else:
                print(f"4. Road accidents are projected to decrease by approximately {int(accident_preds[0] - accident_preds[-1])} incidents over the next {len(years)} years.")
            
            if fatality_rate_change > 0:
                print(f"5. The fatality rate is expected to worsen from {fatality_rates[0]:.2f}% to {fatality_rates[-1]:.2f}% of accidents.")
            else:
                print(f"5. The fatality rate is expected to improve from {fatality_rates[0]:.2f}% to {fatality_rates[-1]:.2f}% of accidents.")
            
            # Identify critical year (if any)
            for i in range(1, len(accident_preds)):
                if (accident_preds[i] - accident_preds[i-1]) / accident_preds[i-1] > 0.05:  # 5% increase
                    print(f"6. Warning: A significant increase in accidents is predicted for {years[i]}.")
                    break
        except Exception as e:
            print(f"Error generating inferences: {e}")
    
    def predict_state_trends(self, years_to_predict=5, display_efficiency=True):
        """
        Predict state-wise road safety trends.
        
        Parameters:
        -----------
        years_to_predict : int
            Number of years to forecast
        display_efficiency : bool
            Whether to print prediction accuracy metrics
            
        Returns:
        --------
        dict
            Dictionary containing prediction results for each state and metric
        """
        if display_efficiency:
            print("\n===== STATE-WISE PREDICTION =====")
            
        metrics = ['Accidents', 'Killed', 'Injured']
        state_predictions = {}
        
        for state in self.state_data['State/UT'].unique():
            state_predictions[state] = {}
            
            if display_efficiency and state == 'Andhra Pradesh':
                # Only display detailed results for first state to avoid output clutter
                print(f"\nDetailed predictions for {state}:")
            
            for metric in metrics:
                # Extract data for this state and metric
                state_row = self.state_data[self.state_data['State/UT'] == state]
                
                # Create a time series
                cols = [f"{metric}_{year}" for year in range(2018, 2023)]
                if all(col in state_row.columns for col in cols):
                    values = state_row[cols].values[0]
                    years = list(range(2018, 2023))
                    
                    # Create dataframe
                    ts_data = pd.DataFrame({'t': values}, index=years)
                    
                    try:
                        # Create lagged features manually
                        data = np.log(ts_data + 1)  # Adding 1 to handle zeros
                        data['t-1'] = data['t'].shift(1)
                        data = data.dropna()
                        
                        if len(data) < 2:  # Need at least 2 data points (t and t-1)
                            raise ValueError(f"Not enough data points for {state} {metric}")
                        
                        # Split into X and y (holding out last year for validation)
                        X_train = data['t-1'].values[:-1].reshape(-1, 1)  # Features
                        y_train = data['t'].values[:-1]  # Target
                        
                        # Validation data
                        X_test = data['t-1'].values[-1:].reshape(-1, 1)
                        y_test = data['t'].values[-1:]
                        
                        # Train the model
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        
                        # Validate on the last year
                        y_pred_log = model.predict(X_test)[0]
                        y_pred = np.exp(y_pred_log) - 1
                        y_actual = np.exp(y_test[0]) - 1
                        
                        # Calculate error
                        mae = abs(y_actual - y_pred)
                        percent_error = (mae/max(1, y_actual))*100
                        
                        # Display validation results for the selected state
                        if display_efficiency and state == 'Andhra Pradesh':
                            print(f"  {metric} - Expected={y_actual:.1f}, Predicted={y_pred:.1f}, Error={percent_error:.2f}%")
                        
                        # Retrain on all data
                        X_full = data['t-1'].values.reshape(-1, 1)
                        y_full = data['t'].values
                        model.fit(X_full, y_full)
                        
                        # Make predictions for future years
                        future_predictions = []
                        last_year = int(ts_data.index.max())
                        last_value = data['t'].iloc[-1]
                        
                        for i in range(years_to_predict):
                            # Use the previous prediction as input
                            next_input = np.array([last_value]).reshape(1, -1)
                            next_log_pred = model.predict(next_input)[0]
                            next_pred = max(0, np.exp(next_log_pred) - 1)  # No negative predictions
                            
                            future_predictions.append(next_pred)
                            last_value = next_log_pred  # Use log prediction for next iteration
                            
                            # Display predictions for the selected state
                            if display_efficiency and state == 'Andhra Pradesh':
                                print(f"    Year {last_year + i + 1} {metric} prediction: {next_pred:.1f}")
                        
                        state_predictions[state][metric] = {
                            'years': list(range(last_year + 1, last_year + years_to_predict + 1)),
                            'predictions': future_predictions,
                            'efficiency': {
                                'mae': mae,
                                'percent_error': percent_error
                            }
                        }
                        
                    except Exception as e:
                        if display_efficiency and state == 'Andhra Pradesh':
                            print(f"  Error predicting {metric} for {state}: {e}")
                        state_predictions[state][metric] = {
                            'years': list(range(2023, 2023 + years_to_predict)),
                            'predictions': [None] * years_to_predict,
                            'efficiency': {'mae': None, 'percent_error': None}
                        }
        
        # Generate inferences from state predictions
        if display_efficiency:
            self._generate_state_inferences(state_predictions)
        
        return state_predictions
    
    def _generate_state_inferences(self, predictions):
        """
        Generate insights from state-level predictions.
        
        Parameters:
        -----------
        predictions : dict
            Dictionary containing state-level prediction results
        """
        print("\n===== STATE-WISE INFERENCES =====")
        
        try:
            # Calculate 2023 predicted accidents for all states
            accident_predictions = {}
            for state, data in predictions.items():
                if 'Accidents' in data and data['Accidents']['predictions'] and data['Accidents']['predictions'][0] is not None:
                    accident_predictions[state] = data['Accidents']['predictions'][0]
            
            # Find states with highest predicted accidents
            top_accident_states = sorted(accident_predictions.items(), key=lambda x: x[1], reverse=True)[:5]
            print("1. States with highest predicted accidents for 2023:")
            for i, (state, value) in enumerate(top_accident_states):
                print(f"   {i+1}. {state}: {value:.1f}")
            
            # Find states with highest growth rate
            growth_rates = {}
            for state, data in predictions.items():
                if 'Accidents' in data and len(data['Accidents']['predictions']) >= 2 and data['Accidents']['predictions'][0] is not None and data['Accidents']['predictions'][-1] is not None:
                    initial = data['Accidents']['predictions'][0]
                    final = data['Accidents']['predictions'][-1]
                    if initial > 0:
                        growth_rate = ((final / initial) ** (1/(len(data['Accidents']['predictions'])-1)) - 1) * 100
                        growth_rates[state] = growth_rate
            
            top_growth_states = sorted(growth_rates.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\n2. States with highest accident growth rates:")
            for i, (state, rate) in enumerate(top_growth_states):
                print(f"   {i+1}. {state}: {rate:.2f}% per year")
            
            # Identify states with improving safety (declining accidents)
            improving_states = sorted(growth_rates.items(), key=lambda x: x[1])[:5]
            print("\n3. States with improving road safety (declining accidents):")
            for i, (state, rate) in enumerate(improving_states):
                print(f"   {i+1}. {state}: {rate:.2f}% per year")
            
            # Calculate fatality rates for 2023
            fatality_rates = {}
            for state, data in predictions.items():
                if ('Accidents' in data and 'Killed' in data and 
                    data['Accidents']['predictions'][0] is not None and 
                    data['Killed']['predictions'][0] is not None and 
                    data['Accidents']['predictions'][0] > 0):
                    fatality_rate = (data['Killed']['predictions'][0] / data['Accidents']['predictions'][0]) * 100
                    fatality_rates[state] = fatality_rate
            
            # States with highest fatality rates
            top_fatality_states = sorted(fatality_rates.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\n4. States with highest predicted fatality rates for 2023:")
            for i, (state, rate) in enumerate(top_fatality_states):
                print(f"   {i+1}. {state}: {rate:.2f}%")
                
        except Exception as e:
            print(f"Error generating state inferences: {e}")
    
    def identify_hotspots(self, display_results=True):
        """
        Identify current hotspots based on black spots data.
        
        Parameters:
        -----------
        display_results : bool
            Whether to print hotspot analysis results
            
        Returns:
        --------
        dict
            Dictionary containing hotspot analysis results
        """
        if display_results:
            print("\n===== CURRENT HOTSPOTS ANALYSIS =====")
        
        # Group by state and count black spots
        hotspot_counts = self.black_spots.groupby('State')['S.no'].count().reset_index()
        hotspot_counts.columns = ['State', 'Black_Spot_Count']
        
        # Get top states with most black spots
        top_hotspot_states = hotspot_counts.sort_values('Black_Spot_Count', ascending=False).head(10)
        
        # Identify the most dangerous spots (highest fatalities)
        dangerous_spots = self.black_spots.sort_values('Number of Fatalities Total of all 3 years', ascending=False).head(20)
        
        # Analyze hotspot characteristics
        highway_counts = self.black_spots['NH/SH/M'].str.extract(r'(NH[- ]?\d*)', expand=False).value_counts().reset_index()
        highway_counts.columns = ['Highway', 'Count']
        
        if display_results:
            print("\nStates with Most Black Spots:")
            for i, (_, row) in enumerate(top_hotspot_states.head(5).iterrows()):
                print(f"{i+1}. {row['State']}: {row['Black_Spot_Count']} black spots")
            
            print("\nMost Dangerous Black Spots (by fatalities):")
            for i, (_, row) in enumerate(dangerous_spots.head(5).iterrows()):
                print(f"{i+1}. {row['State']} - {row['Name of Location Place']}: {row['Number of Fatalities Total of all 3 years']} fatalities")
            
            print("\nMost Common Highway Types in Black Spots:")
            for i, (_, row) in enumerate(highway_counts.head(5).iterrows()):
                print(f"{i+1}. {row['Highway']}: {row['Count']} black spots")
            
            # Generate deeper insights
            self._analyze_hotspot_patterns(dangerous_spots.head(20))
        
        return {
            'state_counts': hotspot_counts,
            'top_states': top_hotspot_states,
            'dangerous_spots': dangerous_spots,
            'highway_analysis': highway_counts
        }
    
    def _analyze_hotspot_patterns(self, dangerous_spots):
        """
        Analyze patterns in the most dangerous hotspots.
        
        Parameters:
        -----------
        dangerous_spots : DataFrame
            DataFrame containing the most dangerous black spots
        """
        print("\n===== HOTSPOT PATTERN ANALYSIS =====")
        
        try:
            # Check for patterns in location types
            location_keywords = ['Junction', 'Bridge', 'Flyover', 'Chowk', 'Cross', 'Bypass', 'Ghat', 'Turn']
            location_counts = {keyword: 0 for keyword in location_keywords}
            
            for _, row in dangerous_spots.iterrows():
                location = str(row['Name of Location Place'])
                for keyword in location_keywords:
                    if keyword.lower() in location.lower():
                        location_counts[keyword] += 1
            
            print("1. Common Location Types in Dangerous Spots:")
            for keyword, count in sorted(location_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"   - {keyword}: {count} spots")
            
            # Analyze fatality trends
            fatality_trends = []
            for _, row in dangerous_spots.iterrows():
                if row['Number of Fatalities 2016'] > 0 and row['Number of Fatalities 2018'] > 0:
                    trend = (row['Number of Fatalities 2018'] - row['Number of Fatalities 2016']) / row['Number of Fatalities 2016']
                    fatality_trends.append(trend)
            
            avg_trend = np.mean(fatality_trends) * 100
            if avg_trend > 0:
                print(f"\n2. Fatality Trend: Average increase of {avg_trend:.2f}% in fatalities from 2016 to 2018")
            else:
                print(f"\n2. Fatality Trend: Average decrease of {abs(avg_trend):.2f}% in fatalities from 2016 to 2018")
            
            # Identify chronic hotspots (consistent high fatalities)
            chronic_spots = dangerous_spots[
                (dangerous_spots['Number of Fatalities 2016'] >= 3) & 
                (dangerous_spots['Number of Fatalities 2017'] >= 3) & 
                (dangerous_spots['Number of Fatalities 2018'] >= 3)
            ]
            
            print(f"\n3. Chronic Hotspots (consistently high fatalities): {len(chronic_spots)} spots")
            for i, (_, row) in enumerate(chronic_spots.head(3).iterrows()):
                print(f"   {i+1}. {row['State']} - {row['Name of Location Place']}: {row['Number of Fatalities 2016']}/{row['Number of Fatalities 2017']}/{row['Number of Fatalities 2018']} fatalities")
                
        except Exception as e:
            print(f"Error analyzing hotspot patterns: {e}")
    
    def predict_future_hotspots(self, display_efficiency=True):
        """
        Predict potential future hotspots based on trends and factors.
        
        Parameters:
        -----------
        display_efficiency : bool
            Whether to print prediction details
            
        Returns:
        --------
        dict
            Dictionary containing future hotspot predictions
        """
        if display_efficiency:
            print("\n===== FUTURE HOTSPOT PREDICTION =====")
            
        try:
            # Step 1: Merge state predictions with location and factor data
            state_accident_trends = self.predict_state_trends(years_to_predict=1, display_efficiency=False)
            
            # Extract 2023 accident predictions for each state
            accident_predictions = {}
            for state, data in state_accident_trends.items():
                if 'Accidents' in data and data['Accidents']['predictions'][0] is not None:
                    accident_predictions[state] = data['Accidents']['predictions'][0]
            
            prediction_df = pd.DataFrame({
                'State/UT': list(accident_predictions.keys()),
                'Predicted_Accidents_2023': list(accident_predictions.values())
            })
            
            # Step 2: Add environmental and infrastructure factors
            weather_data_copy = self.weather_data.copy()
            
            # Ensure column name consistency
            if 'State/UT' not in weather_data_copy.columns:
                if 'States/UTs' in weather_data_copy.columns:
                    weather_data_copy = weather_data_copy.rename(columns={'States/UTs': 'State/UT'})
                else:
                    # Get the name of the first column
                    first_col = weather_data_copy.columns[0]
                    weather_data_copy = weather_data_copy.rename(columns={first_col: 'State/UT'})
            
            merged_df = pd.merge(prediction_df, weather_data_copy, on='State/UT', how='left')
            merged_df = pd.merge(merged_df, self.locations, on='State/UT', how='left')
            
            # Get the latest air quality data (2022)
            air_quality_2022 = self.air_quality[self.air_quality['Year'] == 2022]
            
            # Adjust air quality column names if needed
            air_quality_cols = ['PM10_Annual_Average', 'PM2.5_Annual_Average']
            for col in air_quality_cols:
                if col not in air_quality_2022.columns:
                    # Try variations of column names
                    alt_names = [col, col.replace('_', ' '), f"{col.split('_')[0]} ({' '.join(col.split('_')[1:])})", 'PM10 (Annual Average)', 'PM2.5 (Annual Average)']
                    for alt in alt_names:
                        if alt in air_quality_2022.columns:
                            air_quality_2022[col] = air_quality_2022[alt]
                            break
            
            # Ensure we have required columns for merging
            required_cols = ['State/UT', 'PM10_Annual_Average', 'PM2.5_Annual_Average']
            for col in required_cols:
                if col not in air_quality_2022.columns:
                    air_quality_2022[col] = 0
            
            merged_df = pd.merge(merged_df, air_quality_2022[['State/UT', 'PM10_Annual_Average', 'PM2.5_Annual_Average']], 
                                on='State/UT', how='left')
            
            # Step 3: Calculate risk scores based on multiple factors
            if not merged_df.empty:
                # Look for relevant columns
                fog_columns = [col for col in merged_df.columns if 'Foggy' in col and 'Killed' in col]
                fog_column = fog_columns[0] if fog_columns else None
                
                # Normalize columns for scoring
                for col in ['Predicted_Accidents_2023', 'PM10_Annual_Average', 'PM2.5_Annual_Average']:
                    if col in merged_df.columns:
                        max_val = merged_df[col].max()
                        if max_val > 0:
                            merged_df[col] = merged_df[col] / max_val
                
                if fog_column and fog_column in merged_df.columns:
                    max_val = merged_df[fog_column].max()
                    if max_val > 0:
                        merged_df[fog_column] = merged_df[fog_column] / max_val
                
                # Create composite risk score
                merged_df['Risk_Score'] = (
                    merged_df.get('Predicted_Accidents_2023', pd.Series(0)) * 0.4 + 
                    merged_df.get(fog_column, pd.Series(0)) * 0.3 + 
                    merged_df.get('PM10_Annual_Average', pd.Series(0)) * 0.15 + 
                    merged_df.get('PM2.5_Annual_Average', pd.Series(0)) * 0.15
                ).fillna(0)
                
                # Determine high-risk states
                high_risk_states = merged_df.sort_values('Risk_Score', ascending=False).head(10)
                
                if display_efficiency:
                    print("\nTop High-Risk States for Future Hotspots:")
                    for i, (_, row) in enumerate(high_risk_states.head(5).iterrows()):
                        print(f"{i+1}. {row['State/UT']}: Risk Score = {row['Risk_Score']:.3f}")
                
                # Step 4: Analyze existing black spots for patterns in high-risk states
                high_risk_patterns = {}
                
                for state in high_risk_states['State/UT'].unique():
                    state_spots = self.black_spots[self.black_spots['State'] == state]
                    if not state_spots.empty:
                        # Analyze NH patterns
                        nh_counts = state_spots['NH/SH/M'].str.extract(r'(NH[- ]?\d*)', expand=False).value_counts()
                        
                        # Find common location types
                        location_keywords = ['Junction', 'Bridge', 'Flyover', 'Chowk', 'Cross', 'Bypass']
                        location_types = {}
                        
                        for _, row in state_spots.iterrows():
                            location = str(row['Name of Location Place'])
                            for keyword in location_keywords:
                                if keyword.lower() in location.lower():
                                    location_types[keyword] = location_types.get(keyword, 0) + 1
                        
                        high_risk_patterns[state] = {
                            'common_highways': nh_counts.index.tolist()[:3] if len(nh_counts) >= 3 else nh_counts.index.tolist(),
                            'common_locations': sorted(location_types.items(), key=lambda x: x[1], reverse=True)[:3],
                            'spot_count': len(state_spots)
                        }
                        
                        if display_efficiency and state in list(high_risk_states['State/UT'])[:3]:
                            print(f"\nHotspot Analysis for {state}:")
                            print(f"  Number of existing black spots: {len(state_spots)}")
                            print(f"  Most dangerous highways: {', '.join(high_risk_patterns[state]['common_highways'])}")
                            print("  Common location types:", 
                                  ", ".join([f"{k} ({v})" for k, v in high_risk_patterns[state]['common_locations']]) 
                                  if high_risk_patterns[state]['common_locations'] else "None identified")
                
                # Generate specific inferences for future hotspots
                if display_efficiency:
                    self._generate_hotspot_inferences(high_risk_states, high_risk_patterns)
                
                return {
                    'high_risk_states': high_risk_states,
                    'highway_patterns': high_risk_patterns
                }
            else:
                if display_efficiency:
                    print("No data available for hotspot prediction.")
                return {
                    'high_risk_states': pd.DataFrame(),
                    'highway_patterns': {}
                }
        except Exception as e:
            print(f"Error in predict_future_hotspots: {e}")
            return {
                'high_risk_states': pd.DataFrame(),
                'highway_patterns': {}
            }
    
    def _generate_hotspot_inferences(self, high_risk_states, highway_patterns):
        """
        Generate specific inferences for future hotspots.
        
        Parameters:
        -----------
        high_risk_states : DataFrame
            DataFrame containing high risk states
        highway_patterns : dict
            Dictionary containing highway patterns
        """
        print("\n===== FUTURE HOTSPOT INFERENCES =====")
        
        try:
            # Calculate the overall risk score distribution
            q25 = high_risk_states['Risk_Score'].quantile(0.25)
            q75 = high_risk_states['Risk_Score'].quantile(0.75)
            
            # Very high risk states (above 75th percentile)
            very_high_risk = high_risk_states[high_risk_states['Risk_Score'] > q75]
            
            print("1. States Requiring Immediate Attention:")
            for i, (_, row) in enumerate(very_high_risk.iterrows()):
                print(f"   {i+1}. {row['State/UT']}: Risk Score = {row['Risk_Score']:.3f}")
            
            # Identify states with existing patterns
            print("\n2. Specific High-Risk Road Sections to Monitor:")
            for state in very_high_risk['State/UT']:
                if state in highway_patterns:
                    patterns = highway_patterns[state]
                    highways = patterns['common_highways']
                    locations = patterns['common_locations']
                    
                    print(f"   {state}:")
                    for highway in highways[:2]:
                        location_types = [loc[0] for loc in locations]
                        location_str = " and ".join(location_types) if location_types else "various sections"
                        print(f"     - {highway}: Focus on {location_str}")
            
            # Recommendation for monitoring and intervention
            print("\n3. Recommended Monitoring Approach:")
            print("   - Establish automated traffic monitoring systems at identified high-risk locations")
            print("   - Prioritize enforcement during adverse weather conditions (especially fog and rain)")
            print("   - Deploy mobile road safety teams to high-risk highways during peak accident months")
            
            # Seasonal patterns if available
            if 'Rainy_Killed' in high_risk_states.columns and 'Foggy_Misty_Killed' in high_risk_states.columns:
                rainy_risk = high_risk_states['Rainy_Killed'].sum() > high_risk_states['Foggy_Misty_Killed'].sum()
                if rainy_risk:
                    print("\n4. Seasonal Factor: Rainy season shows higher fatality risk than foggy conditions")
                else:
                    print("\n4. Seasonal Factor: Fog and mist show higher fatality risk than rainy conditions")
            
        except Exception as e:
            print(f"Error generating hotspot inferences: {e}")
    
    def visualize_results(self, output_folder='./outputs'):
        """
        Create visualizations of the analysis results.
        
        Parameters:
        -----------
        output_folder : str
            Path to save visualization outputs
            
        Returns:
        --------
        dict
            Dictionary containing paths to generated visualizations
        """
        print("\n===== CREATING VISUALIZATIONS =====")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        try:
            # 1. National trends visualization
            national_predictions = self.predict_national_trends(years_to_predict=5, display_efficiency=False)
            
            plt.figure(figsize=(14, 8))
            
            metrics = [
                'Total Number of Road Accidents (in numbers)',
                'Total Number of Persons Killed (in numbers)',
                'Total Number of Persons Injured (in numbers)'
            ]
            
            for i, metric in enumerate(metrics):
                plt.subplot(1, 3, i+1)
                
                # Historical data
                hist_data = self.historical_data[['Year', metric]].dropna()
                plt.plot(hist_data['Year'], hist_data[metric], marker='o', linestyle='-', label='Historical')
                
                # Predictions
                preds = national_predictions[metric]
                plt.plot(preds['years'], preds['predictions'], marker='s', linestyle='--', color='red', label='Predicted')
                
                plt.title(metric.split('(')[0].strip())
                plt.xlabel('Year')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
            plt.tight_layout()
            plt.savefig(f'{output_folder}/national_trends.png', dpi=300, bbox_inches='tight')
            print(f"- National trends chart saved to {output_folder}/national_trends.png")
            
            # 2. Create India map with hotspots
            hotspot_data = self.identify_hotspots(display_results=False)
            
            # Create a map centered on India
            india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
            
            # Add heatmap of existing black spots
            if not self.black_spots.empty:
                # We need location data for each black spot
                # For simplicity, we'll use the state center coordinates
                black_spots_with_loc = pd.merge(
                    self.black_spots,
                    self.locations,
                    left_on='State',
                    right_on='State/UT',
                    how='left'
                )
                
                # Create heat data
                heat_data = []
                for _, spot in black_spots_with_loc.iterrows():
                    if pd.notna(spot['latitude']) and pd.notna(spot['longitude']):
                        # Weight by fatality count
                        weight = spot.get('Number of Fatalities Total of all 3 years', 1)
                        heat_data.append([spot['latitude'], spot['longitude'], weight])
                
                # Add the heatmap to the map
                HeatMap(heat_data).add_to(india_map)
            
            # Save the map
            india_map.save(f'{output_folder}/india_hotspots_map.html')
            print(f"- India hotspots map saved to {output_folder}/india_hotspots_map.html")
            
            # 3. Visualize state-wise predictions (top 10 states)
            state_predictions = self.predict_state_trends(years_to_predict=3, display_efficiency=False)
            
            # Extract and sort states by 2023 accident predictions
            state_accident_2023 = {}
            for state, data in state_predictions.items():
                if 'Accidents' in data and data['Accidents']['predictions'][0] is not None:
                    state_accident_2023[state] = data['Accidents']['predictions'][0]
            
            top_10_states = sorted(state_accident_2023.items(), key=lambda x: x[1], reverse=True)[:10]
            
            plt.figure(figsize=(14, 8))
            
            # Plot top 10 states' accident predictions for 2023-2025
            states = [s[0] for s in top_10_states]
            values_2023 = []
            values_2024 = []
            values_2025 = []
            
            for state in states:
                preds = state_predictions[state]['Accidents']['predictions']
                values_2023.append(preds[0] if len(preds) > 0 else 0)
                values_2024.append(preds[1] if len(preds) > 1 else 0)
                values_2025.append(preds[2] if len(preds) > 2 else 0)
            
            x = np.arange(len(states))
            width = 0.25
            
            plt.bar(x - width, values_2023, width, label='2023')
            plt.bar(x, values_2024, width, label='2024')
            plt.bar(x + width, values_2025, width, label='2025')
            
            plt.xlabel('States')
            plt.ylabel('Predicted Accidents')
            plt.title('Top 10 States by Predicted Accidents (2023-2025)')
            plt.xticks(x, states, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(f'{output_folder}/top_states_predictions.png', dpi=300, bbox_inches='tight')
            print(f"- State predictions chart saved to {output_folder}/top_states_predictions.png")
            
            # 4. Create a risk heatmap
            future_hotspots = self.predict_future_hotspots(display_efficiency=False)
            high_risk_states = future_hotspots['high_risk_states']
            
            risk_heatmap_path = None
            if not high_risk_states.empty and 'Risk_Score' in high_risk_states.columns:
                plt.figure(figsize=(12, 8))
                
                # Sort by risk score
                high_risk_states = high_risk_states.sort_values('Risk_Score')
                
                # Create horizontal bar chart
                plt.barh(high_risk_states['State/UT'], high_risk_states['Risk_Score'], color='orange')
                plt.xlabel('Risk Score')
                plt.ylabel('State/UT')
                plt.title('States at Highest Risk for Future Accident Hotspots')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                risk_heatmap_path = f'{output_folder}/risk_heatmap.png'
                plt.savefig(risk_heatmap_path, dpi=300, bbox_inches='tight')
                print(f"- Risk heatmap saved to {output_folder}/risk_heatmap.png")
            
            # 5. Fatality vs Accident rate visualization
            plt.figure(figsize=(12, 8))
            
            # Calculate fatality rates
            latest_data = self.latest_data.copy()
            
            if not latest_data.empty and 'Accidents_2022' in latest_data.columns and 'Killed_2022' in latest_data.columns:
                latest_data['Fatality_Rate'] = latest_data['Killed_2022'] / latest_data['Accidents_2022'] * 100
                
                # Filter out extreme values for better visualization
                scatter_data = latest_data[latest_data['Fatality_Rate'] < 50]
                
                # Create scatter plot
                plt.scatter(scatter_data['Accidents_2022'], scatter_data['Fatality_Rate'], 
                            s=scatter_data['Killed_2022']/10, alpha=0.7)
                
                # Add state labels to points
                for _, row in scatter_data.iterrows():
                    plt.annotate(row['State/UT'], (row['Accidents_2022'], row['Fatality_Rate']), 
                                fontsize=8, ha='center')
                
                plt.xlabel('Number of Accidents (2022)')
                plt.ylabel('Fatality Rate (deaths per 100 accidents)')
                plt.title('Accident Frequency vs. Severity by State (2022)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                fatality_plot_path = f'{output_folder}/fatality_analysis.png'
                plt.savefig(fatality_plot_path, dpi=300, bbox_inches='tight')
                print(f"- Fatality analysis plot saved to {output_folder}/fatality_analysis.png")
            else:
                fatality_plot_path = None
            
            return {
                'national_trends_chart': f'{output_folder}/national_trends.png',
                'hotspot_map': f'{output_folder}/india_hotspots_map.html',
                'state_predictions_chart': f'{output_folder}/top_states_predictions.png',
                'risk_heatmap': risk_heatmap_path,
                'fatality_analysis': fatality_plot_path
            }
        except Exception as e:
            print(f"Error in visualize_results: {e}")
            return {}

    def export_predictions(self, output_folder='./outputs'):
        """
        Export prediction data to Excel files.
        
        Parameters:
        -----------
        output_folder : str
            Path to save output files
            
        Returns:
        --------
        dict
            Dictionary containing paths to generated files
        """
        print("\n===== EXPORTING PREDICTIONS =====")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        try:
            # 1. Export national trends
            national_trends = self.predict_national_trends(display_efficiency=False)
            
            national_df = pd.DataFrame({
                'Year': national_trends['Total Number of Road Accidents (in numbers)']['years']
            })
            
            for metric in national_trends:
                short_name = metric.split('(')[0].strip().replace('Total Number of ', '')
                national_df[short_name] = national_trends[metric]['predictions']
            
            national_df.to_excel(f'{output_folder}/national_predictions.xlsx', index=False)
            print(f"- National predictions saved to {output_folder}/national_predictions.xlsx")
            
            # 2. Export state-wise predictions
            state_predictions = self.predict_state_trends(display_efficiency=False)
            
            # Structure the data for easy export
            for metric in ['Accidents', 'Killed', 'Injured']:
                state_data = []
                
                for state, data in state_predictions.items():
                    if metric in data:
                        state_row = {
                            'State/UT': state
                        }
                        
                        for i, year in enumerate(data[metric]['years']):
                            if i < len(data[metric]['predictions']):
                                state_row[f"{year}"] = data[metric]['predictions'][i]
                        
                        state_data.append(state_row)
                
                metric_df = pd.DataFrame(state_data)
                if not metric_df.empty:
                    metric_df.to_excel(f'{output_folder}/statewise_{metric.lower()}.xlsx', index=False)
                    print(f"- State-wise {metric.lower()} predictions saved to {output_folder}/statewise_{metric.lower()}.xlsx")
            
            # 3. Export hotspot predictions
            hotspot_predictions = self.predict_future_hotspots(display_efficiency=False)
            high_risk_states = hotspot_predictions['high_risk_states']
            
            if not high_risk_states.empty:
                high_risk_states.to_excel(f'{output_folder}/hotspot_predictions.xlsx', index=False)
                print(f"- Hotspot predictions saved to {output_folder}/hotspot_predictions.xlsx")
                hotspot_file = f'{output_folder}/hotspot_predictions.xlsx'
            else:
                hotspot_file = None
            
            # 4. Export comprehensive analysis report
            self._export_comprehensive_report(output_folder)
            
            return {
                'national_file': f'{output_folder}/national_predictions.xlsx',
                'state_files': [
                    f'{output_folder}/statewise_accidents.xlsx',
                    f'{output_folder}/statewise_killed.xlsx',
                    f'{output_folder}/statewise_injured.xlsx',
                ],
                'hotspot_file': hotspot_file,
                'report_file': f'{output_folder}/comprehensive_report.xlsx'
            }
        except Exception as e:
            print(f"Error in export_predictions: {e}")
            return {}
    
    def _export_comprehensive_report(self, output_folder):
        """
        Create and export a comprehensive analysis report.
        
        Parameters:
        -----------
        output_folder : str
            Path to save the report
        """
        try:
            # Create a writer to save multiple sheets
            with pd.ExcelWriter(f'{output_folder}/comprehensive_report.xlsx') as writer:
                # 1. Summary sheet
                summary_data = {
                    'Metric': [
                        'Total States/UTs Analyzed',
                        'Identified Black Spots',
                        'States with Increasing Accident Trend',
                        'States with Decreasing Accident Trend',
                        'National Accident Prediction (2023)',
                        'National Fatality Prediction (2023)'
                    ],
                    'Value': [
                        len(self.state_data['State/UT'].unique()),
                        len(self.black_spots),
                        0,  # Will update later
                        0,  # Will update later
                        0,  # Will update later
                        0   # Will update later
                    ]
                }
                
                # Get national predictions
                national_trends = self.predict_national_trends(display_efficiency=False)
                if national_trends:
                    summary_data['Value'][4] = national_trends['Total Number of Road Accidents (in numbers)']['predictions'][0]
                    summary_data['Value'][5] = national_trends['Total Number of Persons Killed (in numbers)']['predictions'][0]
                
                # Get state trends
                state_predictions = self.predict_state_trends(display_efficiency=False)
                increasing_states = 0
                decreasing_states = 0
                
                for state, data in state_predictions.items():
                    if 'Accidents' in data and len(data['Accidents']['predictions']) >= 2:
                        pred_2023 = data['Accidents']['predictions'][0]
                        pred_2024 = data['Accidents']['predictions'][1]
                        if pred_2023 and pred_2024:
                            if pred_2024 > pred_2023:
                                increasing_states += 1
                            else:
                                decreasing_states += 1
                
                summary_data['Value'][2] = increasing_states
                summary_data['Value'][3] = decreasing_states
                
                # Create and save summary sheet
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # 2. High-Risk States sheet
                future_hotspots = self.predict_future_hotspots(display_efficiency=False)
                high_risk_states = future_hotspots['high_risk_states']
                
                if not high_risk_states.empty:
                    # Clean up the dataframe for export
                    export_cols = ['State/UT', 'Risk_Score', 'Predicted_Accidents_2023']
                    export_df = high_risk_states[export_cols].sort_values('Risk_Score', ascending=False)
                    export_df.columns = ['State/UT', 'Risk Score', 'Predicted Accidents (2023)']
                    export_df.to_excel(writer, sheet_name='High Risk States', index=False)
                
                # 3. Top Black Spots sheet
                dangerous_spots = self.black_spots.sort_values('Number of Fatalities Total of all 3 years', ascending=False).head(20)
                
                if not dangerous_spots.empty:
                    # Select relevant columns
                    spots_cols = [
                        'S.no', 'State', 'Name of Location Place', 'NH/SH/M',
                        'Number of Accidents Total of all 3 years',
                        'Number of Fatalities Total of all 3 years'
                    ]
                    spots_df = dangerous_spots[spots_cols]
                    spots_df.columns = [
                        'ID', 'State', 'Location', 'Highway Type',
                        'Total Accidents (3 years)',
                        'Total Fatalities (3 years)'
                    ]
                    spots_df.to_excel(writer, sheet_name='Top Black Spots', index=False)
                
                # 4. Recommendations sheet
                recommendations = {
                    'Category': [
                        'Infrastructure',
                        'Infrastructure',
                        'Enforcement',
                        'Enforcement',
                        'Technology',
                        'Technology',
                        'Policy',
                        'Policy'
                    ],
                    'Recommendation': [
                        'Implement road safety improvements at identified black spots',
                        'Improve junction designs and signage at high-risk intersections',
                        'Increase enforcement during peak accident times and locations',
                        'Focus on speed enforcement and drunk driving prevention',
                        'Deploy automated traffic monitoring at high-risk locations',
                        'Implement AI-based early warning systems for accident prevention',
                        'Develop state-specific road safety action plans for high-risk states',
                        'Establish trauma care facilities near identified black spots'
                    ],
                    'Priority': [
                        'High',
                        'High',
                        'High',
                        'Medium',
                        'Medium',
                        'Low',
                        'High',
                        'Medium'
                    ]
                }
                
                rec_df = pd.DataFrame(recommendations)
                rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            print(f"- Comprehensive report saved to {output_folder}/comprehensive_report.xlsx")
        except Exception as e:
            print(f"Error creating comprehensive report: {e}")

    def generate_full_report(self, output_folder='./outputs'):
        """
        Generate a comprehensive analytical report including all predictions and insights.
        
        Parameters:
        -----------
        output_folder : str
            Path to save outputs
        """
        print("\n===== GENERATING COMPREHENSIVE REPORT =====")
        
        try:
            # Create visualizations
            viz_paths = self.visualize_results(output_folder)
            
            # Export data files
            data_paths = self.export_predictions(output_folder)
            
            # Generate national predictions with insights
            print("\n===== NATIONAL ROAD SAFETY OUTLOOK =====")
            national_trends = self.predict_national_trends()
            
            # Generate state-level predictions with insights
            print("\n===== STATE-LEVEL ROAD SAFETY PROJECTIONS =====")
            state_predictions = self.predict_state_trends()
            
            # Identify current hotspots with detailed analysis
            print("\n===== BLACK SPOT ANALYSIS =====")
            hotspots = self.identify_hotspots()
            
            # Predict future hotspots with recommendations
            print("\n===== FUTURE RISK AREAS PROJECTION =====")
            future_hotspots = self.predict_future_hotspots()
            
            print("\n===== REPORT GENERATION COMPLETE =====")
            print(f"All outputs have been saved to {output_folder}")
            
            return {
                'visualizations': viz_paths,
                'data_exports': data_paths
            }
        except Exception as e:
            print(f"Error generating full report: {e}")
            return {}


def main():
    """
    Main function to run the Road Safety Analyzer with comprehensive outputs.
    """
    try:
        # Initialize the analyzer
        print("="*80)
        print("ROAD SAFETY ANALYSIS AND PREDICTION SYSTEM")
        print("="*80)
        print("\nInitializing Road Safety Analyzer...")
        
        analyzer = RoadSafetyAnalyzer()
        
        # Generate comprehensive report with all outputs
        analyzer.generate_full_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()