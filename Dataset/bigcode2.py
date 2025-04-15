import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Super-safe numerical operations
def ultra_safe_division(numerator, denominator, default=0.0):
    """Division that handles all edge cases"""
    try:
        if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
            return default
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        if np.isnan(numerator) or np.isinf(numerator):
            return default
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            return default
        if abs(result) > 1e10:  # Cap extremely large values
            return 1e10 if result > 0 else -1e10
        return result
    except:
        return default

def ultra_safe_array(array):
    """Convert any array to a safe array with no NaN or inf values"""
    try:
        array = np.array(array, dtype=float)
        array[~np.isfinite(array)] = 0.0
        return array
    except:
        return np.zeros(len(array) if hasattr(array, '__len__') else 1)

class MinimalRoadSafetyAnalyzer:
    """
    A minimal version of the Road Safety Analyzer focused on numerical stability.
    """
    def __init__(self, data_folder='./'):
        """Initialize with minimal required datasets"""
        self.data_folder = data_folder
        print("Loading datasets...")
        
        try:
            # Load only required datasets
            self.historical_data = pd.read_csv(f'{data_folder}1970-2021 data.csv')
            
            # Immediately clean data
            self._clean_dataframe(self.historical_data)
            
            print("Historical data loaded and cleaned.")
        except Exception as e:
            print(f"Error loading datasets: {e}")
            raise

    def _clean_dataframe(self, df):
        """Aggressively clean a dataframe of any problematic values"""
        for col in df.select_dtypes(include=['number']).columns:
            # Replace inf with NaN then fill with median or 0
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            median = df[col].median()
            if np.isnan(median):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(median)
            
            # Clip extreme values to 99th percentile
            if len(df) > 10:  # Only if we have enough data points
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
                    df[col] = df[col].clip(lower, upper)

    def basic_national_trends(self, years_to_predict=5):
        """
        Generate very basic national trends with minimal processing.
        """
        print("\n===== BASIC NATIONAL TRENDS =====")
        
        results = {}
        # Only focus on accidents for debugging
        column = 'Total Number of Road Accidents (in numbers)'
        
        try:
            # Create a clean copy of the data
            df = self.historical_data[['Year', column]].copy()
            df = df.rename(columns={column: 'value'})
            
            # Drop any problematic rows
            df = df.dropna()
            
            # Sort by year
            df = df.sort_values('Year')
            
            # Extract recent years (last 10 years or all if less)
            recent_years = min(10, len(df))
            recent_data = df.iloc[-recent_years:]
            
            # Calculate average year-to-year change
            recent_data['prev_value'] = recent_data['value'].shift(1)
            recent_data = recent_data.dropna()
            recent_data['change'] = recent_data['value'] - recent_data['prev_value']
            
            # Use median change for stability (instead of mean)
            median_change = recent_data['change'].median()
            if np.isnan(median_change):
                median_change = 0
            
            # Get the most recent value
            latest_year = int(df['Year'].max())
            latest_value = float(df.loc[df['Year'] == latest_year, 'value'].iloc[0])
            
            # Generate predictions
            pred_years = list(range(latest_year + 1, latest_year + years_to_predict + 1))
            predictions = []
            
            current = latest_value
            for _ in range(years_to_predict):
                current += median_change
                # Ensure value is positive and not too large
                current = max(0, min(current, latest_value * 2))
                predictions.append(current)
            
            # Display results
            print(f"\nPredictions for {column.split('(')[0].strip()}:")
            print(f"Historical data range: {df['Year'].min()} to {df['Year'].max()}")
            print(f"Latest value ({latest_year}): {latest_value:.1f}")
            print(f"Projected average annual change: {median_change:.1f}")
            
            for year, pred in zip(pred_years, predictions):
                print(f"Year {year} prediction: {pred:.1f}")
            
            results[column] = {
                'years': pred_years,
                'predictions': predictions
            }
            
            return results
            
        except Exception as e:
            print(f"Error in basic national trends: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def create_minimal_visualization(self, predictions, output_folder='./outputs'):
        """Create minimal visualization with aggressive error checking"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        try:
            if not predictions:
                print("No predictions available to visualize")
                return {}
            
            # Get the first prediction set
            metric = list(predictions.keys())[0]
            pred_data = predictions[metric]
            
            # Verify we have valid prediction data
            if 'years' not in pred_data or 'predictions' not in pred_data:
                print("Prediction data is incomplete")
                return {}
            
            years = pred_data['years']
            preds = ultra_safe_array(pred_data['predictions'])
            
            # Safety check for years
            if not years or len(years) == 0:
                print("No prediction years available")
                return {}
            
            # Create a simple figure
            plt.figure(figsize=(10, 6))
            
            # Historical data
            hist_data = self.historical_data[['Year', metric]].copy()
            self._clean_dataframe(hist_data)
            
            # Plot historical data if available
            if not hist_data.empty:
                plt.plot(hist_data['Year'], hist_data[metric], 'o-', label='Historical')
            
            # Plot predictions
            plt.plot(years, preds, 's--', color='red', label='Predicted')
            
            # Add labels
            plt.title(f"Prediction for {metric.split('(')[0].strip()}")
            plt.xlabel('Year')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Save figure
            output_path = f'{output_folder}/minimal_prediction.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"- Chart saved to {output_path}")
            
            return {'prediction_chart': output_path}
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def export_minimal_data(self, predictions, output_folder='./outputs'):
        """Export minimal prediction data with aggressive error checking"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        try:
            if not predictions:
                print("No predictions available to export")
                return {}
            
            # Create a simple DataFrame for the predictions
            data = {'Year': []}
            
            # Add each prediction set
            for metric, pred_data in predictions.items():
                if 'years' in pred_data and 'predictions' in pred_data:
                    # Only add the first time
                    if not data['Year']:
                        data['Year'] = pred_data['years']
                    
                    # Clean name
                    name = metric.split('(')[0].strip().replace('Total Number of ', '')
                    
                    # Ensure clean values
                    clean_values = []
                    for val in pred_data['predictions']:
                        if isinstance(val, (int, float)) and np.isfinite(val):
                            clean_values.append(val)
                        else:
                            clean_values.append(0.0)
                    
                    # Ensure lengths match
                    if len(clean_values) < len(data['Year']):
                        clean_values.extend([0.0] * (len(data['Year']) - len(clean_values)))
                    elif len(clean_values) > len(data['Year']):
                        clean_values = clean_values[:len(data['Year'])]
                    
                    data[name] = clean_values
            
            # Create DataFrame and export
            if data['Year']:
                df = pd.DataFrame(data)
                output_path = f'{output_folder}/minimal_predictions.xlsx'
                df.to_excel(output_path, index=False)
                print(f"- Prediction data saved to {output_path}")
                return {'prediction_data': output_path}
            else:
                print("No valid prediction data to export")
                return {}
                
        except Exception as e:
            print(f"Error exporting data: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def generate_minimal_report(self, output_folder='./outputs'):
        """Generate a minimal report with aggressive error handling"""
        print("\n===== GENERATING MINIMAL REPORT =====")
        
        try:
            # Generate basic predictions
            predictions = self.basic_national_trends(years_to_predict=5)
            
            # Create visualization
            viz_paths = self.create_minimal_visualization(predictions, output_folder)
            
            # Export data
            data_paths = self.export_minimal_data(predictions, output_folder)
            
            print("\n===== MINIMAL REPORT GENERATION COMPLETE =====")
            print(f"Outputs have been saved to {output_folder}")
            
            return {
                'visualizations': viz_paths,
                'data_exports': data_paths
            }
        except Exception as e:
            print(f"Error generating minimal report: {e}")
            import traceback
            traceback.print_exc()
            return {}

def main():
    """Main function with minimal operations"""
    try:
        print("="*80)
        print("MINIMAL ROAD SAFETY PREDICTION SYSTEM")
        print("="*80)
        print("\nInitializing Minimal Road Safety Analyzer...")
        
        # Create analyzer with minimal functionality
        analyzer = MinimalRoadSafetyAnalyzer()
        
        # Generate minimal report
        analyzer.generate_minimal_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()