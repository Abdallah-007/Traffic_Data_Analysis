# Chicago Traffic Crashes Analysis

This project analyzes Chicago traffic crash data to identify patterns, predict fatalities, and provide safety recommendations.

## Dataset

The analysis uses traffic crash data from the City of Chicago Data Portal (2015-2023):
- 934,000+ crash incidents
- 50+ features per crash (time, location, conditions, severity, etc.)
- Severe class imbalance with only ~0.1% of crashes being fatal

## Project Components

This project consists of several components:

1. **Data Processing**: MongoDB was used for initial data storage and processing
   - MapReduce operations for aggregating crashes by weather, time, etc.
   - Data cleaning and feature engineering

2. **Data Analysis**: Python scripts for analyzing crash patterns
   - Temporal patterns (day of week, hour of day)
   - Environmental factors (weather, road conditions, lighting)
   - Severity factors (conditions associated with fatalities)

3. **Machine Learning**: Predictive modeling for fatal crashes
   - Random Forest classification
   - Class imbalance handling techniques
   - Feature importance analysis

4. **Visualization**: Improved visualizations showing crash patterns
   - Interactive Plotly charts
   - Time series analysis
   - Geographic heatmaps

## Key Files

- `traffic_crash_demo.py`: Streamlit demo application
- `traffic_data_analysis.py`: Main analysis script with visualizations
- `traffic_dashboard.py`: Original dashboard (reference)
- `mapReduceAGG.js`: MongoDB MapReduce operations
- `improved_visuals/`: Directory containing visualization images

## Running the Streamlit Demo

Follow these steps to run the Streamlit demo:

1. **Set up your environment**:
   ```bash
   # Create a virtual environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install the required packages
   pip install streamlit pandas numpy matplotlib seaborn plotly pillow
   ```

2. **Run the Streamlit app**:
   ```bash
   cd /path/to/Big_Data
   streamlit run traffic_crash_demo.py
   ```

3. **Access the app**:
   The app will be available at http://localhost:8501 in your web browser.

## Key Findings

- **Temporal Patterns**: Rush hours (7-9 AM, 4-6 PM) show highest crash volumes, with Friday being the most accident-prone day.
- **Environmental Factors**: Most crashes occur in clear weather and dry road conditions, but adverse conditions show higher risk per exposure hour.
- **Severity Factors**: Night driving (midnight-5 AM) has the highest fatality rate despite fewer crashes.
- **Predictive Insights**: Hour of day and lighting conditions are the strongest predictors of fatal crashes.

## Class Imbalance Challenge

The severe class imbalance (0.1% fatal crashes) presented challenges for predictive modeling:
- Baseline model achieved 99.89% accuracy but completely failed to predict fatal crashes (0% recall)
- With class weighting and threshold optimization, we improved recall for fatal crashes to 9%
- This demonstrates the challenge of predicting rare but critical events

## Recommendations

Based on the analysis, we recommend:

1. **For Traffic Management**:
   - Increase patrol presence during peak crash hours
   - Target Friday afternoon rush hour for special attention
   - Focus on night-time driving safety campaigns

2. **For Infrastructure**:
   - Improve street lighting in high-fatality areas
   - Enhance road drainage in areas prone to weather-related crashes
   - Add traffic calming measures in high-risk corridors

3. **For Public Awareness**:
   - Develop targeted safety campaigns for night drivers
   - Create weather alerts that include driving safety tips
   - Educate on the dangers of rush hour distracted driving 