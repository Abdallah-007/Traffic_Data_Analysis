import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import datetime
import os

# Create a directory for visualizations
os.makedirs('improved_visuals', exist_ok=True)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.figsize'] = (12, 8)
colors = sns.color_palette("viridis", 8)

# Load data from CSV instead of MongoDB
print("Loading data from CSV file...")
try:
    # Try to load the ML-ready data
    df = pd.read_csv('traffic_data.crashes_ml_ready.csv')
    print(f"Loaded {len(df)} records from traffic_data.crashes_ml_ready.csv")
except Exception as e:
    print(f"Error loading ML-ready CSV: {e}")
    try:
        # Try to load the original data
        df = pd.read_csv('Traffic_Crashes_-_Crashes_20250504.csv')
        print(f"Loaded {len(df)} records from Traffic_Crashes_-_Crashes_20250504.csv")
        
        # Process data to match the expected format
        if 'CRASH_DATE' in df.columns:
            df['crash_date'] = pd.to_datetime(df['CRASH_DATE'])
            df['day_of_week'] = df['crash_date'].dt.dayofweek + 1  # 1=Sunday to 7=Saturday
            df['hour'] = df['crash_date'].dt.hour
            
        # Rename columns to match expected format
        column_mapping = {
            'WEATHER_CONDITION': 'weather', 
            'LIGHTING_CONDITION': 'lighting',
            'ROADWAY_SURFACE_COND': 'road_condition',
            'INJURIES_TOTAL': 'injuries',
            'INJURIES_FATAL': 'fatal'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
                
        # Create binary fatal column
        if 'fatal' in df.columns:
            df['fatal'] = (df['fatal'] > 0).astype(int)
    except Exception as e2:
        print(f"Error loading original CSV: {e2}")
        print("Creating sample data for demonstration...")
        
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 10000
        
        # Generate days of week and hours
        days = np.random.randint(1, 8, n_samples)
        hours = np.random.randint(0, 24, n_samples)
        
        # Generate categorical features with realistic distributions
        weather_conditions = np.random.choice(
            ['CLEAR', 'RAIN', 'SNOW', 'CLOUDY/OVERCAST', 'FOG/SMOKE/HAZE', 'SLEET/HAIL', 'UNKNOWN'], 
            n_samples, 
            p=[0.7, 0.1, 0.05, 0.08, 0.03, 0.02, 0.02]
        )
        
        lighting_conditions = np.random.choice(
            ['DAYLIGHT', 'DARKNESS, LIGHTED ROAD', 'DARKNESS', 'DAWN', 'DUSK', 'UNKNOWN'], 
            n_samples, 
            p=[0.6, 0.2, 0.1, 0.04, 0.05, 0.01]
        )
        
        road_conditions = np.random.choice(
            ['DRY', 'WET', 'SNOW OR SLUSH', 'ICE', 'UNKNOWN'], 
            n_samples,
            p=[0.7, 0.15, 0.05, 0.05, 0.05]
        )
        
        # Generate injuries and fatalities
        injuries = np.random.poisson(0.5, n_samples)
        fatal = np.zeros(n_samples)
        fatal_rate = 0.001  # 0.1% fatal rate
        fatal_idx = np.random.choice(n_samples, int(n_samples * fatal_rate), replace=False)
        fatal[fatal_idx] = 1
        
        # Create DataFrame
        df = pd.DataFrame({
            'day_of_week': days,
            'hour': hours,
            'weather': weather_conditions,
            'lighting': lighting_conditions,
            'road_condition': road_conditions,
            'injuries': injuries,
            'fatal': fatal.astype(int)
        })

# Debug: Print all available columns
print("Available columns in the dataset:")
for col in df.columns:
    print(f"- {col}")

# Drop _id column if it exists
if '_id' in df.columns:
    print("Dropping _id column...")
    df = df.drop('_id', axis=1)

# Handle missing values
df = df.fillna({
    'weather': 'UNKNOWN',
    'lighting': 'UNKNOWN',
    'road_condition': 'UNKNOWN',
    'injuries': 0
})

# Feature Engineering
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [1, 7] else 0)  # MongoDB's $dayOfWeek: 1=Sunday, 7=Saturday
df['time_category'] = pd.cut(
    df['hour'], 
    bins=[0, 6, 12, 18, 24], 
    labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']
)

# Visualizations
print("Creating visualizations...")

# 1. Crashes by day of week - IMPROVED
plt.figure(figsize=(12, 8))
day_map = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday', 7: 'Saturday'}
df['day_name'] = df['day_of_week'].map(day_map)
day_counts = df['day_name'].value_counts().reindex(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

# Calculate percentages for annotations
total_crashes = day_counts.sum()
day_percentages = day_counts / total_crashes * 100

# Plot with better colors and annotations
ax = sns.barplot(x=day_counts.index, y=day_counts.values, palette="viridis")
plt.title('Traffic Crashes by Day of Week', fontweight='bold')
plt.ylabel('Number of Crashes', fontweight='bold')
plt.xlabel('Day of Week', fontweight='bold')
plt.xticks(rotation=45)

# Add value labels and percentages on top of bars
for i, (count, percentage) in enumerate(zip(day_counts.values, day_percentages)):
    plt.text(i, count + (total_crashes * 0.01), f'{count:,}', ha='center', fontweight='bold')
    plt.text(i, count/2, f'{percentage:.1f}%', ha='center', color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('improved_visuals/crashes_by_day.png', dpi=300)
plt.close()

# 2. Crashes by hour of day - IMPROVED
plt.figure(figsize=(14, 8))
hour_counts = df['hour'].value_counts().sort_index()

# Calculate moving average for trend line
hour_counts_smooth = hour_counts.rolling(window=3, center=True).mean()

# Plot with enhanced styling
ax = sns.lineplot(x=hour_counts.index, y=hour_counts.values, marker='o', linewidth=3, markersize=10, color=colors[0], label='Hourly crashes')
sns.lineplot(x=hour_counts_smooth.index, y=hour_counts_smooth.values, linewidth=2, linestyle='--', color=colors[5], label='Trend (3-hour moving avg)')

# Mark rush hours
rush_hours_morning = range(7, 10)
rush_hours_evening = range(16, 19)
for h in rush_hours_morning:
    if h in hour_counts.index:
        plt.axvspan(h-0.5, h+0.5, alpha=0.2, color='orange')
for h in rush_hours_evening:
    if h in hour_counts.index:
        plt.axvspan(h-0.5, h+0.5, alpha=0.2, color='orange')

plt.title('Traffic Crashes by Hour of Day', fontweight='bold')
plt.xlabel('Hour (24-hour format)', fontweight='bold')
plt.ylabel('Number of Crashes', fontweight='bold')
plt.xticks(range(0, 24), [f'{h:02d}:00' for h in range(0, 24)], rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

# Annotate peak hours
peak_hour = hour_counts.idxmax()
peak_value = hour_counts.max()
plt.annotate(f'Peak: {peak_value:,} crashes',
             xy=(peak_hour, peak_value), 
             xytext=(peak_hour+1, peak_value+0.1*peak_value),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12,
             fontweight='bold')

plt.tight_layout()
plt.savefig('improved_visuals/crashes_by_hour.png', dpi=300)
plt.close()

# 3. Crashes by weather condition - IMPROVED
plt.figure(figsize=(14, 8))
weather_counts = df['weather'].value_counts().head(10)

# Calculate percentages for the top weather conditions
weather_percentages = (weather_counts / weather_counts.sum() * 100).round(1)

# Create horizontal bar plot with better colors and sorting
ax = sns.barplot(x=weather_counts.values, y=weather_counts.index, palette="viridis", orient='h', order=weather_counts.index)
plt.title('Top 10 Weather Conditions During Crashes', fontweight='bold')
plt.xlabel('Number of Crashes', fontweight='bold')
plt.ylabel('Weather Condition', fontweight='bold')

# Add percentage and count to each bar
for i, (count, pct) in enumerate(zip(weather_counts.values, weather_percentages)):
    plt.text(count + (weather_counts.max() * 0.03), i, f'{count:,} ({pct}%)', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('improved_visuals/crashes_by_weather.png', dpi=300)
plt.close()

# 4. Fatal crashes by time category - IMPROVED
plt.figure(figsize=(12, 8))
fatal_by_time = df.groupby('time_category', observed=True)['fatal'].sum()
total_by_time = df.groupby('time_category', observed=True).size()
percentage_by_time = (fatal_by_time / total_by_time * 100).round(2)

# Create a dual-axis plot - bars for counts, line for percentages
ax1 = sns.barplot(x=fatal_by_time.index, y=fatal_by_time.values, palette="YlOrRd")
plt.title('Fatal Crashes by Time of Day', fontweight='bold')
plt.ylabel('Number of Fatal Crashes', fontweight='bold', color=colors[0])
plt.xlabel('Time of Day', fontweight='bold')

# Create second axis for percentage
ax2 = plt.twinx()
ax2.plot(range(len(percentage_by_time)), percentage_by_time.values, 'o-', linewidth=3, color='darkred')
ax2.set_ylabel('Percentage of Fatal Crashes (%)', fontweight='bold', color='darkred')
ax2.grid(False)

# Add annotations for both counts and percentages
for i, (count, pct) in enumerate(zip(fatal_by_time.values, percentage_by_time.values)):
    ax1.text(i, count + (fatal_by_time.max() * 0.05), f'{count:,}', ha='center', fontweight='bold')
    ax2.text(i, pct + 0.05, f'{pct:.2f}%', ha='center', color='darkred', fontweight='bold')

plt.tight_layout()
plt.savefig('improved_visuals/fatal_by_time.png', dpi=300)
plt.close()

# 5. Crashes by Road Condition - IMPROVED
plt.figure(figsize=(14, 8))
road_counts = df['road_condition'].value_counts().head(10)
road_percentages = (road_counts / road_counts.sum() * 100).round(1)

# Create bar chart with gradient color based on count
ax = sns.barplot(x=road_counts.values, y=road_counts.index, palette="Blues_r", orient='h')
plt.title('Top 10 Road Conditions During Crashes', fontweight='bold')
plt.xlabel('Number of Crashes', fontweight='bold')
plt.ylabel('Road Condition', fontweight='bold')

# Add count and percentage annotations
for i, (count, pct) in enumerate(zip(road_counts.values, road_percentages)):
    plt.text(count + (road_counts.max() * 0.03), i, f'{count:,} ({pct}%)', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('improved_visuals/crashes_by_road_condition.png', dpi=300)
plt.close()

# 6. Crashes by Lighting Condition - IMPROVED
plt.figure(figsize=(14, 8))
light_counts = df['lighting'].value_counts().head(10)
light_percentages = (light_counts / light_counts.sum() * 100).round(1)

# Create a horizontal bar plot with custom coloring
palette = sns.light_palette("purple", n_colors=len(light_counts))
ax = sns.barplot(x=light_counts.values, y=light_counts.index, palette=palette, orient='h')
plt.title('Top 10 Lighting Conditions During Crashes', fontweight='bold')
plt.xlabel('Number of Crashes', fontweight='bold')
plt.ylabel('Lighting Condition', fontweight='bold')

# Add count and percentage annotations
for i, (count, pct) in enumerate(zip(light_counts.values, light_percentages)):
    plt.text(count + (light_counts.max() * 0.03), i, f'{count:,} ({pct}%)', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('improved_visuals/crashes_by_lighting.png', dpi=300)
plt.close()

# Machine Learning: Predicting crash severity (binary: fatal or not)
print("Preparing data for machine learning...")

# Prepare features and target
# Check which categorical features are actually available
available_cat_features = []
for feature in ['weather', 'lighting', 'road_condition', 'time_category']:
    if feature in df.columns:
        available_cat_features.append(feature)
    else:
        print(f"Warning: Feature '{feature}' is not available in the dataset")

print(f"Using these categorical features: {available_cat_features}")

# One-hot encode available categorical features
if available_cat_features:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[available_cat_features])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(available_cat_features))
else:
    print("No categorical features available, skipping one-hot encoding")
    encoded_df = pd.DataFrame()

# Check which numerical features are actually available
available_num_features = []
for feature in ['hour', 'day_of_week', 'is_weekend']:
    if feature in df.columns:
        available_num_features.append(feature)
    else:
        print(f"Warning: Feature '{feature}' is not available in the dataset")

print(f"Using these numerical features: {available_num_features}")

# Combine with numerical features
if available_num_features:
    num_df = df[available_num_features].reset_index(drop=True)
    # Only concatenate if both DataFrames have data
    if not encoded_df.empty:
        X = pd.concat([num_df, encoded_df], axis=1)
    else:
        X = num_df
else:
    if not encoded_df.empty:
        X = encoded_df
    else:
        print("Error: No features available for machine learning")
        exit(1)
y = df['fatal']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training Random Forest model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
try:
    print(classification_report(y_test, y_pred))
except:
    print("Could not generate classification report - likely due to only one class in the test set")

# Create confusion matrix - IMPROVED
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)

# Create a more informative confusion matrix
ax = sns.heatmap(cm, annot=True, fmt=',.0f', cmap='Blues', cbar=False, annot_kws={"size": 16})

# Add descriptive labels and metrics
category_names = ['Non-Fatal', 'Fatal']
plt.ylabel('Actual', fontweight='bold')
plt.xlabel('Predicted', fontweight='bold')
tick_marks = np.arange(len(category_names)) + 0.5
plt.xticks(tick_marks, category_names)
plt.yticks(tick_marks, category_names)

# Add title and labels
plt.title('Confusion Matrix', fontweight='bold', size=16)

# Add performance metrics
tn, fp, fn, tp = cm.ravel()
total = np.sum(cm)
accuracy = (tn + tp) / total
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

plt.text(0.5, -0.15, f"Accuracy: {accuracy:.4f}", ha='center', transform=ax.transAxes, fontsize=14)
plt.text(0.5, -0.2, f"Sensitivity: {sensitivity:.4f}", ha='center', transform=ax.transAxes, fontsize=14)
plt.text(0.5, -0.25, f"Specificity: {specificity:.4f}", ha='center', transform=ax.transAxes, fontsize=14)
plt.text(0.5, -0.3, f"Precision: {precision:.4f}", ha='center', transform=ax.transAxes, fontsize=14)

plt.tight_layout()
plt.savefig('improved_visuals/confusion_matrix.png', dpi=300)
plt.close()

# Feature importance - IMPROVED
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot top 15 features with better visualization
plt.figure(figsize=(14, 10))
sns.set_color_codes("pastel")
ax = sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15), palette='viridis')
plt.title('Top 15 Important Features for Predicting Fatal Crashes', fontweight='bold', size=16)
plt.xlabel('Relative Importance', fontweight='bold')
plt.ylabel('Feature', fontweight='bold')

# Add value annotations
for i, importance in enumerate(feature_importance.head(15)['Importance']):
    plt.text(importance + 0.01, i, f'{importance:.4f}', va='center', fontweight='bold')

# Add grid lines
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('improved_visuals/feature_importance.png', dpi=300)
plt.close()

# 7. Additional analysis: Injury severity by hour - IMPROVED
plt.figure(figsize=(14, 8))
hourly_injuries = df.groupby('hour')['injuries'].mean().reset_index()
hours = hourly_injuries['hour']
injuries = hourly_injuries['injuries']

# Calculate moving average trend
hourly_injuries['injuries_smooth'] = hourly_injuries['injuries'].rolling(window=3, center=True).mean()

# Create a more attractive visualization
ax = sns.lineplot(x='hour', y='injuries', data=hourly_injuries, marker='o', linewidth=3, markersize=10, color=colors[3], label='Average Injuries')
sns.lineplot(x='hour', y='injuries_smooth', data=hourly_injuries, linewidth=2, linestyle='--', color=colors[7], label='Trend (3-hour moving avg)')

# Add shaded regions for night hours
plt.axvspan(0, 6, alpha=0.2, color='gray', label='Night Hours')
plt.axvspan(22, 24, alpha=0.2, color='gray')

# Highlight peak injury hours
peak_hour = hourly_injuries.loc[hourly_injuries['injuries'].idxmax(), 'hour']
peak_injury = hourly_injuries['injuries'].max()
plt.axvline(x=peak_hour, color='red', linestyle='--')
plt.annotate(f'Peak: {peak_injury:.2f} injuries/crash',
             xy=(peak_hour, peak_injury), 
             xytext=(peak_hour+1, peak_injury*1.1),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=12,
             fontweight='bold')

plt.title('Average Injuries per Crash by Hour of Day', fontweight='bold', size=16)
plt.xlabel('Hour (24-hour format)', fontweight='bold')
plt.ylabel('Average Number of Injuries per Crash', fontweight='bold')
plt.xticks(range(0, 24), [f'{h:02d}:00' for h in range(0, 24)], rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('improved_visuals/injuries_by_hour.png', dpi=300)
plt.close()

# 8. Fatal crashes by weather condition - IMPROVED
plt.figure(figsize=(14, 10))
fatal_by_weather = df.groupby('weather')['fatal'].sum().sort_values(ascending=False).head(10)
total_by_weather = df.groupby('weather').size().loc[fatal_by_weather.index]
fatality_rate = (fatal_by_weather / total_by_weather * 1000).round(2)  # Fatalities per 1000 crashes

# Create a single figure
plt.figure(figsize=(14, 10))

# Create primary axis for fatal crash counts
primary_color = 'tab:red'
plt.barh(fatal_by_weather.index, fatal_by_weather.values, color=sns.color_palette("YlOrRd", n_colors=len(fatal_by_weather)))
plt.xlabel('Number of Fatal Crashes', fontweight='bold', color=primary_color)
plt.ylabel('Weather Condition', fontweight='bold')
plt.title('Top 10 Weather Conditions for Fatal Crashes', fontweight='bold', size=16)

# Add count annotations to the bars
for i, count in enumerate(fatal_by_weather.values):
    plt.text(count + max(fatal_by_weather.values) * 0.01, i, f'{count:,}', va='center', fontweight='bold')

# Create a secondary axis for fatality rate
ax2 = plt.gca().twiny()
ax2.set_xlabel('Fatality Rate (per 1,000 crashes)', fontweight='bold', color='darkblue')
ax2.grid(False)

# Normalize fatality rate scale for better visualization
max_rate = max(fatality_rate.values)
max_count = max(fatal_by_weather.values)
scale_factor = max_count / max_rate if max_rate > 0 else 1
normalized_rates = fatality_rate.values * scale_factor

# Plot fatality rate with markers
for i, (rate, norm_rate) in enumerate(zip(fatality_rate.values, normalized_rates)):
    # Add a marker for each rate value
    ax2.plot([0, rate], [i, i], 'o-', linewidth=2, color='darkblue', alpha=0.6)
    # Add text for the rate value
    ax2.text(rate + max_rate * 0.05, i, f'{rate:.2f}', va='center', ha='left', color='darkblue', fontweight='bold')

# Set the limits for the secondary x-axis
ax2.set_xlim(0, max_rate * 1.2)

# Add a legend
from matplotlib.lines import Line2D
legend_elements = [
    plt.Rectangle((0,0), 1, 1, color=sns.color_palette("YlOrRd")[3], label='Fatal Crash Count'),
    Line2D([0], [0], marker='o', color='darkblue', label='Fatality Rate (per 1,000 crashes)', markersize=10, linewidth=2)
]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('improved_visuals/fatal_by_weather.png', dpi=300)
plt.close()

print("Analysis complete! All improved visualizations saved to 'improved_visuals' directory.")